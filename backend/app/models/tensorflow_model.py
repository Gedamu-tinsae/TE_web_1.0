import tensorflow as tf
import numpy as np
import cv2
import os
import logging
import easyocr
import base64
import re
import string
from difflib import SequenceMatcher

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_dirs():
    """Ensure all required directories exist"""
    dirs = [
        "uploads/tensorflow/images",
        "uploads/tensorflow/videos",
        "results/tensorflow/images",
        "results/tensorflow/videos",
        "results/tensorflow/intermediate/images",
        "results/tensorflow/intermediate/videos"
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

# Load the TensorFlow model
model_path = os.path.join(os.path.dirname(__file__), 'saved_model')
if not os.path.exists(model_path):
    logger.error(f"Model path does not exist: {model_path}")
    raise FileNotFoundError(f"Model path does not exist: {model_path}")

model = tf.saved_model.load(model_path)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Define common license plate patterns
LICENSE_PATTERNS = [
    # Format: AB11 ABC
    r'^[A-Z]{2}\d{2}\s?[A-Z]{3}$',
    # Format: ABCDE
    r'^[A-Z]{5}$', 
    # Format: AB11 AB11
    r'^[A-Z]{2}\d{2}\s?[A-Z]{2}\d{2}$',
    # Format: 11 AB 1C 1111
    r'^\d{2}\s?[A-Z]{2}\s?\d{1}[A-Z]{1}\s?\d{4}$',
    # Additional common formats
    r'^[A-Z]{1,3}\d{1,4}$',  # Format: ABC1234
    r'^\d{1,4}[A-Z]{1,3}$',  # Format: 1234ABC
    r'^[A-Z]{2}\d{2}[A-Z]{2}$',  # Format: AB12CD
    r'^[A-Z]{3}\d{3}$',  # Format: ABC123
    r'^[A-Z]{2}\d{3}[A-Z]{2}$',  # Format: AB123CD
    r'^COVID\d{2}$',  # Special case for COVID19
]

def similarity_score(text1, text2):
    """Calculate similarity between two strings."""
    return SequenceMatcher(None, text1, text2).ratio()

def matches_pattern(text):
    """Check if text matches any of the common license plate patterns."""
    # Clean text for pattern matching (remove spaces)
    cleaned_text = ''.join(text.split())
    
    for pattern in LICENSE_PATTERNS:
        if re.match(pattern, cleaned_text):
            return True, pattern
    return False, None

def extract_text_from_plate(plate_region):
    try:
        # Preprocess the plate image for better OCR
        gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
        # Apply some basic image processing
        gray = cv2.resize(gray, None, fx=2, fy=2)  # Upscale
        gray = cv2.GaussianBlur(gray, (5,5), 0)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # Perform OCR with allow_list to restrict to alphanumeric characters
        result = reader.readtext(gray, detail=1, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        
        candidates = []
        
        if result:
            # Process all detected text regions
            for detection in result:
                bbox, text, confidence = detection
                # Clean the text (remove spaces and unwanted characters)
                cleaned_text = ''.join(c for c in text if c.isalnum())
                
                if cleaned_text:  # Only add non-empty text
                    # Get individual character confidences
                    text_region = gray[int(bbox[0][1]):int(bbox[2][1]), int(bbox[0][0]):int(bbox[2][0])]
                    
                    # For each character position, we'll store multiple alternative candidates
                    per_char_data = []
                    
                    # Process each character position
                    for i, char in enumerate(cleaned_text):
                        # Calculate character's horizontal position in the text region
                        width = text_region.shape[1]
                        char_width = width // len(cleaned_text)
                        
                        # Extract character region with some overlap
                        start_x = max(0, i * char_width - 2)
                        end_x = min(width, (i + 1) * char_width + 2)
                        if end_x <= start_x:
                            continue
                        
                        char_img = text_region[:, start_x:end_x]
                        if char_img.size == 0:
                            # Skip empty regions
                            per_char_data.append({
                                "position": i,
                                "candidates": [
                                    {"char": char, "confidence": float(confidence) * 0.8}
                                ]
                            })
                            continue
                        
                        # For this position, analyze all possible characters
                        try:
                            # Run OCR on just this character region with detailed results
                            char_candidates = []
                            
                            # Try to get all possible characters for this position
                            char_results = reader.readtext(
                                char_img, 
                                detail=1,
                                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                                batch_size=True,
                                decoder='beamsearch',
                                beamWidth=10  # This helps get multiple candidates
                            )
                            
                            if char_results:
                                # Process all detected characters for this position
                                seen_chars = set()
                                
                                # First add the expected character from the original text
                                char_candidates.append({
                                    "char": char,
                                    "confidence": float(confidence) * 0.8  # Default confidence
                                })
                                seen_chars.add(char)
                                
                                # Add all alternative detected characters
                                for char_det in char_results:
                                    detected_char = char_det[1].strip()
                                    if detected_char and len(detected_char) == 1 and detected_char not in seen_chars:
                                        char_conf = float(char_det[2])
                                        char_candidates.append({
                                            "char": detected_char,
                                            "confidence": char_conf
                                        })
                                        seen_chars.add(detected_char)
                                
                                # If not enough candidates, generate more by running OCR with different parameters
                                if len(char_candidates) < 5:
                                    # Try with different preprocessing
                                    char_img_inv = cv2.bitwise_not(char_img)
                                    additional_results = reader.readtext(
                                        char_img_inv,
                                        detail=1,
                                        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                                    )
                                    
                                    for char_det in additional_results:
                                        detected_char = char_det[1].strip()
                                        if detected_char and len(detected_char) == 1 and detected_char not in seen_chars:
                                            char_conf = float(char_det[2]) * 0.9  # Slightly lower confidence
                                            char_candidates.append({
                                                "char": detected_char,
                                                "confidence": char_conf
                                            })
                                            seen_chars.add(detected_char)
                                
                                # Generate similar looking characters for remaining slots
                                char_similarity = {
                                    '0': ['O', 'D', 'Q'],
                                    '1': ['I', 'L', 'T'],
                                    '2': ['Z', 'S'],
                                    '5': ['S', '6'],
                                    '8': ['B', '6'],
                                    'B': ['8', 'R', 'E'],
                                    'D': ['O', '0'],
                                    'G': ['6', 'C'],
                                    'I': ['1', 'L'],
                                    'O': ['0', 'Q', 'D'],
                                    'S': ['5', '2'],
                                    'Z': ['2', '7']
                                }
                                
                                if char in char_similarity and len(char_candidates) < 5:
                                    for similar_char in char_similarity[char]:
                                        if similar_char not in seen_chars:
                                            char_candidates.append({
                                                "char": similar_char,
                                                "confidence": float(confidence) * 0.6
                                            })
                                            seen_chars.add(similar_char)
                                            if len(char_candidates) >= 5:
                                                break
                                
                                # Sort candidates by confidence
                                char_candidates.sort(key=lambda x: x["confidence"], reverse=True)
                                
                                # Keep only top 5
                                char_candidates = char_candidates[:5]
                                
                                # Store for this position
                                per_char_data.append({
                                    "position": i,
                                    "candidates": char_candidates
                                })
                            else:
                                # Fallback if no characters detected
                                per_char_data.append({
                                    "position": i,
                                    "candidates": [
                                        {"char": char, "confidence": float(confidence) * 0.7}
                                    ]
                                })
                        except Exception as e:
                            logger.error(f"Error in character OCR: {e}")
                            # On error, use a default confidence
                            per_char_data.append({
                                "position": i,
                                "candidates": [
                                    {"char": char, "confidence": float(confidence) * 0.6}
                                ]
                            })
                    
                    # Check if text matches any common license plate pattern
                    pattern_match, matching_pattern = matches_pattern(cleaned_text)
                    pattern_boost = 0.1 if pattern_match else 0.0
                    
                    # Add to candidates
                    candidates.append({
                        "text": cleaned_text,
                        "confidence": float(confidence) + pattern_boost,
                        "pattern_match": pattern_match,
                        "char_positions": per_char_data  # Now contains multiple candidates per position
                    })
            
            # If we found valid text candidates, return them
            if candidates:
                # Sort by confidence (highest first)
                candidates.sort(key=lambda x: x["confidence"], reverse=True)
                # Return the top candidate text for backwards compatibility
                return candidates[0]["text"], candidates
        
        # Return default values if no valid text found
        return "Unknown", [{"text": "Unknown", "confidence": 0.0, "char_positions": []}]
    except Exception as e:
        logger.error(f"Error in OCR: {e}")
        return "Error", [{"text": "Error", "confidence": 0.0, "char_positions": []}]

def process_image_with_model(file_path):
    try:
        ensure_dirs()
        # Load the image
        original_image = cv2.imread(file_path)
        if original_image is None:
            logger.error("Failed to load image.")
            raise ValueError("Failed to load image.")
        logger.info("Image loaded successfully")

        # Make a copy for detection visualization
        image = original_image.copy()
        image_np = np.array(image)
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.uint8)

        detections = model(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
        detections['num_detections'] = num_detections
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        # Image for showing detections without OCR text
        detection_image = image.copy()
        
        localized_images = []
        extracted_texts = []
        text_candidates = []
        
        for i in range(num_detections):
            if detections['detection_scores'][i] > 0.7:  # Confidence threshold
                box = detections['detection_boxes'][i]
                h, w, _ = image.shape
                y_min, x_min, y_max, x_max = box
                x_min, x_max = int(x_min * w), int(x_max * w)
                y_min, y_max = int(y_min * h), int(y_max * h)
                
                # Draw rectangle on detection image
                cv2.rectangle(detection_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                # Extract and save plate region
                localized_plate = image[y_min:y_max, x_min:x_max]
                localized_images.append(localized_plate)
                
                # Extract text with candidates
                plate_text, candidates = extract_text_from_plate(localized_plate)
                extracted_texts.append(plate_text)
                text_candidates.append(candidates)
                
                # Draw rectangle and text on final image
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(image, plate_text, 
                          (x_min, y_min - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 
                          0.9,
                          (255, 255, 255),
                          2)

        # Save intermediate results
        base_name = os.path.basename(file_path)
        intermediate_dir = os.path.join("results", "tensorflow", "intermediate", "images")
        
        # Save original image
        original_path = os.path.join(intermediate_dir, f"1_original_{base_name}")
        cv2.imwrite(original_path, original_image)
        
        # Save detection image
        detection_path = os.path.join(intermediate_dir, f"2_detection_{base_name}")
        cv2.imwrite(detection_path, detection_image)
        
        # Save plate regions
        plate_paths = []
        for idx, plate in enumerate(localized_images):
            plate_path = os.path.join(intermediate_dir, f"3_plate_{idx}_{base_name}")
            cv2.imwrite(plate_path, plate)
            plate_paths.append(f"/results/tensorflow/intermediate/images/3_plate_{idx}_{base_name}")

        # Save final result
        final_path = os.path.join("results", "tensorflow", "images", base_name)
        cv2.imwrite(final_path, image)

        result = {
            "status": "success",
            "filename": file_path,
            "result_url": f"/results/tensorflow/images/{base_name}",
            "intermediate_steps": {
                "original": f"/results/tensorflow/intermediate/images/1_original_{base_name}",
                "detection": f"/results/tensorflow/intermediate/images/2_detection_{base_name}",
                "plates": plate_paths
            },
            "detected_plates": extracted_texts,
            "license_plate": extracted_texts[0] if extracted_texts else "Unknown",
            "text_candidates": text_candidates[0] if text_candidates else []  # Ensure this is a direct array, not nested
        }

        return result
    except Exception as e:
        logger.error(f"Error processing image with model: {e}")
        raise

def process_video_with_model(file_path):
    try:
        ensure_dirs()
        cap = cv2.VideoCapture(file_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_number = 0
        results = []
        all_extracted_texts = []
        all_intermediate_frames = {
            "original": [],
            "detection": [],
            "plates": []
        }

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_number += 1
            logger.info(f"Processing frame {frame_number}/{frame_count}")

            # Store original frame
            original_frame = frame.copy()

            # Convert frame to tensor and detect plates
            image_np = np.array(frame)
            input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.uint8)
            detections = model(input_tensor)

            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
            detections['num_detections'] = num_detections
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

            # Image for showing detections without OCR text
            detection_frame = frame.copy()
            frame_plates = []
            frame_texts = []
            frame_candidates = []

            for i in range(num_detections):
                if detections['detection_scores'][i] > 0.7:
                    box = detections['detection_boxes'][i]
                    h, w, _ = frame.shape
                    y_min, x_min, y_max, x_max = box
                    x_min, x_max = int(x_min * w), int(x_max * w)
                    y_min, y_max = int(y_min * h), int(y_max * h)

                    # Draw rectangle on detection frame
                    cv2.rectangle(detection_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                    # Extract plate region
                    plate = frame[y_min:y_max, x_min:x_max]
                    frame_plates.append(plate)

                    # Extract text from plate with candidates
                    plate_text, candidates = extract_text_from_plate(plate)
                    frame_texts.append(plate_text)
                    frame_candidates.append(candidates)

                    # Draw rectangle and text on final frame
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(frame, plate_text,
                              (x_min, y_min - 10),
                              cv2.FONT_HERSHEY_SIMPLEX,
                              0.9,
                              (255, 255, 255),
                              2)

            # Store intermediate results for this frame
            all_intermediate_frames["original"].append(original_frame)
            all_intermediate_frames["detection"].append(detection_frame)
            all_intermediate_frames["plates"].extend(frame_plates)
            all_extracted_texts.extend(frame_texts)

            # Add processed frame to results
            results.append(frame)

        cap.release()
        if results:
            # Save the processed video
            result_video_path = os.path.join("results", "tensorflow", "videos", os.path.basename(file_path))
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            frame_size = (results[0].shape[1], results[0].shape[0])
            out = cv2.VideoWriter(result_video_path, fourcc, fps, frame_size)

            for frame in results:
                out.write(frame)
            out.release()
            logger.info(f"Processed video saved at: {result_video_path}")

            # Encode sample frames and plates as base64 for frontend display
            def encode_image(img):
                _, buffer = cv2.imencode('.jpg', img)
                return base64.b64encode(buffer).decode('utf-8')

            # Take a sample frame from each category for display
            sample_frame_index = min(10, len(results) - 1)  # Take 10th frame or last frame if video is shorter
            intermediate_images = {
                "original": encode_image(all_intermediate_frames["original"][sample_frame_index]),
                "detection": encode_image(all_intermediate_frames["detection"][sample_frame_index]),
                "plates": [encode_image(plate) for plate in all_intermediate_frames["plates"][:5]]  # Limit to first 5 plates
            }

            result = {
                "status": "success",
                "filename": file_path,
                "result_url": f"/results/tensorflow/videos/{os.path.basename(file_path)}",
                "result_image": encode_image(results[sample_frame_index]),  # Sample frame from result
                "intermediate_images": intermediate_images,
                "detected_plates": list(set(all_extracted_texts)),  # Remove duplicates
                "license_plate": all_extracted_texts[0] if all_extracted_texts else "Unknown",
                "text_candidates": frame_candidates[0] if frame_candidates else []  # Ensure this is a direct array, not nested
            }

            return result
        else:
            raise Exception("No frames were processed successfully.")
    except Exception as e:
        logger.error(f"Error processing video with model: {e}")
        raise
