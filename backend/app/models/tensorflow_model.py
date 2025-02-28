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
from .plate_correction import correct_candidates

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
        
        # Try multiple preprocessing approaches and select the best result
        preprocessed_images = []
        
        # Approach 1: Minimal preprocessing (similar to OpenCV pipeline)
        # Just use grayscale with minor resize
        img1 = cv2.resize(gray, None, fx=1.2, fy=1.2)
        preprocessed_images.append(img1)
        
        # Approach 2: Current modified approach (reduced scale, no blur)
        img2 = cv2.resize(gray, None, fx=1.5, fy=1.5)
        img2 = cv2.equalizeHist(img2)
        img2 = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        preprocessed_images.append(img2)
        
        # Approach 3: Adaptive thresholding instead of Otsu
        img3 = cv2.resize(gray, None, fx=1.5, fy=1.5)
        img3 = cv2.equalizeHist(img3)
        img3 = cv2.adaptiveThreshold(img3, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        preprocessed_images.append(img3)
        
        # Process each preprocessed image with OCR and keep the best result
        best_confidence = 0.0
        best_result = None
        best_candidates = []
        
        # Define character similarity mappings for suggestions
        char_similarity = {
            '0': ['O', 'D', 'Q', '8', 'C'],
            '1': ['I', 'L', 'T', '7', 'J'],
            '2': ['Z', 'S', '7', 'R'],
            '3': ['8', 'B', 'E', 'S'],
            '4': ['A', 'H', 'M', 'N'],
            '5': ['S', '6', 'G', 'B'],
            '6': ['G', 'C', 'B', '8'],
            '7': ['T', '1', 'L', 'Y'],
            '8': ['B', '6', '3', 'S', '0'],
            '9': ['G', 'Q', 'D', 'P'],
            'A': ['4', 'H', 'M', 'N', 'V'],
            'B': ['8', '3', 'R', 'S', 'D'],
            'C': ['G', 'O', 'Q', '0', 'D'],
            'D': ['O', '0', 'Q', 'C', 'B'],
            'E': ['F', 'B', '3', 'R', 'M'],
            'F': ['E', 'P', 'B', 'R'],
            'G': ['C', '6', 'Q', 'O', '0'],
            'H': ['N', 'M', 'K', 'A', 'X'],
            'I': ['1', 'L', 'T', 'J', '!'],
            'J': ['I', 'L', '1', 'T', 'U'],
            'K': ['X', 'H', 'R', 'N', 'M'],
            'L': ['I', '1', 'T', 'J', 'F'],
            'M': ['N', 'H', 'A', 'W', 'K'],
            'N': ['M', 'H', 'K', 'A', 'X'],
            'O': ['0', 'Q', 'D', 'C', 'G'],
            'P': ['R', 'F', 'D', '9', 'B'],
            'Q': ['O', '0', 'G', 'D', '9'],
            'R': ['B', 'P', 'F', 'K', '2'],
            'S': ['5', '2', '8', 'B', 'Z'],
            'T': ['1', 'I', 'L', '7', 'Y'],
            'U': ['V', 'Y', 'W', 'J', 'O'],
            'V': ['U', 'Y', 'W', 'A', 'N'],
            'W': ['M', 'N', 'V', 'U', 'H'],
            'X': ['K', 'H', 'N', 'M', 'Y'],
            'Y': ['V', 'U', 'T', '7', 'X'],
            'Z': ['2', 'S', '7', 'N', 'M']
        }
        
        for img_idx, processed_img in enumerate(preprocessed_images):
            # Perform OCR with allow_list to restrict to alphanumeric characters
            result = reader.readtext(processed_img, detail=1, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
            
            candidates = []
            
            if result:
                # Process all detected text regions
                for detection in result:
                    bbox, text, confidence = detection
                    # Clean the text (remove spaces and unwanted characters)
                    cleaned_text = ''.join(c for c in text if c.isalnum())
                    
                    if cleaned_text:  # Only add non-empty text
                        # Create a list to hold character positions with their candidates
                        char_positions = []
                        
                        # Extract text region for character-level analysis
                        y1, x1 = int(max(0, bbox[0][1])), int(max(0, bbox[0][0]))
                        y2, x2 = int(min(processed_img.shape[0], bbox[2][1])), int(min(processed_img.shape[1], bbox[2][0]))
                        text_region = processed_img[y1:y2, x1:x2] if y2 > y1 and x2 > x1 else processed_img
                        
                        # Process each character in the text
                        for i, char in enumerate(cleaned_text):
                            # Create an array of character candidates for this position
                            position_candidates = []
                            
                            # Add the primary detected character first
                            position_candidates.append({
                                "char": char,
                                "confidence": float(confidence)
                            })
                            
                            # Calculate character's rough position in text region
                            char_width = text_region.shape[1] // max(1, len(cleaned_text))
                            start_x = max(0, i * char_width - 2)
                            end_x = min(text_region.shape[1], (i + 1) * char_width + 2)
                            
                            # Try to extract just this character for OCR
                            if end_x > start_x and text_region.size > 0:
                                char_img = text_region[:, start_x:end_x]
                                if char_img.size > 0:
                                    try:
                                        # Try to get character-level OCR results
                                        char_results = reader.readtext(char_img, detail=1, 
                                                                    allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                                        
                                        seen_chars = {char}  # Track seen characters to avoid duplicates
                                        
                                        # Add any detected characters from the character region
                                        for char_det in char_results:
                                            detected_char = char_det[1].strip()
                                            if detected_char and len(detected_char) == 1 and detected_char not in seen_chars:
                                                position_candidates.append({
                                                    "char": detected_char,
                                                    "confidence": float(char_det[2])
                                                })
                                                seen_chars.add(detected_char)
                                    except Exception as e:
                                        logger.error(f"Error in character OCR for {char}: {e}")
                            
                            # Add similar looking characters based on mapping
                            if char in char_similarity:
                                for idx, similar_char in enumerate(char_similarity[char]):
                                    if similar_char not in [c["char"] for c in position_candidates]:
                                        # Decrease confidence for each alternative
                                        alt_confidence = max(0.1, float(confidence) * (0.9 - 0.1 * idx))
                                        position_candidates.append({
                                            "char": similar_char,
                                            "confidence": alt_confidence
                                        })
                            
                            # Sort by confidence and keep top 5
                            position_candidates.sort(key=lambda x: x["confidence"], reverse=True)
                            position_candidates = position_candidates[:5]
                            
                            # Add to position list
                            char_positions.append({
                                "position": i,
                                "candidates": position_candidates
                            })
                        
                        # Check if text matches any common license plate pattern
                        pattern_match, matching_pattern = matches_pattern(cleaned_text)
                        pattern_boost = 0.1 if pattern_match else 0.0
                        
                        # Add to candidates
                        candidates.append({
                            "text": cleaned_text,
                            "confidence": float(confidence) + pattern_boost,
                            "pattern_match": pattern_match,
                            "char_positions": char_positions
                        })
                
                # Record the best result based on confidence
                if candidates and candidates[0]["confidence"] > best_confidence:
                    best_confidence = candidates[0]["confidence"]
                    best_result = candidates[0]["text"]
                    best_candidates = candidates
                    
                    # Store which preprocessing method worked best
                    for candidate in best_candidates:
                        candidate["preprocessing_method"] = img_idx
        
        # Apply text correction and pattern matching to our candidates
        if best_candidates:
            corrected_candidates = correct_candidates(best_candidates)
            # Update best result based on corrected candidates
            best_result = corrected_candidates[0]["text"]
            return best_result, corrected_candidates
        
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
