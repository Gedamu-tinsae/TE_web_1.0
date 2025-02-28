import cv2
import numpy as np
import easyocr
import os
import imutils
import logging
import json
import base64
import re
from difflib import SequenceMatcher

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def matches_pattern(text):
    """Check if text matches any of the common license plate patterns."""
    # Clean text for pattern matching (remove spaces)
    cleaned_text = ''.join(text.split())
    
    for pattern in LICENSE_PATTERNS:
        if re.match(pattern, cleaned_text):
            return True, pattern
    return False, None

def ensure_dirs():
    """Ensure all required directories exist"""
    dirs = [
        "uploads/opencv/images",
        "uploads/opencv/videos",
        "results/opencv/images",
        "results/opencv/videos"
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

def process_image(file_path):
    try:
        ensure_dirs()
        # Load the image
        image = cv2.imread(file_path)
        if image is None:
            logger.error("Failed to load image.")
            raise ValueError("Failed to load image.")
        logger.info("Image loaded successfully")

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        logger.info("Converted to grayscale")

        # Apply bilateral filter and Canny edge detection
        bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
        edges = cv2.Canny(bfilter, 30, 200)
        logger.info("Applied bilateral filter and Canny edge detection")

        # Find contours and locate the license plate
        keypoints = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(keypoints)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        logger.info(f"Number of contours found: {len(contours)}")
        location = None
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 10, True)
            logger.info(f"Contour length: {len(approx)}")
            if len(approx) == 4:
                location = approx
                break

        if location is None:
            logger.warning("No valid contour with 4 points found. Trying alternative method.")
            for contour in contours:
                if cv2.contourArea(contour) > 100:  # Filter out small contours
                    location = cv2.convexHull(contour)
                    break

        # Validate the contour points
        if location is not None and location.shape[0] > 0 and location.dtype == np.int32:
            # Create a mask and extract the license plate region
            mask = np.zeros(gray.shape, np.uint8)
            new_image = cv2.drawContours(mask, [location], 0, 255, -1)
            new_image = cv2.bitwise_and(image, image, mask=mask)
            (x, y) = np.where(mask == 255)
            if x.size == 0 or y.size == 0:
                logger.error("Failed to locate the license plate region.")
                raise ValueError("Failed to locate the license plate region.")
            (x1, y1) = (np.min(x), np.min(y))
            (x2, y2) = (np.max(x), np.max(y))
            cropped_image = gray[x1:x2+1, y1:y2+1]
            logger.info("License plate region extracted")

            # Use EasyOCR to read the text with enhanced detail
            reader = easyocr.Reader(['en'])
            result = reader.readtext(cropped_image, detail=1, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
            
            # Process and store all text candidates with confidence
            text_candidates = []
            if result:
                for detection in result:
                    bbox, text, confidence = detection
                    # Clean the text (remove unwanted characters, spaces)
                    cleaned_text = ''.join(c for c in text if c.isalnum())
                    
                    if cleaned_text:
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
                        
                        # For each detected text, analyze characters individually
                        # Create a blank mask to highlight the text region
                        mask = np.zeros(cropped_image.shape, dtype=np.uint8)
                        cv2.rectangle(mask, (int(bbox[0][0]), int(bbox[0][1])), 
                                     (int(bbox[2][0]), int(bbox[2][1])), 255, -1)
                        text_region = cv2.bitwise_and(cropped_image, mask)
                        
                        # Process character-level confidences with alternatives
                        char_positions = []
                        
                        for i, char in enumerate(cleaned_text):
                            # Create a list of candidate characters for this position
                            candidates = []
                            
                            # Add the primary character (from the detected text)
                            candidates.append({
                                "char": char,
                                "confidence": float(confidence)
                            })
                            
                            # Add similar characters based on the mapping
                            if char in char_similarity:
                                for similar_char in char_similarity[char]:
                                    # Add with decreasing confidence
                                    similarity_factor = 0.8 - 0.1 * char_similarity[char].index(similar_char)
                                    candidates.append({
                                        "char": similar_char,
                                        "confidence": float(confidence) * similarity_factor
                                    })
                            
                            # Try to get position-specific OCR if possible
                            try:
                                # Get more detailed character detection
                                if len(text_region.shape) > 0:
                                    width = text_region.shape[1]
                                    char_width = width // len(cleaned_text)
                                    start_x = max(0, i * char_width - 2)
                                    end_x = min(width, (i + 1) * char_width + 2)
                                    if end_x > start_x:
                                        char_img = text_region[:, start_x:end_x]
                                        char_results = reader.readtext(char_img, detail=1, 
                                                                     allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                                        
                                        # Add any new character detections
                                        seen_chars = {candidate["char"] for candidate in candidates}
                                        for char_det in char_results:
                                            detected_char = char_det[1].strip()
                                            if detected_char and len(detected_char) == 1 and detected_char not in seen_chars:
                                                candidates.append({
                                                    "char": detected_char,
                                                    "confidence": float(char_det[2])
                                                })
                                                seen_chars.add(detected_char)
                            except Exception as e:
                                logger.error(f"Error in character OCR: {e}")
                            
                            # Sort by confidence and keep top 5
                            candidates.sort(key=lambda x: x["confidence"], reverse=True)
                            candidates = candidates[:5]
                            
                            # Add to character positions
                            char_positions.append({
                                "position": i,
                                "candidates": candidates
                            })
                        
                        # Check if text matches any common license plate pattern
                        pattern_match, matching_pattern = matches_pattern(cleaned_text)
                        pattern_boost = 0.1 if pattern_match else 0.0
                        
                        # Add to main text candidates
                        text_candidates.append({
                            "text": cleaned_text,
                            "confidence": float(confidence) + pattern_boost,
                            "pattern_match": pattern_match,
                            "char_positions": char_positions
                        })
                
                # Sort by confidence (highest first)
                text_candidates.sort(key=lambda x: x["confidence"], reverse=True)
                
                # Get the highest confidence text
                license_plate = text_candidates[0]["text"] if text_candidates else "Unknown"
            else:
                license_plate = "Unknown"
                text_candidates.append({
                    "text": "Unknown", 
                    "confidence": 0.0,
                    "pattern_match": False,
                    "char_positions": []
                })
                
            logger.info(f"License plate text extracted: {license_plate}")

            # Annotate the image with the license plate text and rectangle
            font = cv2.FONT_HERSHEY_SIMPLEX
            annotated_image = cv2.putText(image, text=license_plate, org=(location[0][0][0], location[1][0][1]+60), fontFace=font, fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
            annotated_image = cv2.rectangle(image, tuple(location[0][0]), tuple(location[2][0]), (0,255,0), 3)
            logger.info("Annotated image with license plate text and rectangle")

            # Save the annotated image in the opencv/images subfolder
            result_image_path = os.path.join("results", "opencv", "images", os.path.basename(file_path))
            cv2.imwrite(result_image_path, annotated_image)
            logger.info(f"Annotated image saved at: {result_image_path}")

            # Load customer data from example.json
            example_json_path = os.path.join(os.path.dirname(__file__), '../api/endpoints/example.json')
            logger.info(f"Loading customer data from: {example_json_path}")
            with open(example_json_path, 'r') as f:
                customer_data = json.load(f)
            logger.info("Customer data loaded successfully")

            # Encode intermediate images as base64 strings
            def encode_image(image):
                _, buffer = cv2.imencode('.jpg', image)
                return base64.b64encode(buffer).decode('utf-8')

            intermediate_images = {
                "gray": encode_image(gray),
                "edge": encode_image(edges),
                "localized": encode_image(new_image),
                "plate": encode_image(cropped_image),
            }

            # Example result - ensure text_candidates is a direct array, not nested in another array
            result = {
                "status": "success",
                "result_url": f"/results/opencv/images/{os.path.basename(result_image_path)}",
                "intermediate_images": intermediate_images,
                "license_plate": license_plate,
                "filename": file_path,
                "customer_data": customer_data,
                "text_candidates": text_candidates[:10]  # Include top 10 candidates as a direct array
            }

            return result
        else:
            logger.error("Invalid or empty contour points.")
            raise ValueError("Invalid or empty contour points.")
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON: {e}")
        raise
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise

def process_video(file_path):
    try:
        ensure_dirs()
        cap = cv2.VideoCapture(file_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0  # Get the original FPS or default to 30.0
        frame_number = 0
        results = []
        max_frames = 40  # Limit the number of frames to process

        while cap.isOpened() and frame_number < max_frames:
            ret, frame = cap.read()
            if not ret or frame is None:  # Add check for None frame
                break

            frame_number += 1
            logger.info(f"Processing frame {frame_number}/{min(frame_count, max_frames)}")

            # Process each frame (similar to process_image function)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
            edges = cv2.Canny(bfilter, 30, 200)
            keypoints = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(keypoints)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
            location = None
            for contour in contours:
                approx = cv2.approxPolyDP(contour, 10, True)
                if len(approx) == 4:
                    location = approx
                    break

            if location is None:
                logger.warning("No valid contour with 4 points found. Trying alternative method.")
                for contour in contours:
                    if cv2.contourArea(contour) > 100:  # Filter out small contours
                        location = cv2.convexHull(contour)
                        break

            if location is not None and location.shape[0] > 0 and location.dtype == np.int32:
                mask = np.zeros(gray.shape, np.uint8)
                new_image = cv2.drawContours(mask, [location], 0, 255, -1)
                new_image = cv2.bitwise_and(frame, frame, mask=mask)
                (x, y) = np.where(mask == 255)
                if x.size == 0 or y.size == 0:
                    logger.error("Failed to locate the license plate region.")
                    continue
                (x1, y1) = (np.min(x), np.min(y))
                (x2, y2) = (np.max(x), np.max(y))
                cropped_image = gray[x1:x2+1, y1:y2+1]
                reader = easyocr.Reader(['en'])
                result = reader.readtext(cropped_image)
                license_plate = result[0][-2] if result else "Unknown"
                font = cv2.FONT_HERSHEY_SIMPLEX
                annotated_frame = cv2.putText(frame, text=license_plate, org=(location[0][0][0], location[1][0][1]+60), fontFace=font, fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
                annotated_frame = cv2.rectangle(frame, tuple(location[0][0]), tuple(location[2][0]), (0,255,0), 3)
                results.append(annotated_frame)

        cap.release()
        if results:
            result_video_path = os.path.join("results", "opencv", "videos", os.path.basename(file_path))
            fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Use H.264 codec
            frame_size = (results[0].shape[1], results[0].shape[0])
            out = cv2.VideoWriter(result_video_path, fourcc, fps, frame_size)

            for frame in results:
                out.write(frame)
            out.release()
            logger.info(f"Video writer released successfully")

            logger.info(f"Annotated video saved at: {result_video_path}")

            result = {
                "status": "success",
                "result_url": f"/results/opencv/videos/{os.path.basename(result_video_path)}",
                "filename": file_path,
            }

            return result
        else:
            raise Exception("No frames were processed successfully.")
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        raise
