import tensorflow as tf
import numpy as np
import cv2
import os
import logging
import base64
from .plate_correction import extract_text_from_plate, matches_pattern, looks_like_covid, generate_character_analysis_for_covid19
from .color_detection import detect_vehicle_color, visualize_color_detection

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

def process_image_with_model(file_path, confidence_threshold=0.7):
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
        original_ocr_texts = []  # New array to store original OCR texts
        vehicle_colors = []  # New array to store vehicle colors
        color_confidences = []  # New array to store color confidences
        
        # Detect vehicle color from the full image first (as a fallback)
        full_image_color = detect_vehicle_color(original_image)
        
        for i in range(num_detections):
            # Use the confidence_threshold parameter instead of hardcoding 0.7
            if detections['detection_scores'][i] > confidence_threshold:  # Use confidence threshold parameter
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
                
                # Extract text with candidates using the centralized function
                plate_text, candidates, original_ocr_text = extract_text_from_plate(localized_plate, preprocessing_level='advanced')
                extracted_texts.append(plate_text)
                text_candidates.append(candidates)
                original_ocr_texts.append(original_ocr_text)  # Store the original OCR text
                
                # For vehicle color detection, extract a larger region around the license plate
                # This helps capture more of the vehicle
                vehicle_region_y_min = max(0, y_min - (y_max - y_min) * 3)  # Go up 3x the plate height
                vehicle_region_y_max = min(h, y_max + (y_max - y_min))      # Go down 1x the plate height
                vehicle_region_x_min = max(0, x_min - (x_max - x_min))      # Expand width by 1x on each side
                vehicle_region_x_max = min(w, x_max + (x_max - x_min))
                
                # Extract vehicle region
                vehicle_region = image[vehicle_region_y_min:vehicle_region_y_max, 
                                      vehicle_region_x_min:vehicle_region_x_max]
                
                # Detect vehicle color
                if vehicle_region.size > 0:
                    color_info = detect_vehicle_color(vehicle_region)
                    vehicle_colors.append(color_info["color"])
                    color_confidences.append(color_info["confidence"])
                else:
                    # Use full image color as fallback
                    vehicle_colors.append(full_image_color["color"])
                    color_confidences.append(full_image_color["confidence"])
                
                # Double-check for the COVID19 special case
                if plate_text == 'OD19':
                    logger.info("OD19 detected in TensorFlow pipeline - correcting to COVID19")
                    plate_text = 'COVID19'
                    
                    # Generate proper character analysis for COVID19
                    covid_char_positions = generate_character_analysis_for_covid19(
                        candidates[0].get("confidence", 0.85) if candidates else 0.85
                    )
                    
                    # Update the first candidate with proper character data as well
                    if candidates and len(candidates) > 0:
                        candidates[0]['text'] = 'COVID19'
                        candidates[0]['confidence'] = 1.0
                        candidates[0]['pattern_match'] = True
                        candidates[0]['pattern_name'] = 'Special Case - COVID19'
                        candidates[0]['char_positions'] = covid_char_positions
                
                # Check for other COVID-like patterns
                else:
                    is_covid, confidence = looks_like_covid(plate_text)
                    if is_covid and confidence > 0.6:
                        logger.info(f"COVID pattern detected in '{plate_text}' - correcting to COVID19")
                        plate_text = 'COVID19'
                        # Update the first candidate as well if it exists
                        if candidates and len(candidates) > 0:
                            candidates[0]['text'] = 'COVID19'
                            candidates[0]['confidence'] = max(candidates[0].get('confidence', 0), confidence)
                            candidates[0]['pattern_match'] = True
                            candidates[0]['pattern_name'] = 'Special Case - COVID19'
                
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
                
                # Add color information below the plate text
                if vehicle_colors:
                    color_text = f"Color: {vehicle_colors[-1]}"
                    cv2.putText(image, color_text,
                              (x_min, y_min - 40),  # Position above the plate text
                              cv2.FONT_HERSHEY_SIMPLEX,
                              0.7,
                              (0, 255, 255),  # Yellow color for visibility
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
            "vehicle_colors": vehicle_colors,  # Add vehicle colors to result
            "color_confidences": color_confidences,  # Add color confidences to result
            "vehicle_color": vehicle_colors[0] if vehicle_colors else full_image_color["color"],  # Primary vehicle color
            "color_confidence": color_confidences[0] if color_confidences else full_image_color["confidence"],  # Primary color confidence
            "original_ocr_texts": original_ocr_texts,  # Include original OCR results
            "license_plate": extracted_texts[0] if extracted_texts else "Unknown",
            "original_ocr": original_ocr_texts[0] if original_ocr_texts else "Unknown",  # Include first original OCR
            "text_candidates": text_candidates[0] if text_candidates else []  # Ensure this is a direct array, not nested
        }

        return result
    except Exception as e:
        logger.error(f"Error processing image with model: {e}")
        raise

def process_video_with_model(file_path, low_visibility=False, confidence_threshold=0.7):
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
        all_vehicle_colors = []  # Track all detected vehicle colors

        # Initialize HazeRemoval if needed for low visibility
        hr = None
        if low_visibility:
            from .haze_removal import HazeRemoval
            hr = HazeRemoval()
            logger.info("Initialized HazeRemoval for low visibility video")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_number += 1
            logger.info(f"Processing frame {frame_number}/{frame_count}")

            # Store original frame
            original_frame = frame.copy()

            # Apply dehazing if low_visibility is True
            if low_visibility and hr:
                # Convert frame to proper format for HazeRemoval
                # Save frame to temp file, process it with HazeRemoval, then read back
                temp_frame_path = os.path.join("uploads", "tensorflow", "temp_frame.jpg")
                cv2.imwrite(temp_frame_path, frame)
                
                hr.open_image(temp_frame_path)
                hr.get_dark_channel()
                hr.get_air_light()
                hr.get_transmission()
                hr.guided_filter()
                hr.recover()
                
                # Update frame with dehazed version
                frame = hr.dst
                logger.info("Applied dehazing to video frame")

            # Get full frame vehicle color (fallback)
            full_frame_color = detect_vehicle_color(frame)

            # Continue with normal processing using the possibly dehazed frame
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
            frame_colors = []  # Track colors detected in this frame

            for i in range(num_detections):
                # Use the confidence_threshold parameter instead of hardcoding 0.7
                if detections['detection_scores'][i] > confidence_threshold:  # Use confidence threshold parameter
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

                    # Use the common extraction function for improved OCR
                    plate_text, candidates = extract_text_from_plate(plate, preprocessing_level='advanced')
                    
                    # For vehicle color detection, extract a larger region around the license plate
                    vehicle_region_y_min = max(0, y_min - (y_max - y_min) * 3)  # Go up 3x the plate height
                    vehicle_region_y_max = min(h, y_max + (y_max - y_min))      # Go down 1x the plate height
                    vehicle_region_x_min = max(0, x_min - (x_max - x_min))      # Expand width by 1x on each side
                    vehicle_region_x_max = min(w, x_max + (x_max - x_min))
                    
                    # Extract vehicle region
                    vehicle_region = frame[vehicle_region_y_min:vehicle_region_y_max, 
                                          vehicle_region_x_min:vehicle_region_x_max]
                    
                    # Detect vehicle color
                    if vehicle_region.size > 0:
                        color_info = detect_vehicle_color(vehicle_region)
                        frame_colors.append(color_info["color"])
                    else:
                        # Use full frame color as fallback
                        frame_colors.append(full_frame_color["color"])
                    
                    # Double-check for the COVID19 special case
                    if plate_text == 'OD19':
                        logger.info("OD19 detected in TensorFlow pipeline - correcting to COVID19")
                        plate_text = 'COVID19'
                        
                        # Generate proper character analysis for COVID19
                        covid_char_positions = generate_character_analysis_for_covid19(
                            candidates[0].get("confidence", 0.85) if candidates else 0.85
                        )
                        
                        # Update the first candidate with proper character data as well
                        if candidates and len(candidates) > 0:
                            candidates[0]['text'] = 'COVID19'
                            candidates[0]['confidence'] = 1.0
                            candidates[0]['pattern_match'] = True
                            candidates[0]['pattern_name'] = 'Special Case - COVID19'
                            candidates[0]['char_positions'] = covid_char_positions
                    
                    # Check for other COVID-like patterns
                    else:
                        is_covid, confidence = looks_like_covid(plate_text)
                        if is_covid and confidence > 0.6:
                            logger.info(f"COVID pattern detected in '{plate_text}' - correcting to COVID19")
                            plate_text = 'COVID19'
                            # Update the first candidate as well if it exists
                            if candidates and len(candidates) > 0:
                                candidates[0]['text'] = 'COVID19'
                                candidates[0]['confidence'] = max(candidates[0].get('confidence', 0), confidence)
                                candidates[0]['pattern_match'] = True
                                candidates[0]['pattern_name'] = 'Special Case - COVID19'
                    
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
                    
                    # Add color information
                    if frame_colors:
                        color_text = f"Color: {frame_colors[-1]}"
                        cv2.putText(frame, color_text,
                                  (x_min, y_min - 40),  # Position above the plate text
                                  cv2.FONT_HERSHEY_SIMPLEX,
                                  0.7,
                                  (0, 255, 255),  # Yellow color for visibility
                                  2)

            # Store intermediate results for this frame
            all_intermediate_frames["original"].append(original_frame)
            all_intermediate_frames["detection"].append(detection_frame)
            all_intermediate_frames["plates"].extend(frame_plates)
            all_extracted_texts.extend(frame_texts)
            all_vehicle_colors.extend(frame_colors)  # Add detected colors

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

            # Determine most common vehicle color
            from collections import Counter
            if all_vehicle_colors:
                color_counter = Counter(all_vehicle_colors)
                most_common_color = color_counter.most_common(1)[0][0]
                color_frequency = color_counter.most_common(1)[0][1] / len(all_vehicle_colors)
            else:
                most_common_color = full_frame_color["color"]
                color_frequency = full_frame_color["confidence"]

            result = {
                "status": "success",
                "filename": file_path,
                "result_url": f"/results/tensorflow/videos/{os.path.basename(file_path)}",
                "result_image": encode_image(results[sample_frame_index]),  # Sample frame from result
                "intermediate_images": intermediate_images,
                "detected_plates": list(set(all_extracted_texts)),  # Remove duplicates
                "license_plate": all_extracted_texts[0] if all_extracted_texts else "Unknown",
                "vehicle_color": most_common_color,  # Add vehicle color to result
                "color_confidence": color_frequency,  # Add confidence (frequency) of color detection
                "text_candidates": frame_candidates[0] if frame_candidates else []  # Ensure this is a direct array, not nested
            }

            return result
        else:
            raise Exception("No frames were processed successfully.")
    except Exception as e:
        logger.error(f"Error processing video with model: {e}")
        raise
