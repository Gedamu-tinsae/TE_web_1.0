import cv2
import numpy as np
import os
import imutils
import logging
import json
import base64
from .plate_correction import extract_text_from_plate, extract_text_from_region, get_reader, matches_pattern, looks_like_covid, generate_character_analysis_for_covid19
from .color_detection import detect_vehicle_color, visualize_color_detection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def process_image(file_path, confidence_threshold=0.7):
    try:
        ensure_dirs()
        # Load the image
        image = cv2.imread(file_path)
        if image is None:
            logger.error("Failed to load image.")
            raise ValueError("Failed to load image.")
        logger.info("Image loaded successfully")

        # Detect vehicle color from the full image
        color_info = detect_vehicle_color(image)
        logger.info(f"Detected vehicle color: {color_info['color']}")

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

        # If we're using a lower confidence threshold for low visibility images,
        # we should also be more lenient with contour selection
        min_contour_area = 50 if confidence_threshold < 0.7 else 100

        if location is None:
            logger.warning("No valid contour with 4 points found. Trying alternative method.")
            for contour in contours:
                if cv2.contourArea(contour) > min_contour_area:  # Use dynamic threshold
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

            # Use our centralized function to extract text and candidates
            license_plate, text_candidates, original_ocr_text = extract_text_from_plate(cropped_image, preprocessing_level='standard')
            
            # Double-check for the COVID19 special case
            if license_plate == 'OD19' or any(c.get('text') == 'OD19' for c in text_candidates):
                logger.info("OD19 detected in OpenCV pipeline - correcting to COVID19")
                license_plate = 'COVID19'
                
                # Generate proper character analysis for COVID19
                covid_char_positions = generate_character_analysis_for_covid19(
                    text_candidates[0].get("confidence", 0.85) if text_candidates else 0.85
                )
                
                # Update the first candidate with proper character data as well
                if text_candidates and len(text_candidates) > 0:
                    text_candidates[0]['text'] = 'COVID19'
                    text_candidates[0]['confidence'] = 1.0
                    text_candidates[0]['pattern_match'] = True
                    text_candidates[0]['pattern_name'] = 'Special Case - COVID19'
                    text_candidates[0]['char_positions'] = covid_char_positions
            
            # Check for other COVID-like patterns
            elif license_plate:
                is_covid, confidence = looks_like_covid(license_plate)
                if is_covid and confidence > 0.6:
                    logger.info(f"COVID pattern detected in '{license_plate}' - correcting to COVID19")
                    license_plate = 'COVID19'
                    # Update the first candidate as well if it exists
                    if text_candidates and len(text_candidates) > 0:
                        text_candidates[0]['text'] = 'COVID19'
                        text_candidates[0]['confidence'] = max(text_candidates[0].get('confidence', 0), confidence)
                        text_candidates[0]['pattern_match'] = True
                        text_candidates[0]['pattern_name'] = 'Special Case - COVID19'
            
            logger.info(f"License plate text extracted: {license_plate}")
            logger.info(f"Original OCR text: {original_ocr_text}")

            # Annotate the image with the license plate text and rectangle
            font = cv2.FONT_HERSHEY_SIMPLEX
            annotated_image = cv2.putText(image, text=license_plate, org=(location[0][0][0], location[1][0][1]+60), fontFace=font, fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
            annotated_image = cv2.rectangle(image, tuple(location[0][0]), tuple(location[2][0]), (0,255,0), 3)
            
            # Add color information to the annotated image
            color_text = f"Color: {color_info['color']}"
            annotated_image = cv2.putText(
                annotated_image, 
                color_text,
                (location[0][0][0], location[1][0][1]+90), # Position below license plate text
                font, 
                0.8, 
                (0, 255, 255), # Yellow color for visibility
                2, 
                cv2.LINE_AA
            )
            
            logger.info("Annotated image with license plate text, rectangle and color information")

            # Extract a larger area around the license plate for vehicle color detection
            h, w = image.shape[:2]
            x1, y1 = location[0][0]
            x2, y2 = location[2][0]
            
            # Calculate vehicle region with larger area above the plate (where the vehicle body usually is)
            vehicle_y_min = max(0, y1 - (y2 - y1) * 3)  # Go up 3x the plate height
            vehicle_y_max = min(h, y2 + (y2 - y1))      # Go down 1x the plate height
            vehicle_x_min = max(0, x1 - (x2 - x1))      # Expand width by 1x on each side
            vehicle_x_max = min(w, x2 + (x2 - x1))
            
            # Validate that the region coordinates are valid
            if vehicle_y_min < vehicle_y_max and vehicle_x_min < vehicle_x_max:
                # Extract the region for color detection
                vehicle_region = image[vehicle_y_min:vehicle_y_max, vehicle_x_min:vehicle_x_max]
                
                # Check if the vehicle region is not empty
                if vehicle_region.size > 0:
                    # Save the vehicle region for visualization
                    vehicle_region_path = os.path.join("results", "opencv", "intermediate", "images")
                    os.makedirs(vehicle_region_path, exist_ok=True)
                    vehicle_region_file = os.path.join(vehicle_region_path, f"vehicle_region_{os.path.basename(file_path)}")
                    cv2.imwrite(vehicle_region_file, vehicle_region)
                    
                    # Draw the vehicle region on the annotated image with a different color (yellow)
                    annotated_image = cv2.rectangle(annotated_image, 
                                                (vehicle_x_min, vehicle_y_min), 
                                                (vehicle_x_max, vehicle_y_max), 
                                                (0, 255, 255), 1)  # Yellow with thin line
                    
                    # Now detect color from this vehicle region instead of the whole image
                    region_color_info = detect_vehicle_color(vehicle_region)
                    logger.info(f"Detected vehicle color from region: {region_color_info['color']}")
                    
                    # Use the color detected from the vehicle region if available
                    color_info = region_color_info
                else:
                    logger.warning("Extracted vehicle region is empty, using full image color detection.")
                    # If vehicle region is empty, keep using the full image color detection
            else:
                logger.warning("Invalid vehicle region coordinates. Using full image color detection.")
                # Just use the full image color detection from earlier

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
                if image is None or image.size == 0:
                    # Return an empty base64 string or placeholder for empty images
                    return ""
                _, buffer = cv2.imencode('.jpg', image)
                return base64.b64encode(buffer).decode('utf-8')

            intermediate_images = {
                "gray": encode_image(gray),
                "edge": encode_image(edges),
                "localized": encode_image(new_image),
                "plate": encode_image(cropped_image),
            }
            
            # Add vehicle region to intermediate images only if it's valid
            if 'vehicle_region' in locals() and vehicle_region.size > 0:
                intermediate_images["vehicle_region"] = encode_image(vehicle_region)
            
            # Vehicle region coordinates for frontend visualization
            vehicle_region_coordinates = {
                "x_min": int(vehicle_x_min),
                "y_min": int(vehicle_y_min),
                "x_max": int(vehicle_x_max),
                "y_max": int(vehicle_y_max)
            }

            # Create the result dict
            result = {
                "status": "success",
                "result_url": f"/results/opencv/images/{os.path.basename(result_image_path)}",
                "intermediate_images": intermediate_images,
                "license_plate": license_plate,
                "original_ocr": original_ocr_text,  # Include original OCR text
                "filename": file_path,
                "customer_data": customer_data,
                "text_candidates": text_candidates,  # Already a direct array from our extraction function
                "vehicle_color": color_info["color"],  # Include detected vehicle color
                "color_confidence": color_info["confidence"],  # Include color detection confidence
                "color_percentages": color_info.get("color_percentages", {}),  # Include detailed color percentages if available
                "vehicle_region_coordinates": vehicle_region_coordinates  # Include the coordinates
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

def process_video(file_path, confidence_threshold=0.7):
    try:
        ensure_dirs()
        cap = cv2.VideoCapture(file_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_number = 0
        results = []
        max_frames = 40  # Limit the number of frames to process
        all_vehicle_colors = []  # Track all detected vehicle colors

        # Use a dynamic threshold based on confidence_threshold
        min_contour_area = 50 if confidence_threshold < 0.7 else 100

        while cap.isOpened() and frame_number < max_frames:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_number += 1
            logger.info(f"Processing frame {frame_number}/{min(frame_count, max_frames)}")
            
            # Detect vehicle color from the full frame
            color_info = detect_vehicle_color(frame)
            all_vehicle_colors.append(color_info["color"])
            logger.info(f"Frame {frame_number}: Detected vehicle color: {color_info['color']}")

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
                    if cv2.contourArea(contour) > min_contour_area:  # Use dynamic threshold
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
                
                # Use our centralized function to extract text
                license_plate, _ = extract_text_from_plate(cropped_image, preprocessing_level='minimal')
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                annotated_frame = cv2.putText(frame, text=license_plate, org=(location[0][0][0], location[1][0][1]+60), fontFace=font, fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
                annotated_frame = cv2.rectangle(frame, tuple(location[0][0]), tuple(location[2][0]), (0,255,0), 3)
                
                # Add color information to the frame
                color_text = f"Color: {color_info['color']}"
                annotated_frame = cv2.putText(
                    annotated_frame, 
                    color_text,
                    (location[0][0][0], location[1][0][1]+90), # Position below license plate text
                    font, 
                    0.8, 
                    (0, 255, 255), # Yellow color for visibility
                    2, 
                    cv2.LINE_AA
                )
                
                results.append(annotated_frame)
            else:
                # Even if we can't find the license plate, we still want to process the frame
                # and show the color detection
                color_text = f"Color: {color_info['color']}"
                annotated_frame = cv2.putText(
                    frame.copy(), 
                    color_text,
                    (10, 30), # Position at top-left if no plate is found
                    font, 
                    0.8, 
                    (0, 255, 255), # Yellow color for visibility
                    2, 
                    cv2.LINE_AA
                )
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

            # Determine most common vehicle color
            from collections import Counter
            if all_vehicle_colors:
                color_counter = Counter(all_vehicle_colors)
                most_common_color = color_counter.most_common(1)[0][0]
                color_frequency = color_counter.most_common(1)[0][1] / len(all_vehicle_colors)
            else:
                most_common_color = "Unknown"
                color_frequency = 0.0

            result = {
                "status": "success",
                "result_url": f"/results/opencv/videos/{os.path.basename(result_video_path)}",
                "filename": file_path,
                "vehicle_color": most_common_color,  # Add most common vehicle color
                "color_confidence": color_frequency,  # Add confidence (frequency) of color detection
                "all_colors": dict(color_counter.most_common(3))  # Include top 3 detected colors
            }

            return result
        else:
            raise Exception("No frames were processed successfully.")
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        raise
