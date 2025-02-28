import cv2
import numpy as np
import easyocr
import os
import imutils
import logging
import json
import base64

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

            # Use EasyOCR to read the text from the license plate with confidence scores
            reader = easyocr.Reader(['en'])
            result = reader.readtext(cropped_image, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
            
            # Process and store all text candidates with confidence
            text_candidates = []
            if result:
                for detection in result:
                    bbox, text, confidence = detection
                    # Clean the text (remove unwanted characters, spaces)
                    text = ''.join(c for c in text if c.isalnum())
                    if text:
                        text_candidates.append({
                            "text": text,
                            "confidence": float(confidence)
                        })
                
                # Sort by confidence (highest first)
                text_candidates.sort(key=lambda x: x["confidence"], reverse=True)
                
                # Get the highest confidence text
                license_plate = text_candidates[0]["text"] if text_candidates else "Unknown"
            else:
                license_plate = "Unknown"
                text_candidates.append({"text": "Unknown", "confidence": 0.0})
                
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
