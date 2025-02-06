import tensorflow as tf
import numpy as np
import cv2
import os
import logging
import easyocr

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the TensorFlow model
model_path = os.path.join(os.path.dirname(__file__), 'saved_model')
if not os.path.exists(model_path):
    logger.error(f"Model path does not exist: {model_path}")
    raise FileNotFoundError(f"Model path does not exist: {model_path}")

model = tf.saved_model.load(model_path)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

def extract_text_from_plate(plate_region):
    try:
        # Preprocess the plate image for better OCR
        gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
        # Apply some basic image processing
        gray = cv2.resize(gray, None, fx=2, fy=2)  # Upscale
        gray = cv2.GaussianBlur(gray, (5,5), 0)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # Perform OCR
        result = reader.readtext(gray)
        
        if result:
            # Combine all detected text pieces
            text = ' '.join([detection[1] for detection in result])
            # Clean the text (remove unwanted characters, spaces)
            text = ''.join(c for c in text if c.isalnum())
            return text
        return "Unknown"
    except Exception as e:
        logger.error(f"Error in OCR: {e}")
        return "Error"

def process_image_with_model(file_path):
    try:
        # Load the image
        original_image = cv2.imread(file_path)
        if original_image is None:
            logger.error("Failed to load image.")
            raise ValueError("Failed to load image.")
        logger.info("Image loaded successfully")

        # Save original image
        original_path = os.path.join("results", "1_original_" + os.path.basename(file_path))
        cv2.imwrite(original_path, original_image)

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
        
        for i in range(num_detections):
            if detections['detection_scores'][i] > 0.8:
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
                
                # Extract text
                plate_text = extract_text_from_plate(localized_plate)
                extracted_texts.append(plate_text)
                
                # Draw rectangle and text on final image
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(image, plate_text, 
                          (x_min, y_min - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 
                          0.9,
                          (255, 255, 255),
                          2)

        # Save intermediate steps
        detection_path = os.path.join("results", "2_detection_" + os.path.basename(file_path))
        cv2.imwrite(detection_path, detection_image)

        # Save individual plate regions
        for idx, plate_img in enumerate(localized_images):
            plate_path = os.path.join("results", f"3_plate_{idx}_" + os.path.basename(file_path))
            cv2.imwrite(plate_path, plate_img)

        # Save final result with OCR
        final_path = os.path.join("results", "4_final_" + os.path.basename(file_path))
        cv2.imwrite(final_path, image)

        result = {
            "status": "success",
            "steps": {
                "original": f"/results/1_original_{os.path.basename(file_path)}",
                "detection": f"/results/2_detection_{os.path.basename(file_path)}",
                "plates": [f"/results/3_plate_{i}_{os.path.basename(file_path)}" for i in range(len(localized_images))],
                "final": f"/results/4_final_{os.path.basename(file_path)}"
            },
            "detected_plates": extracted_texts
        }

        return result
    except Exception as e:
        logger.error(f"Error processing image with model: {e}")
        raise

def process_video_with_model(file_path):
    try:
        cap = cv2.VideoCapture(file_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_number = 0
        results = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_number += 1
            logger.info(f"Processing frame {frame_number}/{frame_count}")

            # Convert frame to tensor and detect plates
            image_np = np.array(frame)
            input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.uint8)
            detections = model(input_tensor)

            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
            detections['num_detections'] = num_detections
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

            for i in range(num_detections):
                if detections['detection_scores'][i] > 0.8:
                    box = detections['detection_boxes'][i]
                    h, w, _ = frame.shape
                    y_min, x_min, y_max, x_max = box
                    x_min, x_max = int(x_min * w), int(x_max * w)
                    y_min, y_max = int(y_min * h), int(y_max * h)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            results.append(frame)

        cap.release()
        if results:
            result_video_path = os.path.join("results", os.path.basename(file_path))
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            frame_size = (results[0].shape[1], results[0].shape[0])
            out = cv2.VideoWriter(result_video_path, fourcc, fps, frame_size)

            for frame in results:
                out.write(frame)
            out.release()
            logger.info(f"Processed video saved at: {result_video_path}")

            result = {
                "status": "success",
                "result_url": f"/results/{os.path.basename(result_video_path)}",
                "filename": file_path,
            }

            return result
        else:
            raise Exception("No frames were processed successfully.")
    except Exception as e:
        logger.error(f"Error processing video with model: {e}")
        raise
