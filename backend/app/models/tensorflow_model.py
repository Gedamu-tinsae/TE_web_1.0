import tensorflow as tf
import numpy as np
import cv2
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the TensorFlow model
model_path = os.path.join(os.path.dirname(__file__), 'saved_model')
if not os.path.exists(model_path):
    logger.error(f"Model path does not exist: {model_path}")
    raise FileNotFoundError(f"Model path does not exist: {model_path}")

model = tf.saved_model.load(model_path)

def process_image_with_model(file_path):
    try:
        # Load the image
        image = cv2.imread(file_path)
        if image is None:
            logger.error("Failed to load image.")
            raise ValueError("Failed to load image.")
        logger.info("Image loaded successfully")

        # Convert image to tensor and detect plates
        image_np = np.array(image)
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.uint8)

        detections = model(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
        detections['num_detections'] = num_detections
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        # Draw bounding boxes around detected plates
        for i in range(num_detections):
            if detections['detection_scores'][i] > 0.8:
                box = detections['detection_boxes'][i]
                h, w, _ = image.shape
                y_min, x_min, y_max, x_max = box
                x_min, x_max = int(x_min * w), int(x_max * w)
                y_min, y_max = int(y_min * h), int(y_max * h)
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Save the processed image
        result_image_path = os.path.join("results", os.path.basename(file_path))
        cv2.imwrite(result_image_path, image)
        logger.info(f"Processed image saved at: {result_image_path}")

        result = {
            "status": "success",
            "result_url": f"/results/{os.path.basename(result_image_path)}",
            "filename": file_path,
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
            input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
            detections = model(input_tensor)

            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
            detections['num_detections'] = num_detections
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

            # Draw bounding boxes around detected plates
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