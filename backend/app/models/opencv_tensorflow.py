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

def process_image(file_path):
    try:
        # Load the image
        image = cv2.imread(file_path)
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
        location = None
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 10, True)
            if len(approx) == 4:
                location = approx
                break
        logger.info("Contours found and license plate located")

        # Create a mask and extract the license plate region
        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [location], 0, 255, -1)
        new_image = cv2.bitwise_and(image, image, mask=mask)
        (x, y) = np.where(mask == 255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))
        cropped_image = gray[x1:x2+1, y1:y2+1]
        logger.info("License plate region extracted")

        # Use EasyOCR to read the text from the license plate
        reader = easyocr.Reader(['en'])
        result = reader.readtext(cropped_image)
        license_plate = result[0][-2] if result else "Unknown"
        logger.info(f"License plate text extracted: {license_plate}")

        # Annotate the image with the license plate text and rectangle
        font = cv2.FONT_HERSHEY_SIMPLEX
        annotated_image = cv2.putText(image, text=license_plate, org=(location[0][0][0], location[1][0][1]+60), fontFace=font, fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
        annotated_image = cv2.rectangle(image, tuple(location[0][0]), tuple(location[2][0]), (0,255,0), 3)
        logger.info("Annotated image with license plate text and rectangle")

        # Save the annotated image
        result_image_path = os.path.join("results", os.path.basename(file_path))
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

        # Example result
        result = {
            "status": "success",
            "result_url": f"/results/{os.path.basename(result_image_path)}",
            "intermediate_images": intermediate_images,
            "license_plate": license_plate,
            "filename": file_path,
            "customer_data": customer_data
        }

        return result
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON: {e}")
        raise
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise
