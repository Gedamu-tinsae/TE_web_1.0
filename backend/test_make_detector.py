import os
import sys
import logging
import cv2
import numpy as np
from app.models.vehicle_make import vehicle_make_detector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_make_detection(image_path):
    """Test vehicle make detection on a single image."""
    logger.info(f"Testing make detection on image: {image_path}")
    
    # Verify file exists
    if not os.path.exists(image_path):
        logger.error(f"Image file not found: {image_path}")
        return
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Failed to load image: {image_path}")
        return
    
    # Try vehicle make detection
    result = vehicle_make_detector.detect(img)
    
    # Print results
    logger.info("Vehicle Make Detection:")
    logger.info(f"Full Image Used for Make Detection:\n")
    logger.info("Full Image Make Detection")
    logger.info(f"Full Image Detection: {result['make']}")
    
    # Extract vehicle region (center of image as fallback)
    h, w = img.shape[:2]
    region_h, region_w = h // 2, w // 2
    x_start = max(0, w // 2 - region_w // 2)
    y_start = max(0, h // 2 - region_h // 2)
    vehicle_region = img[y_start:y_start+region_h, x_start:x_start+region_w]
    
    # Try vehicle make detection on the region
    region_result = vehicle_make_detector.detect(vehicle_region)
    
    logger.info(f"Vehicle Region Used for Make Detection:\n")
    logger.info("Vehicle Make Detection Region")
    logger.info(f"Region Detection: {region_result['make']}")
    
    logger.info(f"Final Vehicle Make Determination:\n")
    if result['confidence'] >= region_result['confidence']:
        logger.info(f"Best Detection: {result['make']} [Source: Full Image]")
    else:
        logger.info(f"Best Detection: {region_result['make']} [Source: Vehicle Region]")

if __name__ == "__main__":
    # Use command line argument for image path or default to a test image
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Default test image - adjust path as needed
        image_path = "uploads/tensorflow/images/image.jpg"
    
    test_make_detection(image_path)
