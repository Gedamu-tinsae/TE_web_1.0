import cv2
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define color ranges in HSV
# These ranges can be adjusted based on your specific needs
COLOR_RANGES = {
    "red": [(0, 70, 50), (10, 255, 255), (160, 70, 50), (180, 255, 255)],  # Red wraps around hue 0
    "orange": [(11, 70, 50), (25, 255, 255)],
    "yellow": [(26, 70, 50), (35, 255, 255)],
    "green": [(36, 70, 50), (85, 255, 255)],
    "blue": [(86, 70, 50), (130, 255, 255)],
    "purple": [(131, 70, 50), (159, 255, 255)],
    "white": [(0, 0, 200), (180, 30, 255)],
    "black": [(0, 0, 0), (180, 255, 30)],
    "gray": [(0, 0, 31), (180, 30, 199)],
    "silver": [(0, 0, 150), (180, 30, 220)],
    "brown": [(0, 20, 20), (30, 150, 150)]
}

def get_dominant_color(image, mask=None):
    """
    Get the dominant color from an image, optionally using a mask.
    Returns the color name and percentage.
    """
    try:
        # Copy image to avoid modifications
        image_copy = image.copy()
        
        # If the image is already cropped to just the vehicle, use it directly
        # Otherwise, create/use a mask to isolate the vehicle region
        if mask is None:
            # Convert to grayscale and apply Otsu's thresholding
            gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Try to focus on the vehicle region by finding contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # Use the largest contour as a mask
                largest_contour = max(contours, key=cv2.contourArea)
                mask = np.zeros_like(gray)
                cv2.drawContours(mask, [largest_contour], 0, 255, -1)
        
        # Convert BGR to HSV
        hsv = cv2.cvtColor(image_copy, cv2.COLOR_BGR2HSV)
        
        # Dictionary to store color percentages
        color_percentages = {}
        
        total_pixels = np.count_nonzero(mask) if mask is not None else hsv.shape[0] * hsv.shape[1]
        if total_pixels == 0:
            logger.warning("No valid pixels found in the image for color detection.")
            return {"color": "unknown", "confidence": 0.0, "color_percentages": {}}
        
        # Check each color range
        for color_name, ranges in COLOR_RANGES.items():
            # Some colors (like red) may have multiple ranges
            if len(ranges) == 4:  # For red which wraps around
                lower1, upper1, lower2, upper2 = ranges
                mask1 = cv2.inRange(hsv, np.array(lower1), np.array(upper1))
                mask2 = cv2.inRange(hsv, np.array(lower2), np.array(upper2))
                color_mask = cv2.bitwise_or(mask1, mask2)
            else:
                lower, upper = ranges
                color_mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            
            # Apply vehicle mask if available
            if mask is not None:
                color_mask = cv2.bitwise_and(color_mask, mask)
            
            # Calculate percentage of pixels that match this color
            color_pixel_count = np.count_nonzero(color_mask)
            percentage = (color_pixel_count / total_pixels) * 100
            color_percentages[color_name] = percentage
        
        # Find the dominant color
        dominant_color = max(color_percentages.items(), key=lambda x: x[1])
        color_name, percentage = dominant_color
        
        logger.info(f"Detected dominant color: {color_name} ({percentage:.2f}%)")
        
        # Calculate confidence based on how dominant the color is
        # If one color is much more dominant than others, we're more confident
        sorted_percentages = sorted(color_percentages.values(), reverse=True)
        if len(sorted_percentages) > 1:
            confidence = min(1.0, (sorted_percentages[0] - sorted_percentages[1]) / 100)
        else:
            confidence = min(1.0, sorted_percentages[0] / 100)
        
        return {
            "color": color_name,
            "confidence": confidence,
            "percentage": percentage,
            "color_percentages": color_percentages
        }
    
    except Exception as e:
        logger.error(f"Error in color detection: {e}")
        return {"color": "unknown", "confidence": 0.0, "color_percentages": {}}

def detect_vehicle_color(image):
    """
    Detect vehicle color from an image.
    This function is a wrapper for get_dominant_color that handles preprocessing.
    """
    try:
        # Resize if image is too large for faster processing
        if image.shape[0] > 800 or image.shape[1] > 800:
            scale = 800 / max(image.shape[0], image.shape[1])
            image = cv2.resize(image, None, fx=scale, fy=scale)
        
        # Get dominant color
        result = get_dominant_color(image)
        
        # Color mapping to standard vehicle colors
        # Combine similar colors for vehicle classifications
        color_mapping = {
            "red": "Red",
            "orange": "Orange/Bronze",
            "yellow": "Yellow/Gold",
            "green": "Green",
            "blue": "Blue",
            "purple": "Purple",
            "white": "White",
            "black": "Black",
            "gray": "Gray",
            "silver": "Silver/Gray", 
            "brown": "Brown"
        }
        
        detected_color = result["color"]
        standard_color = color_mapping.get(detected_color, detected_color.capitalize())
        
        return {
            "color": standard_color,
            "confidence": result["confidence"],
            "raw_color": detected_color,
            "color_percentages": result["color_percentages"]
        }
        
    except Exception as e:
        logger.error(f"Error in vehicle color detection: {e}")
        return {"color": "unknown", "confidence": 0.0}

def visualize_color_detection(image, color_info):
    """
    Add color information to the image for visualization.
    """
    try:
        # Create a copy of the image
        viz_image = image.copy()
        
        # Get the color name and confidence
        color_name = color_info.get("color", "unknown")
        confidence = color_info.get("confidence", 0.0)
        
        # Create text for the visualization
        text = f"Color: {color_name} ({confidence:.2f})"
        
        # Position the text on the image (top-left corner)
        position = (10, 30)
        
        # Add text to the image
        cv2.putText(
            viz_image, text, position,
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA
        )
        
        return viz_image
        
    except Exception as e:
        logger.error(f"Error in color visualization: {e}")
        return image
