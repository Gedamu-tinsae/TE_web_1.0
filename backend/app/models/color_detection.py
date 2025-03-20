import cv2
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def detect_vehicle_color(image):
    """
    Detect the dominant color of a vehicle in an image.
    Returns a dictionary with the color name, confidence, and color percentages.
    """
    try:
        if image is None or image.size == 0:
            return {"color": "Unknown", "confidence": 0.0, "color_percentages": {}}
        
        # Convert to HSV for better color discrimination
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define color ranges in HSV
        color_ranges = {
            # Fix the white detection range - restrict it more to avoid detecting black as white
            "white": [(0, 0, 200), (180, 25, 255)],  # Narrower definition of white - higher value required
            "black": [(0, 0, 0), (180, 45, 40)],     # Expand black range slightly to capture more dark areas
            "gray": [(0, 0, 60), (180, 35, 160)],    # Adjust gray to not overlap with white as much
            "silver": [(0, 0, 160), (180, 30, 220)], # Silver is between gray and white
            "red1": [(0, 70, 50), (10, 255, 255)],   # Red wraps around the hue spectrum
            "red2": [(170, 70, 50), (180, 255, 255)],
            "orange": [(10, 70, 50), (25, 255, 255)],
            "yellow": [(25, 70, 50), (35, 255, 255)],
            "green": [(35, 70, 50), (85, 255, 255)],
            "blue": [(85, 70, 50), (130, 255, 255)],
            "purple": [(130, 70, 50), (170, 255, 255)],
            "brown": [(10, 70, 20), (20, 200, 100)]  # Brown is a difficult color in HSV
        }
        
        # Create masks for each color
        color_counts = {}
        color_masks = {}
        
        # Get total number of pixels
        total_pixels = image.shape[0] * image.shape[1]
        
        # First pass: compute raw color percentages
        for color, (lower, upper) in color_ranges.items():
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            
            # Create mask
            mask = cv2.inRange(hsv_image, lower, upper)
            
            # Count non-zero pixels (matching the color)
            count = cv2.countNonZero(mask)
            percentage = (count / total_pixels) * 100
            
            # Store results
            color_counts[color] = percentage
            color_masks[color] = mask
        
        # Handle the special case of red (combine red1 and red2)
        if "red1" in color_counts and "red2" in color_counts:
            color_counts["red"] = color_counts["red1"] + color_counts["red2"]
            del color_counts["red1"]
            del color_counts["red2"]
            
            # Also combine the masks
            if "red1" in color_masks and "red2" in color_masks:
                color_masks["red"] = cv2.bitwise_or(color_masks["red1"], color_masks["red2"])
                del color_masks["red1"]
                del color_masks["red2"]
        
        # Add heuristic adjustments for common issues:
        
        # 1. White vehicles with reflections might be detected as multiple colors
        # If white percentage is significant, boost it
        if "white" in color_counts and color_counts["white"] > 15:
            color_counts["white"] *= 1.3  # Boost white detection
        
        # 2. Silver is often confused with white or gray
        if "silver" in color_counts and "white" in color_counts:
            if color_counts["silver"] > 10 and color_counts["white"] > 10:
                # If both silver and white are detected strongly, 
                # this is likely a white vehicle with shadows
                if color_counts["silver"] > color_counts["white"]:
                    color_counts["silver"] += color_counts["white"] * 0.5
                else:
                    color_counts["white"] += color_counts["silver"] * 0.5
        
        # 3. Black is often underdetected - boost black detection more
        if "black" in color_counts:
            color_counts["black"] *= 1.5  # Increase the boost for black (was 1.2)
            
        # 4. Handle special case with gray/silver/white/black vehicles
        if (("gray" in color_counts and color_counts["gray"] > 15) or
            ("silver" in color_counts and color_counts["silver"] > 15) or
            ("white" in color_counts and color_counts["white"] > 15) or
            ("black" in color_counts and color_counts["black"] > 10)):
            
            # Compute brightness and saturation statistics
            avg_v = np.mean(hsv_image[:,:,2])
            avg_s = np.mean(hsv_image[:,:,1])
            
            # Add a histogram analysis for better black/white differentiation
            v_channel = hsv_image[:,:,2]
            hist = cv2.calcHist([v_channel], [0], None, [256], [0, 256])
            dark_pixel_ratio = np.sum(hist[:50]) / np.sum(hist)  # Ratio of very dark pixels
            light_pixel_ratio = np.sum(hist[200:]) / np.sum(hist)  # Ratio of very light pixels
            
            logger.info(f"Brightness stats - avg_v: {avg_v}, dark ratio: {dark_pixel_ratio:.2f}, light ratio: {light_pixel_ratio:.2f}")
            
            # Black detection - very low brightness
            if avg_v < 60 or dark_pixel_ratio > 0.5:
                logger.info("Detected likely black vehicle based on low brightness")
                color_counts["black"] = max(color_counts.get("black", 0), 40)  # Stronger bias for black
                # Reduce white if it was incorrectly detected
                if "white" in color_counts:
                    color_counts["white"] *= 0.5
            
            # Very high brightness and low saturation usually means white
            elif avg_v > 200 and avg_s < 30 and light_pixel_ratio > 0.4:
                logger.info("Detected likely white vehicle based on high brightness")
                color_counts["white"] = max(color_counts.get("white", 0), 40)
                # Reduce black if it was incorrectly detected
                if "black" in color_counts:
                    color_counts["black"] *= 0.5
                
            # Medium brightness and low saturation usually means silver
            elif avg_v > 140 and avg_v <= 200 and avg_s < 40:
                color_counts["silver"] = max(color_counts.get("silver", 0), 30)
                
            # Low brightness and low saturation usually means gray or black
            elif avg_v <= 140 and avg_s < 40:
                if avg_v < 80:
                    color_counts["black"] = max(color_counts.get("black", 0), 35)
                else:
                    color_counts["gray"] = max(color_counts.get("gray", 0), 30)
        
        # Find the most dominant color
        dominant_color = max(color_counts.items(), key=lambda x: x[1]) if color_counts else ("Unknown", 0)
        
        # Calculate confidence - normalize to 0-1 range
        total_percentage = sum(color_counts.values())
        confidence = min(dominant_color[1] / max(1, total_percentage) * 1.5, 1.0)
        
        # Calculate percentages for all colors
        color_percentages = {color: round(percentage, 2) for color, percentage in color_counts.items()}
        
        # Add debug information
        logger.info(f"Detected color: {dominant_color[0]} with confidence {confidence:.2f}")
        logger.info(f"Color percentages: {', '.join([f'{c}: {p:.1f}%' for c, p in sorted(color_percentages.items(), key=lambda x: x[1], reverse=True)[:3]])}")
        
        return {
            "color": dominant_color[0],
            "confidence": float(confidence),
            "color_percentages": color_percentages
        }
    except Exception as e:
        logger.error(f"Error in color detection: {e}")
        return {"color": "Error", "confidence": 0.0, "color_percentages": {}}

def visualize_color_detection(image):
    """
    Visualize the color detection process by creating an image with the detected color
    patches and information.
    """
    try:
        if image is None or image.size == 0:
            return None
            
        # Get the color detection result
        result = detect_vehicle_color(image)
        
        # Create a visualization image
        h, w = image.shape[:2]
        vis_height = h + 150  # Add space for color info
        visualization = np.zeros((vis_height, w, 3), dtype=np.uint8)
        
        # Copy the original image to the top
        visualization[:h, :w] = image.copy()
        
        # Draw color percentages as a bar chart
        y_offset = h + 20
        bar_height = 30
        text_offset = 5
        
        # Get the top 5 colors
        top_colors = sorted(result["color_percentages"].items(), key=lambda x: x[1], reverse=True)[:5]
        
        for i, (color_name, percentage) in enumerate(top_colors):
            # Map color name to BGR value
            color_bgr = get_bgr_color(color_name)
            
            # Calculate bar width based on percentage (max 80% of width)
            bar_width = int(w * 0.8 * percentage / 100)
            if bar_width < 5:  # Ensure a minimum visible bar
                bar_width = 5
                
            # Draw the bar
            cv2.rectangle(visualization, 
                          (10, y_offset + i * (bar_height + 5)), 
                          (10 + bar_width, y_offset + i * (bar_height + 5) + bar_height), 
                          color_bgr, -1)
                          
            # Draw the color name and percentage
            cv2.putText(visualization, 
                      f"{color_name}: {percentage:.1f}%", 
                      (15, y_offset + i * (bar_height + 5) + text_offset + bar_height//2), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Draw the dominant color info
        cv2.putText(visualization, 
                  f"Dominant color: {result['color']} (Confidence: {result['confidence']*100:.1f}%)", 
                  (10, h + 10), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                  
        return visualization
    except Exception as e:
        logger.error(f"Error in color visualization: {e}")
        return None

def get_bgr_color(color_name):
    """Map color name to BGR value for visualization"""
    color_map = {
        "red": (0, 0, 255),
        "green": (0, 255, 0),
        "blue": (255, 0, 0),
        "yellow": (0, 255, 255),
        "orange": (0, 165, 255),
        "purple": (255, 0, 255),
        "white": (255, 255, 255),
        "black": (0, 0, 0),
        "gray": (128, 128, 128),
        "silver": (192, 192, 192),
        "brown": (42, 42, 165)
    }
    return color_map.get(color_name, (200, 200, 200))  # Default to light gray
