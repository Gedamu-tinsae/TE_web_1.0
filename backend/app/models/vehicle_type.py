from ultralytics import YOLO
import cv2
import os
import numpy as np
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VehicleTypeDetector:
    def __init__(self):
        self.model = None
        self.model_path = Path(__file__).parent / 'VTD' / 'yolov8n.pt'
        self._load_model()

    def _load_model(self):
        """Load the YOLOv8 model"""
        try:
            if not self.model_path.exists():
                logger.warning("YOLOv8 model not found. Downloading...")
            self.model = YOLO(str(self.model_path))
            logger.info("YOLOv8 model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading YOLOv8 model: {e}")
            self.model = None

    def detect(self, image, conf_threshold=0.25):
        """
        Detect vehicle type from image
        Returns: dict with vehicle_type, confidence, and alternatives
        """
        if self.model is None:
            return {
                "vehicle_type": "Unknown",
                "confidence": 0.0,
                "alternatives": []
            }

        try:
            # Get model predictions
            results = self.model.predict(source=image, conf=conf_threshold)
            
            if not results or len(results) == 0:
                return {
                    "vehicle_type": "Unknown",
                    "confidence": 0.0,
                    "alternatives": []
                }

            # Get class predictions
            boxes = results[0].boxes
            if len(boxes) == 0:
                return {
                    "vehicle_type": "Unknown",
                    "confidence": 0.0,
                    "alternatives": []
                }

            # Get the best prediction
            best_conf = 0.0
            best_type = "Unknown"
            alternatives = []

            # Process all detections
            for box in boxes:
                cls_id = int(box.cls[0].item())
                cls_name = self.model.names[cls_id]
                conf = float(box.conf[0].item())

                # Skip non-vehicle classes
                if cls_name not in ["car", "truck", "bus", "motorcycle", "bicycle"]:
                    continue

                # Record alternative detections
                alternatives.append({
                    "type": cls_name,
                    "confidence": conf
                })

                # Update best detection if confidence is higher
                if conf > best_conf:
                    best_conf = conf
                    best_type = cls_name

            # Sort alternatives by confidence
            alternatives.sort(key=lambda x: x["confidence"], reverse=True)

            return {
                "vehicle_type": best_type,
                "confidence": best_conf,
                "alternatives": alternatives
            }

        except Exception as e:
            logger.error(f"Error detecting vehicle type: {e}")
            return {
                "vehicle_type": "Error",
                "confidence": 0.0,
                "alternatives": []
            }

# Create singleton instance
vehicle_detector = VehicleTypeDetector()
