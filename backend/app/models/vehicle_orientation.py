import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, save_model
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VehicleOrientationDetector:
    def __init__(self):
        # Fix the path calculation
        current_dir = os.path.dirname(os.path.abspath(__file__))  # /backend/app/models
        app_dir = os.path.dirname(current_dir)                    # /backend/app
        backend_dir = os.path.dirname(app_dir)                   # /backend
        
        # Now correctly point to /backend/VOI/models/orientation_model.h5
        self.model_path = os.path.join(backend_dir, 'VOI', 'models', 'orientation_model.h5')
        
        # Log the actual path for debugging
        logger.info(f"Looking for model at: {self.model_path}")
        
        self.model = None
        self.load_model()

    def load_model(self):
        """Load the orientation detection model."""
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found at {self.model_path}")
                raise FileNotFoundError(f"Model file not found at {self.model_path}")
            
            # Try loading with minimal options
            logger.info("Attempting model load with minimal configuration...")
            try:
                self.model = tf.keras.models.load_model(
                    self.model_path,
                    compile=False,
                    custom_objects=None
                )
            except Exception as e:
                # Try alternative loading method
                logger.info("First attempt failed, trying alternative loading...")
                self.model = load_model(
                    self.model_path,
                    compile=False
                )
            
            logger.info("Vehicle orientation model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading vehicle orientation model: {e}")
            return False

    def get_model_input_shape(self):
        """Get the expected input shape from the model."""
        try:
            input_shape = self.model.layers[0].input_shape
            if isinstance(input_shape, list):
                input_shape = input_shape[0]
            
            if input_shape and len(input_shape) == 4:
                return input_shape[1:3]
        except:
            logger.warning("Could not determine model input shape, using default (128, 128)")
        
        return (128, 128)

    def preprocess_image(self, image):
        """
        Preprocess an image for orientation prediction.
        """
        if image is None:
            logger.error("Invalid input image")
            return None
        
        # Convert from BGR to RGB (Keras models typically use RGB)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image to target size
        target_size = self.get_model_input_shape()
        img = cv2.resize(img, target_size)
        
        # Normalize pixel values to [0, 1]
        img = img / 255.0
        
        return img

    def predict(self, image):
        """
        Predict if a vehicle is facing toward the camera (front) or away (rear).
        Returns a dict with orientation and confidence.
        """
        try:
            if self.model is None:
                raise ValueError("Model not loaded")

            # Preprocess the image
            img = self.preprocess_image(image)
            if img is None:
                raise ValueError("Failed to preprocess image")

            # Add batch dimension
            img_batch = np.expand_dims(img, axis=0)
            
            # Make prediction
            prediction = self.model.predict(img_batch)[0][0]
            
            # Convert probability to orientation
            # > 0.5 is front (to_camera=True), <= 0.5 is rear (to_camera=False)
            orientation = "Front-facing" if prediction > 0.5 else "Rear-facing"
            confidence = prediction if prediction > 0.5 else 1 - prediction
            
            return {
                "orientation": orientation,
                "confidence": float(confidence),
                "is_front": bool(prediction > 0.5)
            }

        except Exception as e:
            logger.error(f"Error predicting vehicle orientation: {e}")
            return {
                "orientation": "Unknown",
                "confidence": 0.0,
                "is_front": None,
                "error": str(e)
            }

# Create a singleton instance
vehicle_orientation_detector = VehicleOrientationDetector()
