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
        current_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(current_dir, 'VOI', 'models')
        
        # Try different model formats in order of preference
        self.model_paths = {
            'saved_model': os.path.join(models_dir, 'orientation_model_converted_saved_model'),
            'keras': os.path.join(models_dir, 'orientation_model_converted.keras'),
            'h5': os.path.join(models_dir, 'orientation_model.h5')
        }
        
        self.model = None
        self.model_type = None
        self.load_model()

    def load_model(self):
        """Load the orientation detection model."""
        for model_type, model_path in self.model_paths.items():
            try:
                logger.info(f"Attempting to load {model_type} model from: {model_path}")
                
                if model_type == 'saved_model':
                    self.model = tf.keras.models.load_model(model_path)
                elif model_type == 'keras':
                    self.model = tf.keras.models.load_model(model_path)
                else:  # h5 format
                    self.model = tf.keras.models.load_model(
                        model_path,
                        compile=False,
                        custom_objects=None
                    )
                
                self.model_type = model_type
                logger.info(f"Successfully loaded {model_type} model")
                return True
                
            except Exception as e:
                logger.error(f"Error loading {model_type} model: {e}")
                continue
        
        logger.error("Failed to load model in any format")
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
            
            # Make prediction based on model type
            if self.model_type == 'saved_model':
                prediction = self.model.predict(img_batch, verbose=0)[0][0]
            else:
                prediction = self.model.predict(img_batch, verbose=0)[0][0]
            
            # Convert probability to orientation
            orientation = "Front-facing" if prediction > 0.5 else "Rear-facing"
            confidence = float(prediction if prediction > 0.5 else 1 - prediction)
            
            return {
                "orientation": orientation,
                "confidence": confidence,
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
