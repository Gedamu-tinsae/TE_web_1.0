import os
import sys
import json
import numpy as np
import cv2
import logging
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VehicleMakeDetector:
    def __init__(self):
        # Initialize model paths
        self.models_path = self._get_models_path()
        self.model = None
        self.class_mapping = None
        self.target_size = (224, 224)  # Default target size for model input
        self.base_model = "mobilenet"  # Default base model
        self._load_model()
    
    def _get_models_path(self):
        """Get the path to the models directory"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, 'VMI')
    
    def _load_class_mapping(self):
        """Load class mapping from JSON file."""
        mapping_file = os.path.join(self.models_path, "make_classes.json")
        if not os.path.exists(mapping_file):
            logger.error(f"Error: Class mapping file not found at {mapping_file}")
            return None
        
        with open(mapping_file, 'r') as f:
            class_mapping = json.load(f)
        
        logger.info(f"Loaded {len(class_mapping)} class mappings for vehicle make")
        return class_mapping
    
    def _load_model(self):
        """Load the trained vehicle make classifier."""
        try:
            # Load class mapping to get number of classes
            self.class_mapping = self._load_class_mapping()
            if self.class_mapping is None:
                logger.error("Could not determine number of classes. Cannot load model.")
                return False
                
            num_classes = len(self.class_mapping)
            logger.info(f"Building {self.base_model} model with {num_classes} classes")
            
            # Import model function from the VMI module
            vmi_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(self.models_path))), 'VMI')
            sys.path.append(vmi_path)
            
            from importlib import import_module
            try:
                # First try to import from the VMI directory
                train_module = import_module('VMI.train_model')
            except ModuleNotFoundError:
                # Fall back to importing from a relative path
                sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
                train_module = import_module('VMI.train_model')
            
            build_model = getattr(train_module, 'build_model')
            
            # Temporarily set the global variables needed by build_model
            original_base_model = getattr(train_module, 'BASE_MODEL', self.base_model)
            setattr(train_module, 'BASE_MODEL', self.base_model)
            
            # Build the model
            self.model = build_model(num_classes)
            
            # Restore original values
            setattr(train_module, 'BASE_MODEL', original_base_model)
            
            # Try to load weights
            potential_weights_files = [
                os.path.join(self.models_path, f"make_{self.base_model}_weights.h5"),
                os.path.join(self.models_path, f"final_make_{self.base_model}_weights.h5"),
                os.path.join(self.models_path, f"partial_make_{self.base_model}_weights.h5"),
            ]
            
            weights_loaded = False
            for weights_path in potential_weights_files:
                if os.path.exists(weights_path):
                    logger.info(f"Loading weights from {weights_path}")
                    try:
                        self.model.load_weights(weights_path)
                        logger.info("Successfully loaded weights into built model")
                        weights_loaded = True
                        break
                    except Exception as e:
                        logger.error(f"Error loading weights from {weights_path}: {e}")
            
            if not weights_loaded:
                logger.warning("No weights file found or could be loaded. Using uninitialized model.")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error loading vehicle make model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def preprocess_image(self, img):
        """Preprocess image for model prediction."""
        try:
            # Check if input is a file path or an image array
            if isinstance(img, str):
                if not os.path.exists(img):
                    logger.error(f"Error: Image not found at {img}")
                    return None
                
                # Load and preprocess image
                img = cv2.imread(img)
                if img is None:
                    logger.error(f"Error: Could not read image")
                    return None
            
            # If img is already a numpy array, we'll continue processing
            if not isinstance(img, np.ndarray):
                logger.error(f"Error: Invalid image format, expected numpy array or file path")
                return None
                
            # Convert BGR to RGB (if needed)
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize to target size
            img_resized = cv2.resize(img, self.target_size)
            
            # Normalize to [0, 1]
            img_array = np.array(img_resized) / 255.0
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return None
    
    def detect(self, img, top_k=3):
        """
        Detect vehicle make from an image.
        
        Args:
            img: Image file path or numpy array
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary with vehicle make, confidence, and alternatives
        """
        if self.model is None or self.class_mapping is None:
            return {
                "make": "Unknown",
                "confidence": 0.0,
                "alternatives": []
            }
        
        # Preprocess the image
        img_array = self.preprocess_image(img)
        if img_array is None:
            return {
                "make": "Unknown",
                "confidence": 0.0,
                "alternatives": []
            }
        
        # Make prediction
        try:
            preds = self.model.predict(img_array, verbose=0)
            
            # Get top-k indices and probabilities
            top_indices = np.argsort(preds[0])[-top_k:][::-1]
            top_probs = preds[0][top_indices]
            
            # Map indices to class names
            alternatives = []
            for i, idx in enumerate(top_indices):
                class_name = self.class_mapping.get(str(idx), f"Unknown-{idx}")
                alternatives.append({
                    "make": class_name,
                    "confidence": float(top_probs[i])
                })
            
            # First result is the top prediction
            top_make = alternatives[0]["make"] if alternatives else "Unknown"
            top_confidence = alternatives[0]["confidence"] if alternatives else 0.0
            
            return {
                "make": top_make,
                "confidence": top_confidence,
                "alternatives": alternatives
            }
        except Exception as e:
            logger.error(f"Error detecting vehicle make: {e}")
            return {
                "make": "Error",
                "confidence": 0.0,
                "alternatives": []
            }

# Create singleton instance
vehicle_make_detector = VehicleMakeDetector()
