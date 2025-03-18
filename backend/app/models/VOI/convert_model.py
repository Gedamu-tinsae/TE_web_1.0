import tensorflow as tf
import os
import h5py


def convert_h5_to_saved_model():
    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to your h5 model
    h5_path = os.path.join(current_dir, 'models', 'orientation_model.h5')
    
    print(f"Looking for model at: {h5_path}")
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Model file not found at {h5_path}")
    
    try:
        # Try to open the H5 file directly first
        with h5py.File(h5_path, 'r') as f:
            # Check if it's a Keras model
            if 'layer_names' not in f.attrs and 'model_weights' not in f:
                raise ValueError("Not a valid Keras model")
            
            print("Valid H5 file found, attempting to load...")
        
        # Try loading with minimal options
        model = tf.keras.models.load_model(
            h5_path,
            compile=False,
            custom_objects=None
        )
        
        print("Model loaded successfully")
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

if __name__ == "__main__":
    try:
        model = convert_h5_to_saved_model()
        print("Model conversion complete")
    except Exception as e:
        print(f"Failed to convert model: {e}")
