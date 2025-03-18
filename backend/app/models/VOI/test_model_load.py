import os
import h5py
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

def create_minimal_model():
    """Create a model matching the original architecture from H5 file"""
    model = Sequential([
        # First Conv Block
        Conv2D(24, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)),
        Conv2D(24, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # Second Conv Block
        Conv2D(48, (3, 3), activation='relu', padding='same'),
        Conv2D(48, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # Third Conv Block
        Conv2D(96, (3, 3), activation='relu', padding='same'),
        Conv2D(96, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # Dense Layers
        Flatten(),
        Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    return model

def test_load_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    h5_path = os.path.join(current_dir, 'models', 'orientation_model.h5')
    
    print(f"Looking for model at: {h5_path}")
    
    try:
        # First verify H5 file structure
        with h5py.File(h5_path, 'r') as f:
            print("\nH5 file structure:")
            print("Keys:", list(f.keys()))
            print("Attributes:", list(f.attrs.keys()))
            
            # Print model config to see the architecture
            if 'model_config' in f.attrs:
                print("\nModel config:")
                print(f.attrs['model_config'])
        
        # Create a fresh model with matching architecture
        print("\nCreating fresh model...")
        model = create_minimal_model()
        
        # Try to load weights from H5 file
        print("\nAttempting to load weights...")
        model.load_weights(h5_path)
        
        print("\nModel summary:")
        model.summary()
        
        # Save model in different format
        print("\nSaving model in different formats...")
        save_dir = os.path.join(current_dir, 'models')
        
        # Save as Keras format
        keras_path = os.path.join(save_dir, 'orientation_model.keras')
        model.save(keras_path, save_format='keras')
        print(f"Saved model in Keras format: {keras_path}")
        
        # Save as SavedModel format
        saved_model_path = os.path.join(save_dir, 'orientation_model_saved')
        model.save(saved_model_path, save_format='tf')
        print(f"Saved model in SavedModel format: {saved_model_path}")
        
        return model
            
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

if __name__ == "__main__":
    model = test_load_model()
    if model is not None:
        print("\nModel loaded successfully!")
    else:
        print("\nFailed to load model")
