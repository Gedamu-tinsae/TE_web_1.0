import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras.preprocessing import image
import cv2
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add project root to path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Path configurations
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_path = os.path.join(project_root, "models", "VMI")  # Updated path to VMI subdirectory

# Configuration
TARGET_SIZE = (224, 224)  # Input image size for model
MODEL_TYPE = "make"        # "make" or "model"
BASE_MODEL = "mobilenet"   # "efficientnet", "mobilenet", or "resnet"
TOP_K = 5                 # Number of top predictions to show
DEBUG_MODE = True         # Set to True to enable debug information

def load_class_mapping(model_type="make"):
    """Load class mapping from JSON file."""
    mapping_file = os.path.join(models_path, f"{model_type}_classes.json")
    if not os.path.exists(mapping_file):
        print(f"Error: Class mapping file not found at {mapping_file}")
        return None
    
    with open(mapping_file, 'r') as f:
        class_mapping = json.load(f)
    
    return class_mapping

def load_vehicle_model(model_type="make", base_model="mobilenet"):
    """Load the trained vehicle make/model classifier."""
    # Skip trying to load architecture from JSON - always rebuild the model
    print(f"Building {base_model} model for {model_type} classification...")
    try:
        # Load class mapping to get number of classes
        class_mapping = load_class_mapping(model_type)
        if class_mapping is None:
            print("Could not determine number of classes. Cannot recreate model.")
            return None
            
        num_classes = len(class_mapping)
        print(f"Building {base_model} model with {num_classes} classes")
        
        # Import the build_model function
        from importlib import import_module
        train_module = import_module('VMI.train_model')
        build_model = getattr(train_module, 'build_model')
        
        # Temporarily set the global variables needed by build_model
        original_base_model = train_module.BASE_MODEL
        train_module.BASE_MODEL = base_model
        
        # Build the model
        model = build_model(num_classes)
        
        # Restore original values
        train_module.BASE_MODEL = original_base_model
        
        # Try to load weights
        potential_weights_files = [
            os.path.join(models_path, f"{model_type}_{base_model}_weights.h5"),
            os.path.join(models_path, f"final_{model_type}_{base_model}_weights.h5"),
            os.path.join(models_path, f"partial_{model_type}_{base_model}_weights.h5"),
        ]
        
        weights_loaded = False
        for weights_path in potential_weights_files:
            if os.path.exists(weights_path):
                print(f"Loading weights from {weights_path}")
                try:
                    model.load_weights(weights_path)
                    print("Successfully loaded weights into built model")
                    weights_loaded = True
                    break
                except Exception as e:
                    print(f"Error loading weights from {weights_path}: {e}")
        
        if not weights_loaded:
            print("Warning: No weights file found or could be loaded. Using uninitialized model.")
        
        return model
    except Exception as e:
        print(f"Error building model: {e}")
    
    print("Failed to build and load model.")
    return None

def preprocess_image(img_path, target_size=TARGET_SIZE):
    """Preprocess image for model prediction."""
    # Check if image exists
    if not os.path.exists(img_path):
        print(f"Error: Image not found at {img_path}")
        return None, None
    
    # Load and preprocess image
    try:
        # Using the exact same preprocessing as in training
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Could not read image at {img_path}")
            return None, None
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_img = img.copy()  # Keep original for display
        img = cv2.resize(img, target_size)
        img_array = np.array(img) / 255.0  # Normalize to [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        # Debug information
        if DEBUG_MODE:
            print(f"Preprocessed image shape: {img_array.shape}")
            print(f"Image min/max values: {np.min(img_array)}/{np.max(img_array)}")
        
        return img_array, original_img
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def predict_vehicle(model, img_array, class_mapping, top_k=5):
    """Make prediction and return top-k classes with probabilities."""
    if model is None or img_array is None or class_mapping is None:
        return None
    
    # Make prediction
    try:
        preds = model.predict(img_array, verbose=0)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None
    
    if DEBUG_MODE:
        # Print some debug info about the prediction distribution
        print("\nDebug: Prediction Statistics")
        print(f"Max confidence: {np.max(preds[0]):.4f}")
        print(f"Min confidence: {np.min(preds[0]):.4f}")
        print(f"Mean confidence: {np.mean(preds[0]):.4f}")
        print(f"Confidence std dev: {np.std(preds[0]):.4f}")
        print(f"Number of classes with >0.1 confidence: {np.sum(preds[0] > 0.1)}")
        print(f"Number of classes with >0.01 confidence: {np.sum(preds[0] > 0.01)}")
        
        # Check for uniform distribution (untrained model)
        uniform_threshold = 0.005  # Expected standard deviation for a trained model
        if np.std(preds[0]) < uniform_threshold:
            print("\n⚠️ WARNING: Predictions appear to be nearly uniform!")
            print("This suggests the model has not learned meaningful features.")
            print("Consider retraining with a lower learning rate or different architecture.")
        
        # Check if max probability is too low for confidence
        if np.max(preds[0]) < 0.1:
            print("\n⚠️ WARNING: Maximum prediction confidence is very low!")
            print("The model doesn't seem confident in any class.")
            print("This might indicate insufficient training data or a problem with feature extraction.")
    
    # Get top-k indices and probabilities
    top_indices = np.argsort(preds[0])[-top_k:][::-1]
    top_probs = preds[0][top_indices]
    
    # Map indices to class names
    results = []
    for i, idx in enumerate(top_indices):
        class_name = class_mapping.get(str(idx), f"Unknown ({idx})")
        results.append({
            'class': class_name,
            'probability': float(top_probs[i]),
            'index': int(idx)
        })
    
    return results

def display_prediction(img, results, model_type="make", figsize=(10, 8)):
    """Display image with prediction results."""
    if img is None or results is None:
        return
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Display image
    plt.imshow(img)
    plt.axis('off')
    
    # Create title with predictions
    title = f"Top {len(results)} {model_type.capitalize()} Predictions:\n"
    for i, res in enumerate(results):
        title += f"{i+1}. {res['class']}: {res['probability']:.4f}\n"
    
    plt.title(title)
    plt.tight_layout()
    plt.show()

def analyze_model_weights(model_type="make", base_model="efficientnet"):
    """Analyze the model weights to check for proper training."""
    model = load_vehicle_model(model_type, base_model)
    if model is None:
        print("Could not load model for analysis")
        return
    
    print("\n===== Model Weight Analysis =====")
    
    # Check for "dead" layers (all zeros or very small weights)
    zero_weight_layers = []
    small_weight_layers = []
    
    for layer in model.layers:
        if len(layer.weights) > 0:
            for w in layer.weights:
                weights = w.numpy()
                if np.all(weights == 0):
                    zero_weight_layers.append(layer.name)
                elif np.max(np.abs(weights)) < 0.001:
                    small_weight_layers.append(layer.name)
    
    print(f"Layers with all zero weights: {zero_weight_layers or 'None'}")
    print(f"Layers with very small weights: {small_weight_layers or 'None'}")
    
    # Check for large variance in weights
    large_variance_layers = []
    for layer in model.layers:
        if len(layer.weights) > 0:
            for w in layer.weights:
                weights = w.numpy()
                if np.var(weights) > 1.0:
                    large_variance_layers.append(layer.name)
    
    print(f"Layers with large weight variance: {large_variance_layers or 'None'}")
    
    # Check final layer (output) weights
    if len(model.layers) > 0:
        output_layer = model.layers[-1]
        if len(output_layer.weights) > 0:
            output_weights = output_layer.weights[0].numpy()
            print(f"\nOutput layer weight statistics:")
            print(f"  Mean: {np.mean(output_weights):.6f}")
            print(f"  Std Dev: {np.std(output_weights):.6f}")
            print(f"  Min: {np.min(output_weights)::.6f}")
            print(f"  Max: {np.max(output_weights):.6f}")
            
            # Add histogram visualization of final layer
            plt.figure(figsize=(10, 6))
            plt.hist(output_weights.flatten(), bins=50)
            plt.title('Output Layer Weights Distribution')
            plt.xlabel('Weight Value')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
    
    # Check feature extraction layers (make sure they're working)
    if base_model == "efficientnet":
        # Check the last convolutional layer of EfficientNet
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D) and 'top_conv' in layer.name:
                conv_weights = layer.weights[0].numpy()
                print(f"\nFeature extraction layer statistics ({layer.name}):")
                print(f"  Mean: {np.mean(conv_weights):.6f}")
                print(f"  Std Dev: {np.std(conv_weights):.6f}")
                print(f"  Min: {np.min(conv_weights):.6f}")
                print(f"  Max: {np.max(conv_weights):.6f}")
                break
    
    # Check feature activations on sample image
    print("\nAnalyzing activations on a test image...")
    test_dir = os.path.join(project_root, "test_images")
    test_images = []
    if os.path.exists(test_dir):
        for ext in ['.jpg', '.jpeg', '.png']:
            test_images.extend(list(Path(test_dir).glob(f"*{ext}")))
    
    if test_images:
        img_path = str(test_images[0])
        img_array, _ = preprocess_image(img_path)
        
        if img_array is not None:
            # Create a model that outputs the feature map from the last convolutional layer
            for i, layer in enumerate(model.layers):
                if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
                    feature_layer_index = i - 1
                    break
            else:
                feature_layer_index = len(model.layers) - 5  # Fallback to a reasonable guess
            
            try:
                feature_model = tf.keras.Model(inputs=model.inputs, 
                                           outputs=model.layers[feature_layer_index].output)
                features = feature_model.predict(img_array)
                
                print(f"\nFeature activation statistics:")
                print(f"  Shape: {features.shape}")
                print(f"  Mean: {np.mean(features):.6f}")
                print(f"  Std Dev: {np.std(features):.6f}")
                print(f"  Min: {np.min(features):.6f}")
                print(f"  Max: {np.max(features):.6f}")
                
                # Visualize feature map activations (average across channels)
                plt.figure(figsize=(8, 8))
                plt.imshow(np.mean(features[0], axis=2), cmap='viridis')
                plt.colorbar()
                plt.title('Feature Map Activation (Average)')
                plt.tight_layout()
                plt.show()
                
                # Visualize top 16 feature maps
                if features.shape[-1] >= 16:
                    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
                    for i, ax in enumerate(axes.flat):
                        if i < 16:
                            ax.imshow(features[0, :, :, i], cmap='viridis')
                            ax.set_title(f'Filter {i}')
                        ax.axis('off')
                    plt.tight_layout()
                    plt.suptitle('Top 16 Feature Maps', fontsize=16)
                    plt.subplots_adjust(top=0.95)
                    plt.show()
            except Exception as e:
                print(f"Error visualizing features: {e}")

def predict_image_file(img_path, model_type="make", base_model="mobilenet", top_k=5):
    """Predict vehicle make/model for a single image file."""
    # Load class mapping and model
    class_mapping = load_class_mapping(model_type)
    model = load_vehicle_model(model_type, base_model)
    
    if class_mapping is None or model is None:
        return
    
    # Preprocess image
    img_array, img = preprocess_image(img_path)
    if img_array is None:
        return
    
    # Make prediction
    results = predict_vehicle(model, img_array, class_mapping, top_k)
    if results is None:
        return
    
    # Display results
    print(f"\nPrediction for {os.path.basename(img_path)}:")
    for i, res in enumerate(results):
        print(f"{i+1}. {res['class']}: {res['probability']:.4f}")
    
    # Display image with predictions
    display_prediction(img, results, model_type)
    
    # If in debug mode, show the preprocessed image that's fed to the model
    if DEBUG_MODE:
        print("\nDebug: Showing preprocessed input image")
        plt.figure(figsize=(6, 6))
        plt.imshow(img_array[0])  # Show the actual tensor sent to the model
        plt.title("Preprocessed Input Image")
        plt.axis('off')
        plt.show()
    
    return results

def predict_directory(dir_path, model_type="make", base_model="efficientnet", top_k=5, limit=10):
    """Predict vehicle make/model for all images in a directory."""
    # Load class mapping and model once
    class_mapping = load_class_mapping(model_type)
    model = load_vehicle_model(model_type, base_model)
    
    if class_mapping is None or model is None:
        return
    
    # Get all image files in directory
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    img_files = []
    for ext in image_extensions:
        img_files.extend(list(Path(dir_path).glob(f"*{ext}")))
        img_files.extend(list(Path(dir_path).glob(f"*{ext.upper()}")))
    
    # Limit number of images to process
    img_files = img_files[:limit]
    
    if not img_files:
        print(f"No image files found in {dir_path}")
        return
    
    # Process each image
    for img_path in img_files:
        # Preprocess image
        img_array, img = preprocess_image(str(img_path))
        if img_array is None:
            continue
        
        # Make prediction
        results = predict_vehicle(model, img_array, class_mapping, top_k)
        if results is None:
            continue
        
        # Display results
        print(f"\nPrediction for {img_path.name}:")
        for i, res in enumerate(results):
            print(f"{i+1}. {res['class']}: {res['probability']:.4f}")
        
        # Display image with predictions
        display_prediction(img, results, model_type)

def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict vehicle make/model from image.')
    parser.add_argument('--image', type=str, help='Path to image file')
    parser.add_argument('--dir', type=str, help='Path to directory of images')
    parser.add_argument('--model-type', type=str, default=MODEL_TYPE, choices=['make', 'model'], 
                        help='Whether to predict make only or make and model')
    parser.add_argument('--base-model', type=str, default=BASE_MODEL, 
                        choices=['efficientnet', 'mobilenet', 'resnet'], 
                        help='Base model architecture')
    parser.add_argument('--top-k', type=int, default=TOP_K, 
                        help='Number of top predictions to show')
    parser.add_argument('--limit', type=int, default=10, 
                        help='Limit number of images to process in directory mode')
    parser.add_argument('--debug', action='store_true', 
                        help='Enable debug mode with additional information')
    parser.add_argument('--analyze', action='store_true',
                        help='Analyze model weights for training issues')
    parser.add_argument('--deep-analyze', action='store_true',
                        help='Perform deep analysis of model weights and activations')
    parser.add_argument('--use-mobilenet', action='store_true',
                        help='Use MobileNet instead of EfficientNet (often better for small datasets)')
    
    args = parser.parse_args()
    
    # Set debug mode from command line
    global DEBUG_MODE
    if args.debug:
        DEBUG_MODE = True
        print("Debug mode enabled")
    
    # Override base model if requested
    model_choice = args.base_model
    if args.use_mobilenet:
        model_choice = "mobilenet"
        print("Switching to MobileNet architecture")
    
    # Handle analyze modes
    if args.analyze:
        analyze_model_weights(args.model_type, model_choice)
        return
    
    if args.deep_analyze:
        DEBUG_MODE = True
        analyze_model_weights(args.model_type, model_choice)
        # Then continue with normal prediction to see full debug info
    
    if args.image:
        predict_image_file(args.image, args.model_type, model_choice, args.top_k)
    elif args.dir:
        predict_directory(args.dir, args.model_type, model_choice, args.top_k, args.limit)
    else:
        # Try to use any test image in the test_images directory
        test_dir = os.path.join(project_root, "test_images")
        if os.path.exists(test_dir):
            img_files = []
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                img_files.extend(list(Path(test_dir).glob(f"*{ext}")))
                img_files.extend(list(Path(test_dir).glob(f"*{ext.upper()}")))
            
            if img_files:
                print(f"No image specified, using the first test image found: {img_files[0]}")
                predict_image_file(str(img_files[0]), args.model_type, model_choice, args.top_k)
            else:
                print("No image specified and no images found in test_images directory")
                print("Please use --image or --dir argument to specify input")
        else:
            print("No image specified and test_images directory not found")
            print("Please use --image or --dir argument to specify input")

if __name__ == "__main__":
    main()
