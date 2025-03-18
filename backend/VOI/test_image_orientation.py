import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import argparse

def get_model_input_shape(model):
    """Get the expected input shape from the model."""
    try:
        input_shape = model.layers[0].input_shape
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        if input_shape and len(input_shape) == 4:
            return input_shape[1:3]
    except:
        print("Warning: Could not determine model input shape, using default (128, 128)")    
    return (128, 128)

def validate_image_path(image_path):
    """Validate the image path and file."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
        
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    if not any(image_path.lower().endswith(ext) for ext in valid_extensions):
        raise ValueError(f"Invalid image format. Supported formats: {valid_extensions}")

def preprocess_image(image_path, target_size=(128, 128)):
    """Preprocess an image for prediction."""
    # Validate image path
    validate_image_path(image_path)
    
    # Read and validate image
    img = cv2.imread(image_path)
    if img is None:
        raise IOError(f"Could not read image at {image_path}")
    
    print(f"Original image shape: {img.shape}")
    
    # Convert from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize image to target size
    img = cv2.resize(img, target_size)
    
    # Simple normalization to [0,1]
    img = img.astype(np.float32) / 255.0
    
    # Print preprocessed image stats
    print(f"Preprocessed image shape: {img.shape}")
    print(f"Value range: [{img.min():.3f}, {img.max():.3f}]")
    
    return img

def interpret_prediction(prediction):
    """Interpret model prediction with threshold and confidence calculation."""
    # Raw prediction thresholding without sigmoid
    threshold = 0.15  # Lower threshold since raw values tend to be small
    is_front = prediction >= threshold
    
    # Calculate confidence based on prediction value
    if is_front:
        confidence = min((prediction / threshold) * 100, 100)
    else:
        confidence = min(((threshold - prediction) / threshold) * 100, 100)
    
    orientation = "Front" if is_front else "Rear"
    
    # Print detailed debug info
    print(f"\nDetailed prediction analysis:")
    print(f"Raw prediction value: {prediction:.4f}")
    print(f"Threshold: {threshold}")
    print(f"Is front: {is_front}")
    print(f"Calculated confidence: {confidence:.2f}%")
    
    return orientation, confidence

def main():
    parser = argparse.ArgumentParser(description='Test vehicle orientation on a single image')
    parser.add_argument('--image', type=str, default=r"C:\Users\80\Desktop\sem 8\code\VOI\test_images\test\car416.png",
                       help='Path to the image file')
    parser.add_argument('--model', type=str, 
                        default=r'c:\Users\80\Desktop\sem 8\code\VOI\BoxCars\models\orientation_model.h5',
                        help='Path to the model file')
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model}")
    model = load_model(args.model)
    print(f"Model loaded successfully.")
    
    # Print model summary for debugging
    print("\nModel Summary:")
    model.summary()
    
    # Print model input shape
    input_shape = model.input_shape
    print(f"\nModel input shape: {input_shape}")
    target_size = get_model_input_shape(model)
    print(f"Target size: {target_size}")
    
    # Preprocess image
    print(f"\nProcessing image: {args.image}")
    try:
        img = preprocess_image(args.image, target_size)
        if img is None:
            return
        
        # Add batch dimension
        img_batch = np.expand_dims(img, axis=0)
        print(f"Input batch shape: {img_batch.shape}")
        
        # Make prediction with verbose logging
        print("\nMaking prediction...")
        prediction = model.predict(img_batch, verbose=1)[0][0]
        print(f"Raw prediction value: {prediction}")
        
        # Get orientation and confidence using consistent interpretation
        orientation, confidence = interpret_prediction(prediction)
        print(f"\nPrediction details:")
        print(f"- Raw prediction: {prediction:.4f}")
        print(f"- Orientation: {orientation}")
        print(f"- Confidence: {confidence:.2f}%")
        
        # Visualize the result
        visualize_results(img, prediction, orientation, confidence, args, r"C:\Users\80\Desktop\sem 8\code\VOI\test_images\results")
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise

def visualize_results(img, prediction, orientation, confidence, args, output_dir):
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img)  # Now img should be in correct range [0,1]
    plt.title(f"Image: {os.path.basename(args.image)}")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    # Normalize probabilities for visualization
    if prediction >= 0.15:  # Front
        front_prob = min(prediction, 1.0)
        rear_prob = 1 - front_prob
    else:  # Rear
        rear_prob = min(1 - prediction, 1.0)
        front_prob = 1 - rear_prob
    
    bars = plt.barh(['Front', 'Rear'], [front_prob, rear_prob], color=['blue', 'orange'])
    plt.xlim(0, 1)
    plt.xlabel('Probability')
    plt.title(f"Prediction: {orientation}\nConfidence: {confidence:.2f}%")
    
    # Add confidence values as text with correct probabilities
    for i, v in enumerate([front_prob, rear_prob]):
        plt.text(v + 0.01, i, f"{v:.2%}", va='center')
    
    # Save and show
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(args.image))[0]
    output_path = os.path.join(output_dir, f"{basename}_prediction.png")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Results saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    main()
