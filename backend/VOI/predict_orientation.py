import os
import argparse
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from glob import glob

def get_model_input_shape(model):
    """Get the expected input shape from the model."""
    # This works for most model architectures
    try:
        # Try to get input shape from model's input layer
        input_shape = model.layers[0].input_shape
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        
        # Return shape without batch dimension (e.g., (None, 128, 128, 3) -> (128, 128, 3))
        if input_shape and len(input_shape) == 4:
            return input_shape[1:3]
    except:
        print("Warning: Could not determine model input shape, using default (128, 128)")
    
    return (128, 128)  # Default if we can't determine

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocess an image for prediction.
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return None
    
    # Convert from BGR to RGB (Keras models typically use RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize image to target size
    img = cv2.resize(img, target_size)
    
    # Normalize pixel values to [0, 1]
    img = img / 255.0
    
    return img

def predict_orientation(model, image_path):
    """
    Predict if a vehicle is facing toward the camera (front) or away (rear).
    """
    # Get model input shape
    target_size = get_model_input_shape(model)
    print(f"Model expects input shape: {target_size}")
    
    # Preprocess the image with the correct size
    img = preprocess_image(image_path, target_size)
    if img is None:
        return None, None, None
    
    # Add batch dimension
    img_batch = np.expand_dims(img, axis=0)
    
    # Make prediction
    prediction = model.predict(img_batch)[0][0]
    
    # Convert probability to class label
    # > 0.5 is front (to_camera=True), <= 0.5 is rear (to_camera=False)
    orientation = "Front" if prediction > 0.5 else "Rear"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    
    return orientation, confidence, img

def visualize_prediction(image_path, orientation, confidence, display_image=None):
    """
    Create a visualization of the prediction result.
    """
    # Get filename for display
    filename = os.path.basename(image_path)
    
    # If no display image was provided, read it from the path
    if display_image is None:
        display_image = cv2.imread(image_path)
        display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
    
    # Create a figure for displaying the image and prediction
    plt.figure(figsize=(8, 6))
    plt.imshow(display_image)
    
    # Add prediction information as text
    title = f"{filename}\nPredicted: {orientation} (Confidence: {confidence:.2%})"
    plt.title(title)
    
    # Remove axis ticks
    plt.axis('off')
    
    # Return the figure for saving or displaying
    return plt.gcf()

def process_directory(model, directory, output_dir=None):
    """
    Process all images in a directory and visualize predictions.
    """
    # Find all image files in the directory
    image_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'gif']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob(os.path.join(directory, f"*.{ext}")))
        image_files.extend(glob(os.path.join(directory, f"*.{ext.upper()}")))
    
    if not image_files:
        print(f"No image files found in {directory}")
        return
    
    print(f"Found {len(image_files)} image files in {directory}")
    
    # Process each image
    for image_path in image_files:
        try:
            # Predict orientation
            orientation, confidence, img = predict_orientation(model, image_path)
            
            if orientation is None:
                continue
            
            # Visualize prediction
            fig = visualize_prediction(image_path, orientation, confidence, img)
            
            # Save or display the visualization
            if output_dir:
                # Create output directory if it doesn't exist
                os.makedirs(output_dir, exist_ok=True)
                
                # Create output path
                basename = os.path.splitext(os.path.basename(image_path))[0]
                output_path = os.path.join(output_dir, f"{basename}_pred.png")
                
                # Save figure
                fig.savefig(output_path, bbox_inches='tight')
                plt.close(fig)
                
                print(f"Prediction for {os.path.basename(image_path)}: {orientation} ({confidence:.2%})")
            else:
                plt.show()
        
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Predict vehicle orientation using trained model')
    parser.add_argument('--model', type=str, default='c:\\Users\\80\\Desktop\\sem 8\\code\\VOI\\BoxCars\\models\\orientation_model.h5',
                        help='Path to the trained model')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input image or directory containing images')
    parser.add_argument('--output', type=str, default=None,
                        help='Directory to save prediction visualizations (optional)')
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found at {args.model}")
        return
    
    # Load the model
    print(f"Loading model from {args.model}")
    model = load_model(args.model)
    
    # Check if input is a file or directory
    if os.path.isfile(args.input):
        # Process a single image
        print(f"Processing single image: {args.input}")
        orientation, confidence, img = predict_orientation(model, args.input)
        
        if orientation is not None:
            # Visualize prediction
            fig = visualize_prediction(args.input, orientation, confidence, img)
            
            # Save or display the visualization
            if args.output:
                os.makedirs(args.output, exist_ok=True)
                basename = os.path.splitext(os.path.basename(args.input))[0]
                output_path = os.path.join(args.output, f"{basename}_pred.png")
                fig.savefig(output_path, bbox_inches='tight')
                plt.close(fig)
                print(f"Prediction saved to {output_path}")
            else:
                plt.show()
            
            print(f"Prediction: {orientation} (Confidence: {confidence:.2%})")
        
    elif os.path.isdir(args.input):
        # Process a directory of images
        print(f"Processing directory: {args.input}")
        process_directory(model, args.input, args.output)
        
        if args.output:
            print(f"Predictions saved to {args.output}")
    
    else:
        print(f"Error: Input path {args.input} does not exist")

if __name__ == '__main__':
    main()
