import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2, ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import shutil
from collections import defaultdict, Counter
import sys
import pathlib

# Add project root to path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Path configurations
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(project_root, "data", "Car Make and Model Recognition.v1i.coco")
train_path = os.path.join(data_path, "train")
annotations_path = os.path.join(train_path, "_annotations.coco.json")
models_path = os.path.join(project_root, "models", "VMI") 

# Create processed data directory for training
processed_data_dir = os.path.join(project_root, "data", "processed_data")
train_dir = os.path.join(processed_data_dir, "train")
val_dir = os.path.join(processed_data_dir, "val")
os.makedirs(models_path, exist_ok=True)
os.makedirs(processed_data_dir, exist_ok=True)
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Configuration parameters
TARGET_SIZE = (224, 224)  # Input image size
BATCH_SIZE = 16          # Batch size for training
INITIAL_LR = 0.00005     # Initial learning rate
EPOCHS = 100             # Maximum training epochs
VAL_SPLIT = 0.2         # Validation split ratio
TRAIN_MODE = "make"     # "make" or "model"
BASE_MODEL = "mobilenet"  # "efficientnet", "mobilenet", or "resnet"
TEST_MODE = False       # Quick test with minimal data
QUICK_TEST = False      # Quick test with more data than TEST_MODE
RECREATE_DIRS = True    # Force recreation of directories
CLASS_WEIGHT_BALANCING = True  # Handle class imbalance
USE_DATA_AUGMENTATION = True   # Enable data augmentation
USE_MIXED_PRECISION = True     # Enable mixed precision training

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

class TensorConverterCallback(Callback):
    """Converts TensorFlow tensors to Python native types in logs."""
    
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            converted_logs = {}
            for key, value in logs.items():
                if hasattr(value, 'numpy'):
                    try:
                        numpy_val = value.numpy()
                        if numpy_val.size == 1:
                            converted_logs[key] = float(numpy_val)
                        else:
                            converted_logs[key] = float(numpy_val.mean())
                    except:
                        converted_logs[key] = 0.0
                else:
                    converted_logs[key] = value
            
            logs.update(converted_logs)

def check_tf_gpu():
    """Check if TensorFlow is using GPU and configure it."""
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        print(f"GPU detected: {physical_devices}")
        try:
            for gpu in physical_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Memory growth enabled for GPUs")
        except RuntimeError as e:
            print(f"Error enabling memory growth: {e}")
    else:
        print("No GPU detected. Training will be slow on CPU.")
    
    if USE_MIXED_PRECISION:
        try:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("Mixed precision enabled (float16)")
        except Exception as e:
            print(f"Error enabling mixed precision: {e}")

def plot_training_history(history):
    """Plot training and validation accuracy/loss."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'], loc='upper left')
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'], loc='upper left')
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Save plot
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, f"training_history_{TRAIN_MODE}.png"))
    plt.show()

def load_annotations():
    """Load the COCO annotations file."""
    if not os.path.exists(annotations_path):
        print(f"Error: Annotations file not found at {annotations_path}")
        return None
    
    with open(annotations_path, 'r') as f:
        data = json.load(f)
    return data

def prepare_dataset_structure(data, mode="make"):
    """Prepare dataset structure for training by make or model."""
    if RECREATE_DIRS:
        # Recreate directories
        if os.path.exists(train_dir):
            shutil.rmtree(train_dir)
        if os.path.exists(val_dir):
            shutil.rmtree(val_dir)
        
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        print("Recreated training and validation directories.")
    else:
        # Check if directories already have content
        train_files = os.listdir(train_dir)
        val_files = os.listdir(val_dir)
        if train_files and val_files:
            class_count = sum(1 for item in train_files if os.path.isdir(os.path.join(train_dir, item)))
            print(f"Using existing data splits. Estimated {class_count} classes.")
            return class_count
    
    print(f"Preparing dataset for {mode.upper()} recognition...")
    
    # Get category and image mappings
    categories = {cat["id"]: cat["name"] for cat in data["categories"] if "id" in cat and "name" in cat}
    images = {img["id"]: img["file_name"] for img in data["images"] if "id" in img and "file_name" in img}
    
    # Map each image to its annotations
    image_annotations = defaultdict(list)
    for ann in data["annotations"]:
        if "image_id" in ann and "category_id" in ann:
            image_id = ann["image_id"]
            category_id = ann["category_id"]
            if image_id in images and category_id in categories:
                image_annotations[image_id].append(category_id)
    
    # Split into training and validation sets
    image_ids = list(image_annotations.keys())
    np.random.shuffle(image_ids)
    split_idx = int((1 - VAL_SPLIT) * len(image_ids))
    train_ids = image_ids[:split_idx]
    val_ids = image_ids[split_idx:]
    
    print(f"Total images: {len(image_ids)}")
    print(f"Training images: {len(train_ids)}")
    print(f"Validation images: {len(val_ids)}")
    
    # Process categories based on mode
    if mode == "make":
        # Extract make (first word of category name)
        make_map = {}
        for cat_id, name in categories.items():
            if name and ' ' in name:
                make = name.split()[0]
                make_map[cat_id] = make
            else:
                make_map[cat_id] = name
                
        # Create directory structure
        for make in set(make_map.values()):
            os.makedirs(os.path.join(train_dir, make), exist_ok=True)
            os.makedirs(os.path.join(val_dir, make), exist_ok=True)
        
        # Copy images to appropriate directories
        copied_count = 0
        skipped_count = 0
        
        for image_id in train_ids:
            if image_id in image_annotations and len(image_annotations[image_id]) > 0:
                cat_id = image_annotations[image_id][0]
                make = make_map[cat_id]
                src = os.path.join(train_path, images[image_id])
                dst = os.path.join(train_dir, make, os.path.basename(images[image_id]))
                
                if os.path.exists(src):
                    shutil.copy(src, dst)
                    copied_count += 1
                else:
                    skipped_count += 1
            
        for image_id in val_ids:
            if image_id in image_annotations and len(image_annotations[image_id]) > 0:
                cat_id = image_annotations[image_id][0]
                make = make_map[cat_id]
                src = os.path.join(train_path, images[image_id])
                dst = os.path.join(val_dir, make, os.path.basename(images[image_id]))
                
                if os.path.exists(src):
                    shutil.copy(src, dst)
                    copied_count += 1
                else:
                    skipped_count += 1
        
        return len(set(make_map.values()))
    
    else:  # mode == "model"
        # Use full category names for model-level classification
        for cat_id, name in categories.items():
            class_name = name.replace(" ", "_")  # Replace spaces with underscores
            os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
            os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
        
        # Copy images to appropriate directories
        copied_count = 0
        skipped_count = 0
        
        for image_id in train_ids:
            if image_id in image_annotations and len(image_annotations[image_id]) > 0:
                cat_id = image_annotations[image_id][0]
                class_name = categories[cat_id].replace(" ", "_")
                src = os.path.join(train_path, images[image_id])
                dst = os.path.join(train_dir, class_name, os.path.basename(images[image_id]))
                
                if os.path.exists(src):
                    shutil.copy(src, dst)
                    copied_count += 1
                else:
                    skipped_count += 1
            
        for image_id in val_ids:
            if image_id in image_annotations and len(image_annotations[image_id]) > 0:
                cat_id = image_annotations[image_id][0]
                class_name = categories[cat_id].replace(" ", "_")
                src = os.path.join(train_path, images[image_id])
                dst = os.path.join(val_dir, class_name, os.path.basename(images[image_id]))
                
                if os.path.exists(src):
                    shutil.copy(src, dst)
                    copied_count += 1
                else:
                    skipped_count += 1
    
    print(f"Copied {copied_count} images to {mode} classification directory structure")
    print(f"Skipped {skipped_count} images due to missing files")
    
    return len(categories)

def create_data_generators():
    """Create train and validation data generators with augmentation."""
    if USE_DATA_AUGMENTATION:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.25,
            height_shift_range=0.25,
            shear_range=0.25,
            zoom_range=0.25,
            horizontal_flip=True,
            brightness_range=[0.7, 1.3],
            fill_mode='nearest',
            validation_split=0.2 if VAL_SPLIT > 0 else 0.0
        )
    else:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2 if VAL_SPLIT > 0 else 0.0
        )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Load training data
    if VAL_SPLIT > 0:
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=TARGET_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=True,
            subset='training'
        )
        
        val_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=TARGET_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=False,
            subset='validation'
        )
    else:
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=TARGET_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=True
        )
        
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=TARGET_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )
    
    return train_generator, val_generator

def build_model(num_classes):
    """Build model architecture based on the configuration."""
    # Choose base model
    if BASE_MODEL == "efficientnet":
        base_model = EfficientNetB0(
            include_top=False, 
            weights='imagenet', 
            input_shape=(*TARGET_SIZE, 3)
        )
        fine_tune_at = 50
        
    elif BASE_MODEL == "mobilenet":
        base_model = MobileNetV2(
            include_top=False, 
            weights='imagenet', 
            input_shape=(*TARGET_SIZE, 3)
        )
        fine_tune_at = 50
        
    elif BASE_MODEL == "resnet":
        base_model = ResNet50(
            include_top=False, 
            weights='imagenet', 
            input_shape=(*TARGET_SIZE, 3)
        )
        fine_tune_at = 80
    
    else:
        raise ValueError(f"Unknown base model: {BASE_MODEL}")
    
    # Freeze early layers
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    # Unfreeze later layers
    for layer in base_model.layers[fine_tune_at:]:
        layer.trainable = True
    
    # Add classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile model with a lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=INITIAL_LR),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def compute_class_weights(train_generator):
    """Compute class weights to handle imbalanced classes."""
    if not CLASS_WEIGHT_BALANCING:
        return None
        
    counter = Counter(train_generator.classes)
    max_samples = float(max(counter.values()))
    class_weights = {class_id: max_samples/num_samples for class_id, num_samples in counter.items()}
    
    print(f"Computed class weights for {len(class_weights)} classes")
    return class_weights

def save_class_mapping(train_generator):
    """Save the class mapping to a JSON file."""
    class_indices = train_generator.class_indices
    class_mapping = {v: k for k, v in class_indices.items()}
    
    with open(os.path.join(models_path, f"{TRAIN_MODE}_classes.json"), 'w') as f:
        json.dump(class_mapping, f, indent=4)
    
    print(f"Saved class mapping to {models_path}/{TRAIN_MODE}_classes.json")

def train_model(model, train_generator, val_generator):
    """Train the model with checkpoints and callbacks."""
    # Define paths for saved weights
    weights_path = os.path.join(models_path, f"{TRAIN_MODE}_{BASE_MODEL}_weights.h5")
    
    # Define callbacks
    callbacks = [
        ModelCheckpoint(
            weights_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        TensorConverterCallback()
    ]
    
    # Compute class weights if enabled
    class_weights = compute_class_weights(train_generator) if CLASS_WEIGHT_BALANCING else None
    
    # Calculate steps per epoch and validation steps
    if TEST_MODE:
        steps_per_epoch = 5
        validation_steps = 2
        epochs = 2
        print("RUNNING IN TEST MODE: Using smaller batches and fewer steps for quick testing")
    elif QUICK_TEST:
        steps_per_epoch = 20
        validation_steps = 10
        epochs = 5
        print("RUNNING IN QUICK TEST MODE: Using fewer steps and epochs to validate the entire pipeline")
    else:
        steps_per_epoch = min(200, train_generator.samples // BATCH_SIZE)
        validation_steps = min(50, val_generator.samples // BATCH_SIZE)
        epochs = EPOCHS

    # Ensure at least one step for small datasets
    steps_per_epoch = max(1, steps_per_epoch)
    validation_steps = max(1, validation_steps)
    
    # Print model info
    print(f"\nTraining {TRAIN_MODE} classifier with {BASE_MODEL} backbone")
    print(f"Steps per epoch: {steps_per_epoch}, Validation steps: {validation_steps}")
    
    try:
        # Train model
        history = model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=val_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=2,
            class_weight=class_weights
        )
        
        # Save final weights
        final_weights_path = os.path.join(models_path, f"final_{TRAIN_MODE}_{BASE_MODEL}_weights.h5")
        model.save_weights(final_weights_path)
        print(f"Saved model weights to {final_weights_path}")
        
        return history, model
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        # Try to save partial weights
        try:
            partial_weights_path = os.path.join(models_path, f"partial_{TRAIN_MODE}_{BASE_MODEL}_weights.h5")
            model.save_weights(partial_weights_path)
            print(f"Saved partial weights to {partial_weights_path}")
        except Exception as e2:
            print(f"Could not save weights: {e2}")
            
        return None, model

def evaluate_model(model, val_generator):
    """Evaluate the model on the validation set."""
    evaluation = model.evaluate(val_generator)
    print(f"Validation loss: {evaluation[0]:.4f}")
    print(f"Validation accuracy: {evaluation[1]:.4f}")
    return evaluation

def main():
    """Main function to run the training pipeline."""
    from tensorflow.keras import backend as K
    
    # Check if TensorFlow is using GPU
    check_tf_gpu()
    
    # Print mode status
    if TEST_MODE:
        print("\n===== RUNNING IN TEST MODE =====")
    elif QUICK_TEST:
        print("\n===== RUNNING IN QUICK TEST MODE =====")
    
    # Load annotations
    print("Loading annotations...")
    data = load_annotations()
    if data is None:
        print("Failed to load annotations. Exiting.")
        return
    
    # Check if dataset has already been prepared
    train_dir = pathlib.Path('dataset/MAKE/train')
    val_dir = pathlib.Path('dataset/MAKE/val')
    dataset_prepared = train_dir.exists() and val_dir.exists() and any(train_dir.glob('*')) and any(val_dir.glob('*'))

    if not dataset_prepared:
        print("Preparing dataset for make classification...")
        # Recreate training and validation directories
        import shutil
        if os.path.exists('dataset/MAKE/train'):
            shutil.rmtree('dataset/MAKE/train')
        if os.path.exists('dataset/MAKE/val'):
            shutil.rmtree('dataset/MAKE/val')
        os.makedirs('dataset/MAKE/train', exist_ok=True)
        os.makedirs('dataset/MAKE/val', exist_ok=True)
        print("Recreated training and validation directories.")
        
        # Prepare dataset structure based on TRAIN_MODE
        print(f"Preparing dataset for {TRAIN_MODE} classification...")
        num_classes = prepare_dataset_structure(data, mode=TRAIN_MODE)
        print(f"Number of classes: {num_classes}")
    else:
        print("Dataset already prepared, skipping preparation steps.")
        num_classes = len(os.listdir(train_dir))

    # Create data generators
    print("Creating data generators...")
    train_generator, val_generator = create_data_generators()
    
    # Build model
    print(f"Building {BASE_MODEL} model for {num_classes} classes...")
    model = build_model(num_classes)
    
    # Print summary to file
    with open(os.path.join(models_path, f"{TRAIN_MODE}_{BASE_MODEL}_summary.txt"), "w") as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    # Train model
    print("Training model...")
    history, trained_model = train_model(model, train_generator, val_generator)
    
    # Save class mapping
    save_class_mapping(train_generator)
    
    # Evaluate model
    print("Evaluating model...")
    evaluate_model(trained_model, val_generator)
    
    # Plot training history
    if history is not None:
        plot_training_history(history)
    
    print("Training completed!")

if __name__ == "__main__":
    main()
