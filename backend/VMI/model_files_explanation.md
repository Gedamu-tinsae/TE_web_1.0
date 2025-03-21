# Vehicle Model Files Explanation

## Files Generated During Training

When you run the training script (`train_model.py`), these files are created:

### Core Model Files (Used for Prediction)

1. **`{model_type}_{base_model}_weights.h5`** (e.g., `make_mobilenet_weights.h5`)
   - Contains the best model weights from training (highest validation accuracy)
   - The base_model can be: `mobilenet` (default), `efficientnet`, or `resnet`
   - **Used during prediction**: YES - loaded into the rebuilt model architecture

2. **`final_{model_type}_{base_model}_weights.h5`** (e.g., `final_make_mobilenet_weights.h5`)
   - Contains weights from the final training epoch
   - Used as fallback if best weights aren't available
   - **Used during prediction**: YES - as a backup option

3. **`{model_type}_classes.json`** (e.g., `make_classes.json`)
   - Maps class indices (0, 1, 2...) to actual class names ("Ford", "Toyota", etc.)
   - Needed to interpret predictions from numeric to human-readable form
   - **Used during prediction**: YES - translates predictions to car make names

### Auxiliary Files (Not Used for Prediction)

1. **`{model_type}_{base_model}_summary.txt`** (e.g., `make_mobilenet_summary.txt`)
   - Text description of the model architecture (layers, parameters)
   - For human understanding, not used by the model
   - **Used during prediction**: NO - just for documentation

2. **`partial_{model_type}_{base_model}_weights.h5`** (e.g., `partial_make_mobilenet_weights.h5`)
   - Emergency backup weights saved if training crashes
   - Only created if there's an error during training
   - **Used during prediction**: Only as fallback if other weight files don't exist

## Supported Base Models

Our training pipeline supports three different CNN architectures:

1. **MobileNetV2** (`mobilenet`) - Default choice
   - Lightweight model (~14MB)
   - Best for small datasets and limited GPU memory
   - Fine-tuned from layer 50 onwards

2. **EfficientNetB0** (`efficientnet`)
   - Medium-sized model (~20MB)
   - Good balance between size and accuracy
   - Fine-tuned from layer 50 onwards

3. **ResNet50** (`resnet`)
   - Larger model (~98MB)
   - Best for large datasets with many examples per class
   - Fine-tuned from layer 80 onwards

## How Prediction Works

The `predict.py` script:

1. Loads the **class mapping** from `{model_type}_classes.json` to get the number of classes
2. **Rebuilds the model architecture** using the same code that created it during training
3. Sets the appropriate base model (MobileNet, EfficientNet, or ResNet)
4. Loads the **weights** into the rebuilt model, checking these files in order:
   - `{model_type}_{base_model}_weights.h5` (best weights during training)
   - `final_{model_type}_{base_model}_weights.h5` (weights after training completion)
   - `partial_{model_type}_{base_model}_weights.h5` (emergency backup weights)

This approach eliminates serialization issues because we don't try to save/load the model architecture directly - we just rebuild it each time.

## Why This Approach Works

- **Eliminates JSON serialization errors** by never trying to save the architecture as JSON
- **More reliable** because model code defines the architecture, so it's always consistent
- **Lighter file storage** since we only save the weights, not the full model
- **Same results** since a rebuilt model with the same weights behaves identically
- **Flexible architecture** allows switching between different base models

## Pre-trained Base Model Weights

The model starts with pre-trained ImageNet weights which are downloaded automatically when you first create the model. These weights are the starting point for training and are automatically managed by TensorFlow.

## Why We Need to Rebuild the Architecture During Prediction

The prediction script rebuilds the model architecture for several important reasons:

### 1. TensorFlow Serialization Issues

We couldn't save both the architecture and weights together because TensorFlow encountered serialization errors:

- **Error message:** `Unable to serialize [2.0896919 2.1128857 2.1081853] to JSON. Unrecognized type <class 'tensorflow.python.framework.ops.EagerTensor'>`
- This error occurs because EagerTensors (TensorFlow's dynamic computation objects) can't be properly converted to JSON format

### 2. Benefits of the Rebuild Approach

This approach offers several advantages:

- **Extremely Fast Rebuilding**: Creating the model architecture takes just milliseconds - it's essentially defining the layers in code
- **Guaranteed consistency**: The same code that created the model during training recreates it during prediction
- **Eliminates version compatibility issues**: Different TensorFlow versions might interpret saved architectures differently
- **Avoids complex workarounds**: Other approaches like special JSON encoders still have edge cases that can fail
- **Lightweight storage**: Weight files are smaller than full model files
- **Model switching**: Allows trying different architectures with the same dataset

The model rebuilding process is nearly instantaneous compared to:
- The time it takes to load the weights (which are large files containing millions of parameters)
- The actual prediction computation time
- The image preprocessing time

### 3. Alternative Approaches (And Why They Didn't Work)

We tried several other approaches:

- **Saving the full model in .keras format**: Failed with EagerTensor serialization errors
- **Saving the architecture as JSON**: Failed with the same serialization issues
- **SavedModel format**: Still encountered issues with tensor serialization

The rebuild approach is the most reliable solution, especially when working with complex models like EfficientNet that have many custom layers and operations.

## How the Process Works

1. During training, we save only:
   - The weights (parameters learned by the model)
   - The class mapping (to interpret predictions)
   - Model summary (for documentation)

2. During prediction:
   - We use the same code to rebuild an identical architecture
   - We load the saved weights into this architecture
   - This gives us the exact same model behavior as the original

## Which Model to Use When

- **MobileNetV2**: For datasets with <1000 images or when running on low-memory GPUs (≤4GB)
- **EfficientNetB0**: For medium-sized datasets (1000-5000 images) and GPUs with 4-8GB memory
- **ResNet50**: For larger datasets (>5000 images) and GPUs with >8GB memory

For detailed information about the model architectures, see the `models.md` file in the same directory.
