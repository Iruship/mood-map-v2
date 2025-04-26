# Neural Network for Body Language Detection

This project provides tools to train and run a neural network for body language detection using MediaPipe body landmarks. It replaces the traditional machine learning approach with a deep learning solution.

## Requirements

Install the required packages:

```bash
pip install tensorflow numpy pandas matplotlib opencv-python mediapipe scikit-learn
```

## Files Overview

- `body_language_nn_trainer.py` - Script to train the neural network model
- `body_language_nn_inference.py` - Script to run inference with the trained model
- `coords.csv` - Dataset containing body landmarks extracted using MediaPipe
- `feature_count.txt` - Information about the feature dimensions

## Training the Model

To train a neural network model using the existing `coords.csv` file:

```bash
python body_language_nn_trainer.py --data coords.csv --save-dir body_language_nn_model --epochs 100 --batch-size 32
```

Options:
- `--data`: Path to the input CSV file with landmarks (default: coords.csv)
- `--feature-count`: Path to save feature count information (default: feature_count.txt)
- `--save-dir`: Directory to save the trained model (default: body_language_nn_model)
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size for training (default: 32)

The training will:
1. Load and preprocess the landmark data
2. Build a neural network model with multiple dense layers
3. Train the model with early stopping to prevent overfitting
4. Save the trained model, scalers, label encoders, and training plots
5. Print evaluation metrics on the test set

## Running Inference

After training, you can use the model to detect body language in images or videos:

### Process an image:

```bash
python body_language_nn_inference.py --model-dir body_language_nn_model image path/to/image.jpg --output output.jpg
```

### Process a video:

```bash
python body_language_nn_inference.py --model-dir body_language_nn_model video path/to/video.mp4 --output output.mp4
```

Options:
- `--model-dir`: Directory containing the trained model files

## Neural Network Architecture

The model architecture includes:
- Input layer with a dense layer of 256 neurons with ReLU activation
- Multiple hidden layers with batch normalization and dropout for regularization
- Output layer with softmax activation for classification

The model achieves better generalization than traditional ML approaches through:
1. Batch normalization to stabilize and accelerate training
2. Dropout to prevent overfitting
3. Learning rate scheduling to optimize convergence
4. Early stopping to select the best model

## Using in Google Colab

To run this in Google Colab:

1. Upload the scripts and your `coords.csv` file to Colab
2. Install the required packages:
   ```python
   !pip install tensorflow numpy pandas matplotlib opencv-python mediapipe scikit-learn
   ```
3. Run the training script:
   ```python
   !python body_language_nn_trainer.py --data coords.csv
   ```
4. Run inference on test images/videos:
   ```python
   !python body_language_nn_inference.py image test_image.jpg --output result.jpg
   ```

## Performance Comparison

The neural network approach offers several advantages over the traditional ML model:
- Better feature extraction through deep layers
- Improved generalization to new poses
- Higher accuracy on complex gestures
- More robust to variations in lighting and camera angles

## Extending the Model

To add new body language classes:
1. Collect additional data for the new classes
2. Add the new data to coords.csv (ensure the same feature structure)
3. Retrain the model using the training script

## Troubleshooting

If you encounter issues:
- Ensure MediaPipe is properly installed and working
- Check that your input images/videos are in a standard format
- Verify that the coords.csv file has the correct structure
- Make sure feature dimensions match between training and inference 