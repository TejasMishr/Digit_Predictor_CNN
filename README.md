# Digit Recognition with CNN on MNIST Dataset

This project implements a Convolutional Neural Network (CNN) in TensorFlow/Keras to recognize handwritten digits from the MNIST dataset.

## Project Overview

- **Data Preparation**: The MNIST dataset is loaded, reshaped to a 4D array, and normalized.
- **Model Architecture**: A CNN model is created with multiple convolutional and pooling layers, followed by fully connected dense layers.
- **Training**: The model is trained on the training data and performance metrics are recorded.
- **Evaluation & Visualization**: The model is used to predict digits from test images, and accuracy/loss curves are plotted.
- **Model Saving & Loading**: The trained model is saved as an .h5 file and reloaded for predictions.
- **Predictions**: Predictions are made for individual and multiple test images.

## Code Details

### Libraries

- **tensorflow** and **keras** for building and training the CNN model
- **matplotlib** and **numpy** for data manipulation and visualization

### Model Preparation

#### Load Dataset

The MNIST dataset, consisting of 28x28 grayscale images of handwritten digits (0-9), is loaded.

#### Reshape and Normalize

Data is reshaped to a 4D tensor format and pixel values are normalized.

### Model Architecture

A sequential CNN model with the following layers:

- **Convolutional layers** with ReLU activation
- **MaxPooling layers** to reduce spatial dimensions
- **Flatten layer** to convert 2D matrices to a vector
- **Dense layers** with ReLU activation for fully connected layers
- **Output layer** with sigmoid activation to predict class probabilities

### Model Compilation

The model uses:

- **Optimizer**: Adam
- **Loss**: Sparse Categorical Crossentropy
- **Metrics**: Accuracy

### Training

The model is trained for 10 epochs using the training images and labels.

### Visualization of Training Results

Accuracy and loss for each epoch are plotted using Matplotlib to visualize model performance.

### Saving and Loading Model

The model is saved to `Digit_CNN.h5` after training and is later reloaded for making predictions.

### Making Predictions

#### Single Image Prediction

Predicts the digit for a single test image.

#### Multiple Images Prediction

Predicts digits for a batch of test images.

## Instructions

### Requirements

Install the following Python libraries before running the code:
