"""
Standalone Prediction Module for Face Detection
Assignment 2 - Computer Vision
Author: Muhammad Mahad
Date: November 2024

This module provides the prediction functionality for face detection
using the trained shallow neural network model.
"""

import numpy as np
import json
from PIL import Image
import os

class FacePredictor:
    """
    Face detection predictor using trained neural network.
    """
    
    def __init__(self, model_path='trained_model.json'):
        """
        Initialize the predictor by loading the trained model.
        
        Args:
            model_path: Path to the saved model file
        """
        self.model_path = model_path
        self.model = None
        self.mean = None
        self.std = None
        
        # Load model and normalization parameters
        self.load_model()
        self.load_normalization_params()
        
    def load_model(self):
        """
        Load the trained model from file.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        with open(self.model_path, 'r') as f:
            model_params = json.load(f)
        
        # Reconstruct the model
        self.input_size = model_params['input_size']
        self.hidden_size = model_params['hidden_size']
        self.output_size = model_params['output_size']
        self.W1 = np.array(model_params['W1'])
        self.b1 = np.array(model_params['b1'])
        self.W2 = np.array(model_params['W2'])
        self.b2 = np.array(model_params['b2'])
        
        print(f"Model loaded successfully from {self.model_path}")
        print(f"Model architecture: Input({self.input_size}) -> Hidden({self.hidden_size}) -> Output({self.output_size})")
    
    def load_normalization_params(self):
        """
        Load normalization parameters for feature preprocessing.
        """
        if os.path.exists('normalization_mean.npy'):
            self.mean = np.load('normalization_mean.npy')
            self.std = np.load('normalization_std.npy')
            print("Normalization parameters loaded")
        else:
            print("Warning: Normalization parameters not found. Using default values.")
            self.mean = 0.0
            self.std = 1.0
    
    def relu(self, z):
        """ReLU activation function."""
        return np.maximum(0, z)
    
    def sigmoid(self, z):
        """Sigmoid activation function."""
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))
    
    def forward_pass(self, features):
        """
        Perform forward pass through the network.
        
        Args:
            features: Input features
            
        Returns:
            Output probabilities
        """
        # Hidden layer
        z1 = np.dot(features, self.W1) + self.b1
        a1 = self.relu(z1)
        
        # Output layer
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self.sigmoid(z2)
        
        return a2
    
    def preprocess_image(self, image_path, target_size=(64, 64)):
        """
        Preprocess a single image for prediction.
        
        Args:
            image_path: Path to the image file
            target_size: Target size for the image
            
        Returns:
            Preprocessed features
        """
        # Load and preprocess image
        img = Image.open(image_path)
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize to target size
        img = img.resize(target_size, Image.LANCZOS)
        
        # Convert to numpy array and normalize
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Flatten the image
        features = img_array.reshape(1, -1)
        
        # Apply normalization
        features = (features - self.mean) / self.std
        
        return features
    
    def predict_image(self, image_path):
        """
        Predict whether an image contains a face.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Prediction (0 or 1) and confidence score
        """
        # Preprocess the image
        features = self.preprocess_image(image_path)
        
        # Make prediction
        probability = self.forward_pass(features)
        prediction = int(probability >= 0.5)
        
        return prediction, float(probability[0, 0])
    
    def predict_batch(self, image_paths):
        """
        Predict for multiple images.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of predictions and confidence scores
        """
        results = []
        
        for path in image_paths:
            try:
                pred, conf = self.predict_image(path)
                results.append({
                    'image': path,
                    'prediction': pred,
                    'confidence': conf,
                    'label': 'Face' if pred == 1 else 'No Face'
                })
            except Exception as e:
                results.append({
                    'image': path,
                    'error': str(e)
                })
        
        return results


def prediction(features):
    """
    Required prediction function for the assignment.
    
    Args:
        features: Input features as numpy array
                 Can be a single sample (1D array) or batch (2D array)
    
    Returns:
        Estimated count of faces detected
    """
    # Ensure features is 2D array
    if len(features.shape) == 1:
        features = features.reshape(1, -1)
    
    # Load the trained model parameters
    if not os.path.exists('trained_model.json'):
        raise FileNotFoundError("Trained model not found. Please train the model first.")
    
    with open('trained_model.json', 'r') as f:
        model_params = json.load(f)
    
    # Extract model parameters
    W1 = np.array(model_params['W1'])
    b1 = np.array(model_params['b1'])
    W2 = np.array(model_params['W2'])
    b2 = np.array(model_params['b2'])
    
    # Load normalization parameters if available
    if os.path.exists('normalization_mean.npy'):
        mean = np.load('normalization_mean.npy')
        std = np.load('normalization_std.npy')
        features = (features - mean) / std
    
    # Forward pass
    # Hidden layer with ReLU
    z1 = np.dot(features, W1) + b1
    a1 = np.maximum(0, z1)  # ReLU
    
    # Output layer with Sigmoid
    z2 = np.dot(a1, W2) + b2
    z2 = np.clip(z2, -500, 500)  # Prevent overflow
    a2 = 1.0 / (1.0 + np.exp(-z2))  # Sigmoid
    
    # Convert probabilities to binary predictions
    predictions = (a2 >= 0.5).astype(int)
    
    # Return the count of detected faces
    return np.sum(predictions)


def main():
    """
    Example usage of the prediction module.
    """
    print("=" * 60)
    print("FACE DETECTION PREDICTION MODULE")
    print("=" * 60)
    
    # Check if model exists
    if not os.path.exists('trained_model.json'):
        print("\nError: Trained model not found!")
        print("Please run 'python train_model.py' first to train the model.")
        return
    
    # Initialize predictor
    predictor = FacePredictor()
    
    # Example 1: Predict using the required function interface
    print("\n" + "-" * 40)
    print("Example 1: Using required prediction() function")
    print("-" * 40)
    
    # Create random features for demonstration
    test_features = np.random.randn(5, predictor.input_size)
    count = prediction(test_features)
    print(f"Input: {test_features.shape[0]} samples")
    print(f"Detected faces: {count}")
    
    # Example 2: Predict from image files (if available)
    print("\n" + "-" * 40)
    print("Example 2: Predicting from image files")
    print("-" * 40)
    
    # Check for test images
    test_images = []
    if os.path.exists('./test_images'):
        test_images = [f'./test_images/{f}' for f in os.listdir('./test_images') 
                      if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    if test_images:
        results = predictor.predict_batch(test_images[:5])  # Test first 5 images
        
        print(f"\nPrediction Results:")
        for result in results:
            if 'error' in result:
                print(f"  {result['image']}: Error - {result['error']}")
            else:
                print(f"  {result['image']}: {result['label']} (confidence: {result['confidence']:.2%})")
    else:
        print("No test images found. Create a './test_images' directory with images to test.")
    
    # Example 3: Interactive prediction
    print("\n" + "-" * 40)
    print("Example 3: Interactive prediction")
    print("-" * 40)
    print("\nYou can now enter image paths for prediction.")
    print("Type 'quit' to exit.")
    
    while True:
        image_path = input("\nEnter image path: ").strip()
        
        if image_path.lower() == 'quit':
            break
        
        if not os.path.exists(image_path):
            print(f"Error: File not found - {image_path}")
            continue
        
        try:
            pred, conf = predictor.predict_image(image_path)
            label = "Face" if pred == 1 else "No Face"
            print(f"Prediction: {label}")
            print(f"Confidence: {conf:.2%}")
        except Exception as e:
            print(f"Error processing image: {e}")
    
    print("\nPrediction module demonstration completed!")


if __name__ == "__main__":
    main()
