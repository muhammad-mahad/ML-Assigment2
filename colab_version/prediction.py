"""
ANNEX C: Standalone Prediction Code
Assignment 2 - Face Detection Using Shallow Neural Network

Student Name: Muhammad Mahad
Student ID: 500330

Description:
This file contains the standalone prediction function that can be used
independently to predict whether an image contains a face or not.

Requirements:
- model_weights.npz (trained model weights)
- mean.npy (normalization mean from training)
- std.npy (normalization std from training)

Usage:
    from prediction import prediction
    import numpy as np
    
    # Load and normalize your features
    features = # ... your normalized features
    result = prediction(features)
    # Returns: 1 if face detected, 0 if no face
"""

import numpy as np
from PIL import Image


class FaceDetectionModel:
    """Shallow Neural Network for Face Detection"""
    
    def __init__(self):
        """Initialize empty model - call load_model() to load weights"""
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
        self.input_size = None
        self.hidden_size = None
        self.output_size = None
        
    def relu(self, z):
        """ReLU activation function"""
        return np.maximum(0, z)
    
    def sigmoid(self, z):
        """Sigmoid activation function with numerical stability"""
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))
    
    def forward_propagation(self, X):
        """Forward pass through the network"""
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self.relu(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self.sigmoid(z2)
        return a2
    
    def predict(self, X):
        """Predict class labels (0 or 1)"""
        return (self.forward_propagation(X) >= 0.5).astype(int)
    
    def load_model(self, filename='model_weights.npz'):
        """Load model weights from file"""
        try:
            data = np.load(filename)
            self.W1 = data['W1']
            self.b1 = data['b1']
            self.W2 = data['W2']
            self.b2 = data['b2']
            self.input_size = int(data['input_size'])
            self.hidden_size = int(data['hidden_size'])
            self.output_size = int(data['output_size'])
            print(f"Model loaded successfully from {filename}")
            return True
        except FileNotFoundError:
            print(f"Error: {filename} not found. Please ensure the model file exists.")
            return False
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False


# Global model instance
_model = None


def initialize_model():
    """
    Initialize the model by loading weights.
    This function is called automatically on first prediction.
    """
    global _model
    if _model is None:
        _model = FaceDetectionModel()
        if not _model.load_model('model_weights.npz'):
            raise RuntimeError("Failed to load model weights. Cannot make predictions.")
    return _model


def prediction(features):
    """
    Standalone prediction function for face detection.
    
    This function predicts whether the input features represent a face or not.
    
    Args:
        features: Normalized image features as numpy array
                 - Shape: (n_samples, 12288) for multiple images
                 - Shape: (12288,) for a single image
                 - Must be normalized using the same mean/std from training
    
    Returns:
        estimated_count: Integer value
                        - 1 if face detected
                        - 0 if no face detected
                        - For multiple images, returns numpy array of predictions
    
    Example:
        >>> # Single image
        >>> img_features = normalize_image(image)  # shape: (12288,)
        >>> result = prediction(img_features)
        >>> print(result)  # Output: 1 or 0
        
        >>> # Multiple images
        >>> batch_features = normalize_images(images)  # shape: (n, 12288)
        >>> results = prediction(batch_features)
        >>> print(results)  # Output: array([1, 0, 1, ...])
    
    Raises:
        RuntimeError: If model weights cannot be loaded
        ValueError: If input features have incorrect shape
    """
    # Initialize model if not already loaded
    model = initialize_model()
    
    # Validate input
    if not isinstance(features, np.ndarray):
        raise ValueError("Features must be a numpy array")
    
    # Ensure features is 2D
    if len(features.shape) == 1:
        if features.shape[0] != 12288:
            raise ValueError(f"Single image must have 12288 features, got {features.shape[0]}")
        features = features.reshape(1, -1)
    elif len(features.shape) == 2:
        if features.shape[1] != 12288:
            raise ValueError(f"Features must have 12288 columns, got {features.shape[1]}")
    else:
        raise ValueError(f"Features must be 1D or 2D array, got shape {features.shape}")
    
    # Make prediction
    predictions = model.predict(features)
    
    # Return single value if single image, otherwise return array
    if predictions.shape[0] == 1:
        return int(predictions[0][0])
    else:
        return predictions.flatten().astype(int)


def preprocess_image(image_path, mean_path='mean.npy', std_path='std.npy'):
    """
    Helper function to preprocess an image for prediction.
    
    Args:
        image_path: Path to the image file
        mean_path: Path to the normalization mean file
        std_path: Path to the normalization std file
    
    Returns:
        Normalized features ready for prediction (shape: 12288,)
    """
    try:
        # Load normalization parameters
        mean = np.load(mean_path)
        std = np.load(std_path)
        
        # Load and preprocess image
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((64, 64), Image.LANCZOS)
        
        # Convert to array and normalize
        img_array = np.array(img, dtype=np.float32) / 255.0
        features = img_array.reshape(1, -1)
        normalized_features = (features - mean) / std
        
        return normalized_features.flatten()
        
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Required file not found: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Error preprocessing image: {str(e)}")


# ============================================================================
# DEMONSTRATION AND TESTING
# ============================================================================

if __name__ == "__main__":
    """
    Demonstration of the prediction function.
    Run this file directly to test the prediction functionality.
    """
    
    print("="*70)
    print("ANNEX C: Standalone Prediction Function Test")
    print("="*70)
    
    # Check if required files exist
    import os
    required_files = ['model_weights.npz', 'mean.npy', 'std.npy']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("\nERROR: Missing required files:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nPlease ensure you have run the training notebook first to generate these files.")
    else:
        print("\n✓ All required files found")
        
        # Test 1: Random features (for testing purposes)
        print("\n" + "-"*70)
        print("Test 1: Random Features (Demo)")
        print("-"*70)
        
        # Generate random normalized features
        random_features = np.random.randn(12288)
        result = prediction(random_features)
        print(f"Prediction result: {result}")
        print(f"Interpretation: {'FACE DETECTED' if result == 1 else 'NO FACE'}")
        
        # Test 2: Batch prediction
        print("\n" + "-"*70)
        print("Test 2: Batch Prediction (Demo)")
        print("-"*70)
        
        batch_features = np.random.randn(5, 12288)
        results = prediction(batch_features)
        print(f"Batch predictions: {results}")
        print(f"Faces detected: {np.sum(results)} out of {len(results)}")
        
        # Test 3: If test data exists
        if os.path.exists('test_features.npy') and os.path.exists('test_labels.npy'):
            print("\n" + "-"*70)
            print("Test 3: Real Test Data")
            print("-"*70)
            
            X_test = np.load('test_features.npy')
            y_test = np.load('test_labels.npy')
            
            # Predict on first 10 samples
            sample_size = min(10, len(X_test))
            predictions = prediction(X_test[:sample_size])
            actual = y_test[:sample_size].flatten().astype(int)
            
            print(f"Predictions: {predictions}")
            print(f"Actual:      {actual}")
            print(f"Matches:     {np.sum(predictions == actual)}/{sample_size}")
            print(f"Accuracy:    {np.mean(predictions == actual)*100:.2f}%")
        
        print("\n" + "="*70)
        print("Prediction function test completed successfully! ✓")
        print("="*70)
        
        # Usage instructions
        print("\n" + "="*70)
        print("USAGE INSTRUCTIONS")
        print("="*70)
        print("""
To use this prediction function in your code:

1. Import the function:
   from prediction import prediction, preprocess_image

2. For preprocessed features:
   result = prediction(normalized_features)

3. For image files:
   features = preprocess_image('path/to/image.jpg')
   result = prediction(features)

4. Interpret results:
   - result = 1 means FACE DETECTED
   - result = 0 means NO FACE

Requirements:
   - model_weights.npz
   - mean.npy
   - std.npy
        """)
