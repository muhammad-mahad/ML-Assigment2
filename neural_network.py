"""
Shallow Neural Network Implementation for Face Detection
Assignment 2 - Computer Vision
Author: Muhammad Mahad
Date: November 2024

This module implements a shallow neural network from scratch using only NumPy.
The network has one hidden layer and uses backpropagation for training.
"""

import numpy as np
import json

class ShallowNeuralNetwork:
    """
    A shallow neural network with one hidden layer for binary classification.
    Implemented from scratch using only NumPy.
    """
    
    def __init__(self, input_size, hidden_size, output_size=1, learning_rate=0.01):
        """
        Initialize the neural network.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of neurons in hidden layer
            output_size: Number of output neurons (1 for binary classification)
            learning_rate: Learning rate for gradient descent
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights and biases using He initialization
        # Weights from input to hidden layer
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        
        # Weights from hidden to output layer
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
        
        # Store training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def relu(self, z):
        """
        ReLU activation function.
        
        Args:
            z: Input array
            
        Returns:
            ReLU(z) = max(0, z)
        """
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        """
        Derivative of ReLU activation function.
        
        Args:
            z: Input array
            
        Returns:
            1 if z > 0, else 0
        """
        return (z > 0).astype(float)
    
    def sigmoid(self, z):
        """
        Sigmoid activation function for output layer.
        
        Args:
            z: Input array
            
        Returns:
            1 / (1 + exp(-z))
        """
        # Clip to prevent overflow
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))
    
    def sigmoid_derivative(self, a):
        """
        Derivative of sigmoid function.
        
        Args:
            a: Sigmoid output
            
        Returns:
            a * (1 - a)
        """
        return a * (1 - a)
    
    def forward_propagation(self, X):
        """
        Perform forward propagation through the network.
        
        Args:
            X: Input features (batch_size, input_size)
            
        Returns:
            Dictionary containing all intermediate values
        """
        # Input to hidden layer
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        
        # Hidden to output layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        
        return {
            'z1': self.z1,
            'a1': self.a1,
            'z2': self.z2,
            'a2': self.a2
        }
    
    def compute_loss(self, y_true, y_pred):
        """
        Compute binary cross-entropy loss.
        
        Args:
            y_true: True labels
            y_pred: Predicted probabilities
            
        Returns:
            Average loss value
        """
        m = y_true.shape[0]
        # Add small epsilon to prevent log(0)
        epsilon = 1e-7
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        loss = -1/m * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
    
    def backward_propagation(self, X, y, cache):
        """
        Perform backward propagation to compute gradients.
        
        Args:
            X: Input features
            y: True labels
            cache: Dictionary with forward propagation values
            
        Returns:
            Gradients for all parameters
        """
        m = X.shape[0]
        
        # Output layer gradients
        dz2 = cache['a2'] - y
        dW2 = 1/m * np.dot(cache['a1'].T, dz2)
        db2 = 1/m * np.sum(dz2, axis=0, keepdims=True)
        
        # Hidden layer gradients
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.relu_derivative(cache['z1'])
        dW1 = 1/m * np.dot(X.T, dz1)
        db1 = 1/m * np.sum(dz1, axis=0, keepdims=True)
        
        return {
            'dW1': dW1, 'db1': db1,
            'dW2': dW2, 'db2': db2
        }
    
    def update_parameters(self, gradients):
        """
        Update network parameters using gradient descent.
        
        Args:
            gradients: Dictionary containing gradients
        """
        self.W1 -= self.learning_rate * gradients['dW1']
        self.b1 -= self.learning_rate * gradients['db1']
        self.W2 -= self.learning_rate * gradients['dW2']
        self.b2 -= self.learning_rate * gradients['db2']
    
    def predict_proba(self, X):
        """
        Predict probabilities for input samples.
        
        Args:
            X: Input features
            
        Returns:
            Predicted probabilities
        """
        cache = self.forward_propagation(X)
        return cache['a2']
    
    def predict(self, X):
        """
        Predict binary labels for input samples.
        
        Args:
            X: Input features
            
        Returns:
            Predicted binary labels (0 or 1)
        """
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)
    
    def compute_accuracy(self, X, y):
        """
        Compute classification accuracy.
        
        Args:
            X: Input features
            y: True labels
            
        Returns:
            Accuracy score
        """
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=1000, batch_size=32, verbose=True):
        """
        Train the neural network using mini-batch gradient descent.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Size of mini-batches
            verbose: Whether to print progress
        """
        n_samples = X_train.shape[0]
        n_batches = max(1, n_samples // batch_size)
        
        print(f"Starting training with {epochs} epochs and batch size {batch_size}")
        print(f"Number of batches per epoch: {n_batches}")
        print("-" * 50)
        
        for epoch in range(epochs):
            # Shuffle training data at the start of each epoch
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]
            
            epoch_loss = 0
            
            # Mini-batch gradient descent
            for batch in range(n_batches):
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, n_samples)
                
                X_batch = X_train_shuffled[start_idx:end_idx]
                y_batch = y_train_shuffled[start_idx:end_idx]
                
                # Forward propagation
                cache = self.forward_propagation(X_batch)
                
                # Compute loss
                batch_loss = self.compute_loss(y_batch, cache['a2'])
                epoch_loss += batch_loss
                
                # Backward propagation
                gradients = self.backward_propagation(X_batch, y_batch, cache)
                
                # Update parameters
                self.update_parameters(gradients)
            
            # Average epoch loss
            epoch_loss /= n_batches
            self.train_losses.append(epoch_loss)
            
            # Compute training accuracy
            train_accuracy = self.compute_accuracy(X_train, y_train)
            self.train_accuracies.append(train_accuracy)
            
            # Validation metrics
            if X_val is not None and y_val is not None:
                val_predictions = self.predict_proba(X_val)
                val_loss = self.compute_loss(y_val, val_predictions)
                val_accuracy = self.compute_accuracy(X_val, y_val)
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_accuracy)
            
            # Print progress
            if verbose and (epoch + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{epochs}]")
                print(f"  Train Loss: {epoch_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
                if X_val is not None:
                    print(f"  Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
                print()
        
        print("Training completed!")
    
    def save_model(self, filepath):
        """
        Save model parameters to a file.
        
        Args:
            filepath: Path to save the model
        """
        model_params = {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'learning_rate': self.learning_rate,
            'W1': self.W1.tolist(),
            'b1': self.b1.tolist(),
            'W2': self.W2.tolist(),
            'b2': self.b2.tolist(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_params, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load model parameters from a file.
        
        Args:
            filepath: Path to load the model from
        """
        with open(filepath, 'r') as f:
            model_params = json.load(f)
        
        self.input_size = model_params['input_size']
        self.hidden_size = model_params['hidden_size']
        self.output_size = model_params['output_size']
        self.learning_rate = model_params['learning_rate']
        self.W1 = np.array(model_params['W1'])
        self.b1 = np.array(model_params['b1'])
        self.W2 = np.array(model_params['W2'])
        self.b2 = np.array(model_params['b2'])
        self.train_losses = model_params['train_losses']
        self.val_losses = model_params['val_losses']
        self.train_accuracies = model_params['train_accuracies']
        self.val_accuracies = model_params['val_accuracies']
        
        print(f"Model loaded from {filepath}")

def prediction(features):
    """
    Standalone prediction function for face detection.
    This function loads the trained model and makes predictions.
    
    Args:
        features: Input features (can be single sample or batch)
        
    Returns:
        Predicted count (0 for no face, 1 for face detected)
    """
    # Ensure features is 2D array
    if len(features.shape) == 1:
        features = features.reshape(1, -1)
    
    # Load the trained model
    model = ShallowNeuralNetwork(features.shape[1], 128, 1)
    model.load_model('trained_model.json')
    
    # Make prediction
    predictions = model.predict(features)
    
    # Return the predicted count (sum of predictions for batch)
    return np.sum(predictions)

if __name__ == "__main__":
    # Example usage
    print("Shallow Neural Network for Face Detection")
    print("-" * 50)
    
    # Create synthetic data for demonstration
    np.random.seed(42)
    X_train = np.random.randn(100, 100)
    y_train = np.random.randint(0, 2, (100, 1))
    X_val = np.random.randn(20, 100)
    y_val = np.random.randint(0, 2, (20, 1))
    
    # Create and train model
    model = ShallowNeuralNetwork(
        input_size=100,
        hidden_size=50,
        output_size=1,
        learning_rate=0.01
    )
    
    # Train the model
    model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=100,
        batch_size=16,
        verbose=True
    )
    
    # Save the model
    model.save_model('demo_model.json')
