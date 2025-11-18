"""
Main Training Script for Face Detection Neural Network
Assignment 2 - Computer Vision
Author: Muhammad Mahad
Date: November 2024

This script orchestrates the complete training pipeline:
1. Dataset creation and preprocessing
2. Model training with hyperparameter tuning
3. Evaluation and visualization
4. Model saving
"""

import numpy as np
import matplotlib.pyplot as plt
from face_detection_dataset import FaceDatasetCreator, create_sample_dataset
from neural_network import ShallowNeuralNetwork
import os
import time
import json

class FaceDetectionTrainer:
    """
    Main trainer class that manages the complete training pipeline.
    """
    
    def __init__(self, face_dir=None, non_face_dir=None, use_synthetic=True):
        """
        Initialize the trainer.
        
        Args:
            face_dir: Directory containing face images
            non_face_dir: Directory containing non-face images
            use_synthetic: Use synthetic data if directories not available
        """
        self.face_dir = face_dir
        self.non_face_dir = non_face_dir
        self.use_synthetic = use_synthetic
        
        # Training results storage
        self.results = {
            'dataset_info': {},
            'hyperparameters': {},
            'training_metrics': {},
            'evaluation_metrics': {}
        }
    
    def prepare_dataset(self):
        """
        Prepare the complete dataset for training.
        
        Returns:
            Training, validation, and test sets
        """
        print("=" * 60)
        print("DATASET PREPARATION")
        print("=" * 60)
        
        if self.use_synthetic:
            # Use synthetic data for demonstration
            print("Using synthetic dataset for demonstration...")
            X, y = create_sample_dataset()
        else:
            # Use real image data
            creator = FaceDatasetCreator(
                self.face_dir, 
                self.non_face_dir,
                img_size=(64, 64)
            )
            X, y = creator.create_dataset(augment=True)
            
        # Store dataset info
        self.results['dataset_info']['total_samples'] = X.shape[0]
        self.results['dataset_info']['feature_dim'] = X.shape[1]
        self.results['dataset_info']['positive_samples'] = int(np.sum(y))
        self.results['dataset_info']['negative_samples'] = int(X.shape[0] - np.sum(y))
        
        # Split the dataset (60% train, 20% val, 20% test)
        creator = FaceDatasetCreator(".", ".")  # Dummy paths for splitting
        X_train, X_val, X_test, y_train, y_val, y_test = creator.split_dataset(
            X, y, train_ratio=0.6, val_ratio=0.2
        )
        
        # Normalize features
        X_train, X_val, X_test, mean, std = creator.normalize_features(
            X_train, X_val, X_test
        )
        
        # Save normalization parameters
        np.save('normalization_mean.npy', mean)
        np.save('normalization_std.npy', std)
        
        print(f"\nDataset Statistics:")
        print(f"  Total samples: {self.results['dataset_info']['total_samples']}")
        print(f"  Feature dimension: {self.results['dataset_info']['feature_dim']}")
        print(f"  Positive samples (faces): {self.results['dataset_info']['positive_samples']}")
        print(f"  Negative samples (non-faces): {self.results['dataset_info']['negative_samples']}")
        print(f"  Class balance: {self.results['dataset_info']['positive_samples']/X.shape[0]:.2%} positive")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def hyperparameter_search(self, X_train, y_train, X_val, y_val):
        """
        Perform hyperparameter search to find optimal model configuration.
        
        Args:
            Training and validation data
            
        Returns:
            Best hyperparameters
        """
        print("\n" + "=" * 60)
        print("HYPERPARAMETER SEARCH")
        print("=" * 60)
        
        # Define hyperparameter grid
        hidden_sizes = [32, 64, 128]
        learning_rates = [0.001, 0.01, 0.1]
        
        best_val_accuracy = 0
        best_params = {}
        
        print("Testing different hyperparameter combinations...")
        
        for hidden_size in hidden_sizes:
            for lr in learning_rates:
                print(f"\nTesting: hidden_size={hidden_size}, lr={lr}")
                
                # Create model with current hyperparameters
                model = ShallowNeuralNetwork(
                    input_size=X_train.shape[1],
                    hidden_size=hidden_size,
                    output_size=1,
                    learning_rate=lr
                )
                
                # Train for fewer epochs during search
                model.train(
                    X_train, y_train,
                    X_val, y_val,
                    epochs=200,
                    batch_size=32,
                    verbose=False
                )
                
                # Evaluate on validation set
                val_accuracy = model.compute_accuracy(X_val, y_val)
                print(f"  Validation accuracy: {val_accuracy:.4f}")
                
                # Update best parameters
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_params = {
                        'hidden_size': hidden_size,
                        'learning_rate': lr
                    }
        
        print(f"\nBest hyperparameters found:")
        print(f"  Hidden size: {best_params['hidden_size']}")
        print(f"  Learning rate: {best_params['learning_rate']}")
        print(f"  Validation accuracy: {best_val_accuracy:.4f}")
        
        self.results['hyperparameters'] = best_params
        self.results['hyperparameters']['best_val_accuracy'] = float(best_val_accuracy)
        
        return best_params
    
    def train_final_model(self, X_train, y_train, X_val, y_val, hyperparams):
        """
        Train the final model with best hyperparameters.
        
        Args:
            Training data and hyperparameters
            
        Returns:
            Trained model
        """
        print("\n" + "=" * 60)
        print("TRAINING FINAL MODEL")
        print("=" * 60)
        
        # Create model with best hyperparameters
        model = ShallowNeuralNetwork(
            input_size=X_train.shape[1],
            hidden_size=hyperparams['hidden_size'],
            output_size=1,
            learning_rate=hyperparams['learning_rate']
        )
        
        print(f"Model Architecture:")
        print(f"  Input layer: {X_train.shape[1]} neurons")
        print(f"  Hidden layer: {hyperparams['hidden_size']} neurons (ReLU activation)")
        print(f"  Output layer: 1 neuron (Sigmoid activation)")
        print(f"  Total parameters: {X_train.shape[1] * hyperparams['hidden_size'] + hyperparams['hidden_size'] + hyperparams['hidden_size'] + 1}")
        
        # Train the model for more epochs
        start_time = time.time()
        model.train(
            X_train, y_train,
            X_val, y_val,
            epochs=1000,
            batch_size=32,
            verbose=True
        )
        training_time = time.time() - start_time
        
        print(f"\nTraining completed in {training_time:.2f} seconds")
        
        # Store training metrics
        self.results['training_metrics']['training_time'] = training_time
        self.results['training_metrics']['epochs'] = 1000
        self.results['training_metrics']['batch_size'] = 32
        
        return model
    
    def evaluate_model(self, model, X_train, y_train, X_val, y_val, X_test, y_test):
        """
        Evaluate the model on all datasets and compute metrics.
        
        Args:
            model: Trained model
            Dataset splits
            
        Returns:
            Evaluation metrics
        """
        print("\n" + "=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)
        
        # Compute predictions and metrics for each set
        datasets = {
            'Training': (X_train, y_train),
            'Validation': (X_val, y_val),
            'Test': (X_test, y_test)
        }
        
        metrics = {}
        
        for name, (X, y) in datasets.items():
            # Get predictions
            y_pred_proba = model.predict_proba(X)
            y_pred = model.predict(X)
            
            # Compute metrics
            accuracy = np.mean(y_pred == y)
            loss = model.compute_loss(y, y_pred_proba)
            
            # Compute confusion matrix elements
            tp = np.sum((y == 1) & (y_pred == 1))
            tn = np.sum((y == 0) & (y_pred == 0))
            fp = np.sum((y == 0) & (y_pred == 1))
            fn = np.sum((y == 1) & (y_pred == 0))
            
            # Calculate additional metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Store metrics
            metrics[name] = {
                'accuracy': float(accuracy),
                'loss': float(loss),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1_score),
                'confusion_matrix': {
                    'tp': int(tp),
                    'tn': int(tn),
                    'fp': int(fp),
                    'fn': int(fn)
                }
            }
            
            print(f"\n{name} Set Performance:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Loss: {loss:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1 Score: {f1_score:.4f}")
            print(f"  Confusion Matrix:")
            print(f"    True Positives: {tp}")
            print(f"    True Negatives: {tn}")
            print(f"    False Positives: {fp}")
            print(f"    False Negatives: {fn}")
        
        self.results['evaluation_metrics'] = metrics
        
        return metrics
    
    def plot_training_curves(self, model):
        """
        Plot training and validation curves.
        
        Args:
            model: Trained model with history
        """
        print("\n" + "=" * 60)
        print("GENERATING PLOTS")
        print("=" * 60)
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot loss curves
        axes[0].plot(model.train_losses, label='Training Loss', color='blue')
        if model.val_losses:
            axes[0].plot(model.val_losses, label='Validation Loss', color='red')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot accuracy curves
        axes[1].plot(model.train_accuracies, label='Training Accuracy', color='blue')
        if model.val_accuracies:
            axes[1].plot(model.val_accuracies, label='Validation Accuracy', color='red')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=100)
        print("Training curves saved to 'training_curves.png'")
        
        return fig
    
    def save_results(self):
        """
        Save all training results to a JSON file.
        """
        with open('training_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        print("\nTraining results saved to 'training_results.json'")
    
    def run_complete_training(self):
        """
        Run the complete training pipeline.
        
        Returns:
            Trained model and results
        """
        print("\n" + "=" * 80)
        print(" " * 20 + "FACE DETECTION NEURAL NETWORK TRAINING")
        print("=" * 80)
        
        # Step 1: Prepare dataset
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_dataset()
        
        # Step 2: Hyperparameter search
        best_params = self.hyperparameter_search(X_train, y_train, X_val, y_val)
        
        # Step 3: Train final model
        model = self.train_final_model(X_train, y_train, X_val, y_val, best_params)
        
        # Step 4: Evaluate model
        metrics = self.evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test)
        
        # Step 5: Generate plots
        self.plot_training_curves(model)
        
        # Step 6: Save model and results
        model.save_model('trained_model.json')
        self.save_results()
        
        print("\n" + "=" * 80)
        print(" " * 25 + "TRAINING PIPELINE COMPLETED")
        print("=" * 80)
        
        print("\nSaved files:")
        print("  - trained_model.json (trained model parameters)")
        print("  - normalization_mean.npy (feature normalization mean)")
        print("  - normalization_std.npy (feature normalization std)")
        print("  - training_curves.png (loss and accuracy plots)")
        print("  - training_results.json (complete training metrics)")
        
        return model, self.results


def main():
    """
    Main function to run the training.
    """
    # Check if we have real image directories
    face_dir = "./face_images"
    non_face_dir = "./non_face_images"
    
    if os.path.exists(face_dir) and os.path.exists(non_face_dir):
        print("Found image directories. Using real images for training.")
        trainer = FaceDetectionTrainer(face_dir, non_face_dir, use_synthetic=False)
    else:
        print("Image directories not found. Using synthetic data for demonstration.")
        print("To use real images, create directories:")
        print("  - ./face_images (containing your face images)")
        print("  - ./non_face_images (containing non-face images)")
        trainer = FaceDetectionTrainer(use_synthetic=True)
    
    # Run the complete training pipeline
    model, results = trainer.run_complete_training()
    
    print("\n" + "=" * 80)
    print("Final Summary:")
    print(f"  Best Test Accuracy: {results['evaluation_metrics']['Test']['accuracy']:.2%}")
    print(f"  Best Test F1 Score: {results['evaluation_metrics']['Test']['f1_score']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
