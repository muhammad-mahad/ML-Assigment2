"""
Face Detection Dataset Preparation Module
Assignment 2 - Computer Vision
Author: Muhammad Mahad
Date: November 2024

This module handles dataset creation, preprocessing, and augmentation for face detection
using only NumPy and Pillow libraries.
"""

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import os
import glob
import random

class FaceDatasetCreator:
    """
    Creates and preprocesses a dataset for face detection.
    Handles image loading, augmentation, and train/val/test splitting.
    """
    
    def __init__(self, face_dir, non_face_dir, img_size=(64, 64)):
        """
        Initialize the dataset creator.
        
        Args:
            face_dir: Directory containing your face images
            non_face_dir: Directory containing non-face images
            img_size: Target size for all images (width, height)
        """
        self.face_dir = face_dir
        self.non_face_dir = non_face_dir
        self.img_size = img_size
        self.data = []
        self.labels = []
        
    def load_and_preprocess_image(self, img_path):
        """
        Load an image and preprocess it.
        
        Args:
            img_path: Path to the image file
            
        Returns:
            Preprocessed image as numpy array
        """
        try:
            # Load image using Pillow
            img = Image.open(img_path)
            
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize to target size
            img = img.resize(self.img_size, Image.LANCZOS)
            
            # Convert to numpy array and normalize to [0, 1]
            img_array = np.array(img, dtype=np.float32) / 255.0
            
            return img_array
        
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None
    
    def augment_image(self, img_array):
        """
        Apply data augmentation to increase dataset diversity.
        
        Args:
            img_array: Input image as numpy array
            
        Returns:
            List of augmented images
        """
        augmented = []
        
        # Convert back to PIL Image for augmentation
        img = Image.fromarray((img_array * 255).astype(np.uint8))
        
        # Original image
        augmented.append(img_array)
        
        # Horizontal flip
        flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
        augmented.append(np.array(flipped, dtype=np.float32) / 255.0)
        
        # Brightness variations
        brightness_factors = [0.7, 0.85, 1.15, 1.3]
        for factor in brightness_factors:
            enhancer = ImageEnhance.Brightness(img)
            bright_img = enhancer.enhance(factor)
            augmented.append(np.array(bright_img, dtype=np.float32) / 255.0)
        
        # Contrast variations
        contrast_factors = [0.8, 1.2]
        for factor in contrast_factors:
            enhancer = ImageEnhance.Contrast(img)
            contrast_img = enhancer.enhance(factor)
            augmented.append(np.array(contrast_img, dtype=np.float32) / 255.0)
        
        # Slight rotation
        rotations = [-10, 10]
        for angle in rotations:
            rotated = img.rotate(angle, fillcolor=(128, 128, 128))
            augmented.append(np.array(rotated, dtype=np.float32) / 255.0)
        
        # Slight blur
        blurred = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        augmented.append(np.array(blurred, dtype=np.float32) / 255.0)
        
        return augmented
    
    def create_dataset(self, augment=True):
        """
        Create the complete dataset from directories.
        
        Args:
            augment: Whether to apply data augmentation
            
        Returns:
            X (features), y (labels) as numpy arrays
        """
        print("Loading face images...")
        face_images = []
        face_paths = glob.glob(os.path.join(self.face_dir, '*'))
        
        for path in face_paths:
            img = self.load_and_preprocess_image(path)
            if img is not None:
                if augment:
                    augmented_imgs = self.augment_image(img)
                    face_images.extend(augmented_imgs)
                else:
                    face_images.append(img)
        
        print(f"Loaded {len(face_images)} face images (with augmentation)")
        
        print("Loading non-face images...")
        non_face_images = []
        non_face_paths = glob.glob(os.path.join(self.non_face_dir, '*'))
        
        for path in non_face_paths:
            img = self.load_and_preprocess_image(path)
            if img is not None:
                # Apply less augmentation to non-face images
                non_face_images.append(img)
                if augment:
                    # Only add horizontal flip and one brightness variation
                    img_pil = Image.fromarray((img * 255).astype(np.uint8))
                    flipped = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
                    non_face_images.append(np.array(flipped, dtype=np.float32) / 255.0)
                    
                    enhancer = ImageEnhance.Brightness(img_pil)
                    bright = enhancer.enhance(1.2)
                    non_face_images.append(np.array(bright, dtype=np.float32) / 255.0)
        
        print(f"Loaded {len(non_face_images)} non-face images (with augmentation)")
        
        # Combine and create labels
        X = face_images + non_face_images
        y = [1] * len(face_images) + [0] * len(non_face_images)
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        
        # Flatten images for neural network input
        # Shape: (n_samples, height * width * channels)
        X = X.reshape(X.shape[0], -1)
        
        print(f"Dataset shape: X={X.shape}, y={y.shape}")
        
        return X, y
    
    def split_dataset(self, X, y, train_ratio=0.6, val_ratio=0.2):
        """
        Split dataset into training, validation, and test sets.
        
        Args:
            X: Features array
            y: Labels array
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        # Shuffle the data
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        X = X[indices]
        y = y[indices]
        
        # Calculate split points
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        
        # Split the data
        X_train = X[:n_train]
        y_train = y[:n_train]
        
        X_val = X[n_train:n_train + n_val]
        y_val = y[n_train:n_train + n_val]
        
        X_test = X[n_train + n_val:]
        y_test = y[n_train + n_val:]
        
        print(f"Split sizes: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def normalize_features(self, X_train, X_val, X_test):
        """
        Normalize features using training set statistics.
        
        Args:
            X_train, X_val, X_test: Feature arrays
            
        Returns:
            Normalized arrays and normalization parameters
        """
        # Calculate mean and std from training set
        mean = np.mean(X_train, axis=0, keepdims=True)
        std = np.std(X_train, axis=0, keepdims=True)
        
        # Avoid division by zero
        std[std == 0] = 1.0
        
        # Normalize all sets using training statistics
        X_train_norm = (X_train - mean) / std
        X_val_norm = (X_val - mean) / std
        X_test_norm = (X_test - mean) / std
        
        return X_train_norm, X_val_norm, X_test_norm, mean, std

def create_sample_dataset():
    """
    Create a sample dataset for testing when actual images are not available.
    This generates synthetic data for demonstration purposes.
    """
    print("Creating synthetic dataset for demonstration...")
    
    np.random.seed(42)
    
    # Generate synthetic face features (slightly different distribution)
    n_faces = 300
    face_features = np.random.randn(n_faces, 64 * 64 * 3) * 0.3 + 0.6
    face_features = np.clip(face_features, 0, 1)
    
    # Generate synthetic non-face features
    n_non_faces = 300
    non_face_features = np.random.randn(n_non_faces, 64 * 64 * 3) * 0.4 + 0.4
    non_face_features = np.clip(non_face_features, 0, 1)
    
    # Add some distinguishing patterns
    face_features[:, 1000:1500] += 0.2  # Add pattern for faces
    non_face_features[:, 2000:2500] += 0.2  # Different pattern for non-faces
    
    # Combine
    X = np.vstack([face_features, non_face_features])
    y = np.array([1] * n_faces + [0] * n_non_faces).reshape(-1, 1)
    
    # Shuffle
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    
    return X, y

if __name__ == "__main__":
    # Example usage
    print("Face Detection Dataset Creator")
    print("-" * 50)
    
    # For demonstration, create synthetic dataset
    X, y = create_sample_dataset()
    
    # Create dataset object
    creator = FaceDatasetCreator("./face_images", "./non_face_images")
    
    # Split the dataset
    X_train, X_val, X_test, y_train, y_val, y_test = creator.split_dataset(X, y)
    
    # Normalize features
    X_train_norm, X_val_norm, X_test_norm, mean, std = creator.normalize_features(
        X_train, X_val, X_test
    )
    
    print(f"\nFinal normalized dataset shapes:")
    print(f"X_train: {X_train_norm.shape}")
    print(f"X_val: {X_val_norm.shape}")
    print(f"X_test: {X_test_norm.shape}")
