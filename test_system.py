"""
Test Script for Face Detection Neural Network
Assignment 2 - Computer Vision
Author: Muhammad Mahad
Date: November 2024

This script tests all components of the face detection system.
"""

import numpy as np
import os
import sys
import json

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import numpy as np
        print("  ✓ NumPy imported successfully")
    except ImportError:
        print("  ✗ NumPy not found. Install with: pip install numpy")
        return False
    
    try:
        from PIL import Image
        print("  ✓ Pillow imported successfully")
    except ImportError:
        print("  ✗ Pillow not found. Install with: pip install pillow")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("  ✓ Matplotlib imported successfully")
    except ImportError:
        print("  ✗ Matplotlib not found. Install with: pip install matplotlib")
        return False
    
    # Test custom modules
    try:
        from neural_network import ShallowNeuralNetwork
        print("  ✓ neural_network.py found")
    except ImportError as e:
        print(f"  ✗ Error importing neural_network.py: {e}")
        return False
    
    try:
        from face_detection_dataset import FaceDatasetCreator, create_sample_dataset
        print("  ✓ face_detection_dataset.py found")
    except ImportError as e:
        print(f"  ✗ Error importing face_detection_dataset.py: {e}")
        return False
    
    try:
        from prediction_module import prediction
        print("  ✓ prediction_module.py found")
    except ImportError as e:
        print(f"  ✗ Error importing prediction_module.py: {e}")
        return False
    
    try:
        import train_model
        print("  ✓ train_model.py found")
    except ImportError as e:
        print(f"  ✗ Error importing train_model.py: {e}")
        return False
    
    return True

def test_neural_network():
    """Test the neural network implementation."""
    print("\nTesting Neural Network...")
    
    from neural_network import ShallowNeuralNetwork
    
    # Create a small test network
    input_size = 10
    hidden_size = 5
    batch_size = 4
    
    # Create synthetic data
    X = np.random.randn(batch_size, input_size)
    y = np.random.randint(0, 2, (batch_size, 1))
    
    # Initialize network
    model = ShallowNeuralNetwork(input_size, hidden_size, 1, learning_rate=0.01)
    print(f"  ✓ Network initialized: input={input_size}, hidden={hidden_size}")
    
    # Test forward propagation
    cache = model.forward_propagation(X)
    assert cache['a2'].shape == (batch_size, 1)
    print(f"  ✓ Forward propagation works")
    
    # Test loss computation
    loss = model.compute_loss(y, cache['a2'])
    assert isinstance(loss, (float, np.floating))
    print(f"  ✓ Loss computation works: {loss:.4f}")
    
    # Test backward propagation
    gradients = model.backward_propagation(X, y, cache)
    assert 'dW1' in gradients and 'dW2' in gradients
    print(f"  ✓ Backward propagation works")
    
    # Test prediction
    predictions = model.predict(X)
    assert predictions.shape == y.shape
    print(f"  ✓ Prediction works")
    
    # Test training (brief)
    model.train(X, y, epochs=10, batch_size=2, verbose=False)
    print(f"  ✓ Training works")
    
    # Test save/load
    model.save_model('test_model.json')
    print(f"  ✓ Model saved")
    
    model2 = ShallowNeuralNetwork(input_size, hidden_size, 1)
    model2.load_model('test_model.json')
    print(f"  ✓ Model loaded")
    
    # Clean up
    if os.path.exists('test_model.json'):
        os.remove('test_model.json')
    
    return True

def test_dataset_creation():
    """Test dataset creation and preprocessing."""
    print("\nTesting Dataset Creation...")
    
    from face_detection_dataset import create_sample_dataset, FaceDatasetCreator
    
    # Test synthetic dataset creation
    X, y = create_sample_dataset()
    print(f"  ✓ Synthetic dataset created: X={X.shape}, y={y.shape}")
    
    # Test data splitting
    creator = FaceDatasetCreator(".", ".")
    X_train, X_val, X_test, y_train, y_val, y_test = creator.split_dataset(X, y)
    
    # Verify splits
    total = X.shape[0]
    assert X_train.shape[0] == int(total * 0.6)
    assert X_val.shape[0] == int(total * 0.2)
    print(f"  ✓ Data split correctly: train={X_train.shape[0]}, val={X_val.shape[0]}, test={X_test.shape[0]}")
    
    # Test normalization
    X_train_norm, X_val_norm, X_test_norm, mean, std = creator.normalize_features(
        X_train, X_val, X_test
    )
    assert X_train_norm.shape == X_train.shape
    print(f"  ✓ Normalization works")
    
    return True

def test_prediction_function():
    """Test the standalone prediction function."""
    print("\nTesting Prediction Function...")
    
    # First, create and save a simple model
    from neural_network import ShallowNeuralNetwork
    
    input_size = 100
    model = ShallowNeuralNetwork(input_size, 32, 1)
    model.save_model('trained_model.json')
    
    # Save dummy normalization parameters
    np.save('normalization_mean.npy', np.zeros((1, input_size)))
    np.save('normalization_std.npy', np.ones((1, input_size)))
    
    # Test the prediction function
    from prediction_module import prediction
    
    # Single sample
    features = np.random.randn(input_size)
    count = prediction(features)
    assert isinstance(count, (int, np.integer))
    print(f"  ✓ Single sample prediction works: {count}")
    
    # Batch prediction
    features_batch = np.random.randn(5, input_size)
    count_batch = prediction(features_batch)
    assert isinstance(count_batch, (int, np.integer))
    assert 0 <= count_batch <= 5
    print(f"  ✓ Batch prediction works: {count_batch}/5 faces detected")
    
    # Clean up
    for file in ['trained_model.json', 'normalization_mean.npy', 'normalization_std.npy']:
        if os.path.exists(file):
            os.remove(file)
    
    return True

def test_full_pipeline():
    """Test a minimal version of the full training pipeline."""
    print("\nTesting Full Pipeline...")
    
    from train_model import FaceDetectionTrainer
    
    # Create trainer with synthetic data
    trainer = FaceDetectionTrainer(use_synthetic=True)
    
    # Prepare dataset
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_dataset()
    print(f"  ✓ Dataset prepared")
    
    # Quick hyperparameter search (reduced)
    from neural_network import ShallowNeuralNetwork
    
    model = ShallowNeuralNetwork(
        input_size=X_train.shape[1],
        hidden_size=32,
        output_size=1,
        learning_rate=0.01
    )
    
    # Brief training
    model.train(X_train, y_train, X_val, y_val, epochs=10, batch_size=32, verbose=False)
    print(f"  ✓ Model trained")
    
    # Evaluate
    train_acc = model.compute_accuracy(X_train, y_train)
    val_acc = model.compute_accuracy(X_val, y_val)
    test_acc = model.compute_accuracy(X_test, y_test)
    
    print(f"  ✓ Evaluation complete:")
    print(f"    - Train accuracy: {train_acc:.2%}")
    print(f"    - Val accuracy: {val_acc:.2%}")
    print(f"    - Test accuracy: {test_acc:.2%}")
    
    return True

def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("FACE DETECTION SYSTEM TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Neural Network", test_neural_network),
        ("Dataset Creation", test_dataset_creation),
        ("Prediction Function", test_prediction_function),
        ("Full Pipeline", test_full_pipeline)
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
            if not success:
                print(f"\n✗ {name} test failed")
        except Exception as e:
            print(f"\n✗ {name} test failed with error: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{name:20} {status}")
    
    print("-" * 60)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The system is ready for use.")
        print("\nNext steps:")
        print("1. Prepare your dataset using: python prepare_dataset.py")
        print("2. Train the model using: python train_model.py")
        print("3. Make predictions using: python prediction_module.py")
    else:
        print("\n⚠ Some tests failed. Please fix the issues before proceeding.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
