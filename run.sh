#!/bin/bash

# ============================================================
# Face Detection Neural Network - Automated Setup & Training
# Assignment 2 - Computer Vision
# Unix/Linux/Mac Shell Script
# ============================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo ""
echo "============================================================"
echo "   FACE DETECTION NEURAL NETWORK - AUTOMATED SETUP"
echo "============================================================"
echo ""

# Function to print colored messages
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if Python is installed
print_status "Checking for Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
    PIP_CMD=pip3
elif command -v python &> /dev/null; then
    PYTHON_CMD=python
    PIP_CMD=pip
else
    print_error "Python is not installed!"
    echo "Please install Python 3.7+ from https://www.python.org/"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
print_success "Python $PYTHON_VERSION found"

# Step 1: Create virtual environment
print_status "[1/8] Setting up virtual environment..."
if [ ! -d "venv" ]; then
    print_status "Creating virtual environment..."
    $PYTHON_CMD -m venv venv
    if [ $? -ne 0 ]; then
        print_error "Failed to create virtual environment"
        print_status "Trying alternative method..."
        virtualenv venv
        if [ $? -ne 0 ]; then
            print_error "Please install virtualenv: $PIP_CMD install virtualenv"
            exit 1
        fi
    fi
    print_success "Virtual environment created"
else
    print_status "Virtual environment already exists"
fi

# Step 2: Activate virtual environment
print_status "[2/8] Activating virtual environment..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    print_error "Failed to activate virtual environment"
    exit 1
fi
print_success "Virtual environment activated"

# Step 3: Upgrade pip
print_status "[3/8] Upgrading pip..."
pip install --upgrade pip --quiet
print_success "pip upgraded"

# Step 4: Install requirements
print_status "[4/8] Installing required packages..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt --quiet
    if [ $? -ne 0 ]; then
        print_warning "Some packages failed to install, trying manual installation..."
        pip install numpy Pillow matplotlib --quiet
    fi
else
    print_warning "requirements.txt not found, installing packages manually..."
    pip install numpy Pillow matplotlib --quiet
fi
print_success "Required packages installed"

# Step 5: Create directories
print_status "[5/8] Setting up dataset directories..."
mkdir -p face_images non_face_images test_images outputs
print_success "Directories created"

# Check if user has added their own images
if [ -z "$(ls -A face_images/*.{jpg,jpeg,png} 2>/dev/null)" ]; then
    print_warning "No face images found. Generating synthetic dataset..."
    
    # Create temporary Python script to generate synthetic data
    cat > temp_generate_dataset.py << EOF
import sys
sys.path.append('.')
from prepare_dataset import create_sample_face_images, generate_synthetic_non_face_images

print("Generating synthetic face images...")
create_sample_face_images(50)
print("Generating non-face images...")
generate_synthetic_non_face_images(50)
print("Synthetic dataset created successfully!")
EOF
    
    python temp_generate_dataset.py
    rm temp_generate_dataset.py
    print_success "Synthetic dataset generated"
else
    print_success "Face images found in face_images directory"
fi

# Step 6: Run system tests
echo ""
print_status "[6/8] Running system tests..."
echo ""
python test_system.py
if [ $? -ne 0 ]; then
    print_warning "Some tests failed. Continuing anyway..."
fi

# Step 7: Train the model
echo ""
print_status "[7/8] Starting model training..."
echo ""

# Check if model already exists
if [ -f "trained_model.json" ]; then
    read -p "Trained model already exists. Skip training? (y/n): " skip_training
    if [ "$skip_training" = "y" ] || [ "$skip_training" = "Y" ]; then
        print_status "Skipping training, using existing model"
    else
        print_status "This may take a few minutes depending on your hardware..."
        python train_model.py
        if [ $? -ne 0 ]; then
            print_error "Training failed!"
            exit 1
        fi
        print_success "Model training completed"
    fi
else
    print_status "This may take a few minutes depending on your hardware..."
    python train_model.py
    if [ $? -ne 0 ]; then
        print_error "Training failed!"
        exit 1
    fi
    print_success "Model training completed"
fi

# Step 8: Test predictions
echo ""
print_status "[8/8] Testing prediction module..."
echo ""

# Create test script for predictions
cat > temp_test_prediction.py << EOF
import numpy as np
from prediction_module import prediction, FacePredictor
import os

print("="*60)
print("TESTING PREDICTION MODULE")
print("="*60)

# Test with random features
if os.path.exists('trained_model.json'):
    predictor = FacePredictor()
    test_features = np.random.randn(5, predictor.input_size)
    result = prediction(test_features)
    print(f"\nTest prediction on 5 samples: {result} faces detected")
    print("\nModel is ready for use!")
else:
    print("No trained model found. Please run training first.")
EOF

python temp_test_prediction.py
rm temp_test_prediction.py

# Final summary
echo ""
echo "============================================================"
echo "   SETUP AND TRAINING COMPLETE!"
echo "============================================================"
echo ""
echo -e "${GREEN}Generated files:${NC}"

if [ -f "trained_model.json" ]; then
    echo "  ✓ trained_model.json (model parameters)"
fi
if [ -f "normalization_mean.npy" ]; then
    echo "  ✓ normalization_mean.npy (normalization parameters)"
fi
if [ -f "normalization_std.npy" ]; then
    echo "  ✓ normalization_std.npy (normalization parameters)"
fi
if [ -f "training_curves.png" ]; then
    echo "  ✓ training_curves.png (training plots)"
fi
if [ -f "training_results.json" ]; then
    echo "  ✓ training_results.json (metrics)"
fi

echo ""
echo -e "${BLUE}What to do next:${NC}"
echo "  1. Check the training_curves.png for model performance"
echo "  2. Review training_results.json for detailed metrics"
echo "  3. Test with your own images using prediction_module.py"
echo ""
echo -e "${GREEN}To use the trained model:${NC}"
echo "  python prediction_module.py"
echo ""

# Ask if user wants to keep virtual environment activated
read -p "Keep virtual environment activated? (y/n): " keep_venv
if [ "$keep_venv" = "n" ] || [ "$keep_venv" = "N" ]; then
    deactivate
    print_status "Virtual environment deactivated"
else
    print_success "Virtual environment still active. Type 'deactivate' to exit."
fi

echo ""
print_success "Script execution completed!"
