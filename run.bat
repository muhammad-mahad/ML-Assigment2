@echo off
REM ============================================================
REM Face Detection Neural Network - Automated Setup & Training
REM Assignment 2 - Computer Vision
REM Windows Batch Script
REM ============================================================

echo.
echo ============================================================
echo    FACE DETECTION NEURAL NETWORK - AUTOMATED SETUP
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.7+ from https://www.python.org/
    pause
    exit /b 1
)

echo [1/8] Python found. Checking/creating virtual environment...

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment
        echo Trying with python3...
        python3 -m venv venv
    )
) else (
    echo Virtual environment already exists.
)

REM Activate virtual environment
echo [2/8] Activating virtual environment...
call venv\Scripts\activate
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)

REM Upgrade pip
echo [3/8] Upgrading pip...
python -m pip install --upgrade pip --quiet

REM Install requirements
echo [4/8] Installing required packages...
pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install requirements
    echo Trying manual installation...
    pip install numpy Pillow matplotlib --quiet
)

echo.
echo [5/8] Setting up dataset directories...

REM Create necessary directories
if not exist "face_images" mkdir face_images
if not exist "non_face_images" mkdir non_face_images
if not exist "test_images" mkdir test_images
if not exist "outputs" mkdir outputs

REM Check if user has added their own images
dir /b "face_images\*.jpg" "face_images\*.jpeg" "face_images\*.png" >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo No face images found. Generating synthetic dataset...
    echo.
    
    REM Create a Python script to generate synthetic data
    (
        echo import sys
        echo sys.path.append('.'^^^)
        echo from prepare_dataset import create_sample_face_images, generate_synthetic_non_face_images
        echo print("Generating synthetic face images..."^^^)
        echo create_sample_face_images(50^^^)
        echo print("Generating non-face images..."^^^)
        echo generate_synthetic_non_face_images(50^^^)
        echo print("Synthetic dataset created successfully!"^^^)
    ) > temp_generate_dataset.py
    
    python temp_generate_dataset.py
    del temp_generate_dataset.py
) else (
    echo Face images found in face_images directory.
)

echo.
echo [6/8] Running system tests...
echo.

REM Run tests
python test_system.py
if %errorlevel% neq 0 (
    echo.
    echo [WARNING] Some tests failed. Continuing anyway...
)

echo.
echo [7/8] Starting model training...
echo.
echo This may take a few minutes depending on your hardware...
echo.

REM Check if user wants to skip training if model exists
if exist "trained_model.json" (
    set /p skip_training="Trained model already exists. Skip training? (y/n): "
    if /i "%skip_training%"=="y" goto :test_prediction
)

REM Run training
python train_model.py
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Training failed!
    pause
    exit /b 1
)

:test_prediction
echo.
echo [8/8] Testing prediction module...
echo.

REM Create a test script for predictions
(
    echo import numpy as np
    echo from prediction_module import prediction, FacePredictor
    echo import os
    echo.
    echo print("="*60^^^)
    echo print("TESTING PREDICTION MODULE"^^^)
    echo print("="*60^^^)
    echo.
    echo # Test with random features
    echo if os.path.exists('trained_model.json'^^^):
    echo     predictor = FacePredictor(^^^)
    echo     test_features = np.random.randn(5, predictor.input_size^^^)
    echo     result = prediction(test_features^^^)
    echo     print(f"\nTest prediction on 5 samples: {result} faces detected"^^^)
    echo     print("\nModel is ready for use!"^^^)
    echo else:
    echo     print("No trained model found. Please run training first."^^^)
) > temp_test_prediction.py

python temp_test_prediction.py
del temp_test_prediction.py

echo.
echo ============================================================
echo    SETUP AND TRAINING COMPLETE!
echo ============================================================
echo.
echo Generated files:
if exist "trained_model.json" echo   - trained_model.json (model parameters)
if exist "normalization_mean.npy" echo   - normalization_mean.npy (normalization parameters)
if exist "normalization_std.npy" echo   - normalization_std.npy (normalization parameters)
if exist "training_curves.png" echo   - training_curves.png (training plots)
if exist "training_results.json" echo   - training_results.json (metrics)
echo.
echo What to do next:
echo   1. Check the training_curves.png for model performance
echo   2. Review training_results.json for detailed metrics
echo   3. Test with your own images using prediction_module.py
echo.
echo To use the trained model:
echo   python prediction_module.py
echo.

REM Ask if user wants to keep virtual environment activated
set /p keep_venv="Keep virtual environment activated? (y/n): "
if /i "%keep_venv%"=="n" (
    deactivate
)

echo.
echo Press any key to exit...
pause >nul
