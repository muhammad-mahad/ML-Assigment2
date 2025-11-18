# Makefile for Face Detection Neural Network Project
# Assignment 2 - Computer Vision

# Variables
PYTHON := python3
PIP := pip3
VENV := venv
VENV_ACTIVATE := . $(VENV)/bin/activate

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# Default target
.DEFAULT_GOAL := help

# Phony targets (not files)
.PHONY: all help setup venv install clean train test predict dataset quick full check

# Help target - shows available commands
help:
	@echo "$(BLUE)======================================$(NC)"
	@echo "$(BLUE)  Face Detection Neural Network$(NC)"
	@echo "$(BLUE)  Available Commands:$(NC)"
	@echo "$(BLUE)======================================$(NC)"
	@echo ""
	@echo "$(GREEN)Quick Start:$(NC)"
	@echo "  make quick      - Run complete setup and training"
	@echo "  make all        - Same as 'make quick'"
	@echo ""
	@echo "$(GREEN)Setup:$(NC)"
	@echo "  make venv       - Create virtual environment"
	@echo "  make install    - Install dependencies"
	@echo "  make dataset    - Generate synthetic dataset"
	@echo "  make setup      - Complete setup (venv + install + dataset)"
	@echo ""
	@echo "$(GREEN)Training & Testing:$(NC)"
	@echo "  make train      - Train the model"
	@echo "  make test       - Run system tests"
	@echo "  make predict    - Test prediction module"
	@echo ""
	@echo "$(GREEN)Utilities:$(NC)"
	@echo "  make check      - Check dataset and model status"
	@echo "  make clean      - Remove generated files"
	@echo "  make clean-all  - Remove everything including venv"
	@echo ""
	@echo "$(YELLOW)Usage example: make quick$(NC)"

# Complete setup and training
all: quick

quick: setup train test predict
	@echo "$(GREEN)✓ Complete setup and training finished!$(NC)"
	@echo "$(GREEN)Model is ready for use.$(NC)"

full: clean-all quick
	@echo "$(GREEN)✓ Full rebuild complete!$(NC)"

# Create virtual environment
venv:
	@echo "$(BLUE)[1/8] Creating virtual environment...$(NC)"
	@if [ ! -d "$(VENV)" ]; then \
		$(PYTHON) -m venv $(VENV); \
		echo "$(GREEN)✓ Virtual environment created$(NC)"; \
	else \
		echo "$(YELLOW)⚠ Virtual environment already exists$(NC)"; \
	fi

# Install dependencies
install: venv
	@echo "$(BLUE)[2/8] Installing dependencies...$(NC)"
	@$(VENV_ACTIVATE) && \
		$(PIP) install --upgrade pip --quiet && \
		$(PIP) install numpy Pillow matplotlib --quiet && \
		echo "$(GREEN)✓ Core dependencies installed$(NC)"
	@$(VENV_ACTIVATE) && \
		$(PIP) install opencv-python --quiet 2>/dev/null && \
		echo "$(GREEN)✓ OpenCV installed (optional)$(NC)" || \
		echo "$(YELLOW)⚠ OpenCV not installed (optional)$(NC)"

# Setup directories
dirs:
	@echo "$(BLUE)[3/8] Creating directories...$(NC)"
	@mkdir -p face_images non_face_images test_images outputs
	@echo "$(GREEN)✓ Directories created$(NC)"

# Generate synthetic dataset if needed
dataset: dirs
	@echo "$(BLUE)[4/8] Checking/generating dataset...$(NC)"
	@if [ -z "$$(ls -A face_images/*.{jpg,jpeg,png} 2>/dev/null)" ]; then \
		echo "$(YELLOW)No face images found. Generating synthetic dataset...$(NC)"; \
		$(VENV_ACTIVATE) && $(PYTHON) -c "\
import sys; \
sys.path.append('.'); \
from prepare_dataset import create_sample_face_images, generate_synthetic_non_face_images; \
print('Generating synthetic face images...'); \
create_sample_face_images(50); \
print('Generating non-face images...'); \
generate_synthetic_non_face_images(50); \
print('$(GREEN)✓ Synthetic dataset created$(NC)')"; \
	else \
		echo "$(GREEN)✓ Dataset already exists$(NC)"; \
	fi

# Complete setup
setup: venv install dirs dataset
	@echo "$(GREEN)✓ Setup complete!$(NC)"

# Train the model
train: setup
	@echo "$(BLUE)[5/8] Training model...$(NC)"
	@if [ -f "trained_model.json" ]; then \
		read -p "Model exists. Retrain? (y/n): " retrain; \
		if [ "$$retrain" != "y" ]; then \
			echo "$(YELLOW)⚠ Skipping training$(NC)"; \
			exit 0; \
		fi; \
	fi
	@$(VENV_ACTIVATE) && $(PYTHON) train_model.py
	@echo "$(GREEN)✓ Model training complete!$(NC)"

# Run system tests
test: setup
	@echo "$(BLUE)[6/8] Running system tests...$(NC)"
	@$(VENV_ACTIVATE) && $(PYTHON) test_system.py || true
	@echo "$(GREEN)✓ Tests complete$(NC)"

# Test prediction module
predict: train
	@echo "$(BLUE)[7/8] Testing predictions...$(NC)"
	@$(VENV_ACTIVATE) && $(PYTHON) -c "\
import numpy as np; \
from prediction_module import prediction, FacePredictor; \
import os; \
if os.path.exists('trained_model.json'): \
    predictor = FacePredictor(); \
    test_features = np.random.randn(5, predictor.input_size); \
    result = prediction(test_features); \
    print(f'Test prediction: {result}/5 faces detected'); \
    print('$(GREEN)✓ Prediction module working!$(NC)'); \
else: \
    print('$(RED)✗ No trained model found$(NC)')"

# Check status
check:
	@echo "$(BLUE)======================================$(NC)"
	@echo "$(BLUE)  System Status Check$(NC)"
	@echo "$(BLUE)======================================$(NC)"
	@echo ""
	@echo "$(YELLOW)Virtual Environment:$(NC)"
	@if [ -d "$(VENV)" ]; then \
		echo "  $(GREEN)✓ Exists$(NC)"; \
	else \
		echo "  $(RED)✗ Not found$(NC)"; \
	fi
	@echo ""
	@echo "$(YELLOW)Dataset:$(NC)"
	@echo -n "  Face images: "
	@find face_images -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" 2>/dev/null | wc -l | xargs printf "%d files\n"
	@echo -n "  Non-face images: "
	@find non_face_images -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" 2>/dev/null | wc -l | xargs printf "%d files\n"
	@echo ""
	@echo "$(YELLOW)Model:$(NC)"
	@if [ -f "trained_model.json" ]; then \
		echo "  $(GREEN)✓ Trained model exists$(NC)"; \
		if [ -f "training_results.json" ]; then \
			echo "  $(GREEN)✓ Training results available$(NC)"; \
		fi; \
		if [ -f "training_curves.png" ]; then \
			echo "  $(GREEN)✓ Training plots available$(NC)"; \
		fi; \
	else \
		echo "  $(RED)✗ No trained model$(NC)"; \
	fi

# Clean generated files
clean:
	@echo "$(YELLOW)Cleaning generated files...$(NC)"
	@rm -f trained_model.json normalization_*.npy
	@rm -f training_curves.png training_results.json
	@rm -f test_model.json demo_model.json
	@rm -f temp_*.py
	@echo "$(GREEN)✓ Cleaned generated files$(NC)"

# Clean everything including virtual environment
clean-all: clean
	@echo "$(YELLOW)Removing virtual environment...$(NC)"
	@rm -rf $(VENV)
	@echo "$(YELLOW)Removing all images...$(NC)"
	@rm -rf face_images/* non_face_images/* test_images/*
	@echo "$(GREEN)✓ Complete cleanup done$(NC)"

# Watch for changes and retrain (useful during development)
watch:
	@echo "$(YELLOW)Watching for changes... (Press Ctrl+C to stop)$(NC)"
	@while true; do \
		$(MAKE) train; \
		echo "$(BLUE)Waiting for changes...$(NC)"; \
		sleep 10; \
	done

# Interactive mode
interactive:
	@$(VENV_ACTIVATE) && $(PYTHON) automate.py

# Shortcuts
q: quick
f: full
c: clean
t: test
p: predict
i: interactive
