#!/usr/bin/env python3
"""
Face Detection Neural Network - Automated Setup and Management
Assignment 2 - Computer Vision
Cross-platform automation script
"""

import os
import sys
import subprocess
import platform
import time
import shutil
from pathlib import Path

class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class FaceDetectionSetup:
    """Automated setup and management for face detection project"""
    
    def __init__(self):
        self.system = platform.system()
        self.python_cmd = self.get_python_command()
        self.venv_path = Path("venv")
        self.is_venv_active = self.check_venv_active()
        
    def get_python_command(self):
        """Detect the correct Python command"""
        for cmd in ['python3', 'python']:
            if shutil.which(cmd):
                return cmd
        print(f"{Colors.RED}Error: Python not found in PATH{Colors.ENDC}")
        sys.exit(1)
    
    def check_venv_active(self):
        """Check if virtual environment is active"""
        return hasattr(sys, 'real_prefix') or (
            hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
        )
    
    def run_command(self, command, shell=False):
        """Run a shell command and return the result"""
        try:
            if isinstance(command, str) and not shell:
                command = command.split()
            result = subprocess.run(
                command, 
                capture_output=True, 
                text=True, 
                shell=shell
            )
            return result.returncode == 0, result.stdout, result.stderr
        except Exception as e:
            return False, "", str(e)
    
    def print_header(self, text):
        """Print a formatted header"""
        print(f"\n{Colors.HEADER}{'='*60}{Colors.ENDC}")
        print(f"{Colors.HEADER}{text.center(60)}{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}\n")
    
    def print_status(self, message, status='INFO'):
        """Print a status message with color"""
        colors = {
            'INFO': Colors.BLUE,
            'SUCCESS': Colors.GREEN,
            'WARNING': Colors.YELLOW,
            'ERROR': Colors.RED
        }
        color = colors.get(status, Colors.BLUE)
        print(f"{color}[{status}]{Colors.ENDC} {message}")
    
    def create_virtual_environment(self):
        """Create and activate virtual environment"""
        self.print_header("Setting Up Virtual Environment")
        
        if not self.venv_path.exists():
            self.print_status("Creating virtual environment...")
            success, _, err = self.run_command(
                f"{self.python_cmd} -m venv venv"
            )
            
            if success:
                self.print_status("Virtual environment created", "SUCCESS")
            else:
                self.print_status(f"Failed to create venv: {err}", "ERROR")
                return False
        else:
            self.print_status("Virtual environment already exists", "INFO")
        
        if not self.is_venv_active:
            self.print_status("Please activate the virtual environment:", "WARNING")
            if self.system == "Windows":
                print(f"  Run: {Colors.CYAN}venv\\Scripts\\activate{Colors.ENDC}")
            else:
                print(f"  Run: {Colors.CYAN}source venv/bin/activate{Colors.ENDC}")
            print(f"  Then run this script again\n")
            return False
        
        return True
    
    def install_requirements(self):
        """Install required packages"""
        self.print_header("Installing Requirements")
        
        # Upgrade pip first
        self.print_status("Upgrading pip...")
        self.run_command("pip install --upgrade pip --quiet")
        
        # Install packages
        packages = ["numpy", "Pillow", "matplotlib"]
        for package in packages:
            self.print_status(f"Installing {package}...")
            success, _, _ = self.run_command(f"pip install {package} --quiet")
            if success:
                self.print_status(f"{package} installed", "SUCCESS")
            else:
                self.print_status(f"Failed to install {package}", "WARNING")
        
        # Try to install OpenCV (optional)
        self.print_status("Installing OpenCV (optional)...")
        success, _, _ = self.run_command("pip install opencv-python --quiet")
        if success:
            self.print_status("OpenCV installed (webcam support enabled)", "SUCCESS")
        else:
            self.print_status("OpenCV not installed (webcam disabled)", "WARNING")
        
        return True
    
    def setup_directories(self):
        """Create necessary directories"""
        self.print_header("Setting Up Directories")
        
        directories = ["face_images", "non_face_images", "test_images", "outputs"]
        for dir_name in directories:
            Path(dir_name).mkdir(exist_ok=True)
            self.print_status(f"Created/verified directory: {dir_name}", "SUCCESS")
        
        return True
    
    def generate_synthetic_dataset(self):
        """Generate synthetic dataset"""
        self.print_header("Generating Synthetic Dataset")
        
        try:
            from prepare_dataset import (
                create_sample_face_images, 
                generate_synthetic_non_face_images
            )
            
            self.print_status("Generating synthetic face images...")
            create_sample_face_images(50)
            self.print_status("Generated 50 synthetic face images", "SUCCESS")
            
            self.print_status("Generating non-face images...")
            generate_synthetic_non_face_images(50)
            self.print_status("Generated 50 non-face images", "SUCCESS")
            
            return True
        except Exception as e:
            self.print_status(f"Failed to generate dataset: {e}", "ERROR")
            return False
    
    def train_model(self):
        """Train the neural network model"""
        self.print_header("Training Neural Network")
        
        if Path("trained_model.json").exists():
            response = input("Trained model exists. Retrain? (y/n): ")
            if response.lower() != 'y':
                self.print_status("Skipping training", "INFO")
                return True
        
        self.print_status("Starting training (this may take a few minutes)...")
        success, output, err = self.run_command("python train_model.py")
        
        if success:
            self.print_status("Model training completed", "SUCCESS")
            if Path("training_curves.png").exists():
                self.print_status("Training plots saved to training_curves.png", "SUCCESS")
            return True
        else:
            self.print_status(f"Training failed: {err}", "ERROR")
            return False
    
    def test_system(self):
        """Run system tests"""
        self.print_header("Running System Tests")
        
        success, output, err = self.run_command("python test_system.py")
        if success:
            self.print_status("All tests passed", "SUCCESS")
        else:
            self.print_status("Some tests failed (check output)", "WARNING")
        
        print(output)
        return success
    
    def test_prediction(self):
        """Test the prediction module"""
        self.print_header("Testing Prediction Module")
        
        try:
            import numpy as np
            from prediction_module import prediction, FacePredictor
            
            if Path("trained_model.json").exists():
                predictor = FacePredictor()
                test_features = np.random.randn(5, predictor.input_size)
                result = prediction(test_features)
                
                self.print_status(f"Test prediction: {result}/5 faces detected", "SUCCESS")
                return True
            else:
                self.print_status("No trained model found. Train first!", "ERROR")
                return False
        except Exception as e:
            self.print_status(f"Prediction test failed: {e}", "ERROR")
            return False
    
    def check_dataset_status(self):
        """Check the status of the dataset"""
        self.print_header("Dataset Status")
        
        face_dir = Path("face_images")
        non_face_dir = Path("non_face_images")
        
        # Count images
        face_count = len(list(face_dir.glob("*.jpg")) + 
                        list(face_dir.glob("*.jpeg")) + 
                        list(face_dir.glob("*.png")))
        
        non_face_count = len(list(non_face_dir.glob("*.jpg")) + 
                           list(non_face_dir.glob("*.jpeg")) + 
                           list(non_face_dir.glob("*.png")))
        
        self.print_status(f"Face images: {face_count}", 
                         "SUCCESS" if face_count >= 30 else "WARNING")
        self.print_status(f"Non-face images: {non_face_count}", 
                         "SUCCESS" if non_face_count >= 30 else "WARNING")
        
        if face_count < 30:
            self.print_status("Recommended: At least 30 face images", "WARNING")
        if non_face_count < 30:
            self.print_status("Recommended: At least 30 non-face images", "WARNING")
        
        # Check for trained model
        if Path("trained_model.json").exists():
            self.print_status("Trained model found", "SUCCESS")
        else:
            self.print_status("No trained model", "WARNING")
        
        return face_count, non_face_count
    
    def full_setup(self):
        """Run complete setup process"""
        self.print_header("AUTOMATED FULL SETUP")
        
        steps = [
            ("Virtual Environment", self.create_virtual_environment),
            ("Install Requirements", self.install_requirements),
            ("Setup Directories", self.setup_directories),
            ("Check Dataset", lambda: self.check_dataset_status()[0] > 0),
            ("Generate Dataset", self.generate_synthetic_dataset),
            ("Train Model", self.train_model),
            ("Test System", self.test_system),
            ("Test Prediction", self.test_prediction)
        ]
        
        for step_name, step_func in steps:
            self.print_status(f"Running: {step_name}...", "INFO")
            
            # Skip dataset generation if images exist
            if step_name == "Generate Dataset":
                face_count, _ = self.check_dataset_status()
                if face_count > 0:
                    self.print_status("Dataset exists, skipping generation", "INFO")
                    continue
            
            success = step_func()
            if not success and step_name == "Virtual Environment":
                self.print_status("Cannot continue without virtual environment", "ERROR")
                return False
            
            time.sleep(0.5)  # Brief pause between steps
        
        self.print_header("SETUP COMPLETE!")
        self.print_status("All components are ready!", "SUCCESS")
        self.print_status("You can now use the face detection system", "INFO")
        
        return True
    
    def show_menu(self):
        """Display interactive menu"""
        while True:
            self.print_header("FACE DETECTION - MAIN MENU")
            
            print(f"{Colors.CYAN}Setup Options:{Colors.ENDC}")
            print("  1. Run full automated setup")
            print("  2. Install/update requirements only")
            print("  3. Generate synthetic dataset")
            print("")
            print(f"{Colors.CYAN}Training Options:{Colors.ENDC}")
            print("  4. Train model")
            print("  5. Test prediction module")
            print("")
            print(f"{Colors.CYAN}Information:{Colors.ENDC}")
            print("  6. Check dataset status")
            print("  7. Run system tests")
            print("")
            print(f"{Colors.CYAN}Other:{Colors.ENDC}")
            print("  8. Clean generated files")
            print("  9. Exit")
            print("")
            
            choice = input(f"{Colors.BOLD}Enter choice (1-9): {Colors.ENDC}")
            
            if choice == '1':
                self.full_setup()
            elif choice == '2':
                self.create_virtual_environment()
                if self.is_venv_active:
                    self.install_requirements()
            elif choice == '3':
                self.generate_synthetic_dataset()
            elif choice == '4':
                self.train_model()
            elif choice == '5':
                self.test_prediction()
            elif choice == '6':
                self.check_dataset_status()
            elif choice == '7':
                self.test_system()
            elif choice == '8':
                self.clean_files()
            elif choice == '9':
                print(f"\n{Colors.GREEN}Goodbye!{Colors.ENDC}")
                break
            else:
                self.print_status("Invalid choice. Please try again.", "WARNING")
            
            input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.ENDC}")
    
    def clean_files(self):
        """Clean generated files"""
        self.print_header("Cleaning Generated Files")
        
        files_to_clean = [
            "trained_model.json",
            "normalization_mean.npy", 
            "normalization_std.npy",
            "training_curves.png",
            "training_results.json"
        ]
        
        for file in files_to_clean:
            if Path(file).exists():
                Path(file).unlink()
                self.print_status(f"Removed: {file}", "SUCCESS")
        
        self.print_status("Cleanup complete", "SUCCESS")

def main():
    """Main entry point"""
    setup = FaceDetectionSetup()
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--auto":
            # Run full setup automatically
            setup.full_setup()
        elif sys.argv[1] == "--train":
            # Just train the model
            setup.train_model()
        elif sys.argv[1] == "--test":
            # Just test the system
            setup.test_system()
        elif sys.argv[1] == "--help":
            print("Usage: python automate.py [OPTIONS]")
            print("Options:")
            print("  --auto    Run full automated setup")
            print("  --train   Train the model only")
            print("  --test    Run system tests only")
            print("  --help    Show this help message")
            print("  (no args) Show interactive menu")
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Use --help for available options")
    else:
        # Show interactive menu
        setup.show_menu()

if __name__ == "__main__":
    main()
