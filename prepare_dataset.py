"""
Dataset Preparation Helper Script
Assignment 2 - Computer Vision
Author: Muhammad Mahad
Date: November 2024

This script helps you prepare your face detection dataset by:
1. Creating necessary directories
2. Capturing images from webcam
3. Generating non-face images
4. Organizing the dataset
"""

import os
import numpy as np
from PIL import Image, ImageDraw
import time

def create_directories():
    """Create necessary directories for the dataset."""
    directories = [
        'face_images',
        'non_face_images',
        'test_images',
        'captured_images'
    ]
    
    for dir_name in directories:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"Created directory: {dir_name}")
        else:
            print(f"Directory exists: {dir_name}")

def capture_face_images():
    """
    Capture face images using webcam (if cv2 is available).
    If OpenCV is not available, provide instructions.
    """
    try:
        import cv2
        print("\nOpenCV detected. Starting webcam capture...")
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("\nWebcam Instructions:")
        print("- Press SPACE to capture an image")
        print("- Press Q to quit")
        print("- Try different angles and expressions")
        print("- Aim for at least 30-50 images\n")
        
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Display the frame
            cv2.imshow('Face Capture - Press SPACE to capture, Q to quit', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Space key
                # Save the image
                filename = f'captured_images/face_{count:04d}.jpg'
                cv2.imwrite(filename, frame)
                print(f"Captured: {filename}")
                count += 1
            
            elif key == ord('q'):  # Q key
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\nCaptured {count} images")
        
        # Move images to face_images directory
        if count > 0:
            print("Moving captured images to face_images directory...")
            for i in range(count):
                src = f'captured_images/face_{i:04d}.jpg'
                dst = f'face_images/face_{i:04d}.jpg'
                if os.path.exists(src):
                    os.rename(src, dst)
        
    except ImportError:
        print("\nOpenCV not installed. To capture face images:")
        print("1. Install OpenCV: pip install opencv-python")
        print("2. Or manually take photos using your camera/phone")
        print("3. Place them in the 'face_images' directory")
        print("\nAlternatively, you can use your phone to take selfies and transfer them.")

def generate_synthetic_non_face_images(num_images=50):
    """
    Generate synthetic non-face images for training.
    """
    print(f"\nGenerating {num_images} synthetic non-face images...")
    
    for i in range(num_images):
        # Create random patterns
        img_type = np.random.choice(['gradient', 'noise', 'geometric', 'texture'])
        
        # Create 64x64 RGB image
        img = Image.new('RGB', (64, 64))
        pixels = img.load()
        
        if img_type == 'gradient':
            # Create gradient pattern
            for x in range(64):
                for y in range(64):
                    r = int((x / 63) * 255)
                    g = int((y / 63) * 255)
                    b = int(((x + y) / 126) * 255)
                    pixels[x, y] = (r, g, b)
        
        elif img_type == 'noise':
            # Create random noise
            img_array = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
        
        elif img_type == 'geometric':
            # Create geometric shapes
            draw = ImageDraw.Draw(img)
            # Random background
            bg_color = tuple(np.random.randint(0, 256, 3))
            draw.rectangle([0, 0, 64, 64], fill=bg_color)
            
            # Add random shapes
            for _ in range(np.random.randint(2, 5)):
                shape_type = np.random.choice(['rectangle', 'ellipse', 'line'])
                color = tuple(np.random.randint(0, 256, 3))
                
                if shape_type == 'rectangle':
                    x1, y1 = np.random.randint(0, 32, 2)
                    x2, y2 = np.random.randint(32, 64, 2)
                    draw.rectangle([x1, y1, x2, y2], fill=color)
                elif shape_type == 'ellipse':
                    x1, y1 = np.random.randint(0, 32, 2)
                    x2, y2 = np.random.randint(32, 64, 2)
                    draw.ellipse([x1, y1, x2, y2], fill=color)
                else:
                    x1, y1 = np.random.randint(0, 64, 2)
                    x2, y2 = np.random.randint(0, 64, 2)
                    draw.line([x1, y1, x2, y2], fill=color, width=3)
        
        else:  # texture
            # Create texture pattern
            img_array = np.zeros((64, 64, 3), dtype=np.uint8)
            pattern = np.random.choice(['stripes', 'checkerboard', 'dots'])
            
            if pattern == 'stripes':
                for y in range(64):
                    if y % 8 < 4:
                        img_array[y, :] = np.random.randint(100, 200, 3)
                    else:
                        img_array[y, :] = np.random.randint(50, 100, 3)
            
            elif pattern == 'checkerboard':
                for x in range(0, 64, 8):
                    for y in range(0, 64, 8):
                        if (x // 8 + y // 8) % 2 == 0:
                            img_array[y:y+8, x:x+8] = np.random.randint(150, 255, 3)
                        else:
                            img_array[y:y+8, x:x+8] = np.random.randint(0, 100, 3)
            
            else:  # dots
                img_array[:] = np.random.randint(100, 150, 3)
                for _ in range(20):
                    cx, cy = np.random.randint(5, 59, 2)
                    radius = np.random.randint(2, 5)
                    color = np.random.randint(0, 256, 3)
                    for x in range(max(0, cx-radius), min(64, cx+radius)):
                        for y in range(max(0, cy-radius), min(64, cy+radius)):
                            if (x-cx)**2 + (y-cy)**2 <= radius**2:
                                img_array[y, x] = color
            
            img = Image.fromarray(img_array)
        
        # Save the image
        filename = f'non_face_images/nonface_{i:04d}.png'
        img.save(filename)
        
        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{num_images} images...")
    
    print(f"Successfully generated {num_images} non-face images")

def create_sample_face_images(num_images=30):
    """
    Create synthetic face-like images for demonstration.
    These are simplified representations, not actual faces.
    """
    print(f"\nGenerating {num_images} synthetic face-like images for demonstration...")
    
    for i in range(num_images):
        # Create a simple face-like pattern
        img = Image.new('RGB', (64, 64), color=(255, 220, 177))  # Skin-like color
        draw = ImageDraw.Draw(img)
        
        # Add variations
        skin_variation = np.random.randint(-20, 20, 3)
        base_color = np.array([255, 220, 177]) + skin_variation
        base_color = np.clip(base_color, 0, 255)
        img = Image.new('RGB', (64, 64), color=tuple(base_color))
        draw = ImageDraw.Draw(img)
        
        # Simple face features (eyes)
        eye_y = 20 + np.random.randint(-3, 3)
        eye_spacing = 15 + np.random.randint(-2, 2)
        eye_size = 3 + np.random.randint(-1, 2)
        
        # Left eye
        draw.ellipse([32-eye_spacing-eye_size, eye_y-eye_size, 
                     32-eye_spacing+eye_size, eye_y+eye_size], 
                    fill=(50, 50, 50))
        
        # Right eye
        draw.ellipse([32+eye_spacing-eye_size, eye_y-eye_size,
                     32+eye_spacing+eye_size, eye_y+eye_size],
                    fill=(50, 50, 50))
        
        # Nose (simple line)
        nose_y = 32 + np.random.randint(-2, 2)
        draw.line([32, nose_y-5, 32, nose_y+5], fill=(180, 150, 120), width=2)
        
        # Mouth (simple arc)
        mouth_y = 45 + np.random.randint(-2, 2)
        mouth_width = 10 + np.random.randint(-2, 2)
        draw.arc([32-mouth_width, mouth_y-5, 32+mouth_width, mouth_y+5],
                start=0, end=180, fill=(150, 50, 50), width=2)
        
        # Add some noise for realism
        img_array = np.array(img)
        noise = np.random.normal(0, 5, img_array.shape)
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        
        # Apply random transformations
        if np.random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        
        angle = np.random.uniform(-10, 10)
        img = img.rotate(angle, fillcolor=tuple(base_color))
        
        # Save the image
        filename = f'face_images/synthetic_face_{i:04d}.png'
        img.save(filename)
    
    print(f"Successfully generated {num_images} synthetic face-like images")

def check_dataset_status():
    """Check and report the current dataset status."""
    print("\n" + "=" * 60)
    print("DATASET STATUS")
    print("=" * 60)
    
    # Check face images
    if os.path.exists('face_images'):
        face_files = [f for f in os.listdir('face_images') 
                     if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        print(f"Face images: {len(face_files)} images found")
        if len(face_files) < 30:
            print("  ⚠ Warning: Recommended minimum is 30 face images")
    else:
        print("Face images: Directory not found")
    
    # Check non-face images
    if os.path.exists('non_face_images'):
        non_face_files = [f for f in os.listdir('non_face_images')
                         if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        print(f"Non-face images: {len(non_face_files)} images found")
        if len(non_face_files) < 30:
            print("  ⚠ Warning: Recommended minimum is 30 non-face images")
    else:
        print("Non-face images: Directory not found")
    
    print("\n" + "-" * 60)

def main():
    """Main function to prepare the dataset."""
    print("=" * 60)
    print("FACE DETECTION DATASET PREPARATION")
    print("=" * 60)
    
    # Create directories
    print("\n1. Creating directories...")
    create_directories()
    
    # Check current status
    check_dataset_status()
    
    # Interactive menu
    while True:
        print("\nOptions:")
        print("1. Capture face images from webcam")
        print("2. Generate synthetic face-like images (for testing)")
        print("3. Generate non-face images")
        print("4. Check dataset status")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            capture_face_images()
        
        elif choice == '2':
            num = input("How many synthetic face images? (default: 30): ").strip()
            num = int(num) if num else 30
            create_sample_face_images(num)
        
        elif choice == '3':
            num = input("How many non-face images? (default: 50): ").strip()
            num = int(num) if num else 50
            generate_synthetic_non_face_images(num)
        
        elif choice == '4':
            check_dataset_status()
        
        elif choice == '5':
            print("\nDataset preparation complete!")
            check_dataset_status()
            print("\nNext steps:")
            print("1. Review images in face_images/ and non_face_images/")
            print("2. Remove any incorrectly labeled images")
            print("3. Run 'python train_model.py' to train the model")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
