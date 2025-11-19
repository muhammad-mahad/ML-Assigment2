import cv2
import numpy as np
import os
import random

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# --- Augmentation Functions ---

def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    # Calculate Affine Matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))

def flip_horizontal(image):
    return cv2.flip(image, 1)

def flip_vertical(image):
    return cv2.flip(image, 0)

def add_gaussian_noise(image):
    row, col, ch = image.shape
    mean = 0
    var = 0.05
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + (gauss * 255) # Scale noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def apply_gaussian_blur(image):
    # Kernel size must be odd (5,5)
    return cv2.GaussianBlur(image, (5, 5), 0)

def adjust_brightness(image, value):
    # Convert to HSV to change Value (Brightness) only, preserving color
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    if value >= 0:
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value
    else:
        value = abs(value)
        lim = value
        v[v < lim] = 0
        v[v >= lim] -= value
        
    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

def adjust_contrast(image, alpha):
    # new_img = alpha * old_img + beta
    return cv2.convertScaleAbs(image, alpha=alpha, beta=0)

def shift_image(image, x_shift, y_shift):
    h, w = image.shape[:2]
    M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    return cv2.warpAffine(image, M, (w, h))

def shear_image(image, shear_factor):
    h, w = image.shape[:2]
    # Shearing on X-axis
    M = np.float32([[1, shear_factor, 0], [0, 1, 0]])
    # Adjust w to handle new width or just crop
    return cv2.warpAffine(image, M, (w, h))

def zoom_image(image, scale):
    h, w = image.shape[:2]
    center_x, center_y = w // 2, h // 2
    radius_x, radius_y = w // (2 * scale), h // (2 * scale)
    
    # Crop centered region
    min_x, max_x = int(center_x - radius_x), int(center_x + radius_x)
    min_y, max_y = int(center_y - radius_y), int(center_y + radius_y)
    
    cropped = image[min_y:max_y, min_x:max_x]
    return cv2.resize(cropped, (w, h))

def color_jitter(image):
    # Randomly multply channels by values between 0.8 and 1.2
    image = image.astype(np.float32)
    for i in range(3): # B, G, R
        scale = random.uniform(0.8, 1.2)
        image[:, :, i] *= scale
    return np.clip(image, 0, 255).astype(np.uint8)

def grayscale_convert(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Convert back to BGR so it keeps 3 channels shape for consistency
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# --- Main Generator Script ---

def generate_data(input_folder, output_folder):
    create_directory(output_folder)
    
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp')
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(supported_formats)]
    
    print(f"Found {len(files)} images. Starting augmentation...")

    for filename in files:
        img_path = os.path.join(input_folder, filename)
        original_img = cv2.imread(img_path)
        
        if original_img is None:
            print(f"Could not load {filename}")
            continue
            
        name, ext = os.path.splitext(filename)
        
        # Define the augmentations dictionary
        # Key = suffix for filename, Value = function call
        transformations = {
            "rot_30": rotate_image(original_img, 30),
            "rot_neg30": rotate_image(original_img, -30),
            "flip_h": flip_horizontal(original_img),
            "flip_v": flip_vertical(original_img),
            "noise": add_gaussian_noise(original_img),
            "blur": apply_gaussian_blur(original_img),
            "bright_up": adjust_brightness(original_img, 40),
            "bright_down": adjust_brightness(original_img, -40),
            "contrast": adjust_contrast(original_img, 1.5),
            "shift": shift_image(original_img, 20, 20), # Shift 20px right and down
            "shear": shear_image(original_img, 0.2),
            "zoom": zoom_image(original_img, 1.5),
            "jitter": color_jitter(original_img),
            "gray": grayscale_convert(original_img)
        }

        # Save Original
        cv2.imwrite(os.path.join(output_folder, f"{name}_original{ext}"), original_img)

        # Save Augmented versions
        for suffix, aug_img in transformations.items():
            new_filename = f"{name}_{suffix}{ext}"
            save_path = os.path.join(output_folder, new_filename)
            cv2.imwrite(save_path, aug_img)
            
    print(f"Processing complete. Check '{output_folder}' for results.")

# --- Execution ---
if __name__ == "__main__":
    # UPDATE THESE PATHS
    input_path = "C:\\Users\\muhammad.mahad\\Semester-3\\ML-lec\\Assignment\\ML-Assignment-2\\code\\face_images" 
    output_path = "C:\\Users\\muhammad.mahad\\Semester-3\\ML-lec\\Assignment\\ML-Assignment-2\\code\\face_images\\augmented_data"
    
    # Create dummy folder for testing if it doesn't exist
    if not os.path.exists(input_path):
        os.makedirs(input_path)
        print(f"Created folder '{input_path}'. Please put images there first!")
    else:
        generate_data(input_path, output_path)