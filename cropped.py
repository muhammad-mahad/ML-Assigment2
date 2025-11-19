from PIL import Image
import os

def crop_grid_collage(input_image_path, output_folder, num_rows, num_cols):
    """
    Crops individual cells from a grid collage image.

    Args:
        input_image_path (str): Path to the input collage image.
        output_folder (str): Folder to save the cropped individual images.
        num_rows (int): The number of rows in the grid.
        num_cols (int): The number of columns in the grid.
    """
    try:
        img = Image.open(input_image_path)
        img_width, img_height = img.size

        # Calculate the width and height of each cell
        cell_width = img_width // num_cols
        cell_height = img_height // num_rows

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        base_name = os.path.basename(input_image_path)
        name, ext = os.path.splitext(base_name)

        cell_count = 0
        for r in range(num_rows):
            for c in range(num_cols):
                # Calculate the bounding box for the current cell
                left = c * cell_width
                upper = r * cell_height
                right = left + cell_width
                lower = upper + cell_height

                # Ensure crop box doesn't exceed image dimensions due to integer division
                right = min(right, img_width)
                lower = min(lower, img_height)

                cropped_cell = img.crop((left, upper, right, lower))

                # Saved as name_rowX_colY
                output_image_path = os.path.join(output_folder, f"{name}_row{r}_col{c}{ext}")
                cropped_cell.save(output_image_path)
                cell_count += 1
        
        print(f"Successfully cropped {cell_count} individual images from '{input_image_path}' to '{output_folder}'")

    except FileNotFoundError:
        print(f"Error: Input image not found at '{input_image_path}'")
    except Exception as e:
        print(f"An error occurred while cropping '{input_image_path}': {e}")

def crop_multiple_grid_collages(input_folder, output_folder, num_rows, num_cols):
    """
    Processes all grid collage images in a folder to crop individual cells.
    """
    if not os.path.exists(input_folder):
        print(f"Error: Input folder not found at '{input_folder}'")
        return

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            input_image_path = os.path.join(input_folder, filename)
            print(f"Processing {filename}...")
            crop_grid_collage(input_image_path, output_folder, num_rows, num_cols)

# --- Configuration ---
# Ensure these paths are correct for your local machine
input_images_directory = "C:\\Users\\muhammad.mahad\\Semester-3\\ML-lec\\Assignment\\ML-Assignment-2\\code\\face_images\\generated"
output_cropped_directory = "C:\\Users\\muhammad.mahad\\Semester-3\\ML-lec\\Assignment\\ML-Assignment-2\\code\\face_images"

# Updated dimensions for the provided 3x4 image
NUM_GRID_ROWS = 3
NUM_GRID_COLS = 4

# --- Run the cropping process ---
if __name__ == "__main__":
    crop_multiple_grid_collages(input_images_directory, output_cropped_directory, NUM_GRID_ROWS, NUM_GRID_COLS)