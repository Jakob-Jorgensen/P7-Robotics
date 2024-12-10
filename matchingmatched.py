import os
import shutil

def move_matched_files_by_category_and_id(rgb_folder, hha_folder):
    # Create a new folder for matched HHA files
    matched_hha_folder = hha_folder + '_matched'
    os.makedirs(matched_hha_folder, exist_ok=True)
    
    # Helper function to extract category and identifier from filenames
    def extract_category_and_id(filename, prefix):
        if filename.startswith(prefix) and filename.endswith('.png') or filename.endswith('.tiff'):
            # Remove the prefix and split into category and identifier
            main_part = filename[len(prefix):].rsplit('_', 1)  # Split into category and identifier
            if len(main_part) == 2:
                category = main_part[0]
                identifier = main_part[1].split('.')[0]  # Remove file extension
                return category, identifier
        return None, None

    # Define prefixes for RGB and HHA
    rgb_prefix = "undistored_RGB__"
    hha_prefix = "Depth_"

    # Create a set of category + identifier from the RGB folder
    rgb_keys = set()
    for file in os.listdir(rgb_folder):
        category, identifier = extract_category_and_id(file, rgb_prefix)
        if category and identifier:
            rgb_keys.add((category, identifier))
    
    print(f"Extracted keys from RGB folder: {rgb_keys}")  # Debug

    # Track moved and unmatched files
    moved_files = []
    unmatched_files = []

    # Compare files in the HHA folder
    for hha_file in os.listdir(hha_folder):
        category, identifier = extract_category_and_id(hha_file, hha_prefix)
        if category and identifier:
            if (category, identifier) in rgb_keys:  # Check if match exists
                src_path = os.path.join(hha_folder, hha_file)
                dest_path = os.path.join(matched_hha_folder, hha_file)
                shutil.copy2(src_path, dest_path)  # Copy the matched file
                moved_files.append(hha_file)
            else:
                unmatched_files.append(hha_file)
    
    # Print results for verification
    print(f"Moved matched files to {matched_hha_folder}: {len(moved_files)} files")
    print(f"Matched files: {moved_files}")
    print(f"Unmatched files (not moved): {unmatched_files}")
    print(f"Total unmatched files: {len(unmatched_files)}")

# Define paths to RGB and HHA folders
rgb_folders = [
    r"C:\Users\simao\Documents\aau\1ST SEMESTER\project\final proj dataset\Final_dataset\Testing Downsized\RGB"
]

hha_folders = [
    r"C:\Users\simao\Documents\aau\1ST SEMESTER\project\final proj dataset\Final_dataset\Testing Downsized\new_depth_testing"
]

# Loop over both folder lists and apply the function to each corresponding pair
for rgb_folder, hha_folder in zip(rgb_folders, hha_folders):
    move_matched_files_by_category_and_id(rgb_folder, hha_folder)
