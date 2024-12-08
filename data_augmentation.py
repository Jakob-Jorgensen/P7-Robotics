import os
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random

# Input and output directories
gt_dir = ""
rgb_dir = ""
depth_dir = ""

output_gt_dir = ""
output_rgb_dir = ""
output_depth_dir = ""

# Create output directories if they do not exist
for output_dir in [output_gt_dir, output_rgb_dir, output_depth_dir]:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

# Define a random rotation angle
random_degree = random.uniform(-30, 30)

# Transformation: Rotate and Color Jitter
def get_rgb_depth_transform():
    return transforms.Compose([
        transforms.RandomRotation((-8, 8)),  # Rotate within the same range
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),  # Apply Color Jitter
        transforms.RandomAffine(degrees=0, scale=(0.8, 1.0)),  # Scale between 80% and 120%
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))  # Translate up to 10%
    ])

# Transformation for Ground Truth (only rotation)
def get_gt_transform():
    return transforms.Compose([
        transforms.RandomRotation((-8, 8)),  # Rotate within the same range
        transforms.RandomAffine(degrees=0, scale=(0.8, 1.0)),  # Scale between 80% and 120%
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))  # Translate up to 10%
    ])

# Process directories
for data_dir, output_dir, transform in [
    (gt_dir, output_gt_dir, get_gt_transform()), 
    (rgb_dir, output_rgb_dir, get_rgb_depth_transform()), 
    (depth_dir, output_depth_dir, get_rgb_depth_transform())
]:
    for filename in os.listdir(data_dir):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
            # Load the image
            image_path = os.path.join(data_dir, filename)
            image = Image.open(image_path)

            # Apply the transform
            transformed_image = transform(image)

            # Create a new filename
            new_filename = f"augmented_{filename}"
            new_image_path = os.path.join(output_dir, new_filename)

            # Save the transformed image
            transformed_image.save(new_image_path)

            print(f"Saved: {new_image_path}")

print("All images have been successfully processed and saved.")
