import os
from PIL import Image
import numpy as np
import cv2

def process_image(image_path, output_path):
    # Load the image
    image = Image.open(image_path).convert('L')  # Convert to grayscale

    # Convert image to numpy array
    image_np = np.array(image)

    # Threshold to create a binary image
    _, binary_image = cv2.threshold(image_np, 1, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes and fill the regions
    bounding_box_image = np.zeros_like(binary_image)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(bounding_box_image, (x, y), (x + w, y + h), 255, -1)  # Draw filled rectangle

    # Save the processed image
    output_image = Image.fromarray(bounding_box_image)
    output_image.save(output_path)

def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each image in the folder
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            process_image(input_path, output_path)

# Set input and output folder paths
input_folder = r"C:\Users\Final_Dataset\Testing\GT" 
output_folder = r"C:\Users\Final_Dataset\Testing\GT2"  

# Process all images in the input folder
process_folder(input_folder, output_folder)
