from ultralytics import YOLO
import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt

# Load YOLO model
model = YOLO("YOLO_Training/custom_model313/weights/best.pt")

# Define paths
rgb_path = r"C:\Users\Final_dataset\Testing\RGB2"
depth_images_path = r"C:\Users\Final_Dataset\Testing\Depth2"
output_folder_closest = "closest_box_end"
output_folder_binary = "closest_box_binary_end"  # New folder for binary images
output_folder_initial = "initial_predictions_end"

# Create output folder
os.makedirs(output_folder_closest, exist_ok=True)
os.makedirs(output_folder_initial, exist_ok=True)
os.makedirs(output_folder_binary, exist_ok=True)  # Create output folder for binary images



# Process depth images and find closest predictions
def process_depth_images():
    image_files = sorted(os.listdir(rgb_path))
    print(f"Found {len(image_files)} RGB images.")
    depth_tolerance = 0.7  # Depth tolerance value (adjustable)
    closest_predictions = {}  # {image_id: [list of closest boxes]}

    for idx, img_name in enumerate(image_files):
        image_path = os.path.join(rgb_path, img_name)
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        # Load image and make predictions
        image = cv2.imread(image_path)
        results = model.predict(image)

        # Save initial predictions
        result_img_initial = results[0].plot()  # Generate plot for initial predictions
        initial_output_path = os.path.join(output_folder_initial, f"{img_name}")
        cv2.imwrite(initial_output_path, result_img_initial)
        print(f"Saved initial prediction image {idx} to {initial_output_path}")

        # Load corresponding depth image
        depth_image_path = os.path.join(depth_images_path, img_name)
        depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)

        if depth_image is None:
            print(f"Depth image not found for {img_name}")
            continue

        closest_boxes = []
        closest_depth = float('inf')

        # Find closest bounding boxes
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            if 0 <= center_x < depth_image.shape[1] and 0 <= center_y < depth_image.shape[0]:
                depth_value = depth_image[center_y, center_x]

                if depth_value < closest_depth + depth_tolerance:
                    if depth_value < closest_depth - depth_tolerance:
                        closest_boxes = []
                    closest_boxes.append([x1, y1, x2 - x1, y2 - y1])  # x, y, width, height format
                    closest_depth = min(closest_depth, depth_value)

        closest_predictions[idx] = closest_boxes
        # Draw and save closest bounding boxes
        result_img = image.copy()  # Create a copy of the image for drawing
        binary_img = np.zeros_like(image[:, :, 0])  # Create a binary image (single channel)
        for box in closest_boxes:
            x, y, w, h = box
            cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw green rectangle
            cv2.putText(result_img, "Closest Box", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # Draw white rectangle on binary image
            cv2.rectangle(binary_img, (x, y), (x + w, y + h), 255, -1)  # Filled white rectangle
       
        # Save the image with closest bounding boxes
        output_path = os.path.join(output_folder_closest, f"{img_name}")
        cv2.imwrite(output_path, result_img)
        print(f"Saved closest prediction image {idx} to {output_path}")
        
        # Save the binary image
        binary_output_path = os.path.join(output_folder_binary, f"{img_name}")
        cv2.imwrite(binary_output_path, binary_img)
        print(f"Saved binary image {idx} to {binary_output_path}")
        
    return closest_predictions

# Main function
if __name__ == "__main__":
    closest_predictions = process_depth_images()
