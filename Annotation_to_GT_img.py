import json
import numpy as np
import cv2
import os
import pycocotools.mask as mask_utils

# Load the annotation file
with open('Yolo/testing.json') as f:
    data = json.load(f)

# Directory to save the output masks
output_dir = 'binary_masks'
os.makedirs(output_dir, exist_ok=True)

# Retrieve images and annotations
images = {image["id"]: image for image in data["images"]}
annotations = data["annotations"]

# Create binary masks for each image
for image_id, image_info in images.items():
    height = image_info["height"]
    width = image_info["width"]
    file_name = image_info["file_name"]
    file_name = file_name.split('undistored_RGB__')[1]

    # Create an empty mask (filled with zeros)
    mask = np.zeros((height, width), dtype=np.uint8)

    # There may be multiple annotations for the same image, so we combine them
    for annotation in annotations:
        if annotation["image_id"] == image_id:
            if "segmentation" in annotation and annotation["segmentation"]:
                segmentation = annotation["segmentation"]
                if isinstance(segmentation, list):  # Polygon format
                    rle = mask_utils.frPyObjects(segmentation, height, width)
                    rle = mask_utils.merge(rle) if isinstance(rle, list) else rle
                else:  # RLE format
                    rle = segmentation
                binary_mask = mask_utils.decode(rle)
                mask = np.logical_or(mask, binary_mask).astype(np.uint8)  # Combine masks (logical OR)

    # Save the mask
    output_path = os.path.join(output_dir, f"GT_{os.path.splitext(file_name)[0]}.png")
    cv2.imwrite(output_path, mask * 255)  # Save the mask with values 0 and 255

print("All masks have been created and saved.")
