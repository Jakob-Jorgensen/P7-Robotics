from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import numpy as np
import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import re

#### Settings
track_name = 'testing_logo_1' # For naming the batch of the annotation such that it will be called batch_[batchnumber]_[image number].png
view_image = False # View each image individually 
print_progress = True # Print statements with count of how many images we have process so far
save_images = True # Saves the images in the desired folder
images_path = r"D:\Baked_dataset\Baked_dataset\validating\RGB" # Path to the original pre annotated images
save_dir = r'C:\Users\mikke\Desktop\Universitet\7 Semester\Semester Project\convertion\ground truth [undistorted]\Validation' # Path to the folder which the binary images should be saved in

# Load the COCO annotation file
coco = COCO(r"C:\Users\mikke\Desktop\Universitet\7 Semester\Semester Project\convertion\ground truth [undistorted]\Validation\annotations\instances_default.json")

# Takes the annotation and converts it into a mask for the individual ID 
def mask_maker(image_id): 
    annotations = coco.loadAnns(coco.getAnnIds(imgIds=image_id))

    # Load image metadata
    image_info = coco.loadImgs(image_id)[0]
    image_width = image_info['width']
    image_height = image_info['height']
    image_name = image_info['file_name']
    
    # Removes the .jpg or .png ending from the previous file name loaded in the previous line 
    image_name = re.sub(r".jpg", "", image_name)
    image_name = re.sub(r".png", "", image_name)
    image_name = re.sub(r"undistored_RGB_", "", image_name)


    # Initialize an empty mask
    mask = np.zeros((image_height, image_width), dtype=np.uint8)

    # Loop through each annotation and create a mask
    for annotation in annotations:
        if annotation['iscrowd'] == 1:
            # If the annotation is RLE encoded (iscrowd=1)
            rle = annotation['segmentation']
            m = maskUtils.decode(rle)
        else:
            # If the annotation is polygon segmentation
            m = np.zeros((image_height, image_width), dtype=np.uint8)
            for segment in annotation['segmentation']:
                # Draw polygon using ImageDraw
                poly = np.array(segment).reshape((len(segment) // 2, 2))
                img = Image.fromarray(m)
                draw = ImageDraw.Draw(img)
                draw.polygon([tuple(p) for p in poly], outline=1, fill=1)
                m = np.array(img)  # Convert back to NumPy array

        # Merge the object mask into the main mask (assigning a unique ID or binary)
        mask[m > 0] = annotation['category_id']  # Or use 1 if you want a binary mask

        if save_images == True:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.figure(figsize=(12.8, 7.2),dpi=100)
            plt.imshow(mask, cmap='gray')
            plt.axis('off')
            plt.subplots_adjust(left=0,right=1,top=1,bottom=0)
            filename = f"undistored_GT_{image_name}.png"
            plt.savefig(os.path.join(save_dir, filename), pad_inches=0, dpi = 100)
            plt.close()

        if view_image == True:
            plt.show()

        return mask


# Starts the inital count for the masks as 1
mask_id = 1
counter = 0

# Loads and counts the images contained in the pre-annotated image folder
image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".webp"}
image_count = sum(1 for file in os.listdir(images_path) if os.path.splitext(file)[1].lower() in image_extensions)

# Iterates through all the images and converts them to binary images
while counter < image_count:
    if print_progress == True:
        print("Processing image number:", mask_id, " out of ", image_count)

    mask = mask_maker(mask_id)
    mask_id = mask_id+1
    counter = counter+1
