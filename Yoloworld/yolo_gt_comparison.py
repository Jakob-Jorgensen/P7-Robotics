import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_recall_curve
from PIL import Image
import os

# Load binary images
# Load images from folders on the computer
predict_path = r"C:\Users\Yolo\closest_box_binary_end"  # Folder containing the predicted images
ground_truth_path = r"C:\Users\Final_Dataset\Testing\GT2"  # Folder containing the ground truth images

# Convert images to numpy arrays
predict_images = []
ground_truth_images = []

for filename in os.listdir(predict_path):
    if filename.endswith('.png') or filename.endswith('.jpg'):
        predict_file_path = os.path.join(predict_path, filename)
        ground_truth_file_path = os.path.join(ground_truth_path, filename)

        # Check if ground truth file exists
        if os.path.exists(ground_truth_file_path):
            # Open the predicted image and convert it to grayscale (L mode)
            predict_image = Image.open(predict_file_path).convert('L')
            # Open the ground truth image and convert it to grayscale (L mode)
            ground_truth_image = Image.open(ground_truth_file_path).convert('L')
            
            # Append the numpy array representation of the images to respective lists
            predict_images.append(np.array(predict_image))
            ground_truth_images.append(np.array(ground_truth_image))
        else:
            print(f'Warning: Ground truth file not found for {filename}')

# Create numpy arrays from the lists
predict = np.array(predict_images)
ground_truth = np.array(ground_truth_images)

# Rescale values from 255 to 1 to make them binary (0 or 1)
predict = np.where(predict == 255, 1, 0)
ground_truth = np.where(ground_truth == 255, 1, 0)

# Calculate F1 score with positive label as 1 (white pixels)
f1 = f1_score(ground_truth.flatten(), predict.flatten(), pos_label=1)
print(f'F1 Score: {f1:.2f}')

# Calculate Precision-Recall with positive label as 1 (white pixels)
precision, recall, thresholds = precision_recall_curve(ground_truth.flatten(), predict.flatten(), pos_label=1)

# Calculate F1 scores at each threshold
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8) #The term 1e-8 is added to avoid division by zero

# Best F1 score
best_f1_index = f1_scores.argmax()
best_f1 = f1_scores[best_f1_index]
best_threshold = thresholds[best_f1_index]
print(f"Best F1 Score: {best_f1:.4f} at threshold {best_threshold:.4f}")

# Plot Precision-Recall curve
plt.figure()
plt.plot(recall, precision, marker='.', label='Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid()
plt.show()

# Plot F1 Score curve
plt.figure()
plt.axhline(y=f1, color='r', linestyle='--', label=f'F1 Score: {f1:.2f}')
plt.xlabel('Recall')
plt.ylabel('F1 Score')
plt.title('F1 Score')
plt.legend()
plt.grid()
plt.show()

