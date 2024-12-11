from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.models import clone_model
from tensorflow.keras.models import model_from_json
from sklearn.metrics import precision_recall_curve, average_precision_score
from tensorflow.keras import backend as K
import tensorflow as tf
from keras import layers, Model
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2

                            #### Settings ####

main_path = f"C:/Users/mikke/Downloads/Dataset_3.2/Dataset_3.2"  
weight_path = r"C:\Users\mikke\Desktop\V9 Dice Loss 25 Epochs 36 Batch Augmented Real Gate Real Constants\trainmodel.weights.h5"
json_path = r"C:\Users\mikke\Documents\GitHub\P7-Robotics\model_architecture.json"
image_folder_path = r"C:\Users\mikke\Desktop\V9 Test Images\All Images"


############################# LOADING THE MODEL ################################

# Load architecture
with open(json_path, "r") as f:
    model_architecture = f.read()
loaded_model = model_from_json(model_architecture)

# Load weights
loaded_model.load_weights(weight_path)
loaded_model.summary()

########################### LOADING THE DATA SET ##################################

def preprocess_image(image_path, target_size, Binary_image = False,BGR2RGB = False):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)              # Load the image using OpenCV  
    if Binary_image == True:                                          #If the image is binary
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)                 #Convert the image to grayscale¨
    elif BGR2RGB == True: 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  
    image = cv2.resize(image, target_size)      # Resize the image using OpenCV
    image = image.astype('float32') / 255.0  # Normalize the image to range [0, 1]
    return image

def load_dataset(rgb_folder, saliency_folder, HHA_folder, target_size=(224, 224)):
    rgb_images, saliency_maps, HHA_images = [], [], []   #create empty lists
    rgb_files = sorted(os.listdir(rgb_folder))              #Sorted lists of the files and directories in the specified folders
    saliency_files = sorted(os.listdir(saliency_folder)) 
    HHA_files = sorted(os.listdir(HHA_folder)) 
        
    print(f"Found {len(rgb_files)} RGB images.")            #Printing founded image count from a list length
    print(f"Found {len(saliency_files)} saliency maps.") 
    print(f"Found {len(HHA_files)} HHA images.") 
    
    for i, img_file in enumerate(rgb_files):   
            
        rgb_path = os.path.join(rgb_folder, img_file)       #Make path for each image  
        rgb_image = preprocess_image(rgb_path, target_size,BGR2RGB=True)    
        rgb_images.append(rgb_image)
    
    for i, img_file in enumerate(saliency_files):
        saliency_path = os.path.join(saliency_folder, img_file)  
        saliency_map = preprocess_image(saliency_path, target_size,Binary_image= True)
        saliency_maps.append(saliency_map)

    for i, img_file in enumerate(HHA_files):
        HHA_path = os.path.join(HHA_folder,img_file) 
        HHA_image = preprocess_image(HHA_path, target_size,BGR2RGB=True)
        HHA_images.append(HHA_image)

    return np.array(rgb_images), np.array(saliency_maps), np.array(HHA_images)  #Convert lists to Numpy arrays as an output
    

rgb_folder_test = f"{main_path}/Testing/RGB"
saliency_folder_test = f"{main_path}/Testing/GT" 
HHA_folder_test = f"{main_path}/Testing/HHA"

rgb_images_test, saliency_maps_test,HHA_images_test = load_dataset(rgb_folder_test, saliency_folder_test,HHA_folder_test)  


############################ TESTING THE MODEL #############################

# Function to visualize input and output saliency map
def visualize_saliency(rgb_img, HHA_img, saliency_map, prediction):
    fig, axes = plt.subplots(1, 4, figsize=(16, 8))
    axes[0].imshow(rgb_img)
    axes[0].set_title("RGB Image")
    
    axes[1].imshow(HHA_img) 
    axes[1].set_title("HHA Image")
    
    axes[2].imshow(saliency_map[:, :], cmap='gray')
    axes[2].set_title("Ground Truth Saliency Map")
    
    axes[3].imshow(prediction[:, :, 0], cmap='gray')
    axes[3].set_title("Predicted Saliency Map")
    plt.savefig(output_path, format="png")
    

#BELOW ABOUT PRECISION-RECALL METHOD

# After training the model and obtaining predictions
y_true = saliency_maps_test.flatten()  # Flatten the ground truth saliency maps to a 1D array
# Apply threshold to the ground truth to make it binary (0 or 1)
y_true_binary = (y_true > 0.5).astype(int)  # Assuming values above 0.5 are considered salient
y_scores = loaded_model.predict([rgb_images_test, HHA_images_test]).flatten()  # Flatten the predicted saliency maps to a 1D array


# Calculate precision-recall curve values
precision, recall, thresholds = precision_recall_curve(y_true_binary, y_scores)

# Calculate Average Precision (AP)
average_precision = average_precision_score(y_true_binary, y_scores)
print(f"Average Precision (AP): {average_precision:.4f}")

# Calculate F1 scores at each threshold
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8) #The term 1e-8 is added to avoid division by zero

# Best F1 score
best_f1_index = f1_scores.argmax()
best_f1 = f1_scores[best_f1_index]
best_threshold = thresholds[best_f1_index]
print(f"Best F1 Score: {best_f1:.4f} at threshold {best_threshold:.4f}")

# Plot precision-recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='o')
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid()
plt.show()


############################# TESTING IMAGE FOLDER GENERATOR ###########################

for sample_index in range(len(rgb_images_test)):
    sample_rgb = rgb_images_test[sample_index:sample_index+1]  # Take a single RGB image
    sample_HHA = HHA_images_test[sample_index:sample_index+1]  # Take the corresponding HHA image
    sample_saliency = saliency_maps_test[sample_index]  # Ground truth saliency map for comparison

    # Predict saliency map
    predicted_saliency = loaded_model.predict([sample_rgb, sample_HHA])

    # Ensure the folder exists
    os.makedirs(image_folder_path, exist_ok=True)
    output_file = f"test_plot_{sample_index}.png"
    output_path = os.path.join(image_folder_path, output_file)
    
    visualize_saliency(rgb_img=sample_rgb[0],HHA_img=sample_HHA[0], saliency_map=sample_saliency, prediction=predicted_saliency[0])

print(f"Finished saving plots at: {output_path}")
 