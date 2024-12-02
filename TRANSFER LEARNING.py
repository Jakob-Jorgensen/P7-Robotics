import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import os
import cv2

def load_dataset(rgb_folder, depth_folder, saliency_folder, HHA_folder, target_size=(224, 224)):
    rgb_images, depth_images, saliency_maps,HHA_images = [], [], [], []    #create empty lists

    rgb_files = sorted(os.listdir(rgb_folder))              #Sorted lists of the files and directories in the specified folders
    depth_files = sorted(os.listdir(depth_folder))
    saliency_files = sorted(os.listdir(saliency_folder)) 
    HHA_files = sorted(os.listdir(HHA_folder)) 
    
    print(f"Found {len(rgb_files)} RGB images.")            #Printing founded image count from a list length
    print(f"Found {len(depth_files)} depth images.")
    print(f"Found {len(saliency_files)} saliency maps.") 
    print(f"Found {len(HHA_files)} HHA images.") 
 
    for i, img_file in enumerate(rgb_files):                #For loop for all images
        rgb_path = os.path.join(rgb_folder, img_file)       #Make path for each image 
                
        temp_name = img_file.split('RGB')[1]
        depth_path = os.path.join(depth_folder, 'Depth' + temp_name.split('.')[0] + '.Tiff')
        saliency_path = os.path.join(saliency_folder, 'GT' + img_file.split('RGB')[1]) 
        HHA_path = os.path.join(HHA_folder, 'HHA' + img_file.split('RGB')[1])
        
        # Debugging: Print paths and check file existence
        if not os.path.exists(rgb_path):
            print(f"RGB image not found: {rgb_path}")
            continue
        if not os.path.exists(depth_path):
            print(f"Depth image not found: {depth_path}")
            continue
        if not os.path.exists(saliency_path):
            print(f"Saliency map not found: {saliency_path}")
            continue  
        if not os.path.exists(HHA_path): 
            print(f"HHA image not found: {HHA_path}")
            continue    

        
        
        rgb_image = preprocess_image(rgb_path, target_size,BGR2RGB=True)                                 #Send path and target size to preprocess function
        depth_image = np.expand_dims(preprocess_image(depth_path, target_size), axis=-1)    #Send path and target size to preprocess function
        saliency_map = preprocess_image(saliency_path, target_size,Binary_image= True)  
        HHA_image = preprocess_image(HHA_path, target_size,BGR2RGB=True)                                 #Send path and target size to preprocess function
        rgb_images.append(rgb_image)                                                        #Add preprocessed images to list
        depth_images.append(depth_image)                                                    #Add preprocessed depth images to list
        saliency_maps.append(saliency_map)                                                  #Add preprocessed saliency maps to list     
        HHA_images.append(HHA_image)                                                        #Add preprocessed HHA images to list

        

        
    return np.array(rgb_images), np.array(depth_images), np.array(saliency_maps), np.array(HHA_images)  #Convert lists to Numpy arrays as an output
    

   

# Preprocessing function to load and preprocess both RGB and Depth images
# Preprocessing function to load and preprocess both RGB and Depth images
def preprocess_image(image_path, target_size, Binary_image=False, BGR2RGB=False):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # Load the image using OpenCV
    
    if Binary_image:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert the image to grayscale
        image = cv2.resize(image, target_size)  # Resize the grayscale image
        image = image.astype('float32') / 255.0  # Normalize
        
        # Convert to 3 channels
        image = np.expand_dims(image, axis=-1)  # Shape: (224, 224, 1)
        image = np.concatenate([image] * 3, axis=-1)  # Shape: (224, 224, 3)
    elif BGR2RGB:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, target_size)
        image = image.astype('float32') / 255.0
    else:
        image = cv2.resize(image, target_size)
        image = image.astype('float32') / 255.0


    return image


# Define the model with MobileNetV2 as the baseline
def create_model():
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze the base model

    # RGB and Depth input
    rgb_input = layers.Input(shape=(224, 224, 3), name="rgb_input")
    depth_input = layers.Input(shape=(224, 224, 3), name="depth_input")

    # Pass RGB and Depth through the base model
    rgb_features = base_model(rgb_input)
    depth_features = base_model(depth_input)

    # Add custom layers to combine RGB and Depth features
    x = layers.Concatenate(axis=-1)([rgb_features, depth_features])  # Concatenate RGB and Depth features
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(1, activation='sigmoid')(x)  # For binary classification (adjust if necessary)

    # Define the model
    model = models.Model(inputs=[rgb_input, depth_input], outputs=x)
    
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Train the model
def train_model(model, rgb_images, depth_images, saliency_maps, rgb_images_val, depth_images_val, saliency_maps_val):
    history = model.fit(
        [rgb_images, depth_images], saliency_maps,  # Train with RGB and Depth data
        batch_size=16,  # Adjust batch size based on your system
        epochs=10,  # Adjust epochs as needed
        validation_data=([rgb_images_val, depth_images_val], saliency_maps_val)  # Validation
    )
    return history

# Visualize training results
def plot_training_results(history):
    # Plot the training and validation loss
    plt.figure(figsize=(12, 6))

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

# Visualize predictions
def visualize_predictions(model, rgb_images, depth_images, saliency_maps):
    predictions = model.predict([rgb_images, depth_images])  # Get predictions
    
    # Display a few sample images with their predicted vs actual saliency maps
    num_samples = 5
    plt.figure(figsize=(12, 12))
    
    for i in range(num_samples):
        plt.subplot(num_samples, 3, i*3 + 1)
        plt.imshow(rgb_images[i])  # Display RGB image
        plt.title('RGB Image')
        plt.axis('off')

        plt.subplot(num_samples, 3, i*3 + 2)
        plt.imshow(saliency_maps[i].reshape(224, 224), cmap='jet')  # Actual saliency map
        plt.title('Ground Truth')
        plt.axis('off')

        plt.subplot(num_samples, 3, i*3 + 3)
        plt.imshow(predictions[i].reshape(224, 224), cmap='jet')  # Predicted saliency map
        plt.title('Predicted')
        plt.axis('off')
    
    plt.show()

# Main function to run the full process
def main():
    # Path to dataset folders
    main_path = r"C:\Users\simao\Documents\aau\1ST SEMESTER\project\final proj dataset\Dataset_3.2"

    rgb_folder = os.path.join(main_path, "Training", "RGB")
    depth_folder = os.path.join(main_path, "Training", "Depth")
    saliency_folder = os.path.join(main_path, "Training", "GT")
    HHA_folder = os.path.join(main_path, "Training", "HHA")

    rgb_folder_val = os.path.join(main_path, "Validation", "RGB")
    depth_folder_val = os.path.join(main_path, "Validation", "Depth")
    saliency_folder_val = os.path.join(main_path, "Validation", "GT")
    HHA_folder_val = os.path.join(main_path, "Validation", "HHA")

    # Load the dataset for training and validation
    rgb_images, depth_images, saliency_maps, HHA_images = load_dataset(rgb_folder, depth_folder, saliency_folder, HHA_folder)
    rgb_images_val, depth_images_val, saliency_maps_val, HHA_images_val = load_dataset(rgb_folder_val, depth_folder_val, saliency_folder_val, HHA_folder_val)

    print(f"RGB Images Shape: {rgb_images.shape}")            # Ensure (N, 224, 224, 3)
    print(f"Depth Images Shape: {depth_images.shape}")        # Ensure (N, 224, 224, 3)
    print(f"Saliency Maps Shape: {saliency_maps.shape}")      # Ensure (N, 224, 224, 1)
    # Create the model
    model = create_model()

    # Train the model (correcting the inputs)
    history = train_model(model, rgb_images, HHA_images, saliency_maps, rgb_images_val, HHA_images_val, saliency_maps_val)

    # Visualize the training results
    plot_training_results(history)

    # Visualize the predictions
    visualize_predictions(model, rgb_images_val, depth_images_val, saliency_maps_val)

if __name__ == "__main__":
    main()

