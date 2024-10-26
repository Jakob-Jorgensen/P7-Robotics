import tensorflow as tf 
from keras import models, layers,backend
from sklearn.model_selection import train_test_split  
import matplotlib.pyplot as plt 
import os, glob
import numpy as np   
import cv2 
#import PIL.Image as Image

def build_stream(input_shape,stream_name):
    model = models.Sequential(name=stream_name) 

    model.add(layers.Conv2D(96, (11,11), activation='relu',input_shape=input_shape,padding='valid',strides=4))
    model.add(layers.MaxPooling2D((3, 3),strides=1))
    model.add(layers.BatchNormalization(axis=-1))

    model.add(layers.Conv2D(256, (5, 5),strides=1, dilation_rate=2, activation='relu',padding='same'))
    model.add(layers.BatchNormalization(axis=-1))

    model.add(layers.Conv2D(384, (3, 3), dilation_rate=4, activation='relu',strides=1,padding='same'))
    model.add(layers.Conv2D(384, (3, 3), dilation_rate=4, activation='relu',strides=1,padding='same')) 

    model.add(layers.Conv2D(256, (3, 3), dilation_rate=4, activation='relu',strides=1,padding='same'))   
    model.add(layers.MaxPooling2D((3, 3),strides=1))  

    model.add(layers.Dropout(0.5)) 
    model.add(layers.Conv2D(1, (1, 1), activation='relu',padding='same')) 

    model.summary()
    return model

def build_fusion_model(rgb_shape, depth_shape): 
    # Define the inputs
    rgb_input = layers.Input(shape=rgb_shape, name='rgb_input')
    depth_input = layers.Input(shape=depth_shape, name='depth_input')

    # Build the streams
    rgb_stream = build_stream(rgb_shape, 'rgb_Sequential')
    depth_stream = build_stream(depth_shape, 'depth_Sequential') 

    # Get the features from the streams
    rgb_features = rgb_stream(rgb_input)
    depth_features = depth_stream(depth_input) 

    # Concatenate features from both streams 
    fused = layers.Concatenate(axis=-1)([rgb_features, depth_features]) 
    #Reduce the number of channels 
    fused = layers.Conv2D(3, (1, 1), activation='relu',padding='same')(fused)
     
    # Build the model
    model = models.Model(inputs=[rgb_input, depth_input], outputs=fused, name='fusion_model') 
    #categorical_crossentropy binary_crossentropy
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

# Paths to your datasets
rgb_dir = sorted(glob.glob(os.path.join("dataset","RGB_left", "**")))
depth_dir = sorted(glob.glob(os.path.join("dataset","depth", "**")))
gt_dir =sorted(glob.glob(os.path.join("dataset","GT-1985", "**")) )
# The images must be sorted to match the ground truth images 

#print(rgb_dir) 
#print(depth_dir)

# Define the batch size and image sizes
batch_size = 32
img_size = (224, 224)
gt_size = (50,50)

# Load and resize the images 
print("Resizing images...")

rgb_images = np.array([cv2.resize(cv2.imread(file), img_size) for file in rgb_dir])
#depth_images = np.array([cv2.cvtColor(cv2.resize(cv2.imread(file), img_size),cv2.COLOR_BGR2GRAY) for file in depth_dir]) 
depth_images = np.array([cv2.resize(cv2.imread(file), img_size) for file in depth_dir])
gt_images = np.array([cv2.resize(cv2.imread(file), gt_size) for file in gt_dir])

print("Images resized.")  

debug_bool = False
if debug_bool == True:
    cv2.imshow("RGB",rgb_images[5])  
    cv2.imshow("Depth",depth_images[5])  
    cv2.imshow("GT",gt_images[5])
    cv2.imshow("RGB1",rgb_images[200])  
    cv2.imshow("Depth1",depth_images[200]) 
    cv2.imshow("GT1",gt_images[200])  
    
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
print(depth_images[5].shape)
# Split the dataset into training and validation sets
RGB_train, RGB_valid, depth_train, depth_valid, GT_train, GT_valid=train_test_split(rgb_images, depth_images, gt_images, test_size=0.2, random_state=42, shuffle=True) 
if debug_bool == True:
    cv2.imshow("RGB",RGB_train[5])  
    cv2.imshow("Depth",depth_train[5])  
    cv2.imshow("GT",GT_train[5])
    cv2.imshow("RGB1",RGB_train[200])  
    cv2.imshow("Depth1",depth_train[200]) 
    cv2.imshow("GT1",GT_train[200]) 
    cv2.waitKey(0) 

# Build the fusion model
model = build_fusion_model((img_size[0], img_size[1], 3), (img_size[0], img_size[1], 3))

# Train the model using the optimized dataset
history = model.fit(
    [RGB_train, depth_train],  # Inputs as a list
    GT_train,
    epochs=10,
    batch_size=16, 
    validation_data=([RGB_valid, depth_valid], GT_valid)  #
)
model.save("fusion_model_with_dropout.h5")


# Evaluate the model 
loss, accuracy = model.evaluate([RGB_valid, depth_valid], GT_valid) 
print("Loss: ", loss) 
print("Accuracy: ", accuracy)  

#plot the training and validation accuracy and loss at each epoch 



#Plot the depth rgb image, depth, ground truth and the predicted image 
def plot_image(rgb, depth, gt, pred):
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 4, 1)
    plt.imshow(rgb)
    plt.title("RGB Image")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(depth, cmap="gray")
    plt.title("Depth Image")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(gt, cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(pred, cmap="gray")
    plt.title("Predicted Image")
    plt.axis("off")
    plt.show()

