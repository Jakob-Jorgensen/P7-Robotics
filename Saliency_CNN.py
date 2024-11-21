from keras import models, layers
from sklearn.model_selection import train_test_split  
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt 
import os, glob
import numpy as np   
import cv2 

batchsize = 4
epoch = 5

# Define the stream model:
def build_stream(input_shape,stream_name):
    model = models.Sequential(name=stream_name)  

    model.add(layers.Input(shape=input_shape)) 

    model.add(layers.Conv2D(96, (11,11), activation='relu',padding='valid',strides=4))
    model.add(layers.MaxPooling2D((3, 3),strides=1))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))

    model.add(layers.Conv2D(256, (5, 5),strides=1, dilation_rate=2, activation='relu',padding='same')) 
    model.add(layers.MaxPooling2D((3, 3),strides=1))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))

    #model.add(layers.Conv2D(384, (3, 3), dilation_rate=4, activation='relu',strides=1,padding='same'))
    #model.add(layers.Conv2D(384, (3, 3), dilation_rate=4, activation='relu',strides=1,padding='same')) 

    #model.add(layers.Conv2D(256, (3, 3), dilation_rate=4, activation='relu',strides=1,padding='same'))   
 
    #model.add(layers.Dropout(0.5))  
    #model.add(layers.Conv2D(1, (1, 1), activation='sigmoid')) 

    # Upsampling
    model.add(layers.Conv2DTranspose(128, (4, 4), activation='relu', strides=2, padding='same'))  # 50 -> 100
    model.add(layers.Conv2DTranspose(64, (4, 4), activation='relu', strides=2, padding='same'))   # 100 -> 200
    #model.add(layers.Conv2DTranspose(32, (13, 13), activation='relu', strides=1, padding='valid'))  
    model.add(layers.Conv2DTranspose(3, (25, 25), activation='sigmoid', strides=1, padding='valid'))  
    model.summary()

    return model 

# Define the fusion model:
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
    out_put = layers.Conv2D(1, (1, 1), activation='sigmoid',padding='same')(fused)
     
    # Build the model
    model = models.Model(inputs=[rgb_input, depth_input], outputs=out_put, name='fusion_model')  
    
    #Binary_crossentropy is used because the output is binary 
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

# Paths to the dataset
rgb_train_dir = sorted(glob.glob(os.path.join(r"C:\Users\mikke\Desktop\Final_dataset\Final_dataset\Testing\RGB\*.png")))
rgb_valid_dir = sorted(glob.glob(os.path.join(r"C:\Users\mikke\Desktop\Final_dataset\Final_dataset\Validation\RGB\*.png")))
depth_train_dir = sorted(glob.glob(os.path.join(r"C:\Users\mikke\Desktop\Final_dataset\Final_dataset\Testing\HHA\*.png")))
depth_valid_dir = sorted(glob.glob(os.path.join(r"C:\Users\mikke\Desktop\Final_dataset\Final_dataset\Validation\HHA\*.png")))
gt_train_dir =sorted(glob.glob(os.path.join(r"C:\Users\mikke\Desktop\Final_dataset\Final_dataset\Testing\GT\*.png")) )
gt_valid_dir =sorted(glob.glob(os.path.join(r"C:\Users\mikke\Desktop\Final_dataset\Final_dataset\Validation\GT\*.png")) )

# Define the image sizes
img_size = (224, 224)
gt_size = (224,224)

# Loading and resize the images 
#print("Resizing images...")
#rgb_images = np.array([cv2.resize(cv2.imread(file), img_size).astype('float32')/255.0 for file in rgb_dir]) 
#depth_images = np.array([cv2.cvtColor(cv2.resize(cv2.imread(file), img_size).astype('float32')/255.0, cv2.COLOR_RGB2GRAY) for file in depth_dir])  # Reducing the number of channels to 1 
#gt_images = np.array([cv2.cvtColor(cv2.resize(cv2.imread(file), gt_size).astype('float32')/255.0, cv2.COLOR_RGB2GRAY) for file in gt_dir]) # Reducing the number of channels to 1 
#print("Images resized.")  

print("Resizing images...")
RGB_train = np.array([cv2.resize(cv2.imread(file), img_size).astype('float32')/255.0 for file in rgb_train_dir]) 
RGB_valid = np.array([cv2.resize(cv2.imread(file), img_size).astype('float32')/255.0 for file in rgb_valid_dir]) 
depth_train = np.array([cv2.cvtColor(cv2.resize(cv2.imread(file), img_size).astype('float32')/255.0, cv2.COLOR_RGB2GRAY) for file in depth_train_dir])
depth_valid = np.array([cv2.cvtColor(cv2.resize(cv2.imread(file), img_size).astype('float32')/255.0, cv2.COLOR_RGB2GRAY) for file in depth_valid_dir])
GT_train = np.array([cv2.cvtColor(cv2.resize(cv2.imread(file), gt_size).astype('float32')/255.0, cv2.COLOR_RGB2GRAY) for file in gt_train_dir])
GT_valid = np.array([cv2.cvtColor(cv2.resize(cv2.imread(file), gt_size).astype('float32')/255.0, cv2.COLOR_RGB2GRAY) for file in gt_valid_dir])
print("Images resized.")  

# Split the dataset into training and validation sets
#RGB_train, RGB_valid, depth_train, depth_valid, GT_train, GT_valid=train_test_split(rgb_images, depth_images, gt_images, test_size=0.2,shuffle=True)  

# Build the fusion model
model = build_fusion_model((img_size[0], img_size[1], 3), (img_size[0], img_size[1], 1))

# Train the model using the optimized dataset
history = model.fit(
    [RGB_train, depth_train],  # Training data
    GT_train, # Ground truth labels
    epochs=epoch,
    batch_size=batchsize, 
    validation_freq=1,  
    shuffle=True,
    validation_data=([RGB_valid, depth_valid], GT_valid)  
)  

# Save the model
model.save("jakob_Playground_model.keras")

# Evaluate the model 
loss, accuracy = model.evaluate([RGB_valid, depth_valid], GT_valid) 
print("Loss: ", loss) 
print("Accuracy: ", accuracy)   



y_true = GT_valid.flatten()   
y_true_binary = (y_true > 0.5).astype(int)  # Convert to binary values 
y_scores = model.predict([RGB_valid, depth_valid]).flatten()  # Predict saliency maps and flatten   

precision, recall, thresholds = precision_recall_curve(y_true_binary, y_scores)  
average_precision = average_precision_score(y_true_binary, y_scores) 
print("Average Precision: ", average_precision) 
f1_scores = 2 * (precision * recall) / (precision + recall+ 1e-8) # makes sure we dont divide by zero 


best_f1_index = np.argmax(f1_scores) 
best_f1 = f1_scores[best_f1_index] 
best_threshold = thresholds[best_f1_index] 
print(f"Best F1 Score: {best_f1} at thredshold {best_threshold} " )  

# Plot the training and validation accuracy and loss at each epoch
plt.figure(figsize=(14, 5))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.ylim(0, 1)
plt.xlim(0,epoch-1)

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.ylim(0, 1)
plt.xlim(0,epoch-1)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend() 

plt.figure(figsize=(8, 6)) 
plt.plot(recall, precision, label='Precision-Recall Curve')  
plt.xlabel('Recall') 
plt.ylabel('Precision') 
plt.grid() 
plt.show() 


plt.show()

