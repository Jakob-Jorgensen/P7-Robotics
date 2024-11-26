import tensorflow as tf
from keras import layers, Model
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
import os, glob
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import pandas as pd 


### Settings
visualize_loaded_data = False

# Defines our loss function using weights
def weighted_binary_crossentropy(class_weights):
    def loss(y_true, y_pred):
        weights = y_true * class_weights[1] + (1 - y_true) * class_weights[0]
        return tf.reduce_mean(weights * tf.keras.losses.binary_crossentropy(y_true, y_pred))
    return loss

rgb_train_dir = sorted(glob.glob(os.path.join(r"C:\Users\mikke\Desktop\Final_dataset\Final_dataset\Testing\RGB", '*.png')))
rgb_valid_dir = sorted(glob.glob(os.path.join(r"C:\Users\mikke\Desktop\Final_dataset\Final_dataset\Validation\RGB", '*.png')))
depth_train_dir = sorted(glob.glob(os.path.join(r"C:\Users\mikke\Desktop\Final_dataset\Final_dataset\Testing\HHA", '*.png')))
depth_valid_dir = sorted(glob.glob(os.path.join(r"C:\Users\mikke\Desktop\Final_dataset\Final_dataset\Validation\HHA", '*.png')))
gt_train_dir =sorted(glob.glob(os.path.join(r"C:\Users\mikke\Desktop\Final_dataset\Final_dataset\Testing\GT", '*.png')) )
gt_valid_dir =sorted(glob.glob(os.path.join(r"C:\Users\mikke\Desktop\Final_dataset\Final_dataset\Validation\GT", '*.png')) )

# Define the image sizes
img_size = (224, 224)
gt_size = (224,224)


print("Resizing images...")
rgb_train = np.array([cv2.resize(cv2.imread(file), img_size).astype('float32')/255.0 for file in rgb_train_dir]) 
rgb_val = np.array([cv2.resize(cv2.imread(file), img_size).astype('float32')/255.0 for file in rgb_valid_dir]) 
depth_train = np.array([cv2.resize(cv2.imread(file), img_size).astype('float32')/255.0 for file in depth_train_dir])
depth_val = np.array([cv2.resize(cv2.imread(file), img_size).astype('float32')/255.0 for file in depth_valid_dir])
saliency_train = np.array([cv2.cvtColor(cv2.resize(cv2.imread(file), gt_size).astype('float32')/255.0, cv2.COLOR_RGB2GRAY) for file in gt_train_dir])
saliency_val = np.array([cv2.cvtColor(cv2.resize(cv2.imread(file), gt_size).astype('float32')/255.0, cv2.COLOR_RGB2GRAY) for file in gt_valid_dir])
gt_unsized = np.array([cv2.cvtColor(cv2.imread(file).astype('float32')/255.0, cv2.COLOR_RGB2GRAY) for file in gt_train_dir])
print("Images resized.")  


######### CODE FOR CALCULATING CLASS WEIGHTS!!!
print('Calculating class weights...')

ground_truths = np.stack(gt_unsized)
num_classes = len(np.unique(ground_truths))
pixels_per_class = np.zeros(num_classes)
# [0.00585605 0.99414395]
for class_id in range(num_classes):
    pixels_per_class[class_id] = np.sum(ground_truths == class_id)

total_pixels = np.sum(pixels_per_class)
class_weights = total_pixels / (num_classes * pixels_per_class)

class_weights /= np.sum(class_weights)
print('Finished calculating class weights: ', class_weights)

class_weights_dict = {i: w for i, w in enumerate(class_weights)}

if visualize_loaded_data == True: 
    for i in range(len(rgb_train)):
        cv2.imshow('rgb', rgb_train[0+i])
        cv2.imshow('depth', depth_train[0+i])
        cv2.imshow('gt', saliency_train[0+i])
        cv2.imshow('val1', rgb_val[0+i])
        cv2.imshow('val2', depth_val[0+i])
        cv2.imshow('val3', saliency_val[0+i])
        cv2.waitKey(0)
        i+1

# Define the CNN architecture for RGB-D saliency detection
class Saliency(Model):
    def __init__(self):
        super(Saliency, self).__init__()

        # RGB Stream
        self.rgb_conv1 = layers.Conv2D(2, (3, 3), activation='relu', padding='valid')
        self.rgb_maxpool1 = layers.MaxPooling2D((3, 3), strides=1)
        self.rgb_norm1 =  layers.BatchNormalization()

        self.rgb_dilated_conv2 = layers.Conv2D(256, (3, 3), strides=1 , padding='same', dilation_rate=2, activation='relu')
        self.rgb_maxpool2 = layers.MaxPooling2D(pool_size=(3, 3), strides=1)
        self.rgb_norm2 =  layers.BatchNormalization()

        self.rgb_dilated_conv3 = layers.Conv2D(384, (3, 3), strides=1 , padding='same', dilation_rate=4, activation='relu')
        self.rgb_dilated_conv4 = layers.Conv2D(384, (3, 3), strides=1 , padding='same', dilation_rate=4, activation='relu')
        self.rgb_dilated_conv5 = layers.Conv2D(256, (3, 3), strides=1 , padding='same', dilation_rate=4, activation='relu')

        # Depth (HHA) Stream
        self.depth_conv1 = layers.Conv2D(2, (3, 3), activation='relu', padding='valid')
        self.depth_maxpool1 = layers.MaxPooling2D((3, 3), strides=1)
        self.depth_norm1 =  layers.BatchNormalization()

        self.depth_dilated_conv2 = layers.Conv2D(256, (3, 3), strides=1 , padding='same', dilation_rate=2, activation='relu')
        self.depth_maxpool2 = layers.MaxPooling2D((3, 3), strides=1)
        self.depth_norm2 =  layers.BatchNormalization()

        self.depth_dilated_conv3 = layers.Conv2D(384, (3, 3), strides=1 , padding='same', dilation_rate=4, activation='relu')
        self.depth_dilated_conv4 = layers.Conv2D(384, (3, 3), strides=1 , padding='same', dilation_rate=4, activation='relu')
        self.depth_dilated_conv5 = layers.Conv2D(256, (3, 3), strides=1 , padding='same', dilation_rate=4, activation='relu')

        # Fusion and final layer
        self.fc_conv1 = layers.Conv2D(1, (1, 1), activation='sigmoid')  
        self.fc_conv2 = layers.Conv2D(1, (1, 1), activation='sigmoid') 
        self.trans_conv1 = layers.Conv2DTranspose(1, (5, 5), activation='sigmoid', strides=1, padding='valid')

    # Forward pass for RGB stream
    def forward_rgb(self, x):
        x = self.rgb_conv1(x)
        x = self.rgb_maxpool1(x)
        x = self.rgb_norm1(x)
        #x = self.rgb_dilated_conv2(x)
        #x = self.rgb_maxpool2(x)
        #x = self.rgb_norm2(x)
        #x = self.rgb_dilated_conv3(x)
        #x = self.rgb_dilated_conv4(x)
        #x = self.rgb_dilated_conv5(x)
        #x = self.fc_conv1(x)
        x = self.trans_conv1(x)

        return x

    # Forward pass for Depth stream
    def forward_depth(self, x):
        x = self.depth_conv1(x)
        x = self.depth_maxpool1(x)
        x = self.depth_norm1(x)
        #x = self.depth_dilated_conv2(x)
        #x = self.depth_maxpool2(x)
        #x = self.depth_norm2(x)
        #x = self.depth_dilated_conv3(x)
        #x = self.depth_dilated_conv4(x)
        #x = self.depth_dilated_conv5(x)
        #x = self.fc_conv1(x)
        x = self.trans_conv1(x)

        return x

    # Forward pass combining RGB and Depth (HHA) streams
    def call(self, inputs):
        rgb, depth = inputs 
        # Process RGB and Depth streams separately
        rgb_out = self.forward_rgb(rgb)
        depth_out = self.forward_depth(depth)

        # Fuse high-level features from both streams
        fused = tf.concat([rgb_out, depth_out], axis=-1)

        # Final prediction
        out = self.fc_conv2(fused)
        #fused = fused[:, :, :, :1]  
     
        #print(f"Out shape: {out.shape}")
        
        return out

# Instantiate the model
model = Saliency()

# Define loss function and optimizer (Adam)
loss_fn = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam()

# Compile the model
model.compile(optimizer=optimizer, loss=weighted_binary_crossentropy(class_weights), metrics=['accuracy'])

# model.load_weights('C:/Users/eymen/Documents/project1/trainmodel.keras')        #Load previosly saved model weights


# Train the model
history = model.fit(
    [rgb_train, depth_train],  # Inputs as a list
    saliency_train,            # Targets
    epochs=5, 
    batch_size=16, 
    validation_data=([rgb_val, depth_val], saliency_val)  # Validation data
)


# Print accuracy after each epoch
for epoch, acc in enumerate(history.history['accuracy']):
    print(f"Epoch {epoch+1}: Accuracy: {acc}")

model.summary()
model.save('trainmodel.keras')                                #Save model


# Function to visualize input and output saliency map
def visualize_saliency(rgb_img, depth_img, saliency_map, prediction):

    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    depth_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 4, figsize=(16, 8))
    axes[0].imshow(rgb_img)
    axes[0].set_title("RGB Image")
    
    axes[1].imshow(depth_img)
    axes[1].set_title("Depth Image")
    
    axes[2].imshow(saliency_map[:, :], cmap='gray')
    axes[2].set_title("Ground Truth Saliency Map")
    
    axes[3].imshow(prediction[:, :], cmap='gray')
    axes[3].set_title("Predicted Saliency Map")
    
    plt.show()

# Predict saliency map for a sample image from the validation set
sample_index = 0  # Change this index to visualize different samples
sample_rgb = rgb_val[sample_index:sample_index+1]  # Take a single RGB image
sample_depth = depth_val[sample_index:sample_index+1]  # Take the corresponding depth image
sample_saliency = saliency_val[sample_index]  # Ground truth saliency map for comparison

# Predict saliency map
predicted_saliency = model.predict([sample_rgb, sample_depth])
print(f"predicted shape: {predicted_saliency.shape}")
print("saliency map output values:",predicted_saliency[0,:,:,0])
print("saliency map output values:",predicted_saliency[0,25,25,0])

# Visualize the result
visualize_saliency(sample_rgb[0], sample_depth[0], sample_saliency, predicted_saliency[0])




#BELOW ABOUT PRECISION-RECALL METHOD

# After training the model and obtaining predictions
y_true = saliency_val.flatten()  # Flatten the ground truth saliency maps to a 1D array
# Apply threshold to the ground truth to make it binary (0 or 1)
y_true_binary = (y_true > 0.5).astype(int)  # Assuming values above 0.5 are considered salient
y_scores = model.predict([rgb_val, depth_val]).flatten()  # Flatten the predicted saliency maps to a 1D array


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
