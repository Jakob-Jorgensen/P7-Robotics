
import tensorflow as tf
from keras import layers, Model
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

##############################################################
#main_path = f"C:/Users/jakob/Downloads/Dataset_3.1" 
main_path  = r"D:\CNN_Tensorflow_code\Dataset_3.1"
loss_function = 'dice_loss' # Chose between 'dice_loss' or 'binary_crossentropy'
epochs = 5 
If_trash = False # Chose between trash mode or running the real model
##############################################################

def weighted_binary_crossentropy(pos_weight, neg_weight):
    def loss_fn(y_true, y_pred):
        # Compute binary cross-entropy
        bce = -(pos_weight * y_true * tf.math.log(y_pred + 1e-8) + 
                neg_weight * (1 - y_true) * tf.math.log(1 - y_pred + 1e-8))
        return tf.reduce_mean(bce)
    return loss_fn



def IoC_loss(predicted_mask, ground_truth_mask):
    # Calculate soft intersection and union
    intersection = tf.reduce_sum(predicted_mask * ground_truth_mask)
    union = tf.reduce_sum(predicted_mask + ground_truth_mask - predicted_mask * ground_truth_mask)
    
    # Avoid division by zero
    iou = tf.math.divide_no_nan(intersection, union)
    #print(iou)
    return iou


def weighted_dice_loss(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Flatten the tensors to calculate overlap
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])
    
    # Calculate the number of positive and negative pixels in the ground truth
    num_pos = tf.reduce_sum(y_true_flat)
    num_neg = tf.reduce_sum(1.0 - y_true_flat)
    
    # Calculate positive and negative weights
    pos_weight = tf.cond(tf.greater(num_pos, 0), lambda: num_neg / (num_pos + num_neg), lambda: 1.0)
    neg_weight = tf.cond(tf.greater(num_neg, 0), lambda: num_pos / (num_pos + num_neg), lambda: 1.0)
    
    # Compute weighted true positives, false positives, and false negatives
    intersection = tf.reduce_sum(pos_weight * y_true_flat * y_pred_flat)
    false_negatives = tf.reduce_sum(pos_weight * y_true_flat * (1 - y_pred_flat))
    false_positives = tf.reduce_sum(neg_weight * (1 - y_true_flat) * y_pred_flat)
    
    # Compute the weighted denominator
    denominator = 2 * intersection + false_negatives + false_positives

    # Dice coefficient
    dice_coeff = (2 * intersection + smooth) / (denominator + smooth)
    
    # Dice loss
    return 1.0 - dice_coeff

# Preprocessing function to load and preprocess both RGB and Depth images
def preprocess_image(image_path, target_size, Binary_image = False,BGR2RGB = False):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)              # Load the image using OpenCV 
    if Binary_image == True:                                          #If the image is binary
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)                 #Convert the image to grayscale¨
    elif BGR2RGB == True: 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else: 
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)   
  
    image = cv2.resize(image, target_size)      # Resize the image using OpenCV
    image = image.astype('float32') / 255.0  # Normalize the image to range [0, 1]
    return image

# Load RGB and Depth images from folder
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
        saliency_path = os.path.join(saliency_folder, 'undistored_GT' + img_file.split('RGB')[1]) 
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
    

   
# Dataset folder paths
#main_path = f"C:/Users/astri/Downloads/Dataset_3.0/Dataset_3.0" 

rgb_folder = f"{main_path}/Training/RGB"
depth_folder = f"{main_path}/Training/Depth"
saliency_folder = f"{main_path}/Training/GT" 
HHA_folder = f"{main_path}/Training/HHA" 

rgb_folder_val = f"{main_path}/Validation/RGB"
depth_folder_val = f"{main_path}/Validation/Depth"
saliency_folder_val = f"{main_path}/Validation/GT" 
HHA_folder_val = f"{main_path}/Validation/HHA"

rgb_folder_test = f"{main_path}/Testing/RGB"
depth_folder_test = f"{main_path}/Testing/Depth"
saliency_folder_test = f"{main_path}/Testing/GT" 
HHA_folder_test = f"{main_path}/Testing/HHA"

# Load the dataset
rgb_images, depth_images, saliency_maps,HHA_images = load_dataset(rgb_folder, depth_folder, saliency_folder,HHA_folder)   #Send folder paths to load dataset function
rgb_images_val, depth_images_val, saliency_maps_val,HHA_images_val = load_dataset(rgb_folder_val, depth_folder_val, saliency_folder_val,HHA_folder_val)

#rgb_images_test, depth_images_test, saliency_maps_test,HHA_images_test = load_dataset(rgb_folder_test, depth_folder_test, saliency_folder_test,HHA_folder_test)


# Check dataset shapes
print(f"RGB images shape: {rgb_images.shape}")
print(f"Depth images shape: {depth_images.shape}")
print(f"Saliency maps shape: {saliency_maps.shape}") 
print(f"HHA images shape: {HHA_images.shape}") 

# Split the dataset into training and validation sets (80% train, 20% validation)
#rgb_train, rgb_val, depth_train, depth_val, saliency_train, saliency_val = train_test_split(
#                                                                            rgb_images, depth_images, saliency_maps, test_size=0.2, random_state=42)


pos_weight = np.mean(1 - saliency_maps)  # Mean of non-salient (background) pixels
neg_weight = np.mean(saliency_maps) 

# Define the CNN architecture for RGB-D saliency detection
class Saliency(Model):
    def __init__(self):
        super(Saliency, self).__init__()

        self.conv1 = layers.Conv2D(96, (11, 11), strides=4, activation='relu', padding='valid')
        self.maxpool1 = layers.MaxPooling2D((3, 3), strides=1)
        self.norm1 =  layers.BatchNormalization() 
        

        self.dilated_conv2 = layers.Conv2D(256, (3, 3), strides=1 , padding='same', dilation_rate=2, activation='relu')
        self.maxpool2 = layers.MaxPooling2D(pool_size=(3, 3), strides=1)
        self.norm2 =  layers.BatchNormalization() 
        
        self.dilated_conv3 = layers.Conv2D(384, (3, 3), strides=1 , padding='same', dilation_rate=4, activation='relu')
        self.dilated_conv4 = layers.Conv2D(384, (3, 3), strides=1 , padding='same', dilation_rate=4, activation='relu')
        self.dilated_conv5 = layers.Conv2D(256, (3, 3), strides=1 , padding='same', dilation_rate=4, activation='relu')

        self.trapos1 = layers.Conv2DTranspose(128, (3, 3), strides=2, padding='same', activation='relu')  # 50 > 100
        self.trapos2 = layers.Conv2DTranspose(64, (4, 4), strides=2, padding='same', activation='relu')  # 100 > 200
        self.trapos3 = layers.Conv2DTranspose(3, (25, 25), strides=1, padding='valid', activation='sigmoid') # 224

        # Fusion and final layer
        self.fuse_conv = layers.Conv2D(1, (1, 1), activation='sigmoid') 
        #trash  


        self.drouput = layers.Dropout(0.5) 


        self.trash1 = layers.Conv2D(2, (3,3), activation='relu',padding='valid')
        self.trash2 = layers.MaxPooling2D((3, 3),strides=1)
        self.trash3 = layers.BatchNormalization()
        self.trash4 = layers.Dropout(0.5)
        self.trash5 = layers.Conv2DTranspose(3, (5, 5), activation='sigmoid', strides=1, padding='valid')


    # Forward pass for RGB stream
    if If_trash ==  True: 
        def stream(self, x):
            x = self.trash1(x)
            x = self.trash2(x)
            x = self.trash3(x)
            x = self.trash4(x)
            x = self.trash5(x)
            return x  
    else:
        # Forward pass for Depth stream
        def stream(self, x):
            x = self.conv1(x)
            x = self.maxpool1(x)
            x = self.norm1(x) 

            x = self.dilated_conv2(x)
            x = self.maxpool2(x)
            x = self.norm2(x) 
        
            x = self.dilated_conv3(x)
            x = self.dilated_conv4(x)
            x = self.dilated_conv5(x)

            x = self.drouput(x)

            x = self.trapos1(x) 
            x = self.trapos2(x) 
            x = self.trapos3(x) 

            return x
 
    # Forward pass combining RGB and Depth (HHA) streams
    def call(self, inputs):
        rgb, depth = inputs 
        # Process RGB and Depth streams separately
        rgb_out = self.stream(rgb)
        depth_out = self.stream(depth)

        # Fuse high-level features from both streams
        fused = tf.concat([rgb_out, depth_out], axis=-1)

        # Final prediction
        out = self.fuse_conv(fused)
        #fused = fused[:, :, :, :1]  
     
        #print(f"Out shape: {out.shape}")
        
        return out

# Instantiate the model
model = Saliency()

# Define loss function and optimizer (Adam)
#loss_fn = tf.keras.losses.Dice(reduction = 'sum_over_batch_size', name ='dice')
optimizer = tf.keras.optimizers.Adam()

if loss_function == 'dice_loss': 
    model.compile(optimizer=optimizer, loss=weighted_dice_loss, metrics=['accuracy'])
elif loss_function == 'binary_crossentropy':
    loss_fn = weighted_binary_crossentropy(pos_weight=pos_weight, neg_weight=neg_weight)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy']) 
elif loss_function == 'IoC_Loss': 
    model.compile(optimizer=optimizer, loss=IoC_loss, metrics=['accuracy']) 




#model.load_weights('C:/Users/eymen/Documents/project1/trainmodel.keras')        #Load previosly saved model weights 

# Train the model
history = model.fit(
    [rgb_images, HHA_images],  # Inputs as a list
    saliency_maps,               # Targets
    epochs=epochs, 
    batch_size=16,
    validation_data=([rgb_images_val, HHA_images_val], saliency_maps_val)  # Validation data
)

plt.figure(figsize=(14, 5))
# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy') 
plt.ylim(0, 1)
plt.xlim(0,epochs-1)
plt.title('Training and Validation Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss') 
plt.ylim(0, 1)
plt.xlim(0,epochs-1)
plt.title('Training and Validation Loss')
plt.legend()  


model.summary()
model.save_weights('trainmodel.weights.h5')                                #Save model


# Function to visualize input and output saliency map
def visualize_saliency(rgb_img, depth_img,HHA_img, saliency_map, prediction):

    fig, axes = plt.subplots(1, 5, figsize=(16, 8))
    axes[0].imshow(rgb_img)
    axes[0].set_title("RGB Image")
    
    axes[1].imshow(depth_img)
    axes[1].set_title("Depth Image") 
    
    axes[2].imshow(HHA_img) 
    axes[2].set_title("HHA Image")
    
    axes[3].imshow(saliency_map[:, :], cmap='gray')
    axes[3].set_title("Ground Truth Saliency Map")
    
    axes[4].imshow(prediction[:, :, 0], cmap='gray')
    axes[4].set_title("Predicted Saliency Map") 
    plt.show()
    
   

#BELOW ABOUT PRECISION-RECALL METHOD

# After training the model and obtaining predictions
y_true = saliency_maps_val.flatten()  # Flatten the ground truth saliency maps to a 1D array
# Apply threshold to the ground truth to make it binary (0 or 1)
y_true_binary = (y_true > 0.5).astype(int)  # Assuming values above 0.5 are considered salient
y_scores = model.predict([rgb_images_val, HHA_images_val]).flatten()  # Flatten the predicted saliency maps to a 1D array


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



for sample_index in range(len(rgb_images_val)):
    # Visualize the result 

    # Predict saliency map for a sample image from the validation set
    # Change this index to visualize different samples
    sample_rgb = rgb_images_val[sample_index:sample_index+1]  # Take a single RGB image
    sample_depth = depth_images_val[sample_index:sample_index+1]  # Take the corresponding depth image 
    sample_HHA = HHA_images_val[sample_index:sample_index+1]  # Take the corresponding HHA image
    sample_saliency = saliency_maps_val[sample_index]  # Ground truth saliency map for comparison

    # Predict saliency map
    predicted_saliency = model.predict([sample_rgb, sample_HHA])
    print(f"predicted shape: {predicted_saliency.shape}")
    visualize_saliency(sample_rgb[0], sample_depth[0],sample_HHA[0], sample_saliency, predicted_saliency[0])
