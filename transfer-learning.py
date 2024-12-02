from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.models import clone_model
from sklearn.metrics import precision_recall_curve, average_precision_score
from keras_resnet.models import ResNet18
import tensorflow as tf
from keras import layers, Model
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2


##############################################################

main_path = f"C:/Users/mikke/Downloads/Dataset_3.1/Dataset_3.1"  
loss_function = 'binary_crossentropy' # Chose between 'dice_loss' or 'binary_crossentropy'
epochs = 5 

##############################################################

def weighted_binary_crossentropy(pos_weight, neg_weight):
    def loss_fn(y_true, y_pred):
        # Compute binary cross-entropy
        bce = -(pos_weight * y_true * tf.math.log(y_pred + 1e-8) + 
                neg_weight * (1 - y_true) * tf.math.log(1 - y_pred + 1e-8))
        return tf.reduce_mean(bce)
    return loss_fn

def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
    denominator = tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat)

    dice_coeff = (2.0 * intersection + smooth) / (denominator + smooth)

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
    rgb_images, depth_images, saliency_maps,HHA_images = [], [], [],[]    #create empty lists

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

rgb_images_test, depth_images_test, saliency_maps_test,HHA_images_test = load_dataset(rgb_folder_test, depth_folder_test, saliency_folder_test,HHA_folder_test)


# Check dataset shapes
print(f"RGB images shape: {rgb_images.shape}")
print(f"Depth images shape: {depth_images.shape}")
print(f"Saliency maps shape: {saliency_maps.shape}") 
print(f"HHA images shape: {HHA_images.shape}") 

pos_weight = np.mean(1 - saliency_maps)  # Mean of non-salient (background) pixels
neg_weight = np.mean(saliency_maps) 

print('White weight: ', pos_weight, ' Black Weight: ', neg_weight)
# White weight:  0.9926431  Black Weight:  

# Define unique input layers for RGB and Depth models
rgb_input = Input(shape=(224, 224, 3), name="rgb_input")
depth_input = Input(shape=(224, 224, 3), name="depth_input")

# Load original ResNet50 model for cloning
original_resNet50 = ResNet50(weights='imagenet', include_top=False)

# Clone with unique layer names for RGB stream
resNet50_rgb = clone_model(
    original_resNet50, 
    clone_function=lambda layer: layer.__class__.from_config({
        **layer.get_config(), 
        "name": f"rgb_{layer.name}"
    })
)
resNet50_rgb._name = "resnet50_rgb"  # Explicitly set a unique name for the model
resNet50_rgb.set_weights(original_resNet50.get_weights())  # Transfer weights
for layer in resNet50_rgb.layers:
    layer.trainable = False
rgb_stream = resNet50_rgb(rgb_input)

# Clone with unique layer names for Depth stream
resNet50_depth = clone_model(
    original_resNet50, 
    clone_function=lambda layer: layer.__class__.from_config({
        **layer.get_config(), 
        "name": f"depth_{layer.name}"
    })
)
resNet50_depth._name = "resnet50_depth"  # Explicitly set a unique name for the model
resNet50_depth.set_weights(original_resNet50.get_weights())  # Transfer weights
for layer in resNet50_depth.layers:
    layer.trainable = False
depth_stream = resNet50_depth(depth_input)

# RGB Stream processing
rgb_stream = resNet50_rgb(rgb_input)
#rgb_stream = layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same', activation='relu')(rgb_stream)  # (14, 14, 512)
#rgb_stream = layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', activation='relu')(rgb_stream)  # (28, 28, 256)
#rgb_stream = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', activation='relu')(rgb_stream)  # (56, 56, 128)
#rgb_stream = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation='relu')(rgb_stream)   # (112, 112, 64)
#rgb_stream = layers.Conv2DTranspose(3, (3, 3), strides=(2, 2), padding='same', activation='relu')(rgb_stream)   # (224, 224, 3)

rgb_stream = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(rgb_stream)
rgb_stream = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(rgb_stream)
rgb_stream = layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', activation='relu')(rgb_stream)  # (28, 28, 256)
rgb_stream = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', activation='relu')(rgb_stream)  # (56, 56, 128)
rgb_stream = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation='relu')(rgb_stream)   # (112, 112, 64)
rgb_stream = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu')(rgb_stream)   # (224, 224, 32)
rgb_stream = layers.Conv2D(3, (3, 3), padding='same', activation='sigmoid')(rgb_stream)   # (224, 224, 3) 
# There is a convolutional layer to help preserve details before upsampling



# Depth Stream processing (Fix here: Using depth_stream, not rgb_stream)
depth_stream = resNet50_depth(depth_input)
#depth_stream = layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same', activation='relu')(depth_stream)  # (7, 7, 256)
#depth_stream = layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', activation='relu')(depth_stream)  # (14, 14, 128)
#depth_stream = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', activation='relu')(depth_stream)   # (28, 28, 64)
#depth_stream = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation='relu')(depth_stream)   # (56, 56, 32)
#depth_stream = layers.Conv2DTranspose(3, (3, 3), strides=(2, 2), padding='same', activation='relu')(depth_stream)   # (112, 112, 16)

depth_stream = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(depth_stream)
depth_stream = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(depth_stream)
depth_stream = layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', activation='relu')(depth_stream)  # (28, 28, 256)
depth_stream = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', activation='relu')(depth_stream)  # (56, 56, 128)
depth_stream = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation='relu')(depth_stream)   # (112, 112, 64)
depth_stream = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu')(depth_stream)   # (224, 224, 32)
depth_stream = layers.Conv2D(3, (3, 3), padding='same', activation='sigmoid')(depth_stream)   # (224, 224, 3)
# There is a convolutional layer to help preserve details before upsampling

# Concatenate the RGB and Depth streams
fused = layers.Concatenate()([rgb_stream, depth_stream])

# Apply final convolution to produce the saliency map (binary output)
fused = layers.Conv2D(1, kernel_size=(1, 1), activation='sigmoid')(fused)

# Reshape to match output dimensions
saliency_output = layers.Reshape((224, 224, 1))(fused)




# Define the model with both streams as input
model = Model(inputs=[rgb_input, depth_input], outputs=saliency_output)


if loss_function == 'dice_loss': 
    model.compile(optimizer=tf.keras.optimizer.Adam, loss=dice_loss, metrics=['accuracy'])

elif loss_function == 'binary_crossentropy':
    loss_fn = weighted_binary_crossentropy(pos_weight=pos_weight, neg_weight=neg_weight)
    # Compile the model with binary crossentropy loss
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
              loss=loss_fn, 
              metrics=['accuracy'])




# Summary of the model to ensure everything is correct
model.summary()

# Now you can train the model
history = model.fit(
    [rgb_images, HHA_images],  # Inputs as a list of RGB and Depth images
    saliency_maps,               # Targets (saliency maps)
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
    
   

# Predict saliency map for a sample image from the validation set
sample_index = 0  # Change this index to visualize different samples
sample_rgb = rgb_images_test[sample_index:sample_index+1]  # Take a single RGB image
sample_depth = depth_images_test[sample_index:sample_index+1]  # Take the corresponding depth image 
sample_HHA = HHA_images_test[sample_index:sample_index+1]  # Take the corresponding HHA image
sample_saliency = saliency_maps_test[sample_index]  # Ground truth saliency map for comparison

# Predict saliency map
predicted_saliency = model.predict([sample_rgb, sample_HHA])
print(f"predicted shape: {predicted_saliency.shape}")
print("saliency map output values:",predicted_saliency[0,:,:,0])
print("saliency map output values:",predicted_saliency[0,25,25,0])

# Visualize the result
visualize_saliency(sample_rgb[0], sample_depth[0],sample_HHA[0], sample_saliency, predicted_saliency[0])


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
plt.show()
