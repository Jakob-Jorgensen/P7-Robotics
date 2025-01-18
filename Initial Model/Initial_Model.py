import tensorflow as tf
from keras import layers, Model
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score


# Function to preprocess images
def preprocess_image(image_path, target_size):
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    image = image.astype('float32') / 255.0
    return image

# Function to load datasets
def load_dataset(rgb_folder, depth_folder, gt_folder, target_size=(224, 224)):
    rgb_images, depth_images, saliency_maps = [], [], []

    # List all RGB files (assumes files are named consistently across folders)
    rgb_files = sorted(os.listdir(rgb_folder))

    for file_name in rgb_files:
        # Construct corresponding file paths
        rgb_path = os.path.join(rgb_folder, file_name)
        depth_path = os.path.join(depth_folder, file_name)
        gt_path = os.path.join(gt_folder, file_name)
    

        # Check if all files exist
        if os.path.exists(rgb_path) and os.path.exists(depth_path) and os.path.exists(gt_path):
            # Load and preprocess images
            rgb_image = preprocess_image(rgb_path, target_size)
            depth_image = preprocess_image(depth_path, target_size)
            saliency_map = preprocess_image(gt_path, target_size)[:, :, :1]  # Convert Saliency maps to single-channel format

            # Append to respective lists
            rgb_images.append(rgb_image)
            depth_images.append(depth_image)
            saliency_maps.append(saliency_map)
        else:
            print(f"Warning: Missing file(s) for {file_name}. Skipping...")

    # Convert lists to NumPy arrays
    return np.array(rgb_images), np.array(depth_images), np.array(saliency_maps)

# Dataset folder paths
val_rgb_folder = r"C:\Users\eymen\Documents\project1\Final_dataset_Combined_sent\Validation\RGB"
val_depth_folder = r"C:\Users\eymen\Documents\project1\Final_dataset_Combined_sent\Validation\HHA"
val_saliency_folder = r"C:\Users\eymen\Documents\project1\Final_dataset_Combined_sent\Validation\GT"

# Original and augmented folder paths
train_rgb_folder = r"C:\Users\eymen\Documents\project1\Final_dataset_Combined_sent\Training\RGB"
train_depth_folder = r"C:\Users\eymen\Documents\project1\Final_dataset_Combined_sent\Training\HHA"
train_saliency_folder = r"C:\Users\eymen\Documents\project1\Final_dataset_Combined_sent\Training\GT"


#Loading datasets
rgb_train, depth_train, saliency_train = load_dataset(train_rgb_folder, train_depth_folder, train_saliency_folder)
rgb_val, depth_val, saliency_val = load_dataset(val_rgb_folder, val_depth_folder, val_saliency_folder)


# Check dataset shapes
print(f"RGB Train Shape: {rgb_train.shape}")
print(f"Depth Train Shape: {depth_train.shape}")
print(f"GT Train Shape: {saliency_train.shape}")
print(f"RGB Validation Shape: {rgb_val.shape}")
print(f"Depth Validation Shape: {depth_val.shape}")
print(f"GT Validation Shape: {saliency_val.shape}")


# Define the CNN architecture for RGB-D saliency detection
class Saliency(Model):
    def __init__(self):
        super(Saliency, self).__init__()
        # Shared RGB and Depth Stream
        self.conv1 = layers.Conv2D(96, (11, 11), strides=4, activation='relu', padding='valid')
        self.maxpool1 = layers.MaxPooling2D((3, 3), strides=1)
        self.norm1 = layers.BatchNormalization()

        self.dilated_conv2 = layers.Conv2D(256, (5, 5), strides=1, padding='same', dilation_rate=2, activation='relu')
        self.maxpool2 = layers.MaxPooling2D(pool_size=(3, 3), strides=1)
        self.norm2 = layers.BatchNormalization()

        self.dilated_conv3 = layers.Conv2D(384, (3, 3), strides=1, padding='same', dilation_rate=4, activation='relu')
        self.dilated_conv4 = layers.Conv2D(384, (3, 3), strides=1, padding='same', dilation_rate=4, activation='relu')
        self.dilated_conv5 = layers.Conv2D(256, (3, 3), strides=1, padding='same', dilation_rate=4, activation='relu')

        self.dropout = layers.Dropout(0.3)

        self.trapos1 = layers.Conv2DTranspose(128, (3, 3), strides=2, padding='same', activation='relu')  # 50 > 100
        self.trapos2 = layers.Conv2DTranspose(64, (4, 4), strides=2, padding='same', activation='relu')  # 100 > 200
        self.trapos3 = layers.Conv2DTranspose(3, (25, 25), strides=1, padding='valid', activation='sigmoid') # 224

        self.afterfusion = layers.Conv2D(1,(1,1), activation='sigmoid')
    
    def shared_stream(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.norm1(x)
        x = self.dilated_conv2(x)
        x = self.maxpool2(x)
        x = self.norm2(x)
        x = self.dilated_conv3(x)
        x = self.dilated_conv4(x)
        x = self.dilated_conv5(x)
        x = self.dropout(x)
        x = self.trapos1(x) 
        x = self.trapos2(x) 
        x = self.trapos3(x) 
        return x

    def call(self, inputs):
        rgb, depth = inputs
        rgb_output = self.shared_stream(rgb)
        depth_output = self.shared_stream(depth)
        fused = tf.concat([rgb_output, depth_output], axis=-1)
        out = self.afterfusion(fused)
        return out

# Define a custom callback for visualizing predictions
class VisualizePredictionsCallback(tf.keras.callbacks.Callback):
    def __init__(self, save_dir, sample_rgb, sample_depth, sample_saliency):
        super(VisualizePredictionsCallback, self).__init__()
        self.save_dir = save_dir
        self.sample_rgb = sample_rgb
        self.sample_depth = sample_depth
        self.sample_saliency = sample_saliency
        os.makedirs(self.save_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        # Generate predictions for the batch of samples
        predictions = self.model.predict([self.sample_rgb, self.sample_depth])

        # Visualize and save predictions for each sample
        for i in range(len(self.sample_rgb)):
            self.visualize_and_save(epoch, self.sample_rgb[i], self.sample_depth[i], self.sample_saliency[i], predictions[i], i)

    def visualize_and_save(self, epoch, rgb, depth, ground_truth, prediction, index):
        # Convert RGB for matplotlib
        rgb_img = (rgb * 255).astype('uint8')  # Convert to 0-255 range
        rgb_img = cv2.cvtColor(rgb_img,cv2.COLOR_BGR2RGB)
        depth = cv2.cvtColor(depth,cv2.COLOR_BGR2RGB)

        # Plot RGB, Depth, Ground Truth, and Prediction
        fig, axes = plt.subplots(1, 4, figsize=(16, 8))
        axes[0].imshow(rgb_img)
        axes[0].set_title("RGB Image")
        axes[0].axis('off')

        axes[1].imshow(depth, cmap='gray')
        axes[1].set_title("Depth Image")
        axes[1].axis('off')

        axes[2].imshow(ground_truth[:, :, 0], cmap='gray')
        axes[2].set_title("Ground Truth")
        axes[2].axis('off')

        axes[3].imshow(prediction[:, :, 0] > 0.5, cmap='gray')  # Thresholded predictions
        axes[3].set_title("Prediction")
        axes[3].axis('off')

        # Save the figure for the specific sample
        save_path = os.path.join(self.save_dir, f"epoch_{epoch + 1}_sample_{index + 1}.png")
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Visualization saved for epoch {epoch + 1}, sample {index + 1} at {save_path}")

# Specify 5 images from the validation set
sample_rgb = rgb_val[3:15]  # Take 5 RGB images (indices 5 to 9)
sample_depth = depth_val[3:15]  # Corresponding Depth images
sample_saliency = saliency_val[3:15]  # Corresponding Ground Truth saliency maps

# Instantiate the callback
visualization_callback = VisualizePredictionsCallback(
    save_dir="predictions_visualizations",
    sample_rgb=sample_rgb,
    sample_depth=sample_depth, 
    sample_saliency=sample_saliency
)

# Weighted binary cross-entropy kayıp fonksiyonu
def weighted_binary_crossentropy(y_true, y_pred):
    weight_for_ones = 50.0  # Beyaz piksellerin ağırlığı
    weight_for_zeros = 1.0  # Siyah piksellerin ağırlığı

    # y_true ve y_pred tensorlarının boyutları eşit olmalıdır.
    weights = y_true * weight_for_ones + (1 - y_true) * weight_for_zeros
    
    # Kayıp fonksiyonu her piksel için uygulanmalı
    loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    
    # Boyut uyuşmazlığı problemini çözmek için yayılım (broadcasting) kullanılacak
    weighted_loss = loss * tf.reduce_mean(weights, axis=-1)
    
    return tf.reduce_mean(weighted_loss)

# IoU Metric
def iou_metric(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true + y_pred) - intersection
    return intersection / (union + tf.keras.backend.epsilon())
iou_metric.__name__ = 'iou'     


# Instantiate the model
model = Saliency()

# Compile the model
model.compile(optimizer='adam', loss=weighted_binary_crossentropy, metrics=['accuracy',iou_metric])

# Callbacks
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 0.0001 if epoch < 21 else 0.0001 * tf.math.exp(-0.1 * (epoch - 21)))


# Train the model
training = model.fit(
    [rgb_train, depth_train],
    saliency_train,
    epochs=18,
    batch_size=16,
    validation_data=([rgb_val, depth_val], saliency_val),
    callbacks=[visualization_callback,lr_scheduler]
)


plt.figure(figsize=(14, 5))
# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(training.history['iou'], label='Training Accuracy')
plt.plot(training.history['val_iou'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy') 
plt.ylim(0, 1)
plt.xlim(0,19)
plt.title('Training and Validation Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(training.history['loss'], label='Training Loss')
plt.plot(training.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss') 
plt.ylim(0, 1)
plt.xlim(0,19)
plt.title('Training and Validation Loss')
plt.legend()  


# Function to visualize input and output saliency map
def visualize_saliency(rgb_img, HHA_img, saliency_map, prediction):

    rgb_img = cv2.cvtColor(rgb_img,cv2.COLOR_BGR2RGB)
    HHA_img = cv2.cvtColor(HHA_img,cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 4, figsize=(16, 8))
    axes[0].imshow(rgb_img)
    axes[0].set_title("RGB Image")
    
    axes[1].imshow(HHA_img) 
    axes[1].set_title("HHA Image")
    
    axes[2].imshow(saliency_map[:, :], cmap='gray')
    axes[2].set_title("Ground Truth Saliency Map")
    
    axes[3].imshow(prediction[:, :, 0]>0.5, cmap='gray')
    axes[3].set_title("Predicted Saliency Map")
    
   

# Predict saliency map for a sample image from the validation set
sample_index = 0  # Change this index to visualize different samples
sample_rgb = rgb_val[sample_index:sample_index+1]  # Take a single RGB image
sample_HHA = depth_val[sample_index:sample_index+1]  # Take the corresponding HHA image
sample_saliency = saliency_val[sample_index]  # Ground truth saliency map for comparison

# Predict saliency map
predicted_saliency = model.predict([sample_rgb, sample_HHA])
print(f"predicted shape: {predicted_saliency.shape}")


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



# Summarize model architecture
model.summary()
#save model  
model.save_weights("1stmodelv2.h5") 
model.save_weights("1stmodelv2.weights.h5")
# Save model architecture
with open('1stmodel_architecture.json', 'w') as json_file:
    json_file.write(model.to_json())


# Paths to test folders
test_rgb_folder = r"C:\Users\Final_Dataset\Testing\RGB2"
test_depth_folder = r"C:\Users\Final_Dataset\Testing\HHA2"
test_gt_folder = r"C:\Users\Final_Dataset\Testing\GT3"


rgb_test, depth_test, saliency_test = load_dataset(test_rgb_folder, test_depth_folder, test_gt_folder)

output_folder = "predictions"  # Specify your desired folder to save the output plots

# Function to visualize input and output saliency map
def visualize_saliency(rgb_img, HHA_img, saliency_map, prediction):

    rgb_img = cv2.cvtColor(rgb_img,cv2.COLOR_BGR2RGB)
    HHA_img = cv2.cvtColor(HHA_img,cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 4, figsize=(16, 8))
    axes[0].imshow(rgb_img)
    axes[0].set_title("RGB Image")
    
    axes[1].imshow(HHA_img) 
    axes[1].set_title("HHA Image")
    
    axes[2].imshow(saliency_map[:, :], cmap='gray')
    axes[2].set_title("Ground Truth Saliency Map")
    
   
    axes[3].imshow(prediction[:, :, 0] > 0.5, cmap='gray') 
    axes[3].set_title("Predicted Saliency Map")

    plt.savefig(output_path, format="png")
    #plt.close()

# Predict saliency map for a sample image from the validation set
sample_index = 0  # Change this index to visualize different samples
sample_rgb = rgb_test[sample_index:sample_index+1]  # Take a single RGB image
sample_HHA = depth_test[sample_index:sample_index+1]  # Take the corresponding HHA image
sample_saliency = saliency_test[sample_index]  # Ground truth saliency map for comparison

# Predict saliency map
predicted_saliency = model.predict([sample_rgb, sample_HHA])
print(f"predicted shape: {predicted_saliency.shape}")



# After training the model and obtaining predictions
y_true = saliency_test.flatten()  # Flatten the ground truth saliency maps to a 1D array
# Apply threshold to the ground truth to make it binary (0 or 1)
y_true_binary = (y_true > 0.5).astype(int)  # Assuming values above 0.5 are considered salient
y_scores = model.predict([rgb_test, depth_test]).flatten()  # Flatten the predicted saliency maps to a 1D array


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


output_folder = "testresults"  # Specify your desired folder to save the output plots
for sample_index in range(len(rgb_test)):
    # Visualize the result 

    # Predict saliency map for a sample image from the validation set
    # Change this index to visualize different samples
    sample_rgb = rgb_test[sample_index:sample_index+1]  # Take a single RGB image
    #sample_depth = depth_images_val[sample_index:sample_index+1]  # Take the corresponding depth image 
    sample_HHA = depth_test[sample_index:sample_index+1]  # Take the corresponding HHA image
    sample_saliency = saliency_test[sample_index]  # Ground truth saliency map for comparison

    # Predict saliency map
    predicted_saliency = model.predict([sample_rgb, sample_HHA])
    print(f"predicted shape: {predicted_saliency.shape}")

    # Ensure the folder exists
    os.makedirs(output_folder, exist_ok=True)
    output_file = f"example_plot_{sample_index}.png"
    # Full path to save the plot
    output_path = os.path.join(output_folder, output_file)
    
    visualize_saliency(sample_rgb[0],sample_HHA[0], sample_saliency, predicted_saliency[0])

    print(f"Plot saved at: {output_path}")  

output_folder = "results"  # Specify your desired folder to save the output plots

for sample_index in range(len(rgb_val)):
    # Visualize the result 

    # Predict saliency map for a sample image from the validation set
    # Change this index to visualize different samples
    sample_rgb = rgb_val[sample_index:sample_index+1]  # Take a single RGB image
    #sample_depth = depth_images_val[sample_index:sample_index+1]  # Take the corresponding depth image 
    sample_HHA = depth_val[sample_index:sample_index+1]  # Take the corresponding HHA image
    sample_saliency = saliency_val[sample_index]  # Ground truth saliency map for comparison

    # Predict saliency map
    predicted_saliency = model.predict([sample_rgb, sample_HHA])
    print(f"predicted shape: {predicted_saliency.shape}")

    # Ensure the folder exists
    os.makedirs(output_folder, exist_ok=True)
    output_file = f"example_plot_{sample_index}.png"
    # Full path to save the plot
    output_path = os.path.join(output_folder, output_file)
    
    visualize_saliency(sample_rgb[0],sample_HHA[0], sample_saliency, predicted_saliency[0])

    print(f"Plot saved at: {output_path}")  

