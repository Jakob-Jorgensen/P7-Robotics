import tensorflow as tf
from keras import layers, Model
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import pandas as pd 


# Preprocessing function to load and preprocess both RGB and Depth images
def preprocess_image(image_path, target_size=(224, 224)):
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    # Normalize the image to range [0, 1]
    image = image.astype('float32') / 255.0
    return image

# Load RGB and Depth images from folder
def load_dataset(rgb_folder, depth_folder, saliency_folder, target_size=(224, 224)):
    rgb_images, depth_images, saliency_maps = [], [], []
   
    rgb_files = sorted(os.listdir(rgb_folder))
    depth_files = sorted(os.listdir(depth_folder))
    saliency_files = sorted(os.listdir(saliency_folder))

    print(f"Found {len(rgb_files)} RGB images.")
    print(f"Found {len(depth_files)} depth images.")
    print(f"Found {len(saliency_files)} saliency maps.")

    
    for i, img_file in enumerate(rgb_files):
        rgb_path = os.path.join(rgb_folder, img_file)
        depth_img_file = img_file.replace('.jpg', '.png')
        depth_path = os.path.join(depth_folder, depth_img_file)
        saliency_img_file = img_file.replace('.jpg', '.png')
        saliency_path = os.path.join(saliency_folder, saliency_img_file)

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
        
        rgb_image = preprocess_image(rgb_path, target_size)
        depth_image = preprocess_image(depth_path, target_size)
        saliency_map = preprocess_image(saliency_path, target_size)
        saliency_map = saliency_map[:, :, :1]
        
        rgb_images.append(rgb_image)
        depth_images.append(depth_image)
        saliency_maps.append(saliency_map)
        #print(f"Loaded {len(rgb_images)} RGB images, {len(depth_images)} depth images, {len(saliency_maps)} saliency maps.")
        
    return np.array(rgb_images), np.array(depth_images), np.array(saliency_maps)

# Example dataset folders 
rgb_folder = r"C:\User.."
depth_folder = r"C:\User.."
saliency_folder = r"C:\User.."

# Load the dataset
rgb_images, depth_images, saliency_maps = load_dataset(rgb_folder, depth_folder, saliency_folder)

# Split the dataset into training and validation sets (80% train, 20% validation)
rgb_train, rgb_val, depth_train, depth_val, saliency_train, saliency_val = train_test_split(
    rgb_images, depth_images, saliency_maps, test_size=0.2, random_state=42
)

# Check dataset shapes
print(f"RGB images shape: {rgb_images.shape}")
print(f"Depth images shape: {depth_images.shape}")
print(f"Saliency maps shape: {saliency_maps.shape}")



# Define the CNN architecture for RGB-D saliency detection
class SaliencyNet(Model):
    def __init__(self):
        super(SaliencyNet, self).__init__()

        # RGB Stream
        self.rgb_conv1 = layers.Conv2D(96, (11, 11), strides=4, activation='relu', padding='valid')
        self.rgb_maxpool1 = layers.MaxPooling2D((3, 3), strides=2)
        self.rgb_norm1 =  layers.BatchNormalization()
        self.rgb_dilated_conv2 = layers.Conv2D(256, (5, 5), padding='same', dilation_rate=2, activation='relu')
        self.rgb_maxpool2 = layers.MaxPooling2D(pool_size=(3, 3), strides=2)
        self.rgb_norm2 =  layers.BatchNormalization()
        self.rgb_dilated_conv3 = layers.Conv2D(384, (3, 3), padding='same', dilation_rate=4, activation='relu')
        self.rgb_dilated_conv4 = layers.Conv2D(384, (3, 3), padding='same', dilation_rate=4, activation='relu')
        self.rgb_dilated_conv5 = layers.Conv2D(256, (3, 3), padding='same', dilation_rate=4, activation='relu')
        self.rgb_maxpool3 = layers.MaxPooling2D(pool_size=(3, 3), strides=2)

        # Depth (HHA) Stream
        self.depth_conv1 = layers.Conv2D(96, (11, 11), strides=4, activation='relu', padding='valid')
        self.depth_maxpool1 = layers.MaxPooling2D((3, 3), strides=2)
        self.depth_norm1 =  layers.BatchNormalization()
        self.depth_dilated_conv2 = layers.Conv2D(256, (5, 5), padding='same', dilation_rate=2, activation='relu')
        self.depth_maxpool2 = layers.MaxPooling2D((3, 3), strides=2)
        self.depth_norm2 =  layers.BatchNormalization()
        self.depth_dilated_conv3 = layers.Conv2D(384, (3, 3), padding='same', dilation_rate=4, activation='relu')
        self.depth_dilated_conv4 = layers.Conv2D(384, (3, 3), padding='same', dilation_rate=4, activation='relu')
        self.depth_dilated_conv5 = layers.Conv2D(256, (3, 3), padding='same', dilation_rate=4, activation='relu')
        self.depth_maxpool3 = layers.MaxPooling2D((3, 3), strides=2)

        # Fusion and final layers
        self.fusion_conv = layers.Conv2D(256, (1, 1), activation='relu')
        self.fc_conv = layers.Conv2D(1, (1, 1), activation='sigmoid')  # Final prediction layer

    # Forward pass for RGB stream
    def forward_rgb(self, x):
        x = self.rgb_conv1(x)
        x = self.rgb_maxpool1(x)
        x = self.rgb_norm1(x)
        x = self.rgb_dilated_conv2(x)
        x = self.rgb_maxpool2(x)
        x = self.rgb_norm2(x)
        x = self.rgb_dilated_conv3(x)
        x = self.rgb_dilated_conv4(x)
        x = self.rgb_dilated_conv5(x)
        x = self.rgb_maxpool3(x)

        return x

    # Forward pass for Depth stream
    def forward_depth(self, x):
        x = self.depth_conv1(x)
        x = self.depth_maxpool1(x)
        x = self.depth_norm1(x)
        x = self.depth_dilated_conv2(x)
        x = self.depth_maxpool2(x)
        x = self.depth_norm2(x)
        x = self.depth_dilated_conv3(x)
        x = self.depth_dilated_conv4(x)
        x = self.depth_dilated_conv5(x)
        x = self.depth_maxpool3(x)

        return x

    # Forward pass combining RGB and Depth (HHA) streams
    def call(self, inputs):
        rgb, depth = inputs  # Unpack inputs
        # Process RGB and Depth streams separately
        rgb_out = self.forward_rgb(rgb)
        depth_out = self.forward_depth(depth)

        # Fuse high-level features from both streams
        fused = tf.concat([rgb_out, depth_out], axis=-1)
        
        # Pass through fusion convolutional layer
        fused = self.fusion_conv(fused)

        # Final prediction
        out = self.fc_conv(fused)
        out = tf.image.resize(out, (224, 224))  # Resize to (224, 224)
        
        return out

# Instantiate the model
model = SaliencyNet()

# Example input for RGB and Depth (HHA)
rgb_input = tf.random.normal([1, 224, 224, 3])  # Simulated RGB input (batch size: 1, height: 224, width: 224, channels: 3)
depth_input = tf.random.normal([1, 224, 224, 3])  # Simulated Depth (HHA) input (batch size: 1, height: 224, width: 224, channels: 3)

# Forward pass through the model
output = model([rgb_input, depth_input])
print(output.shape)  # Expected output: (1, 50, 50, 1) (batch size: 1, height: 50, width: 50, channels: 1)

# Define loss function (Binary Cross Entropy) and optimizer (Adam)
loss_fn = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam()

# Compile the model
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])


# Train the model
history = model.fit(
    [rgb_train, depth_train],  # Inputs as a list
    saliency_train,            # Targets
    epochs=10, 
    batch_size=16, 
    validation_data=([rgb_val, depth_val], saliency_val)  # Validation data
)
# Print accuracy after each epoch
for epoch, acc in enumerate(history.history['accuracy']):
    print(f"Epoch {epoch+1}: Accuracy: {acc}")

model.save('trainmodel.h5')

# Function to visualize input and output saliency map
def visualize_saliency(rgb_img, depth_img, saliency_map, prediction):

    rgb_img1 = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 4, figsize=(16, 8))
    axes[0].imshow(rgb_img1)
    axes[0].set_title("RGB Image")
    
    axes[1].imshow(depth_img)
    axes[1].set_title("Depth Image")
    
    axes[2].imshow(saliency_map[:, :, 0], cmap='gray')
    axes[2].set_title("Ground Truth Saliency Map")
    
    axes[3].imshow(prediction[:, :, 0], cmap='gray')
    axes[3].set_title("Predicted Saliency Map")
    
    plt.show()

# Example: Predict saliency map for a sample image from the validation set
sample_index = 0  # Change this index to visualize different samples
sample_rgb = rgb_val[sample_index:sample_index + 1]  # Take a single RGB image
sample_depth = depth_val[sample_index:sample_index + 1]  # Take the corresponding depth image
sample_saliency = saliency_val[sample_index]  # Ground truth saliency map for comparison

# Predict saliency map
predicted_saliency = model.predict([sample_rgb, sample_depth])

# Visualize the result
visualize_saliency(sample_rgb[0], sample_depth[0], sample_saliency, predicted_saliency[0])
model.summary()

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
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

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
