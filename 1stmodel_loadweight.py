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
def load_dataset(rgb_folder, depth_folder, gt_folder, target_size=(360, 180)):
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
val_rgb_folder = r"C:\Users\eymen\Documents\project1\Augmented_Dataset_Version2\Validation\RGB"
val_depth_folder = r"C:\Users\eymen\Documents\project1\Augmented_Dataset_Version2\Validation\HHA"
val_saliency_folder = r"C:\Users\eymen\Documents\project1\Augmented_Dataset_Version2\Validation\GT"

# Original and augmented folder paths
train_rgb_folder = r"C:\Users\eymen\Documents\project1\Augmented_Dataset_Version2\Training\RGB"
train_depth_folder = r"C:\Users\eymen\Documents\project1\Augmented_Dataset_Version2\Training\HHA"
train_saliency_folder = r"C:\Users\eymen\Documents\project1\Augmented_Dataset_Version2\Training\GT"


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



# Instantiate the model
model = Saliency()

# Compile the model
model.compile(optimizer='adam', loss=weighted_binary_crossentropy, metrics=['accuracy',iou_metric])

# Modelin değişkenlerini oluşturmak için bir örnek veriyle çağır
dummy_rgb = np.zeros((1, 180, 360, 3))  # RGB dummy input (batch_size=1)
dummy_depth = np.zeros((1, 180, 360, 3))  # Depth dummy input (batch_size=1)
model([dummy_rgb, dummy_depth])  # Modeli bir kere çağır


model.load_weights('modelv7.keras')


# Function to predict and save predictions
def predict_and_save_predictions(model, test_rgb_folder, test_depth_folder, save_dir, target_size=(360, 180)):
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)

    # List all test RGB files (assumes filenames are consistent across RGB and Depth folders)
    test_rgb_files = sorted(os.listdir(test_rgb_folder))

    for i, file_name in enumerate(test_rgb_files):
        # Construct file paths
        rgb_path = os.path.join(test_rgb_folder, file_name)
        depth_path = os.path.join(test_depth_folder, file_name)

        # Check if both files exist
        if os.path.exists(rgb_path) and os.path.exists(depth_path):
            # Preprocess RGB and Depth images
            rgb_image = preprocess_image(rgb_path, target_size)
            depth_image = preprocess_image(depth_path, target_size)

            # Expand dimensions to match model input shape (batch size of 1)
            rgb_input = np.expand_dims(rgb_image, axis=0)
            depth_input = np.expand_dims(depth_image, axis=0)

            # Generate prediction
            prediction = model([rgb_input, depth_input]).numpy()

            # Post-process prediction (e.g., threshold for binary output)
            prediction = (prediction[0, :, :, 0] > 0.5).astype('uint8') * 255  # Convert to binary mask

            # Save prediction as an image
            save_path = os.path.join(save_dir, f"prediction_{i + 1}.png")
            cv2.imwrite(save_path, prediction)

            print(f"Saved prediction for {file_name} at {save_path}")
        else:
            print(f"Warning: Missing RGB or Depth file for {file_name}. Skipping...")

# Paths to test folders
test_rgb_folder = r"C:\Users\eymen\Documents\project1\Final_Dataset\Testing\RGB2"
test_depth_folder = r"C:\Users\eymen\Documents\project1\Final_Dataset\Testing\HHA2"

# Directory to save predictions
save_predictions_dir = r"C:\Users\eymen\Documents\project1\Predictions"

# Run prediction and save results
predict_and_save_predictions(model, test_rgb_folder, test_depth_folder, save_predictions_dir)

