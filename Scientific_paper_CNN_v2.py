import tensorflow as tf
from keras import layers, Model
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
from sklearn.metrics import precision_score, recall_score, f1_score

# Function to preprocess images
def preprocess_image(image_path, target_size):
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    image = image.astype('float32') / 255.0
    return image

# Function to load datasets
def load_dataset(rgb_folder, depth_folder, gt_folder, target_size=(360,180)):
    rgb_images, depth_images, saliency_maps = [], [], []

    # List all RGB files (assumes files are named consistently across folders)
    rgb_files = sorted(os.listdir(rgb_folder))

    for file_name in rgb_files:
        # Construct corresponding file paths
        rgb_path = os.path.join(rgb_folder, file_name)
        depth_path = os.path.join(depth_folder, file_name.replace("rgb", "depth"))
        gt_path = os.path.join(gt_folder, file_name.replace("rgb", "gt"))

        # Check if all files exist
        if os.path.exists(rgb_path) and os.path.exists(depth_path) and os.path.exists(gt_path):
            # Load and preprocess images
            rgb_image = preprocess_image(rgb_path, target_size)
            depth_image = preprocess_image(depth_path, target_size)
            saliency_map = preprocess_image(gt_path, target_size))[:, :, :1]  # Convert Saliency maps to single-channel format

            # Append to respective lists
            rgb_images.append(rgb_image)
            depth_images.append(depth_image)
            saliency_maps.append(saliency_map)
        else:
            print(f"Warning: Missing file(s) for {file_name}. Skipping...")

    # Convert lists to NumPy arrays
    return np.array(rgb_images), np.array(depth_images), np.array(saliency_maps)

# Dataset folder paths
val_rgb_folder = r"Validation\RGB"
val_depth_folder = r"Validation\HHA"
val_saliency_folder = r"Validation\GT"

# Original and augmented folder paths
train_rgb_folder = r"Training\RGB"
train_depth_folder = r"Training\HHA"
train_saliency_folder = r"Training\GT"

#Loading datasets
rgb_train, depth_train, saliency_train = load_dataset(train_rgb_folder, train_depth_folder, train_saliency_folder)
rgb_val, depth_val, saliency_val = load_dataset(val_rgb_folder, val_depth_folder, val_saliency_folder)


# Check dataset shapes
print(f"RGB Train Shape: {rgb_train.shape}")
print(f"Depth Train Shape: {depth_train.shape}")
print(f"Saliency Train Shape: {saliency_train.shape}")
print(f"RGB Validation Shape: {rgb_val.shape}")
print(f"Depth Validation Shape: {depth_val.shape}")
print(f"Saliency Validation Shape: {saliency_val.shape}")


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
sample_rgb = rgb_val[5:15]  # Take 5 RGB images (indices 5 to 9)
sample_depth = depth_val[5:15]  # Corresponding Depth images
sample_saliency = saliency_val[5:15]  # Corresponding Ground Truth saliency maps

# Instantiate the callback
visualization_callback = VisualizePredictionsCallback(
    save_dir="predictions_visualizations",
    sample_rgb=sample_rgb,
    sample_depth=sample_depth, 
    sample_saliency=sample_saliency
)

# Weighted binary cross-entropy kayıp fonksiyonu
def weighted_binary_crossentropy(y_true, y_pred):
    weight_for_ones = 30.0  # Beyaz piksellerin ağırlığı
    weight_for_zeros = 1.0  # Siyah piksellerin ağırlığı

    # y_true ve y_pred tensorlarının boyutları eşit olmalıdır.
    weights = y_true * weight_for_ones + (1 - y_true) * weight_for_zeros
    
    # Kayıp fonksiyonu her piksel için uygulanmalı
    loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    
    # Boyut uyuşmazlığı problemini çözmek için yayılım (broadcasting) kullanılacak
    weighted_loss = loss * tf.reduce_mean(weights, axis=-1)
    
    return tf.reduce_mean(weighted_loss)

# Instantiate the model
model = Saliency()

# Compile the model
model.compile(optimizer='adam', loss=weighted_binary_crossentropy, metrics=['accuracy'])


# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = tf.keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss')
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 0.00001 if epoch < 21 else 0.00001 * tf.math.exp(-0.1 * (epoch - 10)))


# Train the model
training = model.fit(
    [rgb_train, depth_train],
    saliency_train,
    epochs=20,
    batch_size=16,
    validation_data=([rgb_val, depth_val], saliency_val),
    callbacks=[early_stopping, checkpoint, lr_scheduler, visualization_callback]
)

# Summarize model architecture
model.summary()


#This part for evaluation metrics like F1 score
def evaluate_metrics(y_true, y_pred):
    y_true_flat = y_true.flatten()
    y_pred_flat = (y_pred.flatten() > 0.5).astype(int)
    precision = precision_score(y_true_flat, y_pred_flat)
    recall = recall_score(y_true_flat, y_pred_flat)
    f1 = f1_score(y_true_flat, y_pred_flat)
    return precision, recall, f1

# Predict and visualize
sample_rgb, sample_depth, sample_saliency = rgb_val[0:1], depth_val[0:1], saliency_val[0]
predicted_saliency = model.predict([sample_rgb, sample_depth])
precision, recall, f1 = evaluate_metrics(sample_saliency, predicted_saliency)

print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")