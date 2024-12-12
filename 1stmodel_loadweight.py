import tensorflow as tf
from keras import layers, Model
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score


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




