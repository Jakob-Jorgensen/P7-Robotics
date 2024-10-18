
import tensorflow as tf
from keras import models,layers 
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator

# Set up the ImageDataGenerator for training and validation
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)  # 20% for validation

# Define paths to your dataset
train_data_dir = 'depth/'

# Generate training and validation batches from your image directories
train_generator = train_datagen.flow_from_directory(train_data_dir,target_size=(224, 224),batch_size=32,class_mode='categorical',subset='training')

validation_generator = train_datagen.flow_from_directory( train_data_dir,target_size=(224, 224),batch_size=32,class_mode='categorical',subset='validation')



# RGB Stream with 1x1 convolution for 50x50 output
def build_rgb_stream(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))

    # Add more layers including dilated convolutions 
    model.add(layers.BatchNormalization(axis=3))        

    model.add(layers.Conv2D(128, (3, 3), dilation_rate=2, activation='relu'))   
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))) 

    model.add(layers.BatchNormalization(axis=3))   
    model.add(layers.Conv2D(256, (3, 3), dilation_rate=4, activation='relu'))
    
    model.add(layers.Conv2D(256, (3, 3), dilation_rate=4, activation='relu')) 

    model.add(layers.Conv2D(256, (3, 3), dilation_rate=4, activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))) 

    # Final 1x1 convolution to produce a feature map close to 50x50
    model.add(layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same'))

    return model

# Depth Stream with 1x1 convolution for 50x50 output
def build_depth_stream(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))

    # Add more layers including dilated convolutions 
    model.add(layers.BatchNormalization(axis=3))    
    
    model.add(layers.Conv2D(128, (3, 3), dilation_rate=2, activation='relu'))   
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))) 

    model.add(layers.BatchNormalization(axis=3))   
    model.add(layers.Conv2D(256, (3, 3), dilation_rate=4, activation='relu'))
    
    model.add(layers.Conv2D(256, (3, 3), dilation_rate=4, activation='relu'))
    model.add(layers.Conv2D(256, (3, 3), dilation_rate=4, activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    # Final 1x1 convolution to produce a feature map close to 50x50
    model.add(layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same'))

    return model

# Fusion Model with final Conv2D layer for saliency prediction
def build_fusion_model(rgb_input_shape, depth_input_shape):
    # Build the two streams
    rgb_stream = build_rgb_stream(rgb_input_shape)
    depth_stream = build_depth_stream(depth_input_shape)

    # Define inputs
    rgb_input = layers.Input(shape=rgb_input_shape)
    depth_input = layers.Input(shape=depth_input_shape)

    # Extract features from both streams
    rgb_features = rgb_stream(rgb_input)
    depth_features = depth_stream(depth_input)

    # Late fusion - concatenate feature maps from both streams
    fused = layers.Concatenate()([rgb_features, depth_features])

    # Final convolution layer to reduce the concatenated map into one channel
    fused = layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')(fused)

    # Ensure the output is exactly 50x50
    fused = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(fused)

    # Build and compile the model
    model = models.Model(inputs=[rgb_input, depth_input], outputs=fused)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 
    model.summary()
    return model

""" 
rgb_input_shape = (224, 224, 3)  # RGB input shapeimport tensorflow as tf
depth_input_shape = (224, 224, 3)  # HHA depth input shape (3 channels)

model = build_fusion_model(rgb_input_shape, depth_input_shape)  

# Train the model using the data generators 
history = model.fit(train_generator,epochs=10, validation_data=validation_generator) 

test_generator = train_datagen.flow_from_directory(train_data_dir,target_size=(224, 224),batch_size=32,class_mode='categorical')

test_loss, test_acc = model.evaluate(test_generator)
print(f"Test accuracy: {test_acc}")
"""
