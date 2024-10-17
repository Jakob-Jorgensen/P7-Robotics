import tensorflow as tf
from keras import models,layers


def Miniproject():  
    
    model=models.Sequential() 

    # 1st Convolutional Layer Of Alexnet
    model.add(layers.Conv2D(96, (11, 11), strides=(4, 4), activation='relu', input_shape=(224, 224, 3)))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    # 2st layer: 1.Normalisation
    model.add(layers.LayerNormalization(axis=3))   

    # 3st layer: 1.Dilated Convolutional  
    model.add(layers.Conv2D(256, (5, 5),dilation_rate=(2,2), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    
    # 4st layer: 2.Normalisation
    model.add(layers.LayerNormalization(axis=3))   

    # 5st layer: 2.Dilated Convolutional
    model.add(layers.Conv2D(384, (3, 3),dilation_rate=(4,4), padding='same', activation='relu'))

    # 6th Layer: 3.Dilated Convolutional
    model.add(layers.Conv2D(384, (3, 3),dilation_rate=(4,4), padding='same', activation='relu'))

    # 7th Layer: 4.Dilated Convolutional 
    model.add(layers.Conv2D(256, (3, 3),dilation_rate=(4,4), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    
    # Flatten the convolutional output
    model.add(layers.Flatten())  

    # 1st Fully Connected Layer
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))  # Dropout layer for regularization

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Model summary
    model.summary() 






def AlexNet(): 
    #Initialize the AlexNet model
    model = models.Sequential()

    # 1st Convolutional Layer
    model.add(layers.Conv2D(96, (11, 11), strides=(4, 4), activation='relu', input_shape=(227, 227, 3)))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    # 2nd Convolutional Layer
    model.add(layers.Conv2D(256, (5, 5), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    # 3rd Convolutional Layer
    model.add(layers.Conv2D(384, (3, 3), padding='same', activation='relu'))

    # 4th Convolutional Layer
    model.add(layers.Conv2D(384, (3, 3), padding='same', activation='relu'))

    # 5th Convolutional Layer
    model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    # Flatten the convolutional output
    model.add(layers.Flatten())

    # 1st Fully Connected Layer
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))  # Dropout layer for regularization

    # 2nd Fully Connected Layer
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))  # Dropout layer for regularization

    # Output Layer (1000 classes for ImageNet)
    model.add(layers.Dense(1000, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Model summary
    model.summary() 



Miniproject()
