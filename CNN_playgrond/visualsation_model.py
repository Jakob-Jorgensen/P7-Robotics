import matplotlib.pyplot as plt 
import numpy as np 
import tensorflow as tf 
from keras import models, layers,utils 
import cv2, os, glob

mode_swith = 1

#Loading the model 
  


# swtiching between the two modes 
# print the layers and the model summary 
# or display the model processing an sicliancy image
if mode_swith == 0:  
    #Summarising the model  
    model = models.load_model('playground_model_small.keras')    
    model.summary() 

    #Visualising the model 
    utils.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)  
    img = plt.imread('model_plot.png') 
    plt.imshow(img) 
    plt.show() 
else:     
    print("Loading the model...") 
    model = models.load_model('Saliency_model.h5')     
    
    print("Model loaded.")
    # Paths to the dataset
    rgb_dir = sorted(glob.glob(os.path.join("dataset","RGB_left", "**")))
    depth_dir = sorted(glob.glob(os.path.join("dataset","depth", "**")))
    gt_dir =sorted(glob.glob(os.path.join("dataset","GT-1985", "**")) ) 

    # Define the batch size and image sizes
    img_size = (224, 224)
    gt_size = (50,50) 
    img_numb = 6

    # Loading and resize the images 
    print("Resizing images...")
    rgb_images = np.array([cv2.cvtColor(cv2.resize(cv2.imread(file), img_size).astype('float32')/255.0,  cv2.COLOR_BGR2RGB ) for file in rgb_dir]) 
    depth_images = np.array([cv2.cvtColor(cv2.resize(cv2.imread(file), img_size).astype('float32')/255.0, cv2.COLOR_RGB2GRAY) for file in depth_dir])  # Reducing the number of channels to 1 
    gt_images = np.array([cv2.cvtColor(cv2.resize(cv2.imread(file), gt_size).astype('float32')/255.0, cv2.COLOR_RGB2GRAY) for file in gt_dir]) # Reducing the number of channels to 1 
    print("Images resized.")  

    
    # Add batch dimension for a single sample to make them compatible with evaluate and predict
    rgb_sample = np.expand_dims(rgb_images[img_numb], axis=0)
    depth_sample = np.expand_dims(depth_images[img_numb], axis=0)
    gt_sample = np.expand_dims(gt_images[img_numb], axis=0)

     # Ensure depth images and gt_images are correctly shaped (e.g., HxWx1 for grayscale)
    depth_sample = np.expand_dims(depth_sample, axis=-1)  # Adds a channel dimension to depth image
    gt_sample = np.expand_dims(gt_sample, axis=-1)        # Adds a channel dimension to ground truth

    # Evaluate the model on the sample
    #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  
    #loss, *metrics = model.evaluate([rgb_images, depth_images], gt_images, verbose=1)
    #print(f"Loss: {loss}, Metrics: {metrics}")
    model.evaluate([rgb_sample, depth_sample], gt_sample) 
    # Predict with the model on the sample
    predicted_image = model.predict([rgb_sample, depth_sample])

    # Print shapes to confirm dimensions
    print("RGB Sample shape:", rgb_sample.shape)
    print("Depth Sample shape:", depth_sample.shape)
    print("GT Sample shape:", gt_sample.shape)
    print("Predicted Image shape:", predicted_image.shape)
    
    rgb_image = rgb_images[img_numb]                  # Should be (224, 224, 3) for RGB
    depth_image = np.squeeze(depth_sample)  # Should be (224, 224) for grayscale
    gt_image = np.squeeze(gt_sample)        # Should be (50, 50) for ground truth
    processed_image = np.squeeze(predicted_image).astype('float32')  # Should be (50, 50) for processed/predicted

    # Display images
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    ax[0].imshow(rgb_image)
    ax[0].set_title('RGB Image')
    ax[0].axis('off')

    ax[1].imshow(depth_image, cmap='gray')
    ax[1].set_title('Depth Image')
    ax[1].axis('off')

    ax[2].imshow(gt_image, cmap='gray')
    ax[2].set_title('Ground Truth')
    ax[2].axis('off')

    ax[3].imshow(processed_image, cmap='viridis')  # Adjust the colormap as needed
    ax[3].set_title('Processed Image')
    ax[3].axis('off')

    #cv2.imshow('Proced img',processed_image)     
    
    plt.show() 
    #cv2.waitKey(0)




