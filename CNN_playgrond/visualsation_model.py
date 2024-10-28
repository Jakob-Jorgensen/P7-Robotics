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
    model = models.load_model('test_model.h5')    
    model.summary() 

    #Visualising the model 
    utils.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)  
    img = plt.imread('model_plot.png') 
    plt.imshow(img) 
    plt.show() 
else:     
    print("Loading the model...") 
    model = models.load_model('test_model.h5')     
    print("Model loaded.")
    # Paths to the dataset
    rgb_dir = sorted(glob.glob(os.path.join("dataset","RGB_left", "**")))
    depth_dir = sorted(glob.glob(os.path.join("dataset","depth", "**")))
    gt_dir =sorted(glob.glob(os.path.join("dataset","GT-1985", "**")) ) 

    # Define the batch size and image sizes
    img_size = (224, 224)
    gt_size = (50,50)

    # Loading and resize the images 
    print("Resizing images...")
    rgb_images = np.array([cv2.resize(cv2.imread(file), img_size).astype('float32')/255.0 for file in rgb_dir]) 
    depth_images = np.array([cv2.cvtColor(cv2.resize(cv2.imread(file), img_size).astype('float32')/255.0, cv2.COLOR_RGB2GRAY) for file in depth_dir])  # Reducing the number of channels to 1 
    gt_images = np.array([cv2.cvtColor(cv2.resize(cv2.imread(file), gt_size).astype('float32')/255.0, cv2.COLOR_RGB2GRAY) for file in gt_dir]) # Reducing the number of channels to 1 
    print("Images resized.")  

    #print(rgb_images[0].shape) 
    #print(depth_images[0].shape)

    proced_image=model.predict([rgb_images[0], depth_images[0]]) 
    
 
    # Display images
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(rgb_images[0])  
    ax[0].set_title('RGB Image')
    ax[0].axis('off')

    ax[1].imshow(depth_images[0], cmap='gray')  
    ax[1].set_title('Depth Image')
    ax[1].axis('off')

    ax[2].imshow(proced_image, cmap='viridis')  # Adjust the colormap as needed
    ax[2].set_title('Processed Image')
    ax[2].axis('off')

    plt.show()






