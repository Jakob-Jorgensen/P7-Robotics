import matplotlib.pyplot as plt 
import numpy as np 
import tensorflow as tf 
from keras import models, layers,utils  
from sklearn.metrics import precision_recall_curve, average_precision_score
import cv2 
import os, glob 

mode_swith = 1

print("Loading the model...") 
model = models.load_model('jakob_Playground_model.keras') 
print("Model loaded.") 


#img_numb = 7

# swtiching between the two modes 
# print the layers and the model summary 
# or display the model processing an sicliancy image
if mode_swith == 0:  
    #Summarising the model  
    
    model.summary() 
    #Visualising the model 
    utils.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)  
    img = plt.imread('model_plot.png') 
    plt.imshow(img) 
    plt.show() 
elif mode_swith == 1:     
    # Paths to the dataset
    rgb_dir = sorted(glob.glob(os.path.join("dataset","RGB_left", "**")))
    depth_dir = sorted(glob.glob(os.path.join("dataset","depth", "**")))
    gt_dir =sorted(glob.glob(os.path.join("dataset","GT-1985", "**")) ) 

    # Define the batch size and image sizes
    img_size = (224, 224)
    gt_size = (50,50) 

    # Loading and resize the images 
    print("Resizing images...")
    rgb_images = np.array([cv2.cvtColor(cv2.resize(cv2.imread(file), img_size).astype('float32')/255.0,  cv2.COLOR_BGR2RGB ) for file in rgb_dir]) 
    depth_images = np.array([cv2.cvtColor(cv2.resize(cv2.imread(file), img_size).astype('float32')/255.0, cv2.COLOR_RGB2GRAY) for file in depth_dir])  # Reducing the number of channels to 1 
    gt_images = np.array([cv2.cvtColor(cv2.resize(cv2.imread(file), gt_size).astype('float32')/255.0, cv2.COLOR_RGB2GRAY) for file in gt_dir]) # Reducing the number of channels to 1 
    print("Images resized.")   
    #Compiling the  model's evaluation matrix
    #model.evaluate([rgb_images, depth_images], gt_images) 
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    for  img_numb in range(len(rgb_images)):
       
        # Add batch dimension for a single sample to make them compatible with evaluate and predict
        rgb_sample = np.expand_dims(rgb_images[img_numb], axis=0)
        depth_sample = np.expand_dims(depth_images[img_numb], axis=0)
        gt_sample = np.expand_dims(gt_images[img_numb], axis=0)

        # Ensure depth images and gt_images are correctly shaped (e.g., HxWx1 for grayscale)
        #depth_sample = np.expand_dims(depth_sample, axis=-1)  # Adds a channel dimension to depth image
        #gt_sample = np.expand_dims(gt_sample, axis=-1)        # Adds a channel dimension to ground truth

        # Evaluate the model on the sample
       
        # Predict with the model on the sample
        predicted_image = model.predict([rgb_sample, depth_sample])

        # Print shapes to confirm dimensions
        print("RGB Sample shape:", rgb_sample.shape)
        print("Depth Sample shape:", depth_sample.shape)
        print("GT Sample shape:", gt_sample.shape)
        print("Predicted Image shape:", predicted_image.shape)
        
        # Display images
        fig, ax = plt.subplots(1, 4, figsize=(16, 4))
        ax[0].imshow(rgb_sample[0,:,:,:], cmap='cividis')  # Adjust the colormap as needed
        ax[0].set_title('RGB Image')
        ax[0].axis('off')

        ax[1].imshow(depth_sample[0,:,:], cmap='gray')
        ax[1].set_title('Depth Image')
        ax[1].axis('off')

        ax[2].imshow(gt_sample[0,:,:], cmap='gray')
        ax[2].set_title('Ground Truth')
        ax[2].axis('off')

        ax[3].imshow(predicted_image[0,:,:,0], cmap='gray')  # Adjust the colormap as needed
        ax[3].set_title('Saliency Map')
        ax[3].axis('off')   
        plt.show()
        #plt.savefig('Mini_project/processed_images/saliency_map_'+str(img_numb)+'.png') 
elif mode_swith == 2:  

    # Paths to the dataset
    rgb_dir = sorted(glob.glob(os.path.join("dataset","RGB_left", "**")))
    depth_dir = sorted(glob.glob(os.path.join("dataset","depth", "**")))
    gt_dir =sorted(glob.glob(os.path.join("dataset","GT-1985", "**")) ) 

    # Define the batch size and image sizes
    img_size = (224, 224)
    gt_size = (224,224) 

    # Loading and resize the images 
    print("Resizing images...")
    rgb_images = np.array([cv2.cvtColor(cv2.resize(cv2.imread(file), img_size).astype('float32')/255.0,  cv2.COLOR_BGR2RGB ) for file in rgb_dir]) 
    depth_images = np.array([cv2.cvtColor(cv2.resize(cv2.imread(file), img_size).astype('float32')/255.0, cv2.COLOR_RGB2GRAY) for file in depth_dir])  # Reducing the number of channels to 1 
    gt_images = np.array([cv2.cvtColor(cv2.resize(cv2.imread(file), gt_size).astype('float32')/255.0, cv2.COLOR_RGB2GRAY) for file in gt_dir]) # Reducing the number of channels to 1 
    print("Images resized.")  
    y_true = gt_images.flatten()   
    y_true_binary = (y_true > 0.5).astype(int)  # Convert to binary values 
    y_scores = model.predict([rgb_images, depth_images]).flatten()  # Predict saliency maps and flatten   
    
    precision, recall, thresholds = precision_recall_curve(y_true_binary, y_scores)  
    average_precision = average_precision_score(y_true_binary, y_scores) 
    print("Average Precision: ", average_precision) 
    f1_scores = 2 * (precision * recall) / (precision + recall) 


    best_f1_index = np.argmax(f1_scores) 
    best_f1 = f1_scores[best_f1_index] 
    best_threshold = thresholds[best_f1_index] 
    print(f"Best F1 Score: {best_f1} at thredshold {best_threshold} " )  

    plt.figure(figsize=(8, 6)) 
    plt.plot(recall, precision, label='Precision-Recall Curve')  
    plt.xlabel('Recall') 
    plt.ylabel('Precision') 
    plt.grid() 
    plt.show() 
