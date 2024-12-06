import numpy as np 
import os 
import cv2
import shutil

for modes in ['Validation']:

    Main_path = "C:/Users/jakob/Desktop/Projec/pos/Unfiltert_Dataset_3.0/"
    end_path  = "C:/Users/jakob/Desktop/Projec/pos/Dataset_3.0/" 
    GT_path = f"{modes}/"+"GT/" 
    RGB_path = f"{modes}/"+"RGB/"  
    HHA_path = f"{modes}/"+"HHA/" 
    RGB_masked_path = f"{modes}/"+"RGB_masked/" 
    Depth_path = f"{modes}/"+"Depth/"
    count = 0
    for RGBs in os.listdir(Main_path+RGB_path):  
        count += 1 
        #shutil.copyfile(Main_path+RGB_path+RGBs, end_path+RGB_path+f"RGB{RGBs.split('RGB_')[1]}") 
       
        #shutil.copyfile(Main_path+HHA_path+f"HHA{RGBs.split('RGB_')[1]}", end_path+HHA_path+f"HHA{RGBs.split('RGB_')[1]}")  
        tempoary_depth_name = RGBs.split('RGB_')[1] 
        tempoary_new_depth_name = RGBs.split('RGB_')[1] 
        #print(tempoary_depth_name.split('.')[0])
        #shutil.copyfile(Main_path+Depth_path+f"Depth{tempoary_depth_name.split('.')[0]}"+".tiff", end_path+Depth_path+f"Depth{tempoary_new_depth_name.split('.')[0]}"+".tiff") 
        #shutil.copyfile(Main_path+RGB_masked_path+f"masked_RGB{RGBs.split('RGB')[1]}", end_path+RGB_masked_path+f"masked_RGB{RGBs.split('RGB')[1]}")  
        
        if not os.path.exists(Main_path+GT_path+f"undistored_GT{RGBs.split('RGB')[1]}"):  
            cv2.imwrite(end_path+GT_path+"undistored_GT"+RGBs.split("RGB_")[1], np.zeros((720,1280, 1)))  
            print(f" interating over {count}/{len(os.listdir(Main_path+RGB_path))} in {modes} mode -> {RGBs}")       
        else: 
            shutil.copyfile(Main_path+GT_path+f"undistored_GT{RGBs.split('RGB')[1]}", end_path+GT_path+f"undistored_GT{RGBs.split('RGB_')[1]}") 
        
