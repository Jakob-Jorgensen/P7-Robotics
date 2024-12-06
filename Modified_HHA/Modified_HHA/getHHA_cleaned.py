# --*-- coding:utf-8 --*--
import math
import cv2
import os
import math 
import shutil
import glob
from utils.rgbd_util import *
from utils.getCameraParam import *


########## Interfacing with the dataset ########## 

write_to_path = "Path/TO"
Write_mode = "False" # True or False for writing to path 

########## Interfacing with the dataset ##########



'''
must use 'colour_BGR2GRAY' here, or you will get a different gray-value with what MATLAB gets.
''' 
def getImage(i,vej_depth,vej_rgb):
    D = cv2.imread(os.path.join(vej_depth, f"{i}"),cv2.COLOR_BGR2GRAY)
    RGB = cv2.imread(os.path.join(vej_rgb, f"RAW_RGB{i.split('Depth')[1]}")) 
    
    return D ,RGB 

'''
C: Camera matrix
D: Depth image, the unit of each element in it is "meter"
RD: Raw depth image, the unit of each element in it is "meter"
'''
def getHHA(C, D, RD):
    missingMask = (RD == 0)
    pc, N, yDir, h, pcRot, NRot = processDepthImage(D * 100, missingMask, C)

    tmp = np.multiply(N, yDir)
    acosValue = np.minimum(1,np.maximum(-1,np.sum(tmp, axis=2)))
    angle = np.array([math.degrees(math.acos(x)) for x in acosValue.flatten()])
    angle = np.reshape(angle, h.shape)

    '''
    Must convert nan to 180 as the MATLAB program actually does. 
    Or we will get a HHA image whose border region is different
    with that of MATLAB program's output.
    '''
    angle[np.isnan(angle)] = 180        


    pc[:,:,2] = np.maximum(pc[:,:,2], 100)
    I = np.zeros(pc.shape)

    # opencv-python save the picture in BGR order.
    I[:,:,2] = 31000/pc[:,:,2]
    I[:,:,1] = h
    I[:,:,0] = (angle + 128-90)

    # print(np.isnan(angle))

    '''
    np.uint8 seems to use 'floor', but in matlab, it seems to use 'round'.
    So I convert it to integer myself.
    '''
    I = np.rint(I)

    # np.uint8: 256->1, but in MATLAB, uint8: 256->255 
    if True :     
        I[I>255] = 255 
    else: 
        I[I>65535] = 65535
    HHA = I.astype(np.uint8)
    return HHA

if __name__ == "__main__":  

    parameters = {
            "Mikkeline": {
                "K_depth": np.array([
                    [647.6559448242188, 0.0, 643.5599365234375],
                    [0.0, 647.6559448242188, 362.2444152832031],
                    [0.0, 0.0, 1.0]
                ]),
                "D_depth": np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
                "K_colour": np.array([
                    [642.18603515625, 0.0, 644.2113647460938],
                    [0.0, 641.4976806640625, 362.7994079589844],
                    [0.0, 0.0, 1.0]
                ]),
                "D_colour": np.array([-0.05690142139792442, 0.06686285883188248, 0.0004544386174529791, 0.0006704007973894477, -0.021477429196238518])
            },
            "Jakobs": {
                "K_depth": np.array([
                    [647.6559448242188, 0.0, 643.5599365234375],
                    [0.0, 647.6559448242188, 362.3144226074219],
                    [0.0, 0.0, 1.0]
                ]),
                "D_depth": np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
                "K_colour": np.array([
                    [642.18603515625, 0.0, 644.2113647460938],
                    [0.0, 641.4976806640625, 362.7994079589844],
                    [0.0, 0.0, 1.0]
                ]),
                "D_colour": np.array([-0.05690142139792442, 0.06686285883188248, 0.0004544386174529791, 0.0006704007973894477, -0.021477429196238518])
            },
            "Daniel": {
                "K_depth": np.array([
                    [647.6559448242188, 0.0, 643.5599365234375],
                    [0.0, 647.6559448242188, 362.3144226074219],
                    [0.0, 0.0, 1.0]
                ]),
                "D_depth": np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
                "K_colour": np.array([
                    [641.8356323242188, 0.0, 644.2113647460938],
                    [0.0, 641.147705078125, 362.7994079589844],
                    [0.0, 0.0, 1.0]
                ], ),
                "D_colour": np.array([-0.05690142139792442, 0.06686285883188248, 0.0004544386174529791, 0.0006704007973894477, -0.021477429196238518])
            }
        }  
    

    for modes in ["Training","Testing","Validation"]:   
            
        main_road = f'Unfiltert_Dataset_3.0/'
        vej_depth = main_road + f'{modes}/Depth'     
        vej_rgb =  main_road+ f'{modes}/RGB'    


        count = 0
        for filnames in os.listdir(vej_depth) :    

            # Here we just gets the images D for depth and RGB for colour   
            D,RGB=getImage(filnames,vej_depth,vej_rgb)  
            
      
            # Based on the name of the file we know which camera was used to record the data
            if modes == "Training" :  
                if filnames.find("without_line") != -1:  pc_recorded = "Jakobs"
                if filnames.find("lines") != -1: pc_recorded = "Daniel" 
                if filnames.find("logo") != -1: pc_recorded = "Jakobs"  
            elif modes == "Testing" :  
                if filnames.find("without_line") != -1:  pc_recorded = "Daniel"
                if  filnames.find("lines") != -1: pc_recorded = "Daniel" 
                if filnames.find("logo") != -1: pc_recorded = "Daniel"   
            elif modes == "Validation" :  
                if filnames.find("without_line") != -1:  pc_recorded = "Jakobs"
                if  filnames.find("lines") != -1: pc_recorded = "Jakobs" 
                if filnames.find("logo") != -1: pc_recorded = "Jakobs"    
            elif filnames.find("red_line")  != -1: 
                pc_recorded = "Mikkeline" 
            
          
            h,  w = D.shape[:2] 

            # Undistort the depth and RGB images
            new_K_depth, _ = cv2.getOptimalNewCameraMatrix(parameters[pc_recorded]["K_depth"], parameters[pc_recorded]["D_depth"], (w,h), 1, (w,h))
            depth_image_undistorted = cv2.undistort(D, parameters[pc_recorded]["K_depth"], parameters[pc_recorded]["D_depth"], None, new_K_depth)
            new_K_RGB, _ = cv2.getOptimalNewCameraMatrix(parameters[pc_recorded]["K_colour"], parameters[pc_recorded]["D_colour"], (w, h), 1, (w, h))
            RGB_image_undistorted = cv2.undistort(RGB, parameters[pc_recorded]["K_colour"], parameters[pc_recorded]["D_colour"], None, new_K_RGB)

            # Calculate the FoV of the depth and RGB camera
            fov_depth_h = 2 * np.arctan(w / (2 * parameters[pc_recorded]["K_depth"][0][0])) * 180 / np.pi  # Horizontal FoV in degrees
            fov_rgb_h = 2 * np.arctan(w / (2 * parameters[pc_recorded]["K_colour"][0][0])) * 180 / np.pi
            fov_depth_v = 2 * np.arctan(h / (2 * parameters[pc_recorded]["K_depth"][1][1])) * 180 / np.pi  # Vertical FoV in degrees
            fov_rgb_v = 2 * np.arctan(h / (2 * parameters[pc_recorded]["K_colour"][1][1])) * 180 / np.pi

            # Crop depth image to match RGB camera's narrower FoV, print the FOV of the depth and RGB camera to see the difference
            crop_factor_h = min(1,fov_rgb_h / fov_depth_h)
            crop_factor_v = min(1,fov_rgb_v / fov_depth_v)
            crop_width = int(w * crop_factor_h)
            crop_height = int(h * crop_factor_v)

            x_min = (w - crop_width) // 2
            x_max = x_min + crop_width
            y_min = (h - crop_height) // 2
            y_max = y_min + crop_height
            

            #Crop the images  
            depth_image_undistorted = depth_image_undistorted[y_min:y_max, x_min:x_max] 
            
          
            #D455f Tranlation and orientation from depth to colour image   
            Rotation_Matrix = np.array([[0.999988,0.000502921,0.00483117], 
                                        [-0.000513652,0.999997,0.00222027], 
                                        [-0.00483004,-0.00222273,0.999986]])
             
            Translation_Vector = np.array([-0.0591291524469852, -7.45405777706765e-05, 0.00055361760314554])

            h, w = depth_image_undistorted.shape 
            # Create meshgrid for 3D points
            i, j = np.meshgrid(np.arange(w), np.arange(h))

            # Convert depth image to 3D points
            z = depth_image_undistorted*0.0010000000474974513 
            x = (j - parameters[pc_recorded]["K_depth"][0][2]) * z / parameters[pc_recorded]["K_depth"][0][0]
            y = (i - parameters[pc_recorded]["K_depth"][1][2]) * z / parameters[pc_recorded]["K_depth"][1][1]

            points_3d = np.stack((x, y, z), axis=-1).reshape(-1, 3)  # Reshape to a list of 3D points 

            #align the depth image with the RGB image
            points_3d_rgb = (Rotation_Matrix @ points_3d.T).T + Translation_Vector

            # Project points to RGB image plane
            valid_indices = points_3d_rgb[:, 2] >= 0  # Only process points with positive depth
            points_3d_rgb = points_3d_rgb[valid_indices]
            
            x_proj = (points_3d_rgb[:, 0] * parameters[pc_recorded]["K_colour"][0][0] / points_3d_rgb[:, 2]) + parameters[pc_recorded]["K_colour"][0][2]
            y_proj = (points_3d_rgb[:, 1] * parameters[pc_recorded]["K_colour"][1][1] / points_3d_rgb[:, 2]) + parameters[pc_recorded]["K_colour"][1][2]

            # Create aligned depth map
            aligned_depth = np.zeros((h, w), dtype=np.float32)  # Same size as RGB image

            # Populate aligned depth map
            for idx, (y_p, x_p, z_p) in enumerate(zip(x_proj, y_proj, points_3d_rgb[:, 2])):
                if 0 <= int(x_p) < w and 0 <= int(y_p) < h:
                    if aligned_depth[int(y_p), int(x_p)] == 0 or z_p < aligned_depth[int(y_p), int(x_p)]:
                        aligned_depth[int(y_p), int(x_p)] = z_p  # Keep the closest depth value 

            min_depth = 0 # 
            max_depth = 6 # 6 meters thredshold
            filtered_depth_image = np.where(((aligned_depth>=min_depth) & (aligned_depth <= max_depth)), aligned_depth, 0) 
           
            # Create a mask for the depth image to see the alligment of depth and RGB image
            mask = (filtered_depth_image > 0).astype(np.uint8) * 255
            masked_rgb = cv2.bitwise_and(RGB_image_undistorted, RGB_image_undistorted, mask=mask)
           
            #save the images to the path 
            if Write_mode == True:
                hha_complete = getHHA(parameters[pc_recorded]["K_depth"],filtered_depth_image ,filtered_depth_image) 
                tempory_filname=filnames.split("Depth")[1]  
                cv2.imwrite(f'{write_to_path}/{modes}/RGB/'+f'undistored_RGB{filnames.split("Depth")[1]}',RGB_image_undistorted)
                cv2.imwrite(f'{write_to_path}/{modes}/HHA/'+f'HHA{filnames.split("Depth")[1]}'+".png", hha_complete)   
                cv2.imwrite(f'{write_to_path}/{modes}/Depth/'+f'filtered_depth_image{tempory_filname.split(".")[0]}'+".tiff",filtered_depth_image) 
                cv2.imwrite(f'{write_to_path}/{modes}/RGB_masked/'+f'masked_RGB{filnames.split("Depth")[1]}',masked_rgb)  
                
            count += 1  
            print(f"Iterration {count}/{len(os.listdir(vej_depth))} doing: {modes} ")
        
