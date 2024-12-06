# --*-- coding:utf-8 --*--
import math
import cv2
import os
import math 
import shutil
import glob
from utils.rgbd_util import *
from utils.getCameraParam import *


#pc_recorded = "Daniel" # Jakobs, Daniel
#data_category = "training"  # validating, testing, training
#track_name =  "lines"  # without_line, lines, logo, red_line
#Jakobs train "logo, without_line" Done


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
    #camera_matrix = getCameraParam('colour')    

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
    for modes in ["Testing"]:   
            
        main_road = f'Unfiltert_Dataset_3.0/'
        vej_depth = main_road + f'{modes}/Depth'     
        vej_rgb =  main_road+ f'{modes}/RGB'    


        count = 0
        for filnames in os.listdir(vej_depth) :    

            # Here we just gets the images D for depth and RGB for colour   
            D,RGB=getImage(filnames,vej_depth,vej_rgb)  
            """ 
            extrisinc_matrix = np.array([[3458.041299 ,0.000000 ,149.621634],
                                        [0.000000,2874.851107,96.824471],
                                        [0.000000,0.000000,1.000000]],dtype=np.float32)
            distort =np.array([0.187651,-16.754147,-0.020581,0.104681,134.489414],dtype=np.float32) 



            extrisinc_matrix_50 = np.array([[1719.507059,0.000000,201.517313],[0.000000,1275.760777,318.041236],[0.000000,0.000000,1.000000]],dtype=np.float32)

            distort_50 = np.array([0.328611,0.574656,-0.012306,-0.271428,-0.500519],dtype=np.float32)

            
            h,  w = D.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(extrisinc_matrix, distort, (w,h), 1, (w,h))

            """
            
            #print(np.max(R),np.max(G),np.max(B))
            #D = D * 0.0010000000474974513 # converte to meter  
          
            if modes == "training" :  
                if filnames.find("without_line") != -1:  pc_recorded = "Jakobs"
                if filnames.find("lines") != -1: pc_recorded = "Daniel" 
                if filnames.find("logo") != -1: pc_recorded = "Jakobs"  
            elif modes == "Testing" :  
                if filnames.find("without_line") != -1:  pc_recorded = "Daniel"
                if  filnames.find("lines") != -1: pc_recorded = "Daniel" 
                if filnames.find("logo") != -1: pc_recorded = "Daniel"   
            elif modes == "validating" :  
                if filnames.find("without_line") != -1:  pc_recorded = "Jakobs"
                if  filnames.find("lines") != -1: pc_recorded = "Jakobs" 
                if filnames.find("logo") != -1: pc_recorded = "Jakobs"    
            elif filnames.find("red_line")  != -1: 
                pc_recorded = "Mikkeline" 
            
            """
            h,  w = D.shape[:2] 

            new_K_depth, _ = cv2.getOptimalNewCameraMatrix(parameters[pc_recorded]["K_depth"], parameters[pc_recorded]["D_depth"], (w,h), 1, (w,h))
            depth_image_undistorted = cv2.undistort(D, parameters[pc_recorded]["K_depth"], parameters[pc_recorded]["D_depth"], None, new_K_depth)
        

            new_K_RGB, _ = cv2.getOptimalNewCameraMatrix(parameters[pc_recorded]["K_colour"], parameters[pc_recorded]["D_colour"], (w, h), 1, (w, h))
            RGB_image_undistorted = cv2.undistort(RGB, parameters[pc_recorded]["K_colour"], parameters[pc_recorded]["D_colour"], None, new_K_RGB)

            fov_depth_h = 2 * np.arctan(w / (2 * parameters[pc_recorded]["K_depth"][0][0])) * 180 / np.pi  # Horizontal FoV in degrees
            fov_rgb_h = 2 * np.arctan(w / (2 * parameters[pc_recorded]["K_colour"][0][0])) * 180 / np.pi
            fov_depth_v = 2 * np.arctan(h / (2 * parameters[pc_recorded]["K_depth"][1][1])) * 180 / np.pi  # Vertical FoV in degrees
            fov_rgb_v = 2 * np.arctan(h / (2 * parameters[pc_recorded]["K_colour"][1][1])) * 180 / np.pi


            #print(f"Depth Camera FoV: Horizontal {fov_depth_h:.2f}, Vertical {fov_depth_v:.2f}")
            #print(f"RGB Camera FoV: Horizontal {fov_rgb_h:.2f}, Vertical {fov_rgb_v:.2f}")
            # Crop depth image to match RGB camera's narrower FoV
            crop_factor_h = min(1,fov_rgb_h / fov_depth_h)
            crop_factor_v = min(1,fov_rgb_v / fov_depth_v)
            crop_width = int(w * crop_factor_h)
            crop_height = int(h * crop_factor_v)

            x_min = (w - crop_width) // 2
            x_max = x_min + crop_width
            y_min = (h - crop_height) // 2
            y_max = y_min + crop_height
            


            depth_image_undistorted = depth_image_undistorted[y_min:y_max, x_min:x_max] 
            #print(f"depth_image_undistorted: {depth_image_undistorted.shape}")

     
            """ 
            Translation_Vector = np.array([-0.0589994452893734 ,-0.000220542555325665 ,0.00041276979027316 ])    

            Rotation_Matrix = np.array([[0.999984,-0.00521408,0.00214487], 
                                        [0.00520871 ,0.999983,0.00250408], 
                                        [-0.00215789,-0.00249287,0.999995]])  
                 
            Rotation_Matrix = np.array([[0.999997,0.00218754,0.000459686], 
                                        [-0.00218882,0.999994,0.0027873], 
                                        [-0.000453586,-0.0027883,0.999996]])

            Translation_Vector = np.array([-0.0591273009777069-30, 0.000230809804634191, 0.000316781341098249]) 
            """ 
            #D455f Tranlation and orientation from depth to colour     
            Rotation_Matrix = np.array([[0.999988,0.000502921,0.00483117], 
                                        [-0.000513652,0.999997,0.00222027], 
                                        [-0.00483004,-0.00222273,0.999986]])
             
            Translation_Vector = np.array([-0.0591291524469852, -7.45405777706765e-05, 0.00055361760314554])

            h, w = depth_image_undistorted.shape
            i, j = np.meshgrid(np.arange(w), np.arange(h))

            # Convert depth image to 3D points
            z = depth_image_undistorted.astype(np.float32)
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

            #depth_points = np.array([[1000, 105], [1068, 111], [1530, 177], [978, 171]])  # Points in depth image
            #rgb_points = np.array([[994,120], [1061, 126], [1040, 187], [976, 179]])  # Corresponding points in RGB image

            # Compute the homography matrix
            #H, _ = cv2.findHomography(depth_points, rgb_points)

            # Warp the depth image to align better
            #aligned_depth = cv2.warpPerspective(aligned_depth, H, (w, h))


        
            #dx = 88  # Horizontal translation in pixels
            #dy = 108  # Vertical translation in pixels 
            #dx = -20
            #dy = 50 
            #translation = np.float32([[1, 0, dx], [0, 1, dy]]) 
            #aligned_depth = cv2.warpAffine(aligned_depth, translation, (aligned_depth.shape[1], aligned_depth.shape[0]))
        
        
            aligned_depth = aligned_depth* 0.0010000000474974513 # converte to meter
            min_depth = 0 # 
            max_depth = 6 # 6 meters thredshold
            filtered_depth_image = np.where(((aligned_depth>=min_depth) & (aligned_depth <= max_depth)), aligned_depth, 0) 
            #filtered_depth_image = filtered_depth_image.astype(np.float32)  
            # Combine RGB and depth
            #combined = cv2.addWeighted(RGB_image_undistorted, 0.7, depth_colormap, 0.3, 0)
            #print(f"RGB_image_undistorted: {RGB_image_undistorted.shape}") 
            #print(f"aligned_depth: {aligned_depth.shape}")  

            #cv2.imshow("Undistored RGB", RGB_image_undistorted)   
        
            # cv2.imshow("raw_depth", pull)
            #cv2.imshow("RAW_RGB", RGB)
            #cv2.imshow("Depth", filtered_depth_image)   

            mask = (filtered_depth_image > 0).astype(np.uint8) * 255
            masked_rgb = cv2.bitwise_and(RGB_image_undistorted, RGB_image_undistorted, mask=mask)
            #cv2.imshow("masked_rgb", masked_rgb) 

            
            #cv2.imshow("filtered_depth_image", cv2.cvtColor( filtered_depth_image,cv2.IMREAD_GRAYSCALE)) 

            #depth_colored = cv2.applyColorMap((filtered_depth_image*10).astype(np.uint8), cv2.COLORMAP_JET) 
            #cv2.imshow("depth_colored", depth_colored) 
            

            # Create a mask for invalid depth values 
            #holes_mask = (filtered_depth_image == 0) | np.isnan(filtered_depth_image)
            #holes_mask = holes_mask.astype(np.uint8) * 255
            #cv2.imshow("holes_mask", holes_mask )
            # Apply inpainting (e.g., Navier-Stokes or Telea method)
            #inpainted_depth = cv2.inpaint(filtered_depth_image.astype(np.uint16), holes_mask, inpaintRadius=4, flags=cv2.INPAINT_TELEA)
            #filtered_depth = cv2.bilateralFilter(filtered_depth_image.astype(np.float32), d=4, sigmaColor=75, sigmaSpace=75)
            #filtered_depth = cv2.medianBlur(filtered_depth_image.astype(np.float32), 5) 
            #filtered_depth = cv2.GaussianBlur(filtered_depth_image.astype(np.float32), (5, 5), 0) 
            # moprph  =  cv2.morphologyEx(filtered_depth_image.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
            # cv2.imshow("filtered_depth", moprph)

            #depth_colored = cv2.cvtColor((filtered_depth_image).astype(np.uint8), cv2.COLOR_GRAY2BGR)

            # Perform alpha blending
            #alpha = filtered_depth_image[..., np.newaxis]  # Add a channel dimension for broadcasting
            #blended_image = cv2.convertScaleAbs(alpha * RGB_image_undistorted + (1 - alpha) * depth_colored)

            #cv2.imshow("blended_image.png", blended_image) 
            #print('max gray value: ', np.max(D))       # make sure that the image is in 'meter 
            #print('max therdsholded value: ', np.max(filtered_depth_image))   

            count += 1
            # get me the size of os.listdir(vej_depth) as a number so I can see how many images I have left to process 
            print(f"Iterration {count}/{len(os.listdir(vej_depth))} doing: {modes} ")
            
            

            
            
            depth_test = cv2.undistort(D, extrisinc_matrix, distort, None, newcameramtx)
            #DEpth_trehds_test=cv2.undistort(filtered_depth_image, extrisinc_matrix, distort, None, newcameramtx) 
            RGB_test=cv2.undistort(RGB, extrisinc_matrix,distort, None, newcameramtx)    
            #parameters[pc_recorded]["D_colour"]
            x, y, w, h = roi
            RGB_test = RGB_test[y:y+h, x:x+w]  
        
            depth_test = depth_test[y:y+h, x:x+w]    
            # reduce   
            """
            """ 
            #cv2.imshow("RAW_RGB",RGB) 
            #cv2.imshow("RAW_Depth",D)
            #cv2.imshow("Undistored_RGB",RGB_test) 
            #cv2.imshow("Undistored_Depth",depth_test)   
            
            #cv2.imshow("Depth",depth_test) 
            #cv2.imshow("Alpha_blend",A)
            #filtered_depth_image = filtered_depth_image.astype(np.float32) 
            #D = D* 0.0010000000474974513    

            """  

            RockAndRoll = True
            tempory_filname=filnames.split("Depth")[1]  
            
            if RockAndRoll == True:
                hha_complete = getHHA(parameters[pc_recorded]["K_depth"],D ,D) 
                
                #cv2.imwrite(f'Turf_tankBaked_dataset/{modes}/RGB/'+f'undistored_RGB_{filnames.split("Depth")[1]}',RGB_image_undistorted)
                cv2.imwrite(f'Dataset_3.0/{modes}/HHA/'+f'HHA{tempory_filname.split(".")[0]}'+".png", hha_complete)   
                #cv2.imwrite(f'Turf_tankBaked_dataset/{modes}/Depth/'+f'filtered_depth_image_{filnames.split("Depth")[1]}',filtered_depth_image) 
                #cv2.imwrite(f'Turf_tankBaked_dataset/{modes}/RGB_masked/'+f'masked_RGB_{filnames.split("Depth")[1]}',masked_rgb)  
                count += 1
                print(f"  files {count} / {len(os.listdir(vej_depth))}") 
                
            #cv2.imwrite(f'Test/'+ filnames,D) 

            
            #cv2.imshow("HHA_img",hha_complete) 
            #cv2.imshow("Raw_depth",D)   
            #cv2.imshow("RAW_RGB",RGB)
            #cv2.imshow("thredshold",filtered_depth_image)  
            #cv2.imshow("undistored_RGB",RGB_image_undistorted)   
            #cv2.imshow("masked_rgb",masked_rgb)
            #cv2.imshow("H",A) 
            #cv2.imshow("Raw_HHA",RAW_HHA)
            #cv2.waitKey() 
            
            
    """
    D455 

        Extrinsic from "Depth"	  To	  "Color" :
    Rotation Matrix:
    0.999984        -0.00521408       0.00214487    
    0.00520871       0.999983         0.00250408    
    -0.00215789      -0.00249287       0.999995      

    Translation Vector: -0.0589994452893734  -0.000220542555325665  0.00041276979027316  



    D455f 
    Extrinsic from "Depth"	  To	  "Color" :
    Rotation Matrix:
    0.999997         0.00218754       0.000459686   
    -0.00218882       0.999994         0.0027873     
    -0.000453586     -0.0027883        0.999996      

    Translation Vector: -0.0591273009777069  0.000230809804634191  0.000316781341098249  





    """