import cv2  
import matplotlib.pyplot as plt
import numpy as np  



RGB_image = cv2.imread("RGB.png")  
Depth_image = cv2.imread("Depth.png") 

mask = cv2.inRange(Depth_image,(0,0,0),(4,4,4))  

mask_3d = cv2.merge([mask,mask,mask]) 

 
fig, ax =  plt.subplots(2,2,figsize=(18,8)) 
ax[0,0].imshow(Depth_image)  
ax[0,0].set_title('Depth Image') 
ax[0,0].axis('off') 
ax[1,0].imshow(RGB_image) 
ax[1,0].set_title('RGB Image') 
ax[1,0].axis('off')
ax[0,1].imshow(mask) 
ax[0,1].set_title('Mask') 
ax[0,1].axis('off') 

ax[1,1].imshow(cv2.bitwise_and(RGB_image,mask_3d))
ax[1,1].set_title('Masked Image')
ax[1,1].axis('off') 

plt.tight_layout()  
plt.show()


cv2.waitKey()  
