# Import necessary libraries
import pyrealsense2 as rs
import cv2  
import numpy as np

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure it to stream
config = rs.config()

# This is the minimal recommended configuration for D435 Depth Camera
config.enable_stream(rs.stream.depth,640 ,480, rs.format.z16, 15)
config.enable_stream(rs.stream.color,640, 480, rs.format.bgr8, 15)

# Start streaming 
try: 
    profile = pipeline.start(config)
except Exception as e: 
    print(f"Failed to connect to camera: {e}")



# Create an align object
deth_sensor = profile.get_device().first_depth_sensor() 
depth_scale = deth_sensor.get_depth_scale() 
align_to = rs.stream.color
align = rs.align(align_to) 
# Get camera intrinsics
profile = pipeline.get_active_profile()
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()

# Get camera intrinsics
print(" depth scale : %s" %depth_scale)

# Streaming loop 
try:
    while True:
        # Get frameset of color and depth 
        frames = pipeline.wait_for_frames()  
        
        # Align the depth frame to color frame
        aligned_frames =align.process(frames) 

        # Get aligned frames  
        depth_frame = aligned_frames.get_depth_frame()
        color_frame =aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue 
        
        # Extraxting depth and color images from frames
        depth_image = np.array(depth_frame.get_data())
        color_image = np.array(color_frame.get_data())  
        depth = depth_image.astype(float)  
        
        depth_image_3d = cv2.cvtColor(depth_image, cv2.COLOR_GRAY2RGB)
        combined= np.hstack((color_image,depth_image_3d))
        cv2.imshow('Color',color_image) 
        cv2.imshow('Depth_image',combined)
        cv2.imshow('RealSense',depth)
      
        if  cv2.waitKey(1) & 0xFF == ord('q') or  cv2.waitKey(1) == 27 : 
            cv2.destroyAllWindows()
            break 
           
finally:
    pipeline.stop()