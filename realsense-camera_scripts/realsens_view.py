# Import necessary libraries
import pyrealsense2 as rs
import numpy as np   
import cv2 

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
        frames = pipeline.wait_for_frames(5)  
        
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
        
        #center of the camera - coordinates xy
        center_kamera = [int(color_image.shape[0]/2), int(color_image.shape[1]/2)]
        
        # two types of depth calculation, in the center of the camera's resolution
        distance = depth_image * depth_scale    
        distance2 = depth_frame.get_distance(center_kamera[1], center_kamera[0]) 
        
        # Map pixel coordinates to real-world coordinates
        depth_value = distance2  # coordinates are in (y, x) format
        Real_world_coordinates = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [center_kamera[0], center_kamera[1]],depth_value)
        print("Real-world coordinates (x, y, z): %s ,  distance: %s " % ( Real_world_coordinates,distance[center_kamera[0], center_kamera[1]]))
        #print(Real_world_coordinates[2]-distance[center_kamera[1], center_kamera[0]])
        # Display the aligned frames in a window and drawing a line in the center of the camera 
        #print()
        cv2.line(color_image, (center_kamera[1],center_kamera[0]), (center_kamera[1],center_kamera[0]), (255, 0, 0), 3) 
        cv2.line(distance, (center_kamera[1],center_kamera[0]), (center_kamera[1],center_kamera[0]), (255, 0, 0), 3)  
        
        #combined = np.hstack((color_image, distance))  
        
        cv2.imshow('RealSense',color_image)
        #cv2.imshow('RealSense', color_image) 
        #cv2.imshow('depth',distance)
        if  cv2.waitKey(1) & 0xFF == ord('q') or  cv2.waitKey(1) == 27 : 
            cv2.destroyAllWindows()
            break 
           
finally:
    pipeline.stop()