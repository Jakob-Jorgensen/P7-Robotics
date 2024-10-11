# Import necessary libraries
import pyrealsense2 as rs
import numpy as np
import cv2  # OpenCV for displaying the video feed
class RealSense_came():  
    # Create a pipeline
    pipeline = rs.pipeline()

    # Create a config and configure it to stream at 1280x720 resolution
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)  # Set depth stream to 1280x720 at 30 FPS
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)  # Set color stream to 1280x720 at 30 FPS

    # Start streaming
    try: 
        profile = pipeline.start(config)
    except Exception as e: 
        print(f"Failed to connect to camera: {e}")

    # Create an align object to align depth to color frame
    depth_sensor = profile.get_device().first_depth_sensor() 
    depth_scale = depth_sensor.get_depth_scale()  # Get depth scale (meters per depth unit)
    align_to = rs.stream.color
    align = rs.align(align_to) 

    # Streaming loop
    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            
            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames (depth and color)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            # Ensure frames are valid
            if not depth_frame or not color_frame:
                continue

            # Convert depth and color frames to numpy arrays
            depth_data = np.asanyarray(depth_frame.get_data())  # Raw depth data, no normalization
            color_image = np.asanyarray(color_frame.get_data())

            # Apply depth scale to convert depth values to meters (raw depth data for computations)
            depth_image_in_meters = depth_data * depth_scale

            # Gentle scaling for visualization: Clip depth values between a defined range (e.g., 0.5 to 4 meters)
            min_depth_meters = 0.6  # Minimum depth (closer than this will be clipped)
            max_depth_meters = 5.0  # Maximum depth (farther than this will be clipped)

            # Clip the depth data within the defined range
            depth_clipped = np.clip(depth_image_in_meters, min_depth_meters, max_depth_meters)

            # Rescale the clipped depth data to the 16-bit range for visualization
            depth_scaled = ((depth_clipped - min_depth_meters) / (max_depth_meters - min_depth_meters)) * 2**16
            depth_16bit = depth_scaled.astype(np.uint16)

            # Apply a median filter to the 16-bit depth image
            # The kernel size must be an odd number (e.g., 5)
            depth_filtered = cv2.medianBlur(depth_16bit, 5)  # Apply a median filter with a 5x5 kernel

            # Display the filtered 16-bit depth image in OpenCV (grayscale 16-bit)
            cv2.imshow('RealSense Depth Feed (Filtered 16-bit)', depth_filtered)

            # Display the color frame as well
            cv2.imshow('RealSense Color Feed', color_image)

            # Check for key press, exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Stop the pipeline and close the windows
        pipeline.stop()
        cv2.destroyAllWindows()
