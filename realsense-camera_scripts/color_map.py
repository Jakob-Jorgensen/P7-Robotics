import pyrealsense2 as rs
import numpy as np
import cv2

# Create a RealSense pipeline
pipeline = rs.pipeline()

# Create a configuration for the pipeline
config = rs.config()

# Enable both depth and color streams
config.enable_stream(rs.stream.depth, 648, 480, rs.format.z16, 15)  # Depth stream
config.enable_stream(rs.stream.color, 648, 480, rs.format.bgr8, 15)  # RGB stream

# Start streaming 
try: 
    profile = pipeline.start(config)
except Exception as e: 
    print(f"Failed to connect to camera: {e}")



try:
    while True:
        # Wait for frames from the camera
        frames = pipeline.wait_for_frames(50)

        # Get depth and color frames
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # Check if both frames are valid
        if not depth_frame or not color_frame:
            continue

        # Convert depth frame to a numpy array and normalize for visualization
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Convert color frame to a numpy array
        color_image = np.asanyarray(color_frame.get_data())

        # Stack both images horizontally
        images = np.hstack((color_image, depth_colormap))

        # Display the stacked images
        cv2.imshow('RGB and Depth', images)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop the pipeline and close all OpenCV windows
    pipeline.stop()
    cv2.destroyAllWindows()
