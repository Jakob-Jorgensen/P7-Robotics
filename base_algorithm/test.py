import pyrealsense2 as rs
import numpy as np
import cv2

# Create a pipeline
pipeline = rs.pipeline()

# Configure the pipeline to stream Depth, RGB, and Infrared data at the same resolution
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)  # Left IR stream

# Start the pipeline
pipeline.start(config)

# Align object: Align depth to color
align_to = rs.stream.color
align = rs.align(align_to)

try:
    while True:
        # Wait for frames
        frames = pipeline.wait_for_frames()
        
        # Align depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        ir_frame = frames.get_infrared_frame()

        # Ensure that frames are valid
        if not depth_frame or not color_frame or not ir_frame:
            continue

        # Convert frames to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())  # Raw depth data (16-bit values)
        color_image = np.asanyarray(color_frame.get_data())  # RGB color image
        ir_image = np.asanyarray(ir_frame.get_data())        # IR data (grayscale)

        # Apply colormap to depth image for visualization
        # Normalize depth image for colormap (convert 16-bit depth to 8-bit)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Convert IR image to 3-channel for visualization (optional)
        ir_image_colored = cv2.cvtColor(ir_image, cv2.COLOR_GRAY2BGR)

        # Combine the RGB, Depth (colormap), and IR streams into one image
        # Stack them horizontally (RGB | Depth | IR)
        combined_image = np.hstack((color_image, depth_colormap, ir_image_colored))

        # Display the combined image
        cv2.imshow('Combined (RGB | Depth | IR)', combined_image)

        # Break loop on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop the pipeline
    pipeline.stop()
    cv2.destroyAllWindows()

