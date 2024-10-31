import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import sys
import os
import subprocess
import numpy as np

class DualStreamRecorder(Node):
    def __init__(self, bag_name):
        super().__init__('dual_stream_recorder')

        # Initialize OpenCV Bridge
        self.bridge = CvBridge()
        
        # Use the bag name as a prefix for video files
        self.bag_name = bag_name

        # Set up subscribers for each stream and camera info topics
        self.subscription1 = self.create_subscription(
            Image,
            '/camera/depth/image_rect_raw',
            self.image_callback1,
            10
        )
        self.subscription2 = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback2,
            10
        )
        self.depth_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/depth/camera_info',
            self.depth_camera_info_callback,
            10
        )
        self.color_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/color/camera_info',
            self.color_camera_info_callback,
            10
        )

        # VideoWriter placeholders
        self.out1 = None  # For depth stream
        self.out2 = None  # For color stream

        # Set fixed frame rate
        self.frame_rate = 30.0

        # Initialize intrinsic matrices and distortion coefficients for depth and color
        self.K_depth = None
        self.D_depth = None
        self.K_color = None
        self.D_color = None

        # Frame dimensions for depth and color
        self.frame_width_depth = None
        self.frame_height_depth = None
        self.frame_width_color = None
        self.frame_height_color = None

    def depth_camera_info_callback(self, msg):
        if self.K_depth is None:
            self.K_depth = np.array(msg.k, dtype=np.float32).reshape((3, 3))  # Intrinsic matrix for depth
            self.D_depth = np.array(msg.d[:5], dtype=np.float32)               # Distortion coefficients for depth
            self.frame_width_depth = int(msg.width)
            self.frame_height_depth = int(msg.height)
            self.get_logger().info(f'Depth Intrinsic Matrix (K_depth):\n{self.K_depth}')
            self.get_logger().info(f'Depth Distortion Coefficients (D_depth):\n{self.D_depth}')

    def color_camera_info_callback(self, msg):
        if self.K_color is None:
            self.K_color = np.array(msg.k, dtype=np.float32).reshape((3, 3))  # Intrinsic matrix for color
            self.D_color = np.array(msg.d[:5], dtype=np.float32)               # Distortion coefficients for color
            self.frame_width_color = int(msg.width)
            self.frame_height_color = int(msg.height)
            self.get_logger().info(f'Color Intrinsic Matrix (K_color):\n{self.K_color}')
            self.get_logger().info(f'Color Distortion Coefficients (D_color):\n{self.D_color}')

    def initialize_video_writer(self, stream_name, width, height):
        width, height = int(width), int(height)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        filename = f'{self.bag_name}_{stream_name}_stream.mp4'
        video_writer = cv2.VideoWriter(filename, fourcc, self.frame_rate, (width, height))
        self.get_logger().info(f'Started recording {stream_name} stream to {filename} at {self.frame_rate} FPS with resolution {width}x{height}.')
        return video_writer

    def image_callback1(self, msg):
        try:
            depth_bgr = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            if self.K_depth is not None and self.D_depth is not None:
                depth_bgr = cv2.undistort(depth_bgr, self.K_depth, self.D_depth)

            if self.out1 is None:
                self.out1 = self.initialize_video_writer("Depth", self.frame_width_depth, self.frame_height_depth)

            self.out1.write(depth_bgr)

        except Exception as e:
            self.get_logger().error(f'Error processing depth image: {e}')

    def image_callback2(self, msg):
        try:
            # Ensure frame dimensions are set before processing
            if self.frame_width_color is None or self.frame_height_color is None:
                self.get_logger().warn('Color frame dimensions are not set. Waiting for camera info.')
                return

            color_frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            if self.K_color is not None and self.D_color is not None:
                color_frame = cv2.undistort(color_frame, self.K_color, self.D_color)

            if self.out2 is None:
                self.out2 = self.initialize_video_writer("Color", self.frame_width_color, self.frame_height_color)

            self.out2.write(color_frame)

        except Exception as e:
            self.get_logger().error(f'Error processing color image: {e}')

    def stop_recording(self):
        if self.out1 is not None:
            self.out1.release()
            self.get_logger().info('Stopped recording depth stream.')
        if self.out2 is not None:
            self.out2.release()
            self.get_logger().info('Stopped recording color stream.')

def start_recording(bag_path):
    bag_name = os.path.splitext(os.path.basename(bag_path))[0]
    play_command = ['ros2', 'bag', 'play', bag_path, '--rate', '1.0']
    play_process = subprocess.Popen(play_command)

    rclpy.init()
    recorder = DualStreamRecorder(bag_name)

    try:
        while rclpy.ok() and play_process.poll() is None:
            rclpy.spin_once(recorder, timeout_sec=0.1)
    except KeyboardInterrupt:
        recorder.get_logger().info('Recording interrupted by user.')
    finally:
        recorder.stop_recording()
        recorder.destroy_node()
        play_process.terminate()
        play_process.wait()

    rclpy.shutdown()

def main():
    bag_files = sys.argv[1:]
    
    if not bag_files:
        print("Usage: ros2 run vision_toolbox Bag2mp4.Converter <bag_name1> <bag_name2> ...")
        sys.exit(1)

    for bag_path in bag_files:
        print(f"Processing bag file: {bag_path}")
        start_recording(bag_path)

if __name__ == '__main__':
    main()
