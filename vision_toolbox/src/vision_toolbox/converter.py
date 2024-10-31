import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import sys
import os
import subprocess
import numpy as np
from collections import deque


class DualStreamRecorder(Node):
    def __init__(self, bag_name):
        super().__init__('dual_stream_recorder')
        self.bridge = CvBridge()
        self.bag_name = bag_name
        self.resize_dim = (640, 480)  # Set resize dimensions

        # Define a default QoS profile
        qos_profile = QoSProfile(depth=10)

        # Subscribers for each stream and camera info topics
        self.subscription1 = self.create_subscription(
            Image,
            '/camera/depth/image_rect_raw',
            self.image_callback1,
            qos_profile
        )
        self.subscription2 = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback2,
            qos_profile
        )
        self.depth_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/depth/camera_info',
            self.depth_camera_info_callback,
            qos_profile
        )
        self.color_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/color/camera_info',
            self.color_camera_info_callback,
            qos_profile
        )

        # VideoWriter placeholders for raw streams
        self.raw_depth_writer = None
        self.raw_color_writer = None

        # Flags and intrinsic parameters for undistortion
        self.depth_info_ready = False
        self.color_info_ready = False
        self.K_depth = self.D_depth = None
        self.K_color = self.D_color = None

        # Frame queues for synchronization
        self.depth_queue = deque()
        self.color_queue = deque()

        # Temporary storage paths
        self.raw_depth_path = f'{self.bag_name}_raw_depth_stream.mp4'
        self.raw_color_path = f'{self.bag_name}_raw_color_stream.mp4'

    def depth_camera_info_callback(self, msg):
        if not self.depth_info_ready:
            self.K_depth = np.array(msg.k, dtype=np.float32).reshape((3, 3))
            self.D_depth = np.array(msg.d[:5], dtype=np.float32)
            self.depth_info_ready = True
            self.get_logger().info('Depth Camera Info ready') 


    def color_camera_info_callback(self, msg):
        if not self.color_info_ready:
            self.K_color = np.array(msg.k, dtype=np.float32).reshape((3, 3))
            self.D_color = np.array(msg.d[:5], dtype=np.float32)
            self.color_info_ready = True
            self.get_logger().info('Color Camera Info ready')

    def initialize_raw_writer(self, path, width, height):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        return cv2.VideoWriter(path, fourcc, 30.0, (width, height))

    def get_timestamp_seconds(self, msg):
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def sync_and_write_frames(self): 
        
        while self.depth_queue and self.color_queue:
            depth_msg, depth_frame = self.depth_queue[0]
            color_msg, color_frame = self.color_queue[0]

            # Get timestamps in seconds
            depth_timestamp = self.get_timestamp_seconds(depth_msg)
            color_timestamp = self.get_timestamp_seconds(color_msg)

            # Sync frames within 0.05 seconds tolerance
            if abs(depth_timestamp - color_timestamp) < 0.05:
                # Initialize VideoWriters only once
                if self.raw_depth_writer is None: 
                    self.get_logger().info(f'Initializing Depth Stream VideoWriter with resolution {depth_frame.shape[1]}x{depth_frame.shape[0]}.')
                    self.raw_depth_writer = self.initialize_raw_writer(self.raw_depth_path, depth_frame.shape[1], depth_frame.shape[0]) 
                    
                if self.raw_color_writer is None: 
                    self.get_logger().info(f'Initializing Color Stream VideoWriter with resolution {color_frame.shape[1]}x{color_frame.shape[0]}.')
                    self.raw_color_writer = self.initialize_raw_writer(self.raw_color_path, color_frame.shape[1], color_frame.shape[0])

                # Write synchronized frames
                self.raw_depth_writer.write(depth_frame)
                self.raw_color_writer.write(color_frame)
                

                # Pop frames from both queues
                self.depth_queue.popleft()
                self.color_queue.popleft()
            else:
                # If frames are out of sync, discard the older frame
                if depth_timestamp < color_timestamp:
                    self.depth_queue.popleft()
                else:
                    self.color_queue.popleft()

    def image_callback1(self, msg):
        try:
            depth_frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.depth_queue.append((msg, depth_frame)) 
            
            self.sync_and_write_frames()
        except Exception as e:
            self.get_logger().error(f'Error processing depth image: {e}')

    def image_callback2(self, msg):
        try:
            color_frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.color_queue.append((msg, color_frame)) 
            self.sync_and_write_frames()
        except Exception as e:
            self.get_logger().error(f'Error processing color image: {e}')

    def stop_recording(self):
        if self.raw_depth_writer:
            self.raw_depth_writer.release()
            self.get_logger().info('Stopped raw depth recording.')

        if self.raw_color_writer:
            self.raw_color_writer.release()
            self.get_logger().info('Stopped raw color recording.')

    def undistort_and_resize(self, input_path, output_path, K, D):
        cap = cv2.VideoCapture(input_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30.0, self.resize_dim)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Apply undistortion directly
            undistorted = cv2.undistort(frame, K, D)

            # Resize to target dimensions if specified
            resized_frame = cv2.resize(undistorted, self.resize_dim)
            out.write(resized_frame)

        cap.release()
        out.release()
        self.get_logger().info(f'Processed video saved as {output_path}')

    def process_videos(self):
        # Process depth and color videos after recording
        if self.depth_info_ready:
            self.undistort_and_resize(self.raw_depth_path, f'{self.bag_name}_processed_depth_stream.mp4', self.K_depth, self.D_depth)
        if self.color_info_ready:
            self.undistort_and_resize(self.raw_color_path, f'{self.bag_name}_processed_color_stream.mp4', self.K_color, self.D_color)

        # Delete raw video files
        if os.path.exists(self.raw_depth_path):
            os.remove(self.raw_depth_path)
            self.get_logger().info(f'Deleted raw depth video: {self.raw_depth_path}')
        if os.path.exists(self.raw_color_path):
            os.remove(self.raw_color_path)
            self.get_logger().info(f'Deleted raw color video: {self.raw_color_path}')

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

    # Post-process the videos
    recorder.process_videos()
    rclpy.shutdown()

def main():
    bag_files = sys.argv[1:]
    if not bag_files:
        print("Usage: ros2 run vision_toolbox Bag2mp4.converter  <bag_name1> <bag_name2> ... or all bags in location with *.db3 ")
        sys.exit(1)

    for bag_path in bag_files:
        print(f"Processing bag file: {bag_path}")
        start_recording(bag_path)

if __name__ == '__main__':
    main()
