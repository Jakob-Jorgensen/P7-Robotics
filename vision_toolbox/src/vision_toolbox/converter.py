import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import sys
import subprocess
import numpy as np
from collections import deque

class DualStreamRecorder(Node):
    def __init__(self, bag_name):
        super().__init__('dual_stream_recorder')
        self.bridge = CvBridge()
        self.bag_name = bag_name
        
        # Define folders for saving images
        self.depth_folder = os.path.join(os.getcwd(), f"{self.bag_name}_depth_images")
        self.color_folder = os.path.join(os.getcwd(), f"{self.bag_name}_color_images")

        # Create directories if they don't exist
        os.makedirs(self.depth_folder, exist_ok=True)
        os.makedirs(self.color_folder, exist_ok=True)

        # Define a default QoS profile
        qos_profile = QoSProfile(depth=50)

        # Subscribers for each stream
        self.subscription1 = self.create_subscription(
            Image,
            '/camera/camera/depth/image_rect_raw',
            self.image_callback1,
            qos_profile
        ) 
        
        self.subscription2 = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback2,
            qos_profile
        ) 

        # Frame counters for naming files
        self.depth_frame_count = 1
        self.color_frame_count = 1

        # Frame queues for synchronization
        self.depth_queue = deque()
        self.color_queue = deque()

    def sync_and_write_frames(self): 
        while self.depth_queue and self.color_queue:
            depth_msg, depth_frame = self.depth_queue[0]
            color_msg, color_frame = self.color_queue[0]

            # Sync frames within 0.05 seconds tolerance
            depth_timestamp = depth_msg.header.stamp.sec + depth_msg.header.stamp.nanosec * 1e-9
            color_timestamp = color_msg.header.stamp.sec + color_msg.header.stamp.nanosec * 1e-9

            if abs(depth_timestamp - color_timestamp) < 0.05:
                # Save the synchronized depth and color frames
                self.save_depth_frame(depth_frame)
                self.save_color_frame(color_frame)

                # Pop frames from both queues
                self.depth_queue.popleft()
                self.color_queue.popleft()
            else:
                # If frames are out of sync, discard the older frame
                if depth_timestamp < color_timestamp:
                    self.depth_queue.popleft()
                else:
                    self.color_queue.popleft()

    def save_depth_frame(self, depth_frame):
        output_filename = os.path.join(self.depth_folder, f"depth_{self.depth_frame_count:04d}.png") 
        cv2.imwrite(output_filename, depth_frame).astype(np.uint16)
        self.get_logger().info(f'Saved depth frame as {output_filename}')
        self.depth_frame_count += 1

    def save_color_frame(self, color_frame):
        output_filename = os.path.join(self.color_folder, f"color_{self.color_frame_count:04d}.png") 
        cv2.imwrite(output_filename, color_frame)
        self.get_logger().info(f'Saved color frame as {output_filename}')
        self.color_frame_count += 1

    def image_callback1(self, msg):
        try:
            # Convert depth image to 16-bit single-channel image
            depth_frame = self.bridge.imgmsg_to_cv2(msg, '16UC1')
            # Append the depth frame to the queue
            self.depth_queue.append((msg, depth_frame))
            self.sync_and_write_frames()
        except Exception as e:
            self.get_logger().error(f'Error processing depth image: {e}')

    def image_callback2(self, msg):
        try:
            # Convert color image to 8-bit 3-channel image
            color_frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            # Append the color frame to the queue
            self.color_queue.append((msg, color_frame))
            self.sync_and_write_frames()
        except Exception as e:
            self.get_logger().error(f'Error processing color image: {e}')

def start_recording(bag_path, enable_resize=True):
    bag_name = os.path.splitext(os.path.basename(bag_path))[0]

    play_command = ['ros2', 'bag', 'play', bag_path, '--rate', '0.2',"--read-ahead-queue-size", "500"]
    play_process = subprocess.Popen(play_command)

    rclpy.init()
    recorder = DualStreamRecorder(bag_name) 
    
    try:
        while rclpy.ok() and play_process.poll() is None:
            rclpy.spin_once(recorder,timeout_sec=0.000000001) 
    except KeyboardInterrupt:
        recorder.get_logger().info('Recording interrupted by user.')
    finally:
        recorder.destroy_node()
        play_process.terminate()
        play_process.wait()
        rclpy.shutdown()


def main():
    bag_files = sys.argv[1:]
    if not bag_files:
        print("Usage: ros2 run vision_toolbox Bag2png.converter  <bag_name1> <bag_name2> ... or all bags in location with *.db3 ")
        sys.exit(1)

    for bag_path in bag_files:
        print(f"Processing bag file: {bag_path}")
        start_recording(bag_path)

if __name__ == '__main__':
    main()
