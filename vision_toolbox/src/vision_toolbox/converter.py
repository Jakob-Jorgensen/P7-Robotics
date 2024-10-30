import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import sys
import os
import subprocess
import time

class DualStreamRecorder(Node):
    def __init__(self, bag_name, play_process):
        super().__init__('dual_stream_recorder')

        # Initialize OpenCV Bridge and store bag playback process
        self.bridge = CvBridge()
        self.play_process = play_process
        
        # Use the bag name as a prefix for video files
        self.bag_name = bag_name

        # Set up subscribers for each stream
        self.subscription1 = self.create_subscription(
            Image,
            '/camera/depth/image_rect_raw',  # Topic for depth stream
            self.image_callback1,
            10
        )
        self.subscription2 = self.create_subscription(
            Image,
            '/camera/color/image_raw',  # Topic for color stream
            self.image_callback2,
            10
        )

        # VideoWriter placeholders (initialized on first frame)
        self.out1 = None  # For depth stream
        self.out2 = None  # For color stream
        self.frame_rate = 30.0  # Set the frame rate to 30 FPS

    def image_callback1(self, msg):
        try:
            # Convert "depth" image to BGR format directly
            depth_bgr = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            height, width = depth_bgr.shape[:2]

            # Initialize VideoWriter for depth stream with the correct resolution and frame rate
            if self.out1 is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                depth_filename = f'{self.bag_name}_Depth_stream_bgr.mp4'
                self.out1 = cv2.VideoWriter(depth_filename, fourcc, self.frame_rate, (width, height))
                self.get_logger().info(f'Started recording depth stream to {depth_filename} with resolution {width}x{height} at {self.frame_rate} FPS.')

            # Write BGR depth frame to file
            self.out1.write(depth_bgr)

        except Exception as e:
            self.get_logger().error(f'Error processing depth image: {e}')

    def image_callback2(self, msg):
        try:
            # Convert ROS Image message to OpenCV color image in BGR format directly
            color_frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            height, width = color_frame.shape[:2]

            # Initialize VideoWriter for color stream with the correct resolution and frame rate
            if self.out2 is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                color_filename = f'{self.bag_name}_Color_stream.mp4'
                self.out2 = cv2.VideoWriter(color_filename, fourcc, self.frame_rate, (width, height))
                self.get_logger().info(f'Started recording color stream to {color_filename} with resolution {width}x{height} at {self.frame_rate} FPS.')

            # Write frame to file
            self.out2.write(color_frame)

        except Exception as e:
            self.get_logger().error(f'Error processing color image: {e}')

    def stop_recording(self):
        # Release VideoWriters when recording is done
        if self.out1 is not None:
            self.out1.release()
            self.get_logger().info('Stopped recording depth stream.')
        if self.out2 is not None:
            self.out2.release()
            self.get_logger().info('Stopped recording color stream.')

def process_bag(bag_path):
    # Get the bag file name without extension
    bag_name = os.path.splitext(os.path.basename(bag_path))[0]
    
    # Run the ros2 bag play command in a subprocess
    play_command = ['ros2', 'bag', 'play', bag_path]
    play_process = subprocess.Popen(play_command)

    # Initialize ROS 2 context and node for each bag, passing the playback process
    rclpy.init()
    recorder = DualStreamRecorder(bag_name, play_process)

    try:
        # Continuously check if bag playback is still running
        while rclpy.ok() and play_process.poll() is None:
            rclpy.spin_once(recorder, timeout_sec=0.1)
            time.sleep(0.1)  # Add a small delay to reduce CPU usage

    except KeyboardInterrupt:
        recorder.get_logger().info('Recording interrupted by user.')
    finally:
        # Stop recording and clean up
        recorder.stop_recording()
        recorder.destroy_node()

        # Ensure the bag play process is terminated if it's still running
        play_process.terminate()
        play_process.wait()

        rclpy.shutdown()

def main():
    # Get list of bag files from command-line arguments
    bag_files = sys.argv[1:]
    
    if not bag_files:
        print("Usage: ros2 run <your_package> <your_node> <bag_name1> <bag_name2> ...")
        sys.exit(1)

    # Process each bag file sequentially
    for bag_path in bag_files:
        print(f"Processing bag file: {bag_path}")
        process_bag(bag_path)

if __name__ == '__main__':
    main()
