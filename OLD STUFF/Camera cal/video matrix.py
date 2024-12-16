import cv2 as cv
import numpy as np
import glob
import pickle
import os

# Load the camera matrix, distortion coefficients, and median extrinsic matrix
cameraMatrix = pickle.load(open(r"C:\Users\simao\Documents\aau\1ST SEMESTER\project\camera cal\CameraCalibration\cameraMatrix.pkl", "rb"))
dist = pickle.load(open(r"C:\Users\simao\Documents\aau\1ST SEMESTER\project\camera cal\CameraCalibration\dist.pkl", "rb"))
median_extrinsic = pickle.load(open(r"C:\Users\simao\Documents\aau\1ST SEMESTER\project\camera cal\CameraCalibration\median_extrinsic.pkl", "rb"))

# Directory containing input frames and for output
input_frames_dir = r'C:\Users\simao\Documents\aau\1ST SEMESTER\project\camera cal\input_frames'
output_dir = r'C:\Users\simao\Documents\aau\1ST SEMESTER\project\camera cal\output_frames'

# Create the output directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to apply extrinsic matrix transformation to a frame
def apply_extrinsic(frame, extrinsic_matrix):
    h, w = frame.shape[:2]

    # Generate dummy 3D points based on the image size
    corners_3d = np.array([
        [0, 0, 0, 1],
        [w, 0, 0, 1],
        [w, h, 0, 1],
        [0, h, 0, 1]
    ])

    # Transform 3D points using the extrinsic matrix
    transformed_corners = (extrinsic_matrix @ corners_3d.T).T

    # Convert to 2D by dividing by the z-coordinate
    transformed_2d = transformed_corners[:, :2] / transformed_corners[:, 2][:, np.newaxis]

    # Overlay transformed points onto the frame
    for point in transformed_2d.astype(int):
        cv.circle(frame, tuple(point), 5, (0, 255, 0), -1)

    return frame

# Process each frame
frame_files = glob.glob(os.path.join(input_frames_dir, '*.jpg'))  # Adjust if frames are in another format
for frame_file in frame_files:
    frame = cv.imread(frame_file)
    if frame is None:
        continue

    # Apply the extrinsic matrix transformation
    transformed_frame = apply_extrinsic(frame, median_extrinsic)

    # Save the transformed frame to the output directory
    frame_name = os.path.basename(frame_file)
    output_path = os.path.join(output_dir, f"transformed_{frame_name}")
    cv.imwrite(output_path, transformed_frame)


cv.destroyAllWindows()
