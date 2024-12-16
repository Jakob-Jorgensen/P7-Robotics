import numpy as np
import cv2 as cv
import glob
import pickle

################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################

chessboardSize = (8, 6)
frameSize = (404, 240)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 1, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)

size_of_chessboard_squares_mm = 20
objp = objp * size_of_chessboard_squares_mm

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

images = glob.glob(r"C:\Users\simao\Documents\aau\1ST SEMESTER\project\camera cal\CameraCalibration\images\Checkerboard\chekerboard_0_color_images50\*.png")

for image in images:
    img = cv.imread(image)
    if img is None:
        print(f"Error loading image: {image}")
        continue

    # Convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners with additional flags
    ret, corners = cv.findChessboardCorners(
        gray, 
        chessboardSize, 
        cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK
    )
    print(f"Corners found in {image}: {ret}")

    # Visual debugging to display each image with corners
    cv.imshow('Gray Image', gray)
    if ret:
        # Refine and save corner positions
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
    cv.imshow('Corners', img)
    cv.waitKey(1000)

cv.destroyAllWindows()

############## CALIBRATION #######################################################

# Ensure we have detected corners in at least some images
if len(objpoints) == 0 or len(imgpoints) == 0:
    print("No corners were detected in any image. Check chessboard pattern or camera calibration setup.")
    exit()

print('Calibrating... Obtaining the matrixes')
ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)
print('Calibration complete!')

# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
# Make sure to include the file name in the path
np.savetxt(r"C:\Users\simao\Documents\aau\1ST SEMESTER\project\camera cal\CameraCalibration\cameraMatrix.txt", cameraMatrix, fmt='%.6f')
np.savetxt(r"C:\Users\simao\Documents\aau\1ST SEMESTER\project\camera cal\CameraCalibration\dist.txt", dist, fmt='%.6f')




############## UNDISTORTION #####################################################

img = cv.imread(r'C:\Users\simao\Documents\aau\1ST SEMESTER\project\camera cal\CameraCalibration\images\img0.png')
if img is not None:
    h, w = img.shape[:2]
    newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w, h), 1, (w, h))

    # Undistort
    dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)

    # Crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    cv.imwrite('caliResult1.png', dst)

    # Undistort with Remapping
    mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w, h), 5)
    dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

    # Crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    cv.imwrite('caliResult2.png', dst)

else:
    print("Error: Sample undistortion image 'cali5.png' not found.")

# Reprojection Error
mean_error = 0

for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    mean_error += error

print("Total reprojection error: {:.4f}".format(mean_error / len(objpoints)))
extrinsics = []
max_error_threshold = 1.0  # Adjust this threshold as necessary

for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    
    if error < max_error_threshold:
        # Convert rvec to rotation matrix and form the extrinsic matrix
        R, _ = cv.Rodrigues(rvecs[i])
        extrinsic_matrix = np.hstack((R, tvecs[i]))
        extrinsics.append(extrinsic_matrix)
        print(f"Valid extrinsic matrix for image {i+1}:\n", extrinsic_matrix)
    else:
        print(f"Image {i+1} has high error ({error:.4f}), excluding from extrinsic calculation.")

# Calculate the median extrinsic matrix
if extrinsics:
    median_extrinsic = np.median(np.array(extrinsics), axis=0)
    np.savetxt(r"C:\Users\simao\Documents\aau\1ST SEMESTER\project\camera cal\CameraCalibration\median_extrinsic.txt", median_extrinsic, fmt='%.6f')
    print("Median extrinsic matrix:\n", median_extrinsic)
else:
    print("No valid extrinsic matrices were found within the error threshold.")
