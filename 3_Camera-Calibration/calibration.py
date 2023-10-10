import numpy as np
import cv2
import glob
import pickle



chessboardSize = (9,6)
frameSize = (640,480)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

size_of_chessboard_squares_mm = 20
objp = objp * size_of_chessboard_squares_mm

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


images = glob.glob("./3_Camera-Calibration/imgs/*.png")

for image in images:

    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, chessboardSize, None)

    # If found, add object points, image points (after refining them)
    if ret == True:

        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, chessboardSize, corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(1000)


cv2.destroyAllWindows()

ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

pickle.dump((cameraMatrix, dist), open( "./3_Camera-Calibration/results/calibration.pkl", "wb" ))
pickle.dump(cameraMatrix, open( "./3_Camera-Calibration/results/cameraMatrix.pkl", "wb" ))
pickle.dump(dist, open( "./3_Camera-Calibration/results/dist.pkl", "wb" ))
pickle.dump(rvecs, open( "./3_Camera-Calibration/results/rvecs.pkl", "wb" ))
pickle.dump(tvecs, open( "./3_Camera-Calibration/results/tvecs.pkl", "wb" ))
pickle.dump(objpoints, open( "./3_Camera-Calibration/results/objpoints.pkl", "wb" ))
pickle.dump(imgpoints, open( "./3_Camera-Calibration/results/imgpoints.pkl", "wb" ))

# Saving in 4_3D-Pose-Distance estimation folder
pickle.dump(cameraMatrix, open( "./4_3D-Pose-Distance/Cal-data/cameraMatrix.pkl", "wb" ))
pickle.dump(dist, open( "./4_3D-Pose-Distance/Cal-data/dist.pkl", "wb" ))
