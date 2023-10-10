import pickle
import cv2


with open("./3_Camera-Calibration/results/cameraMatrix.pkl", 'rb') as f:
    cameraMatrix = pickle.load(f)
with open("./3_Camera-Calibration/results/dist.pkl", 'rb') as f:
    dist = pickle.load(f)
with open("./3_Camera-Calibration/results/rvecs.pkl", 'rb') as f:
    rvecs = pickle.load(f)
with open("./3_Camera-Calibration/results/tvecs.pkl", 'rb') as f:
    tvecs = pickle.load(f)
with open("./3_Camera-Calibration/results/objpoints.pkl", 'rb') as f:
    objpoints = pickle.load(f)
with open("./3_Camera-Calibration/results/imgpoints.pkl", 'rb') as f:
    imgpoints = pickle.load(f)


img = cv2.imread("./3_Camera-Calibration/imgs/img3.png")
h,  w = img.shape[:2]
newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))

# Undistort
dst = cv2.undistort(img, cameraMatrix, dist, None, newCameraMatrix)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite("./3_Camera-Calibration/imgs/caliResult_undistort.png", dst)

# Undistort with Remapping
mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w,h), 5)
dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite("./3_Camera-Calibration/imgs/caliResult_remap.png", dst)

# Reprojection Error
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error

print( "total error: {}".format(mean_error/len(objpoints)) )