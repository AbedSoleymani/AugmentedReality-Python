from load_cal_data import load_cal_data
from process_video_feed import process_video_feed

calibration_data = load_cal_data()

# cam_mat, dist_coef = calibration_data
# print(cam_mat)
# print(dist_coef)

marker_length = 2  # in centimeters
marker_size, totalMarkers = 6, 250

process_video_feed(marker_length=marker_length,
                   calibration_data=calibration_data,
                   marker_size=marker_size,
                   totalMarkers=totalMarkers)