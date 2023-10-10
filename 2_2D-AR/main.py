import os
import cv2

from process_video_feed import process_video_feed

os.system("clear")

overlay_image = cv2.imread("./2_2D-AR/imgs/castle.jpg")


process_video_feed(overlay_image=overlay_image,
                   marker_size=4)