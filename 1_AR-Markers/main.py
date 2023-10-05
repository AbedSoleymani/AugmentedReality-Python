import os
import cv2
import numpy as np

from gen_marker_img import gen_marker_img
from find_markers import find_markers
from show_markers import show_markers

os.system("clear")

marker_size = 5

gen_marker_img(marker_id=13,
               marker_size=marker_size,
               show=False,
               save=False)

frame = cv2.imread("./1_AR-Markers/imgs/{}x{}.png".format(marker_size,marker_size))

marker_corners, marker_ids = find_markers(image=frame,
                                          marker_size=marker_size,
                                          totalMarkers=50)

show_markers(frame=frame,
             marker_ids=marker_ids,
             marker_corners=marker_corners)