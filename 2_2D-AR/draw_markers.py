import cv2
import numpy as np

def draw_markers(frame, marker_ids, marker_corners):
    for ids, corners in zip(marker_ids, marker_corners):
        cv2.polylines(img=frame,
                    pts=[corners.astype(np.int32)],
                    isClosed=True,
                    color=(0,0,255),
                    thickness=5,
                    lineType=cv2.LINE_AA)
        
        corners =corners.reshape(4, 2)
        corners = corners.astype(np.int32)
        top_right = corners[0].ravel()
        cv2.putText(img=frame,
                    text=f"id: ({ids[0]})",
                    org=top_right,
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=3,
                    thickness=4,
                    color=(200,100,0))

    return frame
