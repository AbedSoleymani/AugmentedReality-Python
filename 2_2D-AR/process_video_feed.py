import cv2

from find_markers import find_markers
from draw_markers import draw_markers
from super_impose import super_impose

def process_video_feed(overlay_image, marker_size=4):

    cap = cv2.VideoCapture(0)

    video_height = 480
    video_width = 640
    overlay_image = cv2.resize(overlay_image, (video_width, video_height))

    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break
        
        marker_corners, marker_ids = find_markers(image=frame,
                                                  marker_size=marker_size,
                                                  totalMarkers=50)

        if marker_ids is not None:
            frame = super_impose(frame, (marker_corners, marker_ids),
                                                    overlay_image, video_width,
                                                    video_height)
            frame = draw_markers(frame=frame,
                                 marker_ids=marker_ids,
                                 marker_corners=marker_corners)
        
        cv2.imshow('Processed Camera Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
