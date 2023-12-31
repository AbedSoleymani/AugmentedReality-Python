import cv2
import numpy as np

def super_impose(video_frame,
                 aruco_markers,
                 overlay_image,
                 video_width,
                 video_height):
    
    frame_height, frame_width = video_frame.shape[:2]

    if len(aruco_markers[0]) != 0:
        for i, marker_corner in enumerate(aruco_markers[0]):
            marker_corners = marker_corner.reshape((4, 2)).astype(np.int32)

            # Draw a polygon around the marker corners
            cv2.polylines(video_frame, [marker_corners], True, (0, 255, 0), 2)

            # Add marker ID as text on the top-left corner of the marker
            cv2.putText(video_frame, str(aruco_markers[1][i]),
                        tuple(marker_corners[0]),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 2)

            # Find the homography matrix to map the overlay image onto the marker
            homography_matrix, _ = cv2.findHomography(
                np.array([[0, 0], [video_width, 0], [video_width, video_height],
                          [0, video_height]], dtype="float32"), marker_corners)

            # Warp the overlay image to align with the marker using homography matrix
            warped_image = cv2.warpPerspective(overlay_image, homography_matrix,
                                               (frame_width, frame_height))

            # Create a mask to apply the warped image only on the marker area
            mask = np.zeros((frame_height, frame_width), dtype="uint8")
            cv2.fillConvexPoly(mask, marker_corners, (255, 255, 255), cv2.LINE_AA)

            masked_warped_image = cv2.bitwise_and(warped_image, warped_image,
                                                 mask=mask)

            # Apply the inverse mask to the video frame
            masked_video_frame = cv2.bitwise_and(video_frame, video_frame,
                                                mask=cv2.bitwise_not(mask))

            # Combine the masked warped image and masked video frame
            video_frame = cv2.add(masked_warped_image, masked_video_frame)

    return video_frame