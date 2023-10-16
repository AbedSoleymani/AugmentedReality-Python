import cv2
import numpy as np

def process_video_feed(marker_length, calibration_data, marker_size=6, totalMarkers=250):

    cam_mat, dist_coef = calibration_data

    dictionary_key = getattr(cv2.aruco, f'DICT_{marker_size}X'
                                        f'{marker_size}_{totalMarkers}')

    cv2.aruco_dictionary = cv2.aruco.getPredefinedDictionary(dictionary_key)
    cv2.aruco_params = cv2.aruco.DetectorParameters()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        marker_corners, marker_IDs, _ = cv2.aruco.detectMarkers(gray_frame,
                                                                cv2.aruco_dictionary,
                                                                parameters=cv2.aruco_params)
        
        if marker_corners:
            rVec, tVec, _ = cv2.aruco.estimatePoseSingleMarkers(corners=marker_corners,
                                                                markerLength=marker_length,
                                                                cameraMatrix=cam_mat,
                                                                distCoeffs=dist_coef)
            
            total_markers = range(0, marker_IDs.size)

            for ids, corners, i in zip(marker_IDs, marker_corners, total_markers):

                cv2.polylines(frame, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv2.LINE_AA)

                corners = corners.reshape(4, 2)
                corners = corners.astype(int)
                top_right = corners[0].ravel()
                top_left = corners[1].ravel()
                bottom_right = corners[2].ravel()
                bottom_left = corners[3].ravel()

                # distance = np.sqrt(tVec[i][0][2] ** 2 + tVec[i][0][0] ** 2 + tVec[i][0][1] ** 2)
                distance = np.linalg.norm(tVec[i])
                
                # Drawing the marker pose
                point = cv2.drawFrameAxes(image=frame,
                                          cameraMatrix=cam_mat,
                                          distCoeffs=dist_coef,
                                          rvec=rVec[i],
                                          tvec=tVec[i],
                                          length=2,
                                          thickness=4)

                cv2.putText(frame,
                            f"id: {ids[0]} Dist: {round(distance, 2)}",
                            top_right,
                            cv2.FONT_HERSHEY_PLAIN,
                            1.3,
                            (0, 0, 255),
                            2,
                            cv2.LINE_AA,)
                
                cv2.putText(frame,
                            f"x:{round(tVec[i][0][0],1)} y: {round(tVec[i][0][1],1)} ",
                            bottom_right,
                            cv2.FONT_HERSHEY_PLAIN,
                            1.0,
                            (0, 0, 255),
                            2,
                            cv2.LINE_AA,)
                
                # print(ids, "  ", corners)

        cv2.imshow("Processed Camera Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()