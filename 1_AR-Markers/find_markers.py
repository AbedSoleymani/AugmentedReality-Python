import cv2

def find_markers(image, marker_size=6, totalMarkers=250):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    dictionary_key = getattr(cv2.aruco, f'DICT_{marker_size}X'
                                        f'{marker_size}_{totalMarkers}')

    aruco_dictionary = cv2.aruco.getPredefinedDictionary(dictionary_key)

    aruco_params = cv2.aruco.DetectorParameters()

    marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray,
                                                            aruco_dictionary,
                                                            parameters=aruco_params)

    return marker_corners, marker_ids