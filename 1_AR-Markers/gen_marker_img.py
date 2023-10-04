import cv2

def gen_marker_img(window_size=400,
                   marker_id=0,
                   show=True,
                   save=True,
                   marker_size=4, # 6
                   total_markers=250, # 250
                   ):
    
    dictionary_key = getattr(cv2.aruco,
                             f'DICT_{marker_size}X'f'{marker_size}_{total_markers}')
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_key)
    
    marker_img = cv2.aruco.generateImageMarker(aruco_dict,
                                               marker_id,
                                               window_size)
    if save:
        cv2.imwrite("./1_AR-Markers/markers/marker_{}.png".format(marker_id), marker_img)

    if show:
        cv2.imshow("Marker", marker_img)
        print("Dimensions:", marker_img.shape)
        cv2.waitKey(0)