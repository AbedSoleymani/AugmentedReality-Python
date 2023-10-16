import pickle

def load_cal_data():
    with open("./4_3D-Pose-Distance/Cal-data/cameraMatrix.pkl", 'rb') as f:
        cam_mat = pickle.load(f)
    with open("./4_3D-Pose-Distance/Cal-data/dist.pkl", 'rb') as f:
        dist_coef = pickle.load(f)

    return cam_mat, dist_coef