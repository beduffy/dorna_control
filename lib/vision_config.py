import os

import numpy as np

try:
    import open3d as o3d
    open3d_import_success = True
except Exception as e:
    print(e)
    print('Tried to import open3d but not installed')
    open3d_import_success = False


# globals
# with open(os.path.abspath(os.path.join(__file__, '..', 'resources', 'coco_labels_91.txt')), 'r') as f:
with open(os.path.abspath(os.path.join(__file__, '..', '..', 'resources', 'coco_labels_91.txt')), 'r') as f:
    class_names = [line.strip() for line in f.readlines()]
    class_name_to_id = {n: idx for idx, n in enumerate(class_names)}

# intrinsics
# from better calibration RMS 0.18 and mean error:  0.02500
width, height, fx, fy, ppx, ppy = (640.0, 480.0, 612.14801862, 611.29345062, 340.03640321,
                                   230.06928807)
if open3d_import_success:
    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            int(width), int(height), fx, fy, ppx, ppy)
    
# old intrinsic calibration done by me
camera_matrix = np.array([[612.14801862, 0., 340.03640321],
                            [0., 611.29345062, 230.06928807],
                            [0., 0., 1.]])
dist_coeffs = np.array(
    [1.80764862e-02, 1.09549436e+00, -3.38044260e-03, 4.04543459e-03, -4.26585263e+00])

