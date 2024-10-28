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
    
# TODO measure reprojection error of this. And compare why the above is so different?
# old intrinsic calibration done by me
# camera_matrix = np.array([[612.14801862, 0., 340.03640321],
#                             [0., 611.29345062, 230.06928807],
#                             [0., 0., 1.]])
# dist_coeffs = np.array(
#     [1.80764862e-02, 1.09549436e+00, -3.38044260e-03, 4.04543459e-03, -4.26585263e+00])

# october 27 2024 directly from using grippers. easured later 0.79 pixels reprojection error!!!
# camera_matrix = np.array([[639.97875221,   0.        , 319.90874536],
#        [  0.        , 479.91357711, 239.9565378 ],
#        [  0.        ,   0.        ,   1.        ]])
# dist_coeffs = np.array([[-0.00025456,  0.0004778 ,  0.00003447, -0.00002476, -0.00068842]])

# october 28th, 21 images,  mean_error: 0.21742727005639262. But maybe I don't have enough angles?
camera_matrix = np.array([[611.61332568  , 0.    ,     340.21606321],
                        [  0.      ,   610.36491859 ,239.2910932 ],
                        [  0.    ,       0.      ,     1.        ]])
dist_coeffs = np.array([ 0.09702268, -0.04537982,  0.00111311,  0.00708548, -0.5604325 ])


# directly from depth/color intrinsics from factory
# camera_matrix = np.array([[depth_intrin.fx, 0., depth_intrin.ppx],
#                           [0., depth_intrin.fy, depth_intrin.ppy],
#                           [0., 0., 1.]])

# camera_matrix = np.array([[color_intrin.fx, 0., color_intrin.ppx],
#                           [0., color_intrin.fy, color_intrin.ppy],
#                           [0., 0., 1.]])
