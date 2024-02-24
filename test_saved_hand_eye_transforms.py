from __future__ import print_function
import os
import sys
import argparse
import time
import math
import traceback
from glob import glob

import open3d as o3d
import requests
import pyrealsense2 as rs
from cv2 import aruco
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib as mpl
from scipy import optimize

from lib.vision import get_full_pcd_from_rgbd
from lib.vision import get_camera_coordinate, create_homogenous_transformations, convert_pixel_to_arm_coordinate, convert_cam_pcd_to_arm_pcd
from lib.vision import isclose, dist, isRotationMatrix, rotationMatrixToEulerAngles
from lib.vision_config import pinhole_camera_intrinsic
from lib.vision_config import camera_matrix, dist_coeffs
from lib.realsense_helper import setup_start_realsense, realsense_get_frames, run_10_frames_to_wait_for_auto_exposure
from lib.aruco_helper import create_aruco_params, aruco_detect_draw_get_transforms
from lib.handeye_opencv_wrapper import handeye_calibrate_opencv, load_all_handeye_data, plot_all_handeye_data
from lib.dorna_kinematics import i_k, f_k
from lib.open3d_plot_dorna import plot_open3d_Dorna
from lib.aruco_image_text import OpenCvArucoImageText


folder_name = 'handeye_jan_10th'
# folder_name = '24_02_2024_14_13_53'
folder_name = '24_02_2024_14_19_13'
# folder_name = '24_02_2024_14_35_34'

handeye_data_dict = load_all_handeye_data(folder_name)
print('handeye dict:\n', handeye_data_dict)
# plot_all_handeye_data(handeye_data_dict)
handeye_calibrate_opencv(handeye_data_dict, folder_name)

# TODO why load from file again, why not just return from function?
cam2arm = np.loadtxt('data/{}/latest_cv2_cam2arm.txt'.format(folder_name), delimiter=' ')
saved_cam2arm = cam2arm

if handeye_data_dict['color_images']:
    camera_color_img = handeye_data_dict['color_images'][0]
    camera_depth_img = handeye_data_dict['depth_images'][0]

cam_pcd = get_full_pcd_from_rgbd(camera_color_img, camera_depth_img, pinhole_camera_intrinsic, visualise=False)
full_arm_pcd, full_pcd_numpy = convert_cam_pcd_to_arm_pcd(cam_pcd, saved_cam2arm, in_milimetres=False)

# TODO cam_pcd points are all between these min and max. Because cv2.imwrite for depth is wrong?
'''
np.asarray(cam_pcd.points).min()
(Pdb) -6.535066973443431e-06
np.asarray(cam_pcd.points).max()
(Pdb) 1.1764705959649291e-05
'''
# ahh img = cv2.imread(fp, cv2.IMREAD_UNCHANGED) solved it 

# plot_all_handeye_data(handeye_data_dict, cam_pcd=full_arm_pcd)
plot_all_handeye_data(handeye_data_dict, cam_pcd=cam_pcd)
# plot_all_handeye_data(handeye_data_dict)

size = 0.1
origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0.0, 0.0, 0.0])

transformed_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size * 2.5, origin=[0.0, 0.0, 0.0])
transformed_frame.transform(saved_cam2arm)

all_target2cam_rotation_mats = handeye_data_dict['all_target2cam_rotation_mats']
all_target2cam_tvecs = handeye_data_dict['all_target2cam_tvecs']

geometry_to_plot = []
for idx, cam2target_tvec in enumerate(all_target2cam_tvecs):
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                                    size=0.1, origin=[cam2target_tvec[0], cam2target_tvec[1], cam2target_tvec[2]])
    coordinate_frame.rotate(all_target2cam_rotation_mats[idx])

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    # TODO WHY is the sphere always a bit higher than the origin of the coordinate frame?
    sphere.translate([cam2target_tvec[0], cam2target_tvec[1], cam2target_tvec[2]])
    geometry_to_plot.append(sphere)
    geometry_to_plot.append(coordinate_frame)

print('Visualising origin, transformed frame and spheres and coordinate frames')  # TODO what are we doing here?
# TODO rename transformed frame and understand which frame is which frame
list_of_geometry_elements = [origin_frame, transformed_frame] + geometry_to_plot
# list_of_geometry_elements = [origin_frame_transformed_from_camera_frame, camera_coordinate_frame] + arm_position_coord_frames
o3d.visualization.draw_geometries(list_of_geometry_elements)