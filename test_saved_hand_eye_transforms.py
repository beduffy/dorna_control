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


np.set_printoptions(suppress=True)


# TODO make entire file cleaner
# TODO could visualise color (+ maybe depth) images

# folder_name = 'handeye_jan_10th'
# folder_name = '24_02_2024_14_13_53'
# folder_name = '24_02_2024_14_19_13'
# folder_name = '24_02_2024_14_35_34'
# folder_name = '29_02_2024_16_30_07'  # finally moved real dorna 4 times only base angle with 12 aruco on cardboard
folder_name = '31_03_2024_17_34_45'  # 24 transforms, stronger flatter mini pupper cardboard, more ducttape
# overall goal is that I really I just want cam2arm to be similar in position to one aruco to camera?


handeye_data_dict = load_all_handeye_data(folder_name)
handeye_calibrate_opencv(handeye_data_dict, folder_name)

# TODO why load from file again, why not just return from function?
cam2arm = np.loadtxt('data/{}/latest_cv2_cam2arm.txt'.format(folder_name), delimiter=' ')
saved_cam2arm = cam2arm
handeye_data_dict['saved_cam2arm'] = saved_cam2arm

# use color and depth images to create point clouds
if handeye_data_dict['color_images']:
    camera_color_img = handeye_data_dict['color_images'][0]
    camera_depth_img = handeye_data_dict['depth_images'][0]
cam_pcd = get_full_pcd_from_rgbd(camera_color_img, camera_depth_img, pinhole_camera_intrinsic, visualise=False)
full_arm_pcd, full_pcd_numpy = convert_cam_pcd_to_arm_pcd(cam_pcd, saved_cam2arm, in_milimetres=False)

# plotting two different things in here
plot_all_handeye_data(handeye_data_dict, cam_pcd=cam_pcd)






'''
Below I am visualising origin (in camera coordinates) and the arm frame.
And pointcloud from camera transformed to arm frame... but that does not make sense?

'''

all_target2cam_rotation_mats = handeye_data_dict['all_target2cam_rotation_mats']
all_target2cam_tvecs = handeye_data_dict['all_target2cam_tvecs']

all_target2cam_transforms = []
for R, t in zip(all_target2cam_rotation_mats, all_target2cam_tvecs):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    all_target2cam_transforms.append(T)

all_gripper_rotation_mats = handeye_data_dict['all_gripper_rotation_mats']
all_gripper_tvecs = handeye_data_dict['all_gripper_tvecs']

all_gripper2base_transforms = []
for R, t in zip(all_gripper_rotation_mats, all_gripper_tvecs):
    T = np.eye(4)
    # TODO validate that this code works
    T[:3, :3] = R
    T[:3, 3] = t
    all_gripper2base_transforms.append(T)


frame_size = 0.1
sphere_size = 0.01
# Create a red sphere at the origin frame for clear identification
origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size, origin=[0.0, 0.0, 0.0])
origin_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_size, resolution=20)
origin_sphere.paint_uniform_color([1, 0, 0])  # Red
origin_sphere.translate([0.0, 0.0, 0.0])  # redundant but very clear

# Create a green sphere at the transformed frame for clear identification
transformed_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size, origin=[0.0, 0.0, 0.0])
transformed_frame.transform(saved_cam2arm)
transformed_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_size, resolution=20)
transformed_sphere.paint_uniform_color([0, 1, 0])  # Green
transformed_sphere.transform(saved_cam2arm)  # TODO what am i doing here. I assume transforming origin in camera frame, brings us to arm frame so this coordinate frame should be in base of arm

geometry_to_plot = []
# given transformed frame, now i can also plot all gripper transformations after saved_cam2_arm to see where that frame is
for idx, homo_transform in enumerate(all_gripper2base_transforms):
    # Create coordinate frame for each gripper transform
    gripper_coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                                    size=frame_size, origin=[0.0, 0.0, 0.0])
    # combined_transform = np.dot(saved_cam2arm, homo_transform)
    combined_transform = homo_transform @ saved_cam2arm
    gripper_coordinate_frame.transform(combined_transform)
    # gripper_coordinate_frame.transform(saved_cam2arm)
    # gripper_coordinate_frame.transform(homo_transform)

    # Create a sphere for each gripper transform
    gripper_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_size)
    gripper_sphere.paint_uniform_color([0, 0, 1])  # Blue for distinction
    gripper_sphere.transform(saved_cam2arm)
    gripper_sphere.transform(homo_transform)

    # Add the created geometries to the list for plotting
    geometry_to_plot.append(gripper_sphere)
    geometry_to_plot.append(gripper_coordinate_frame)


for idx, homo_transform in enumerate(all_target2cam_transforms):
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                                    size=frame_size, origin=[0.0, 0.0, 0.0])
    coordinate_frame.transform(homo_transform)

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_size)
    sphere.transform(homo_transform)
    geometry_to_plot.append(sphere)
    geometry_to_plot.append(coordinate_frame)

    # # Adding text to the plot for better identification
    # text_position = np.array(homo_transform)[0:3, 3] + np.array([0, 0, sphere_size * 2])  # Positioning text above the sphere
    # text = f"Frame {idx}"
    # text_3d = o3d.geometry.Text3D(text, position=text_position, font_size=10, density=1, font_path="OpenSans-Regular.ttf")
    # geometry_to_plot.append(text_3d)

print('Visualising origin, transformed frame and spheres and coordinate frames')  # TODO what are we doing here?
# TODO rename transformed frame and understand which frame is which frame
# list_of_geometry_elements = [origin_frame, transformed_frame, origin_sphere, transformed_sphere] + geometry_to_plot
# list_of_geometry_elements = [full_arm_pcd, origin_frame, transformed_frame, origin_sphere, transformed_sphere] + geometry_to_plot
list_of_geometry_elements = [cam_pcd, origin_frame, transformed_frame, origin_sphere, transformed_sphere] + geometry_to_plot
# list_of_geometry_elements = [origin_frame_transformed_from_camera_frame, camera_coordinate_frame] + arm_position_coord_frames
o3d.visualization.draw_geometries(list_of_geometry_elements)
