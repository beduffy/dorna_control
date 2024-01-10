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




handeye_data_dict = load_all_handeye_data()
# plot_all_handeye_data(handeye_data_dict)
handeye_calibrate_opencv(handeye_data_dict)

# TODO why load from file again, why not just return from function?
cam2arm = np.loadtxt('data/handeye/latest_cv2_cam2arm.txt', delimiter=' ')
saved_cam2arm = cam2arm

# TODO what the hell am I doing, of course saved cam2arm is fucked up. The only way to is to use cam_pcd
# TODO should save camera image
# cam_pcd = get_full_pcd_from_rgbd(camera_color_img, camera_depth_img, pinhole_camera_intrinsic, visualise=False)
# full_arm_pcd, full_pcd_numpy = convert_cam_pcd_to_arm_pcd(cam_pcd, saved_cam2arm, in_milimetres=False)

# plot_all_handeye_data(handeye_data_dict, cam_pcd=full_arm_pcd)
plot_all_handeye_data(handeye_data_dict, cam_pcd=cam_pcd)