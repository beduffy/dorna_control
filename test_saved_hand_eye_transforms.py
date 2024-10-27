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


# TODO could visualise color (+ maybe depth) images. no need since we have pointcloud. remove TODO

# folder_name = 'handeye_jan_10th'
# folder_name = '24_02_2024_14_13_53'
# folder_name = '24_02_2024_14_19_13'
# folder_name = '24_02_2024_14_35_34'
# folder_name = '29_02_2024_16_30_07'  # finally moved real dorna 4 times only base angle with 12 aruco on cardboard
# folder_name = '31_03_2024_17_34_45'  # 24 transforms, stronger flatter mini pupper cardboard, more ducttape
# overall goal is that I really I just want cam2arm to be similar in position to one aruco to camera?

folder_name = '27_10_2024_13_59_14'  # first time realsense on gripper servo, so need to change all opencv stuff

handeye_data_dict = load_all_handeye_data(folder_name)
handeye_calibrate_opencv(handeye_data_dict, folder_name)

# import pdb;pdb.set_trace()
# plotting two different things in here: gripper frames in arm frame + aruco frames in camera frame
plot_all_handeye_data(handeye_data_dict)