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

from lib.handeye_opencv_wrapper import handeye_calibrate_opencv, \
    load_all_handeye_data, plot_all_handeye_data, test_transformations


np.set_printoptions(suppress=True)


# TODO could visualise color (+ maybe depth) images. no need since we have pointcloud. remove TODO

# eye_in_hand=False
# folder_name = 'handeye_jan_10th'
# folder_name = '24_02_2024_14_13_53'
# folder_name = '24_02_2024_14_19_13'
# folder_name = '24_02_2024_14_35_34'
# folder_name = '29_02_2024_16_30_07'  # finally moved real dorna 4 times only base angle with 12 aruco on cardboard
# folder_name = '31_03_2024_17_34_45'  # 24 transforms, stronger flatter mini pupper cardboard, more ducttape
# overall goal is that I really I just want cam2arm to be similar in position to one aruco to camera?

eye_in_hand=True
folder_name = '27_10_2024_13_59_14'  # first time realsense on gripper servo, so need to change all opencv stuff
folder_name = '27_10_2024_15_50_00'  # some base rotation, then rotated wrist, then pitched wrist and more shoulder stuff
folder_name = '27_10_2024_16_38_10'  # 10 images


handeye_data_dict = load_all_handeye_data(folder_name)

test_transformations(handeye_data_dict)

# handeye_calibrate_opencv(handeye_data_dict, folder_name, eye_in_hand=eye_in_hand)

# manually specified eye-in-hand transform instead of the above
manually_measured_transform = np.eye(4)
# T[:3, :3] = R
manually_measured_transform[:3, 3] = [0.0, 0.03, 0.05]
handeye_data_dict['saved_cam2arm'] = manually_measured_transform
print('manually measured cam2gripper \n', handeye_data_dict['saved_cam2arm'])

# plotting three different things in here: 
# 1. gripper frames in arm frame
# 2. aruco frames in camera frame
# 3. transformation of things but problems, unclear TODO
plot_all_handeye_data(handeye_data_dict, eye_in_hand=eye_in_hand)