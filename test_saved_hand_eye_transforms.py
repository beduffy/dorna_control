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
    load_all_handeye_data, plot_all_handeye_data, test_transformations, \
        verify_calibration, optimize_cam2gripper_transform, verify_transform_chain

np.set_printoptions(suppress=True)

# eye_in_hand=False
# folder_name = 'handeye_jan_10th'
# folder_name = '24_02_2024_14_13_53'
# folder_name = '24_02_2024_14_19_13'
# folder_name = '24_02_2024_14_35_34'
# folder_name = '29_02_2024_16_30_07'  # finally moved real dorna 4 times only base angle with 12 aruco on cardboard
# folder_name = '31_03_2024_17_34_45'  # 24 transforms, stronger flatter mini pupper cardboard, more ducttape
# overall goal is that I really I just want cam2arm to be similar in position to one aruco to camera?

eye_in_hand = True
folder_name = '27_10_2024_13_59_14'  # first time realsense on gripper servo, so need to change all opencv stuff
folder_name = '27_10_2024_15_50_00'  # some base rotation, then rotated wrist, then pitched wrist and more shoulder stuff
folder_name = '27_10_2024_16_38_10'  # 10 images
folder_name = '27_10_2024_18_09_05'  # 22 images, took aruco off cardboard and taped to ground so it is flatter
folder_name = '30_10_2024_11_02_54'  # 30 images, much  more dorna movement and tried to get more variation. sometimes aruco was doing axis flip thing but that was in text not visually (i'm not drawing aruco though). Also set toolhead x to 0 since wrist pitch can and innacuracies here can compound

handeye_data_dict = load_all_handeye_data(folder_name)

# TODO could recalculate all opencv transforms, and with undistortion TODO make sure RGB and not BGR?
test_transformations(handeye_data_dict)  # TODO just moved above handeye but maybe should separate recalculate transformations with undistortion with new func


handeye_calibrate_opencv(handeye_data_dict, folder_name, eye_in_hand=eye_in_hand)

saved_cam2arm = handeye_data_dict['saved_cam2arm']

R_cam2gripper = saved_cam2arm[:3, :3]
t_cam2gripper = saved_cam2arm[:3, 3]
verify_calibration(handeye_data_dict, R_cam2gripper, t_cam2gripper)
verify_transform_chain(handeye_data_dict, saved_cam2arm)



#########################################
# # manually specified eye-in-hand transform instead of the above
# Create rotation matrix for camera orientation (z forward, x right, y down)
Rx = np.array([[0, 0, 1],
               [0, -1, 0],
               [1, 0, 0]])  # Rotate to get x right, y down, z forward

# Create rotation matrix for 90 degree yaw (rotation around z axis)
Rz = np.array([[0, -1, 0],
               [1, 0, 0], 
               [0, 0, 1]])

# R = Rx @ Ry  # Combined rotation
R = Rx @ Rz  # Apply Rx first, then rotate 90 degrees in yaw


# TODO im measuring gripper2cam but need cam2gripper, can visualisation deceive me i.e. visualisation looks good but it's wrong


manually_measured_transform = np.eye(4)
manually_measured_transform[:3, :3] = R
manually_measured_transform[:3, 3] = [0.025, 0.03, 0.05]  
# manually_measured_transform[:3, 3] = [-0.025, -0.03, -0.05]  # invert numbers
handeye_data_dict['saved_cam2arm'] = manually_measured_transform
print('manually measured cam2gripper \n', handeye_data_dict['saved_cam2arm'])
saved_cam2arm = handeye_data_dict['saved_cam2arm']

# R_cam2gripper_manual = saved_cam2arm[:3, :3]
# t_cam2gripper_manual = saved_cam2arm[:3, 3]
R_gripper2cam_manual = saved_cam2arm[:3, :3]
t_gripper2cam_manual = saved_cam2arm[:3, 3]

# Convert gripper2cam to cam2gripper
gripper2cam = np.eye(4)
gripper2cam[:3, :3] = R_gripper2cam_manual
gripper2cam[:3, 3] = t_gripper2cam_manual

# Invert to get cam2gripper
cam2gripper = np.linalg.inv(gripper2cam)
R_cam2gripper_manual = cam2gripper[:3, :3]
t_cam2gripper_manual = cam2gripper[:3, 3]
manually_measured_transform = cam2gripper
#########################################



#########################################

def verify_optimise_verify_manual_calibration(handeye_data_dict, R_cam2gripper_manual, t_cam2gripper_manual):
    verify_calibration(handeye_data_dict, R_cam2gripper_manual, t_cam2gripper_manual)
    # TODO build_transform function since im always :3, 3 in
    # TODO my manual ruler gave 2.37 norm error whereas before is was closer to 3
    # TODO could do a random grid search or optimisation or gradient descent to find those 3 numbers
    # TODO more functions above and clean whole file again

    R_optimized, t_optimized = optimize_cam2gripper_transform(handeye_data_dict, R_cam2gripper_manual, t_cam2gripper_manual)

    # Create final transform
    final_transform = np.eye(4)
    final_transform[:3, :3] = R_optimized
    final_transform[:3, 3] = t_optimized

    print("Optimization results:")
    print("Translation (metres):", t_optimized)
    print("Rotation matrix:\n", R_optimized)
    # TODO omg the below is actually lower translation error to everything else...
    verify_calibration(handeye_data_dict, R_optimized, t_optimized)
    # TODO also verify this optimised transform
    # import pdb;pdb.set_trace()
    
verify_optimise_verify_manual_calibration(handeye_data_dict, R_cam2gripper_manual, t_cam2gripper_manual)

#########################################


# plotting four different things in here: 
# 1. gripper frames in arm frame + with arm links
# 2. aruco frames in camera frame
# 3. transformation of things but problems, unclear TODO
plot_all_handeye_data(handeye_data_dict, eye_in_hand=eye_in_hand)

# TODO function to check all aruco poses with all cam_pcds
# TODO could do ICP between camera pointclouds to find true transformation of what?
# TODO could plot two cam pcds over each other but I need to transform each one based on how much we moved e.g. from ICP
# TODO could I put camera and aruco board at 90 degree angles and just sanity check my aruco transformation is 90 and then camera should be at 90 degree to gripper or something?

# TODO before the above two things, what can I visualise better to see how off I am? e.g.
'''
e.g. could I do multiple positions and fuse pointclouds somehow
e.g. could I click aruco pose in image (automatically) and what?
e.g. AX = XB. How can I optimise this for one image? 
e.g. in eye in hand, ICP of 1st transform to 2nd, is the same for gripper as it is for camera?
e.g. when arm moves, pointcloud should move with it. gripper transformation from pose 1 to 2 is similar enough to camera transformation (camera can rotate with wrist pitch)

if we are eye-in-hand, and I click a point in image (e.g. aruco center), will multiplying that point by
cam2gripper bring it to base coordinates or should it be cam2base (cam2gripper combined with gripper2base)? 

Ideally should be plotting all of this in rviz rather than Open3D
'''