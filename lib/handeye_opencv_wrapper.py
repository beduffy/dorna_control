from __future__ import print_function
import os
import sys
import argparse
import time
import math
import traceback
from glob import glob

import numpy as np
import cv2

def handeye_calibrate_opencv():
    # os.listdir('data/handeye/')
    gripper_transform_files = glob('data/handeye/gripper2base*')
    cam2target_files = glob('data/handeye/target2cam*')

    # TODO make sure we load each transform in the same order

    import pdb;pdb.set_trace()
    all_gripper_rotation_mats = []
    all_gripper_tvecs = []
    for fp in gripper_transform_files:
        gripper_transform = np.loadtxt(fp, delimiter=' ')
        print('Loaded {} transform'.format(fp))
        gripper2base_rot = gripper_transform[:3, :3]
        all_gripper_rotation_mats.append(gripper2base_rot)
        all_gripper_tvecs.append(gripper_transform[:3, 3])
    R_gripper2base = np.array(all_gripper_rotation_mats)
    t_gripper2base = np.array(all_gripper_tvecs)

    all_target2cam_rotation_mats = []
    all_target2cam_tvecs = []
    for fp in cam2target_files:
        cam2target_transform = np.loadtxt(fp, delimiter=' ')
        print('Loaded {} transform'.format(fp))
        cam2target_rot = cam2target_transform[:3, :3]
        all_target2cam_rotation_mats.append(cam2target_rot)
        all_target2cam_tvecs.append(cam2target_transform[:3, 3])
    R_target2cam = np.array(all_target2cam_rotation_mats)
    t_target2cam = np.array(all_target2cam_tvecs)
    import pdb;pdb.set_trace()

    # eye-in-hand (according to default opencv2 params and weak documentation. "inputting the suitable transformations to the function" for eye-to-hand)
    # first formula has b_T_c for X so that's what comes out of function. It expects b_T_g and c_T_t so gripper2base and target2cam
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam)
    # eye-to-hand
    # second formula has b_T_c for X so camera2base actually is what comes out of function. It expects g_T_b (base2gripper) and c_T_t (target2cam)
    # R_camera2base, t_camera2base = cv2.calibrateHandEye(R_base2gripper, t_base2gripper, R_target2cam, t_target2cam)
    # TODO could rename nothing and just change inputs to be right, of course I could. 

    cam2arm = np.identity(4)
    cam2arm[:3, :3] = R_cam2gripper
    cam2arm[:3, 3] = t_cam2gripper.squeeze()
    print('Saving handeye (cv2) cam2arm {}\n'.format(cam2arm))
    np.savetxt('data/handeye/latest_cv2_cam2arm.txt', cam2arm, delimiter=' ')


if __name__ == '__main__':
    handeye_calibrate_opencv()