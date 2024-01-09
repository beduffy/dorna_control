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
import open3d as o3d

from lib.vision import get_inverse_homogenous_transform

def load_all_handeye_data():
    gripper_transform_files = sorted(glob('data/handeye/gripper2base*'))
    cam2target_files = sorted(glob('data/handeye/target2cam*'))

    all_gripper_rotation_mats = []
    all_gripper_tvecs = []
    for fp in gripper_transform_files:
        gripper_transform = np.loadtxt(fp, delimiter=' ')
        print('Loaded {} transform'.format(fp))
        gripper2base_rot = gripper_transform[:3, :3]
        all_gripper_rotation_mats.append(gripper2base_rot)
        # all_gripper_tvecs.append(gripper_transform[:3, 3] * 1000.0)
        all_gripper_tvecs.append(gripper_transform[:3, 3] / 1000.0)
    R_gripper2base = np.array(all_gripper_rotation_mats)
    t_gripper2base = np.array(all_gripper_tvecs)

    all_target2cam_rotation_mats = []
    all_target2cam_tvecs = []
    for fp in cam2target_files:
        cam2target_transform = np.loadtxt(fp, delimiter=' ')

        # Just testing but not right
        # target2cam = get_inverse_homogenous_transform(cam2target_transform)
        # cam2target_transform = target2cam

        print('Loaded {} transform'.format(fp))
        cam2target_rot = cam2target_transform[:3, :3]
        all_target2cam_rotation_mats.append(cam2target_rot)
        all_target2cam_tvecs.append(cam2target_transform[:3, 3])
        # all_target2cam_tvecs.append(cam2target_transform[:3, 3] / 1000.0)  # TODO or multiply gripper coord by 1000?
    R_target2cam = np.array(all_target2cam_rotation_mats)
    t_target2cam = np.array(all_target2cam_tvecs)

    handeye_data_dict = {
        'all_gripper_rotation_mats': all_gripper_rotation_mats,
        'all_gripper_tvecs': all_gripper_tvecs,
        'R_gripper2base': R_gripper2base,
        't_gripper2base': t_gripper2base,
        'all_target2cam_rotation_mats': all_target2cam_rotation_mats,
        'all_target2cam_tvecs': all_target2cam_tvecs,
        'R_target2cam': R_target2cam,
        't_target2cam': t_target2cam,
    }
    return handeye_data_dict


def plot_all_handeye_data(handeye_data_dict, cam_pcd=None):
    all_gripper_rotation_mats = handeye_data_dict['all_gripper_rotation_mats']
    all_gripper_tvecs = handeye_data_dict['all_gripper_tvecs']
    R_gripper2base = handeye_data_dict['R_gripper2base']
    t_gripper2base = handeye_data_dict['t_gripper2base']
    all_target2cam_rotation_mats = handeye_data_dict['all_target2cam_rotation_mats']
    all_target2cam_tvecs = handeye_data_dict['all_target2cam_tvecs']
    R_target2cam = handeye_data_dict['R_target2cam']
    t_target2cam = handeye_data_dict['t_target2cam']

    geometry_to_plot = []
    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                                        size=0.2, origin=[0.0, 0.0, 0.0])
    geometry_to_plot.append(origin_frame)
    for idx, base2gripper_tvec in enumerate(all_gripper_tvecs):
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                                        # size=0.1, origin=[cam2target_tvec[0], cam2target_tvec[1], cam2target_tvec[2]])
                                        size=0.1, origin=[base2gripper_tvec[0], base2gripper_tvec[1], base2gripper_tvec[2]])
        coordinate_frame.rotate(all_gripper_rotation_mats[idx])

        # sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        # sphere.translate([cam2target_tvec[0], cam2target_tvec[1], cam2target_tvec[2]])
        # geometry_to_plot.append(sphere)
        geometry_to_plot.append(coordinate_frame)

    o3d.visualization.draw_geometries(geometry_to_plot)

    # TODO do the below for gripper poses as well, they should perfectly align rotation-wise to aruco poses
    geometry_to_plot = []
    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                                        size=0.2, origin=[0.0, 0.0, 0.0])
    # TODO ideally I'd visualise a frustum. matplotlib?
    # TODO draw mini plane of all arucos rather than coordinate frames. https://github.com/isl-org/Open3D/issues/3618
    geometry_to_plot.append(origin_frame)
    for idx, cam2target_tvec in enumerate(all_target2cam_tvecs):
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                                        # size=0.1, origin=[cam2target_tvec[0], cam2target_tvec[1], cam2target_tvec[2]])
                                        size=0.1, origin=[cam2target_tvec[0], cam2target_tvec[1], cam2target_tvec[2]])
        coordinate_frame.rotate(all_target2cam_rotation_mats[idx])

        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        # todo WHY is the sphere always a bit higher than the origin of the coordinate frame?
        sphere.translate([cam2target_tvec[0], cam2target_tvec[1], cam2target_tvec[2]])
        geometry_to_plot.append(sphere)
        geometry_to_plot.append(coordinate_frame)
        # TODO need better way to visualise? images? pointclouds? Origin should be different or bigger and point x outward?

    if cam_pcd is not None:
        geometry_to_plot.append(cam_pcd)
    o3d.visualization.draw_geometries(geometry_to_plot)


def handeye_calibrate_opencv(handeye_data_dict):
    # all_gripper_rotation_mats = handeye_data_dict['all_gripper_rotation_mats']
    # all_gripper_tvecs = handeye_data_dict['all_gripper_tvecs']
    R_gripper2base = handeye_data_dict['R_gripper2base']
    t_gripper2base = handeye_data_dict['t_gripper2base']
    # all_target2cam_rotation_mats = handeye_data_dict['all_target2cam_rotation_mats']
    # all_target2cam_tvecs = handeye_data_dict['all_target2cam_tvecs']
    R_target2cam = handeye_data_dict['R_target2cam']
    t_target2cam = handeye_data_dict['t_target2cam']

    method = cv2.CALIB_HAND_EYE_DANIILIDIS
    # TODO try others
    # method = 
    # eye-in-hand (according to default opencv2 params and weak documentation. "inputting the suitable transformations to the function" for eye-to-hand)
    # first formula has b_T_c for X so that's what comes out of function. It expects b_T_g and c_T_t so gripper2base and target2cam
    # R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, method=method)
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(R_gripper2base=R_gripper2base, 
                                                        t_gripper2base=t_gripper2base, 
                                                        R_target2cam=R_target2cam, 
                                                        t_target2cam=t_target2cam, method=method)
    # eye-to-hand
    # second formula has b_T_c for X so camera2base actually is what comes out of function. It expects g_T_b (base2gripper) and c_T_t (target2cam)
    # R_camera2base, t_camera2base = cv2.calibrateHandEye(R_base2gripper, t_base2gripper, R_target2cam, t_target2cam)
    # TODO could rename nothing and just change inputs to be right, of course I could. 

    cam2arm = np.identity(4)
    cam2arm[:3, :3] = R_cam2gripper
    cam2arm[:3, 3] = t_cam2gripper.squeeze()
    print('Saving handeye (cv2) cam2arm \n{}'.format(cam2arm))
    np.savetxt('data/handeye/latest_cv2_cam2arm.txt', cam2arm, delimiter=' ')

    # pos_camera = np.dot(-R_cam2gripper, np.matrix(t_cam2gripper).T)
    # TODO why does pos_camera seem to have the better position and similarity to my old cam2arms?
    pos_camera = np.dot(-R_cam2gripper, np.matrix(t_cam2gripper))
    cam2arm_local = np.identity(4)
    cam2arm_local[:3, :3] = R_cam2gripper.T
    # cam2arm_local[:3, :3] = R_cam2gripper
    cam2arm_local[0, 3] = pos_camera[0]
    cam2arm_local[1, 3] = pos_camera[1]
    cam2arm_local[2, 3] = pos_camera[2]
    # print('cam2arm inverse:\n{}'.format(cam2arm_local))
    print('handeye (cv2) cam2arm inverse \n{}'.format(cam2arm_local))
    # print('Saving handeye (cv2) cam2arm inverse \n{}'.format(cam2arm_local))
    # np.savetxt('data/handeye/latest_cv2_cam2arm.txt', cam2arm_local, delimiter=' ')


# Eduardo's code from here: https://forum.opencv.org/t/eye-to-hand-calibration/5690/10
# TODO maybe it's nice to just have a param for eye-to-hand like that so I can keep everything pretty similar?
def calibrate_eye_hand(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, eye_to_hand=True):
    if eye_to_hand:
        # change coordinates from gripper2base to base2gripper
        R_base2gripper, t_base2gripper = [], []
        for R, t in zip(R_gripper2base, t_gripper2base):
            R_b2g = R.T
            t_b2g = -R_b2g @ t
            R_base2gripper.append(R_b2g)
            t_base2gripper.append(t_b2g)
        
        # change parameters values
        R_gripper2base = R_base2gripper
        t_gripper2base = t_base2gripper

    # calibrate
    R, t = cv2.calibrateHandEye(
        R_gripper2base=R_gripper2base,
        t_gripper2base=t_gripper2base,
        R_target2cam=R_target2cam,
        t_target2cam=t_target2cam,
    )

    return R, t

if __name__ == '__main__':
    handeye_calibrate_opencv()