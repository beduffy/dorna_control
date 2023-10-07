from __future__ import print_function
import os
import sys
import argparse
import time
import math
import traceback
from glob import glob

try:
    import open3d as o3d
    from skimage.measure import find_contours
except Exception as e:
    print(e)
    print('Tried to import open3d or skimage but not installed')
import requests
import pyrealsense2 as rs
from cv2 import aruco
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib as mpl
from scipy import optimize

from lib.vision import isclose, dist, isRotationMatrix, rotationMatrixToEulerAngles, create_homogenous_transformations
from lib.vision import get_full_pcd_from_rgbd, convert_cam_pcd_to_arm_pcd
from lib.vision_config import camera_matrix, dist_coeffs, pinhole_camera_intrinsic
from lib.realsense_helper import setup_start_realsense, realsense_get_frames, run_10_frames_to_wait_for_auto_exposure
from lib.aruco_helper import create_aruco_params, aruco_detect_draw_get_transforms
from lib.dorna_kinematics import i_k, f_k

# export PYTHONPATH=$PYTHONPATH:/home/ben/all_projects/dorna_control
# TODO could I put most of below common things into some function/library?
# TODO too much copied from basic_aruco_example.py

if __name__ == '__main__':

    depth_intrin, color_intrin, depth_scale, pipeline, align, spatial = setup_start_realsense()

    board, parameters, aruco_dict, marker_length = create_aruco_params()

    size = 0.1
    # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        # size=25.0, origin=[0.0, 0.0, 0.0])
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=size, origin=[0.0, 0.0, 0.0])
    # coordinate_frame_shoulder_height = o3d.geometry.TriangleMesh.create_coordinate_frame(
    #     size=25.0, origin=[0.0, 0.0, 206.01940000000002])

    cam2arm = np.identity(4)
    saved_cam2arm = cam2arm

    run_10_frames_to_wait_for_auto_exposure(pipeline, align)

    frame_count = 0
    print('Starting loop')
    while True:
        try:
            color_frame, depth_frame = realsense_get_frames(pipeline, align, spatial)
            
            camera_depth_img = np.asanyarray(depth_frame.get_data())
            camera_color_img = np.asanyarray(color_frame.get_data())
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(camera_depth_img, alpha=0.03),
                                                cv2.COLORMAP_JET)  # todo why does it look so bad, add more contrast?
            
            bgr_color_data = cv2.cvtColor(camera_color_img, cv2.COLOR_RGB2BGR)
            gray_data = cv2.cvtColor(bgr_color_data, cv2.COLOR_RGB2GRAY)

            ids, all_rvec, all_tvec = aruco_detect_draw_get_transforms(gray_data, camera_color_img, aruco_dict, parameters, marker_length, camera_matrix, dist_coeffs)

            if ids is not None:
                print(ids)
                ids_list = [l[0] for l in ids.tolist()]
                for list_idx, corner_id in enumerate(ids_list):
                    rvec_aruco, tvec_aruco = all_rvec[list_idx, 0, :], all_tvec[list_idx, 0, :]
                    # rvec_aruco, tvec_aruco = all_rvec[corner_id, 0, :], all_tvec[corner_id, 0, :]
                    # aruco.drawAxis(camera_color_img, camera_matrix, dist_coeffs, rvec_aruco, tvec_aruco, marker_length)
                    # https://stackoverflow.com/questions/72702953/attributeerror-module-cv2-aruco-has-no-attribute-drawframeaxes

                tvec, rvec = tvec_aruco, rvec_aruco  # for easier assignment if multiple markers later.
                cam2arm, arm2cam, R_tc, R_ct, pos_camera = create_homogenous_transformations(tvec, rvec)
                saved_cam2arm = cam2arm
            
            images = np.hstack((camera_color_img, depth_colormap))
            # images = np.hstack((camera_color_img, depth_colormap))
            # images = camera_color_img
            camera_color_img = cv2.cvtColor(camera_color_img, cv2.COLOR_BGR2RGB)  # for open3D

            cv2.imshow("image", images)
            k = cv2.waitKey(1)

            if k == ord('q'):
                cv2.destroyAllWindows()
                pipeline.stop()
                break

            if k == ord('o'):
                cam_pcd = get_full_pcd_from_rgbd(camera_color_img, camera_depth_img,
                                            pinhole_camera_intrinsic, visualise=False)
                full_arm_pcd, full_pcd_numpy = convert_cam_pcd_to_arm_pcd(cam_pcd, saved_cam2arm, 0.0)  # TODO do I need this or not?

                # gripper_coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                #                         size=25.0, origin=[0.0, 0.0, 0.0])
                # gripper_base_transform = get_gripper_base_transformation(joint_angles)
                # gripper_coordinate_frame.transform(gripper_base_transform)

                aruco_coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                                        size=size, origin=[0.0, 0.0, 0.0])
                # aruco_coordinate_frame.transform(cam2arm)  # didn't work. I lack understanding TODO
                aruco_coordinate_frame.transform(arm2cam)  # on the spot but rotated wrong...  ahh did it again and it looked good. it's aruco flaws... maybe this caused all my problems? I should use charuco board?

                # plot_open3d_Dorna(xyz_positions_of_all_joints, extra_geometry_elements=[full_arm_pcd, coordinate_frame, coordinate_frame_shoulder_height])
                # list_of_geometry_elements = [full_arm_pcd, coordinate_frame, coordinate_frame_shoulder_height]
                # list_of_geometry_elements = [full_arm_pcd]
                # list_of_geometry_elements = [full_arm_pcd, coordinate_frame]
                list_of_geometry_elements = [cam_pcd, coordinate_frame, aruco_coordinate_frame]
                o3d.visualization.draw_geometries(list_of_geometry_elements)
                # import pdb;pdb.set_trace()

            frame_count += 1
        except ValueError as e:
            print('Error in main loop')
            print(e)
            print(traceback.format_exc())
            print('sys exc info')
            print(sys.exc_info()[2])

            # Stop streaming
            pipeline.stop()
            cv2.destroyAllWindows()