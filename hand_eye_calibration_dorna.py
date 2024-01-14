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


# TODO seems very different to depth.intrinsics. Use depth intrinsics.....
# TODO .translate is easy for FK, what I've been doing, then the only other angles that matter is wrist pitch, wrist roll and base? pitch, roll and yaw I suppose
# TODO so I need to find the 4x4 matrix which will make .transform of coordinate frame onto gripper. And it's gripper to base or base 2 gripper?
# TODO once I have 4x4, I actually only need 3x1 or 3x3 rvec and 3x1 tvec
# TODO tvec and rvec come straight from aruco
# TODO why do we get cam2gripper and not cam2base though? But it's the gripper expressed in base coordinates soooooo base?
# TODO When estimate_pose = False I could still visualise the marker finding but only optimise and save after key a is pressed
# TODO FileNotFoundError: [Errno 2] No such file or directory: 'data/best_aruco_cam2arm.txt'
# TODO how to ensure everything is run from root directory.... Absolute paths.
# TODO how to ensure marker/found frame is flat? the world is flattened I mean. Wrong assumption because it isn;t?
# TODO could find relative pose transformations between multiple markers and then use them to create absolute ground truth instead of using a ruler
# TODO most important: make it work from 70cm away. This is perfect. Do I need 2 extra markers?
# TODO for this I might need to try different optimisers or parameters. Also need to understand PnP better
# TODO stop indenting so much, fix it with classes and stuff
# TODO im not using depth for 3D or 2D points!!!!!!?


def get_joint_angles_from_dorna_flask():
    r = requests.get('http://localhost:8080/get_xyz_joint')
    robot_data = r.json()
    joint_angles = robot_data['robot_joint_angles']

    return joint_angles

def transform_dict_of_xyz(xyz_dict, transform):

    # transforming below dict of lists with transform
    # xyz_positions_of_all_joints = {'shoulder': [shoulder_x, shoulder_y, shoulder_z], 
    #                                    'elbow': [elbow_x, elbow_y, elbow_z], 
    #                                    'wrist': [wrist_x, wrist_y, wrist_z], 
    #                                    'toolhead': xyz_toolhead_pos}

    new_dict = {}

    for key in xyz_dict.keys():
        arr = np.ones(4)  # just to make last element a 1
        arr[:3] = xyz_dict[key]
        transformed_arr = np.dot(transform, arr)
        new_dict[key] = [transformed_arr[0], transformed_arr[1], transformed_arr[2]]

    return new_dict

def get_gripper_base_transformation(joint_angles):
    full_toolhead_fk, xyz_positions_of_all_joints = f_k(joint_angles)

    # TODO are things weird because of dorna's weird angle coordinate system?
    # because dorna's j1 is measure relative to ground plane, j2 is then relative to j1, j3 relative to j2. But also inverse direction right?
    joint_angles[3] = -joint_angles[3]
    joint_angles[2] = -joint_angles[2]
    joint_angles[1] = -joint_angles[1]

    joint_angles_rad = [math.radians(j) for j in joint_angles]

    # homo_array = np.zeros((4, 4))
    homo_array = np.identity(4)
    # arm2cam_local[:3, :3] = R_ct
    homo_array[0, 3] = full_toolhead_fk[0]
    homo_array[1, 3] = full_toolhead_fk[1]
    homo_array[2, 3] = full_toolhead_fk[2]

    # TODO, wait it's the toolhead bottom which rotates, not the gripper tip, does this affect anything?
    wrist_pitch = np.sum(joint_angles_rad[1:4])
    wrist_roll = joint_angles_rad[4]
    base_yaw = joint_angles_rad[0]  # and the only way we can yaw
    # rot_mat = o3d.geometry.get_rotation_matrix_from_xyz(np.array([wrist_roll, wrist_pitch, base_yaw]))
    # rot_mat = o3d.geometry.get_rotation_matrix_from_zyx(np.array([wrist_roll, wrist_pitch, base_yaw]))
    rot_mat = o3d.geometry.get_rotation_matrix_from_zyx(np.array([base_yaw, wrist_pitch, wrist_roll]))
    # rot_mat = o3d.geometry.get_rotation_matrix_from_zyx(np.array([wrist_pitch, base_yaw, wrist_roll]))
    homo_array[:3, :3] = rot_mat

    # # -- Now get Position and attitude f the camera respect to the marker
    # # pos_camera = -R_tc * np.matrix(tvec).T  # TODO how could element-wise possibly work!?!?!?!?
    # pos_camera = np.dot(-R_tc, np.matrix(tvec_in).T)

    # # cam_position = np.dot(-arm2cam_rotation, tvec_arm2cam)  # .T   # arm2cam
    # cam2arm_local = np.identity(4)
    # cam2arm_local[:3, :3] = R_tc
    # cam2arm_local[0, 3] = pos_camera[0]
    # cam2arm_local[1, 3] = pos_camera[1]
    # cam2arm_local[2, 3] = pos_camera[2]

    return homo_array


def get_rigid_transform_error(joined_input_array, cam_3d_coords):
    # TODO make less global
    tvec_opt = joined_input_array[:3].copy()  # TODO do I need to copy these?
    rvec_opt = joined_input_array[3:].copy()

    # R_ct = np.matrix(cv2.Rodrigues(rvec_opt)[0])  # TODO confirm that rvec makes sense
    # # TODO all geometric vision, slam, pose estimation is about consistency with yourself... fundamental.
    # # TODO Can we turn everything into optimisation? study it more
    #
    # # TODO replace this with homogenous transform function given tvec and rvec?
    # arm2cam_opt = np.identity(4)
    # arm2cam_opt[:3, :3] = R_ct
    # arm2cam_opt[0, 3] = tvec_opt[0]
    # arm2cam_opt[1, 3] = tvec_opt[1]
    # arm2cam_opt[2, 3] = tvec_opt[2]
    # pos_camera = np.dot(-R_tc, np.matrix(tvec_opt).T)
    # cam2arm_opt = np.identity(4)
    # cam2arm_opt[:3, :3] = R_tc
    # cam2arm_opt[0, 3] = pos_camera[0]
    # cam2arm_opt[1, 3] = pos_camera[1]
    # cam2arm_opt[2, 3] = pos_camera[2]

    cam2arm_opt, arm2cam_opt, \
            R_tc, R_tc, pos_camera = create_homogenous_transformations(tvec_opt, rvec_opt)

    half_marker_len = marker_length / 2
    origin = np.array([0., 0., 0.])
    top_left = np.array([-half_marker_len, half_marker_len, 0.])
    top_right = np.array([half_marker_len, half_marker_len, 0.])
    bottom_right = np.array([half_marker_len, -half_marker_len, 0.])
    bottom_left = np.array([-half_marker_len, -half_marker_len, 0.])
    ground_truth = [origin, top_left, top_right, bottom_right, bottom_left]

    assert len(ground_truth) == cam_3d_coords.shape[0]

    error = 0
    for i in range(cam_3d_coords.shape[0]):
        # cam_coord_opt = np.array([x_cam_input, y_cam_input, z_cam_input, 1])
        cam_coord_opt = cam_3d_coords[i]
        arm_coord_opt = np.dot(cam2arm_opt, cam_coord_opt)[:3]

        error += dist(ground_truth[i], arm_coord_opt)

    error = error / cam_3d_coords.shape[0]

    # TODO could do above with nice big distance function across all array dimensions. Did this in interactive_click_fabrik.py for side view
    # print('tvec: {}. rvec: {}. Error: {:.5f}'.format(tvec_opt, rvec_opt, error))
    return error


def optimise_transformation_to_origin(cam_3d_coords, init_tvec, init_rvec):  # TODO first param isn't used....
    # TODO remove first param
    # TODO fix params and globals and make cleaner
    # TODO add more "world points". With 3 markers we would have 3x5=15 points and should help optimisation find correct transform from a distance
    # TODO instead of rvec axis-angle optimise quaternions or dual quaternions or something continuous

    # TODO remove most of below
    # the output of np.dot(cam2arm_opt, cam_coord) should be 0, 0, 0
    # cam2arm_opt is made from init_tvec and init_rvec which are optimised

    # global cam_coords  # so get_rigid_transform_error can access it
    #
    # cam_coords = []
    # for pixel in pixel_pos:
    #     cam_coord_spec = get_camera_coordinate(camera_depth_img, pixel[0], pixel[1])
    #     cam_coords.append(cam_coord_spec.reshape(1, 4))
    #
    # cam_coords = np.concatenate(cam_coords)

    init_tvec = init_tvec.copy()
    init_rvec = init_rvec.copy()
    # global init_rvec  # TODO just for now

    print('Starting optimisation with init_tvec: {}. init_rvec: {}'.format(init_tvec, init_rvec))

    joined_input_array = np.hstack((init_tvec, init_rvec))

    # def zero_con(t):
    #     return t[0]
    # cons = [{'type': 'eq', 'fun': zero_con}]  # TODO how do I specify the output of arm_coord z-axis to be 0 with constraints?

    # optim_result = optimize.minimize(get_rigid_transform_error, args=(init_tvec, init_rvec),  # TODO use args?
    # optim_result = optimize.minimize(get_rigid_transform_error, joined_input_array,
    optim_result = optimize.minimize(get_rigid_transform_error, joined_input_array, args=(cam_3d_coords, ),
                                     options={'max_iter': 500, 'disp': True},
                                     method='Nelder-Mead')
                                     # method='BFGS')
                                     # method='Newton-CG')
                                     # method='L-BFGS-B')
                                     # method='SLSQP')  #constraints=cons
    # TODO try other optimisers? Least squares? Should learn about all of them anyway!
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    # TODO will more points help or make it harder??!?!
    # TODO could do constrained optimsation of forcing z to 0!!!!!!

    tvec_opt = optim_result.x[:3]
    rvec_opt = optim_result.x[3:]
    print(optim_result.success)
    print(optim_result.status)
    print(optim_result.message)

    final_optimised_error = get_rigid_transform_error(np.hstack((tvec_opt, rvec_opt)), cam_3d_coords)

    return tvec_opt, rvec_opt, final_optimised_error


def click_callback(event, x, y, flags, param):
    global mouseX, mouseY
    global curr_arm_xyz, prev_arm_xyz, click_pix_coord
    global FK_gripper_tip_projected_to_image, FK_wrist_projected_to_image, FK_elbow_projected_to_image, FK_shoulder_projected_to_image
    if event == cv2.EVENT_LBUTTONUP:
        mouseX, mouseY = x, y

        click_pix_coord = (int(round(mouseX)), int(round(mouseY)))

        if saved_cam2arm is not None:
            curr_arm_xyz = convert_pixel_to_arm_coordinate(camera_depth_img, mouseX, mouseY, saved_cam2arm, verbose=True)
            if curr_arm_xyz is not None:
                curr_arm_xyz = curr_arm_xyz * 1000

                camera_coord = get_camera_coordinate(camera_depth_img, mouseX, mouseY, verbose=True)
                print('camera_coord using depth and intrinsics:', camera_coord)
                print('saved tvec:', saved_tvec)  # TODO how different is saved_tvec to inverse saved
                print('hand-eye click in arm coords: ', [x for x in curr_arm_xyz])

                joint_angles = get_joint_angles_from_dorna_flask()

                # TODO remove prints for f_k or have param
                full_toolhead_fk, xyz_positions_of_all_joints = f_k(joint_angles)
                dist_btwn_FK_and_cam2arm_transformed_click = dist(full_toolhead_fk[:3], curr_arm_xyz)

                # TODO round all prints below, too hard to read
                print('FK xyz: ', full_toolhead_fk[:3])
                print('dist_btwn_FK_and_cam2arm_transformed_click: {}'.format(dist_btwn_FK_and_cam2arm_transformed_click))
                
                # TODO do I not need arm2cam? or in other words why the hell did not doing it make such a good result? does projectPoints do its own thing? yes
                gripper_tip_fk_metric = np.array(full_toolhead_fk[:3]) / 1000
                # print('gripper_tip_fk_metric: ', gripper_tip_fk_metric)
                imagePointsFK, jacobian = cv2.projectPoints(gripper_tip_fk_metric, saved_rvec, saved_tvec, camera_matrix, dist_coeffs)
                x_FK_pix, y_FK_pix = imagePointsFK.squeeze().tolist()
                FK_gripper_tip_projected_to_image = (int(round(x_FK_pix)), int(round(y_FK_pix)))
                print('click_pix_coord: {}, FK_gripper_tip_projected_to_image: {}'.format(click_pix_coord, FK_gripper_tip_projected_to_image))

                wrist_fk_metric = np.array(xyz_positions_of_all_joints['wrist']) / 1000
                imagePointsFK, jacobian = cv2.projectPoints(wrist_fk_metric, saved_rvec, saved_tvec, camera_matrix, dist_coeffs)
                x_FK_pix, y_FK_pix = imagePointsFK.squeeze().tolist()
                FK_wrist_projected_to_image = (int(round(x_FK_pix)), int(round(y_FK_pix)))

                elbow_fk_metric = np.array(xyz_positions_of_all_joints['elbow']) / 1000
                imagePointsFK, jacobian = cv2.projectPoints(elbow_fk_metric, saved_rvec, saved_tvec, camera_matrix, dist_coeffs)
                x_FK_pix, y_FK_pix = imagePointsFK.squeeze().tolist()
                FK_elbow_projected_to_image = (int(round(x_FK_pix)), int(round(y_FK_pix)))

                shoulder_fk_metric = np.array(xyz_positions_of_all_joints['shoulder']) / 1000
                imagePointsFK, jacobian = cv2.projectPoints(shoulder_fk_metric, saved_rvec, saved_tvec, camera_matrix, dist_coeffs)
                x_FK_pix, y_FK_pix = imagePointsFK.squeeze().tolist()
                FK_shoulder_projected_to_image = (int(round(x_FK_pix)), int(round(y_FK_pix)))
            else:
                print('curr_arm_xyz is None')
        else:
            print('no saved_cam2arm. Needed for click')

    # if event == cv2.EVENT_MBUTTONUP:
    #     # cv2.circle(camera_color_img, (x, y), 100, (255, 255, 0), -1)  # TODO if I ever want this
    #     mouseX, mouseY = x, y

    #     curr_arm_xyz = convert_pixel_to_arm_coordinate(camera_depth_img, mouseX, mouseY, cam2arm, verbose=True)

    #     print('Distance to previous arm xyz 3D: {}. 2D distance: {}'.format(dist(curr_arm_xyz, prev_arm_xyz), dist(curr_arm_xyz[:2], prev_arm_xyz[:2])))
    #     prev_arm_xyz = curr_arm_xyz
    #     # TODO try except
    #     # IndexError: index 1258 is out of bounds for axis 0 with size 640


def estimate_cam2arm_on_frame(color_img, depth_img, estimate_pose=True):
    global ids, corners, all_rvec, all_tvec

    color_img = color_img.copy()
    depth_img = depth_img.copy()
    bgr_color_data = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
    gray_data = cv2.cvtColor(bgr_color_data, cv2.COLOR_RGB2GRAY)

    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray_data, aruco_dict,
                                                                parameters=parameters)
    frame_markers = aruco.drawDetectedMarkers(color_img, corners, ids)
    all_rvec, all_tvec, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)

    if all_rvec is not None:
        ids_list = [l[0] for l in ids.tolist()]

        if len(ids_list) >= 1:
            corners_reordered = []
            for corner_id in [1, 2, 3, 4]:
                # TODO what am I doing below?
                # for corner_id in [4]:  # TODO make it not crash if other aruco etc!!!
                if corner_id in ids_list:
                    corner_index = ids_list.index(corner_id) 
                    corners_reordered.append(corners[corner_index])
                    rvec_aruco, tvec_aruco = all_rvec[corner_index, 0, :], all_tvec[corner_index, 0, :]
                    # aruco.drawAxis(color_img, camera_matrix, dist_coeffs, rvec_aruco, tvec_aruco, marker_length)

        found_correct_marker = False
        if id_on_shoulder_motor in ids:
            # TODO is this even correct?!?!?! since 1 index vs 0 index?!?!? ahh because I found correct index?
            shoulder_motor_marker_id = [l[0] for l in ids.tolist()].index(id_on_shoulder_motor) 
            rvec, tvec = all_rvec[shoulder_motor_marker_id, 0, :], all_tvec[shoulder_motor_marker_id, 0, :]  # get first marker
            found_correct_marker = True
        else:
            # print('Did not find shoulder marker')
            pass
        # if found_correct_marker and rvec is not None and len(corners) == 4:  # TODO do not forget
        # if found_correct_marker and rvec is not None and len(corners) == 2:  # TODO do not forget
        if found_correct_marker and rvec is not None and len(corners) == 1:  # TODO do not forget

            # draw arm origin axis as well
            # roughly 96mm to arm and 41mm from there to centre of arm

            # with some slightly off x-axis origin
            # y_offset_for_id_1 = -0.13
            # y_offset_for_id_3 = 0.127
            # extra_y_offset_for_id_2 = -0.167
            # extra_y_offset_for_id_4 = 0.173
            # x_offset_for_id_2 = 0.2
            # x_offset_for_id_4 = -0.2

            # found with depth camera clicking. # TODO are the below the very specific marker positions relative to arm?
            y_offset_for_id_1 = -0.175  # for big marker. # 0.0765 is dorna square width, so if I measure from edge of dorna instead (easier). 
            # 0.175 - (0.0765 / 2) = 0.175 - 0.03825  = 0.13675m
            extra_y_offset_for_id_2 = -0.166
            y_offset_for_id_3 = 0.1214
            extra_y_offset_for_id_4 = 0.169
            x_offset_for_id_2 = 0.0
            x_offset_for_id_4 = 0.0



            y_offset_for_id_1 = 0
            
            # project center into image with cyan? TODO what are we projecting, name it better
            imagePoints, jacobian = cv2.projectPoints(np.array([0.0, y_offset_for_id_1, 0.0]), rvec, tvec, camera_matrix, dist_coeffs)
            x, y = imagePoints.squeeze().tolist()
            cv2.circle(color_img, (int(x), int(y)), 5, (255, 255, 0), -1)
            
            half_marker_len = marker_length / 2
            top_left_first = np.array([-half_marker_len, half_marker_len + y_offset_for_id_1, 0.])
            top_right_first = np.array([half_marker_len, half_marker_len + y_offset_for_id_1, 0.])
            bottom_right_first = np.array([half_marker_len, -half_marker_len + y_offset_for_id_1, 0.])
            bottom_left_first = np.array([-half_marker_len, -half_marker_len + y_offset_for_id_1, 0.])

            # top_left_second = np.array([-half_marker_len + x_offset_for_id_2, half_marker_len + extra_y_offset_for_id_2, 0.])
            # top_right_second = np.array([half_marker_len + x_offset_for_id_2, half_marker_len + extra_y_offset_for_id_2, 0.])
            # bottom_right_second = np.array([half_marker_len + x_offset_for_id_2, -half_marker_len + extra_y_offset_for_id_2, 0.])
            # bottom_left_second = np.array([-half_marker_len + x_offset_for_id_2, -half_marker_len + extra_y_offset_for_id_2, 0.])

            # top_left_third = np.array([-half_marker_len, half_marker_len + y_offset_for_id_3, 0.])
            # top_right_third = np.array([half_marker_len, half_marker_len + y_offset_for_id_3, 0.])
            # bottom_right_third = np.array([half_marker_len, -half_marker_len + y_offset_for_id_3, 0.])
            # bottom_left_third = np.array([-half_marker_len, -half_marker_len + y_offset_for_id_3, 0.])

            # top_left_fourth = np.array([-half_marker_len + x_offset_for_id_4, half_marker_len + extra_y_offset_for_id_4, 0.])
            # top_right_fourth = np.array([half_marker_len + x_offset_for_id_4, half_marker_len + extra_y_offset_for_id_4, 0.])
            # bottom_right_fourth = np.array([half_marker_len + x_offset_for_id_4, -half_marker_len + extra_y_offset_for_id_4, 0.])
            # bottom_left_fourth = np.array([-half_marker_len + x_offset_for_id_4, -half_marker_len + extra_y_offset_for_id_4, 0.])
            
            # the below ends up being the same as corners[shoulder_motor_marker_id]. As it should be
            # top_left_first = np.array([-half_marker_len, half_marker_len, 0.])
            # top_right_first = np.array([half_marker_len, half_marker_len, 0.])
            # bottom_right_first = np.array([half_marker_len, -half_marker_len, 0.])
            # bottom_left_first = np.array([-half_marker_len, -half_marker_len, 0.])

            # TODO make sure order is correct if top left is always first and x is forward?!?!?!
            corners_3d_points = np.array([top_left_first, top_right_first, bottom_right_first, bottom_left_first])
            # corners_3d_points = np.array([top_left_first, top_right_first, bottom_right_first, bottom_left_first, 
            #                               top_left_second, top_right_second, bottom_right_second, bottom_left_second])
            # corners_3d_points = np.array([top_left_first, top_right_first, bottom_right_first, bottom_left_first, 
            #                             top_left_second, top_right_second, bottom_right_second, bottom_left_second,
            #                             top_left_third, top_right_third, bottom_right_third, bottom_left_third,
            #                             top_left_fourth, top_right_fourth, bottom_right_fourth, bottom_left_fourth])
            imagePointsCorners, jacobian = cv2.projectPoints(corners_3d_points, rvec, tvec, camera_matrix, dist_coeffs)
            for x, y in imagePointsCorners.squeeze().tolist():
                cv2.circle(color_img, (int(x), int(y)), 5, (255, 255, 255), -1)
            # x, y = imagePointsCorners.squeeze().tolist()
            # cv2.circle(color_img, (int(x), int(y)), 5, (0, 0, 255), -1)

            # TODO give initial guess. Did it help?
            # marker_indices = [shoulder_motor_marker_id, 4]
            # outval, rvec_arm, tvec_arm = cv2.solvePnP(corners_3d_points, corners[shoulder_motor_marker_id], camera_matrix, dist_coeffs)
            # outval, rvec_arm, tvec_arm = cv2.solvePnP(corners_3d_points, corners[shoulder_motor_marker_id], camera_matrix, dist_coeffs, rvec, tvec)
            # outval, rvec_arm, tvec_arm = cv2.solvePnP(corners_3d_points, corners[marker_indices], camera_matrix, dist_coeffs, rvec, tvec)
            # if there are only two, wrong could be wrong order
            
            ids_list = [l[0] for l in ids.tolist()]

            if len(ids_list) > 1:
                corners_reordered = []
                # for corner_id in [x[0] for x in ids.tolist()]:
                for corner_id in [1, 2, 3, 4]:
                # for corner_id in [x[0] - 1 for x in ids.tolist()]:
                    corner_index = ids_list.index(corner_id) 
                    # corners_reordered.append(corners[corner_id])
                    corners_reordered.append(corners[corner_index])
                    rvec_aruco, tvec_aruco = all_rvec[corner_index, 0, :], all_tvec[corner_index, 0, :]
                    # aruco.drawAxis(color_img, camera_matrix, dist_coeffs, rvec_aruco, tvec_aruco, marker_length)
            else:
                corners_reordered = corners
                # aruco.drawAxis(color_img, camera_matrix, dist_coeffs, rvec, tvec, marker_length)

            # TODO should not be running solvePnP with only 1 point... Commenting 
            # outval, rvec_arm, tvec_arm = cv2.solvePnP(corners_3d_points, corners[shoulder_motor_marker_id], camera_matrix, dist_coeffs, rvec, tvec, useExtrinsicGuess=True)
            # rvec = rvec_arm.squeeze()
            # tvec = tvec_arm.squeeze()

            # TODO how to minimise eyeball error? glue and pencil?
            # TODO why do some angles not work well? 

            # draw circle over arm origin in camera image
            imagePoints, jacobian = cv2.projectPoints(np.array([0.0, 0.0, 0.0]), rvec, tvec, camera_matrix, dist_coeffs)
            x, y = imagePoints.squeeze().tolist()
            cv2.circle(color_img, (int(x), int(y)), 5, (0, 0, 255), -1)

            # TODO would it ever be nice to always have it projected or only on click?
            # imagePointsFK, jacobian = cv2.projectPoints(, rvec, tvec, camera_matrix, dist_coeffs)
            # x_FK_pix, y_FK_pix = imagePointsFK.squeeze().tolist()
            # FK_gripper_tip_projected_to_image = (int(round(x_FK_pix)), int(round(y_FK_pix)))
        else:
            # print('Not doing solvePnP')
            tvec, rvec = None, None
        
        # TODO ahhhhhhh more markers doesn't help the above. It'd be better for me to collect the marker positions of 4-10 markers and run into solvepnp
        # TODO as this says: https://stackoverflow.com/questions/51709522/unstable-values-in-aruco-pose-estimation
        # TODO https://github.com/opencv/opencv_contrib/blob/master/modules/aruco/src/aruco.cpp
        # https://stackoverflow.com/questions/51709522/unstable-values-in-aruco-pose-estimation
        # TODO could use cube!!!
        # https://pypi.org/project/apriltag/

        # TODO read this: https://www.learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
        """ 
        As mentioned earlier, an approximate estimate of the pose ( \mathbf{R} and \mathbf{t} ) can be found using the DLT solution. 
        A naive way to improve the DLT solution would be to randomly change the pose ( \mathbf{R} and \mathbf{t} ) slightly and 
        check if the reprojection error decreases. 
        If it does, we can accept the new estimate of the pose. 
        We can keep perturbing \mathbf{R} and \mathbf{t} again and again to find better estimates. 
        While this procedure will work, it will be very slow. 
        Turns out there are principled ways to iteratively change the values of \mathbf{R} and \mathbf{t} so that the reprojection error decreases. 
        One such method is called Levenberg-Marquardt optimization. Check out more details on Wikipedia.
        """

        # TODO smart way to weigh the aruco markers by their distance and fuse all the pose estimations.

        # if estimate_pose:  # further refine pose of specific marker by
        #     # TODO does the above get better with more markers or not?!?!?!?! it doesn't because it's individual markers

        #     found_correct_marker = False
        #     # TODO use initial guess as last all_rvec? This might help it be less jumpy?

        #     if ids is not None: # and (id_on_shoulder_motor in ids):
        #         # retval, rvec, tvec = cv2.aruco.estimatePoseBoard
        #         # -- ret = [rvec, tvec, ?]
        #         # -- array of rotation and position of each marker in camera frame
        #         # -- rvec = [[rvec_1], [rvec_2], ...]    attitude of the marker respect to camera frame
        #         # -- tvec = [[tvec_1], [tvec_2], ...]    position of the marker in camera frame

    #             if id_on_shoulder_motor in ids:
    #                 shoulder_motor_marker_id = [l[0] for l in ids.tolist()].index(id_on_shoulder_motor)
    #                 # TODO calculate tvec distance between all adjacent marker middles. Should be 0.0033!!!!!!!!!!!!!!! optimise something to make it so?
    #                 # TODO and maybe intrinsics too?

    #                 rvec, tvec = all_rvec[shoulder_motor_marker_id, 0, :], all_tvec[shoulder_motor_marker_id, 0, :]  # get first marker
    #                 found_correct_marker = True

        #         # TODO if no markers it crashes still
        #         if 'tvec' in locals():  # TODO how to avoid this?
        #             cam2arm, arm2cam, R_tc, R_ct, pos_camera = create_homogenous_transformations(tvec, rvec)

        #             # -- Get the attitude in terms of euler 321 (Needs to be flipped first)
        #             roll_marker, pitch_marker, yaw_marker = rotationMatrixToEulerAngles(
        #                 R_flip * R_tc)

        #                 if found_correct_marker:
        #                     aruco.drawAxis(color_img, camera_matrix, dist_coeffs, rvec, tvec, marker_length / 2)  # last param is axis length
        #                     # experimentation with how tvec changes everything? a bit to the right
        #                     # aruco.drawAxis(image, camera_matrix, dist_coeffs, rvec[i], tvec[i] + np.array([0.01, 0.01, 0]), marker_length)

        #                     # get all 4 corners and ensure that their xyz in arm coordinates are correct
        #                     c = corners[shoulder_motor_marker_id][0]  # should only be the shoulder motor marker id

        #                     middle_pixel = np.array([int(round(c[:, 0].mean())), int(round(c[:, 1].mean()))])
        #                     # convert_pixel_to_arm_coordinate(camera_depth_img, middle_pixel[0], middle_pixel[1], cam2arm)
        #                     # marker_xyz_middle = convert_pixel_to_arm_coordinate(camera_depth_img, middle_pixel[0], middle_pixel[1], cam2arm, verbose=True)
        #                     marker_xyz_middle = convert_pixel_to_arm_coordinate(camera_depth_img, middle_pixel[0], middle_pixel[1], cam2arm)

        #                     # TODO optimise all 4 corners and then if that still doesn't work all 4x12 corners
        #                     # TODO see how much aruco board error are
        #                     # TODO is rounding bad since everyone talks about sub-pixel accuracy? how else would i index?

        #                     # 4 corners
        #                     top_left_pixel = (int(round(c[0, 0])), int(round(c[0, 1])))
        #                     top_right_pixel = (int(round(c[1, 0])), int(round(c[1, 1])))
        #                     bottom_right_pixel = (int(round(c[2, 0])), int(round(c[2, 1])))
        #                     bottom_left_pixel = (int(round(c[3, 0])), int(round(c[3, 1])))

        #                     marker_xyz_top_left = convert_pixel_to_arm_coordinate(camera_depth_img, top_left_pixel[0], top_left_pixel[1], cam2arm)
        #                     marker_xyz_top_right = convert_pixel_to_arm_coordinate(camera_depth_img, top_right_pixel[0], top_right_pixel[1], cam2arm)
        #                     marker_xyz_bottom_right = convert_pixel_to_arm_coordinate(camera_depth_img, bottom_right_pixel[0], bottom_right_pixel[1], cam2arm)
        #                     marker_xyz_bottom_left = convert_pixel_to_arm_coordinate(camera_depth_img, bottom_left_pixel[0], bottom_left_pixel[1], cam2arm)

        #                     # print('\nMiddle marker xyz before optimisation: {}'.format(marker_xyz_middle))
        #                     # print('Top left marker xyz before optimisation: {}'.format(marker_xyz_top_left))
        #                     # print('Top Right marker xyz before optimisation: {}'.format(marker_xyz_top_right))
        #                     # print('Bottom Right marker xyz before optimisation: {}'.format(marker_xyz_bottom_right))
        #                     # print('Bottom Left marker xyz before optimisation: {}'.format(marker_xyz_bottom_left))

        #                     pixel_positions_to_optimise = [middle_pixel, top_left_pixel, top_right_pixel,
        #                                                    bottom_right_pixel, bottom_left_pixel]

        #                     joined_input_array = np.hstack((tvec, rvec))

        #                     # global cam_coords  # so get_rigid_transform_error can access it
        #                     cam_coords = []
        #                     for pixel in pixel_positions_to_optimise:
        #                         cam_coord_spec = get_camera_coordinate(camera_depth_img, pixel[0], pixel[1])
        #                         try:
        #                             cam_coords.append(cam_coord_spec.reshape(1, 4))
        #                         except Exception as e:
        #                             print(e)
        #                             print('0 z-depth value at camera coordinate. continuing loop')
        #                             continue  # TODO not a loop anymore. But returning will break everything
        #                             # return
        #                         # TODO AttributeError: 'NoneType' object has no attribute 'reshape'
        #                         # TODO probably because there is no depth at that part of the image. But it wasn't a problem before?

        #                     # TODO should see how much my rigid body transform error changes with the same cam2arm!!!!
        #                     cam_coords = np.concatenate(cam_coords)

        #                     rigid_body_error_local = get_rigid_transform_error(joined_input_array, cam_coords)
        #                     print('Rigid body transform error: {}'.format(rigid_body_error))


        # TODO i want to remove the below because I'm not using charuco but there's something about using last point and allowed difference and enforcing z
        #             else:  # TODO fix all below so charuco works as well
        #                 corner_3d_positions = []
        #                 all_dist_2ds = []
        #                 all_dist_3ds = []
        #                 # print('Calculating error for each corner')
        #                 if charuco_corners is not None:
        #                     for cor in charuco_corners:
        #                     # for cor in corners[0]:
        #                         cv2.circle(color_img, tuple(cor[0]), 4, color=(0, 255, 0))

        #                         corner_pixel = (int(round(cor[0, 0])), int(round(cor[0, 1])))
        #                         corner_xyz_arm_coord = convert_pixel_to_arm_coordinate(camera_depth_img, corner_pixel[0],
        #                                                                                corner_pixel[1], cam2arm)

        #                         if corner_xyz_arm_coord is not None:
        #                             # cv2.imshow("image", color_img)
        #                             # cv2.waitKey(1)

        #                             # TODO how to enforce z is 0? for everything...
        #                             if len(corner_3d_positions) > 0:
        #                                 allowable_y_difference = 0.0008
        #                                 allowable_y_difference = 0.0012
        #                                 # TODO TypeError: 'NoneType' object is not subscriptable happens if I cover board. Why?
        #                                 if abs(corner_3d_positions[-1][1] - corner_xyz_arm_coord[1]) > (charuco_square_length - allowable_y_difference):
        #                                     # print('Jumped row!!!')  # TODO not detecting all. Now it is since i changed to 0.0012?
        #                                     jumped_row = True
        #                                 else:
        #                                     jumped_row = False

        #                                 dist_3d_to_last_point = dist(corner_xyz_arm_coord, corner_3d_positions[-1])
        #                                 dist_2d = dist(corner_xyz_arm_coord[:2], corner_3d_positions[-1][:2])
        #                                 # print('Dist 3D {:.4f}. Dist 2D {:.4f}. Absolute error 3D: {:.4f}. Absolute error 2D: {:.4f}'.format(dist_3d_to_last_point, dist_2d, abs(dist_3d_to_last_point - distance_between_adjacent_corners), abs(dist_2d - distance_between_adjacent_corners)))
        #                                 # TODO this happened once: TypeError: unsupported operand type(s) for -: 'NoneType' and 'float'
        #                                 # if dist_2d < 0.05:  # to avoid adding cases when we swap row
        #                                 if not jumped_row:  # to avoid adding cases when we swap row
        #                                     all_dist_2ds.append(dist_2d)
        #                                     all_dist_3ds.append(dist_3d_to_last_point)
        #                                     if corner_3d_positions[-1][0] > corner_xyz_arm_coord[0]:
        #                                         pass
        #                                         # print('X to the right is smaller than the left!!!')  # TODO is it really that bad to happen 1-10% of the time?

        #                             corner_3d_positions.append(corner_xyz_arm_coord)

        #                     # print('Mean 3D absolute error: {}. Mean 2D absolute error: {}'.format(sum(all_dist_3ds) / len(all_dist_3ds),
        #                     #                                         sum(all_dist_2ds) / len(all_dist_2ds)))

        if tvec is None or rvec is None:
            # print('tvec or vec is none')
            found_correct_marker = False

        if found_correct_marker:
            # is this id1 or not. nope it isn't.
            # tvec, rvec = all_tvec[0].squeeze(), all_rvec[0].squeeze()

            # TODO typo below??? WILL IT BREAK THINGS and overwrite cam2arm or not?
            # TODO the below repeats from basic_aruco_example.py
            cam2arm, arm2cam, R_tc, R_ct, pos_camera = create_homogenous_transformations(tvec, rvec)
            # cam2arm, arm2cam, R_tc, R_ct, pos_camera = create_homogenous_transformations(tvec, rvec)

            # -- Get the attitude in terms of euler 321 (Needs to be flipped first)
            roll_marker, pitch_marker, yaw_marker = rotationMatrixToEulerAngles(opencv_aruco_image_text.R_flip * R_tc)
            # -- Get the attitude of the camera respect to the frame
            roll_camera, pitch_camera, yaw_camera = rotationMatrixToEulerAngles(opencv_aruco_image_text.R_flip * R_ct)  # todo no flip needed?

            opencv_aruco_image_text.put_marker_text(camera_color_img, tvec, roll_marker, pitch_marker, yaw_marker)
            opencv_aruco_image_text.put_camera_text(camera_color_img, pos_camera, roll_camera, pitch_camera, yaw_camera)
            # opencv_aruco_image_text.put_avg_marker_text(camera_color_img, avg_6dof_pose)
        
            # return rigid_body_error_local, color_img, depth_img, pixel_positions_to_optimise, tvec, rvec, cam_coords
            # TODO what to return? and to not make it ugly?
            # return tvec, rvec
    else:
        tvec, rvec = None, None

    return color_img, depth_img, tvec, rvec


if __name__ == '__main__':
    np.set_printoptions(precision=6, suppress=True)
    opencv_aruco_image_text = OpenCvArucoImageText()

    prev_arm_xyz = np.array([0., 0., 0.])
    cam2arm = np.identity(4)
    saved_cam2arm = None
    saved_arm2cam = None
    curr_arm_xyz = None
    # cam_coords = []
    click_pix_coord = None
    FK_gripper_tip_projected_to_image = None
    FK_wrist_projected_to_image = None
    FK_elbow_projected_to_image = None
    FK_shoulder_projected_to_image = None
    saved_rvec = None
    saved_tvec = None

    # label_3D = o3d.visualization.gui.Label3D('hey', np.array([0, 0, 0]))
    # o3d.visualization.draw_geometries([label_3D])
    # https://stackoverflow.com/questions/71959737/display-text-on-point-cloud-vertex-in-pyrender-open3d

    depth_intrin, color_intrin, depth_scale, pipeline, align, spatial = setup_start_realsense()

    shoulder_height = 206.01940000000002

    # calibration and marker params
    optimise_origin = False
    # optimise_origin = True
    # estimate_pose = True
    estimate_pose = False

    board, parameters, aruco_dict, marker_length = create_aruco_params()
    half_marker_len = marker_length / 2
    colors = ((255, 255, 255), (255, 0, 0), (0, 0, 0), (0, 0, 255))
    marker_separation = 0.006
    marker_separation = 0.0065
    ids, corners, all_rvec, all_tvec = None, None, None, None  # TODO global remove

    mouseX, mouseY = 0, 0
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', click_callback)

    check_corner_frame_count = 0
    frame_count = 1
    num_saved_handeye_transforms = 0
    marker_top_left_x_bigger_than_top_right = 0
    marker_top_left_x_bigger_than_bottom_right = 0
    marker_top_left_y_bigger_than_bottom_left = 0
    marker_top_left_y_bigger_than_bottom_right = 0

    id_on_shoulder_motor = 1

    # if estimate_pose:  # TODO think about this
    # command = '/home/beduffy/anaconda/envs/py36/bin/python control/scripts/send_arm_to_home_position.py'
    # print('Running command: {}'.format(command))
    # os.system(command)
    # time.sleep(1)

    lowest_error = 1000000
    lowest_optimised_error = 1000000

    run_10_frames_to_wait_for_auto_exposure(pipeline, align)

    print('Starting main loop')
    while True:
        try:
            color_frame, depth_frame = realsense_get_frames(pipeline, align, spatial)

            camera_depth_img = np.asanyarray(depth_frame.get_data())
            camera_color_img = np.asanyarray(color_frame.get_data())
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(camera_depth_img, alpha=0.03),
                                               cv2.COLORMAP_JET)  # TODO why does it look so bad, add more contrast?

            # rigid_body_error, color_img, depth_img, \
            #     pixel_positions_to_optimise, \
            #     tvec, rvec, cam_coords = estimate_cam2arm_on_frame(camera_color_img, camera_depth_img, estimate_pose=estimate_pose)

            color_img, depth_img, tvec, rvec = estimate_cam2arm_on_frame(camera_color_img, camera_depth_img, estimate_pose=estimate_pose)

            # TODO would optimising the transform with depth info solve most of my problems?
            # if pixel_positions_to_optimise:
            #     # middle_pixel = pixel_positions_to_optimise[0]  # TODO first is a numpy array, the rest are lists....
            #     (middle_pixel, top_left_pixel, top_right_pixel,
            #      bottom_right_pixel, bottom_left_pixel) = pixel_positions_to_optimise

            #     hit_lowest_error = False
            #     if rigid_body_error < lowest_error:  # TODO if optimise_origin works calculate lowest_error after or create 3rd cam2arm?
            #         lowest_error = rigid_body_error
            #         print('Saving best aruco cam2arm: \n{}'.format(cam2arm))
            #         print('Best error so far: {}'.format(rigid_body_error))
            #         np.savetxt('data/best_aruco_cam2arm.txt', cam2arm, delimiter=' ')
            #         hit_lowest_error = True

            #     # if optimise_origin: # TODO do this better
            #     if hit_lowest_error:
            #         # TODO making the assumption that the lowest error above will get the lowest possible error here!!!
            #         # TODO why much better error when some markers are hidden? I'm closer? I don't think other markers help? prove it
            #         # TODO do many tests!!!
            #         # TODO since arm is in home position, we could plot in open3D right here!!!!!!!!
            #         # TODO show pointcloud view with open3d (blocking vs non-blocking) to test how good all the frames are!!!!!!!!!!!!!!!!!!!!!!!
            #         # TODO could do it on q key!! or every time here. Would help me visualise coordinate frame, pose and error
            #         # TODO COULD OPTIMISE ALL MARKER CORNERS for all markers ??!?!? but we don't know exactly where they are but we could find the plane and use vectors hmmmm

            #         tvec, rvec, optimised_error = optimise_transformation_to_origin(
            #             cam_coords, tvec, rvec)

            if tvec is not None and rvec is not None:
                cam2arm, arm2cam, R_tc, R_ct, pos_camera = create_homogenous_transformations(tvec, rvec)

            # print('Rigid body transform error: {}'.format(optimised_error))

            # if optimised_error < lowest_optimised_error:
            #     lowest_optimised_error = optimised_error
            #     print('Saving best optimised aruco cam2arm with error {}\n{}'.format(
            #         lowest_optimised_error, cam2arm))
            #     np.savetxt('data/best_optimised_aruco_cam2arm.txt', cam2arm, delimiter=' ')
            #
            # marker_xyz_middle = convert_pixel_to_arm_coordinate(camera_depth_img,
            #                                                     middle_pixel[0],
            #                                                     middle_pixel[1], cam2arm,
            #                                                     verbose=True)
            #
            # # after optimisation see if everything is correct
            # marker_xyz_top_left = convert_pixel_to_arm_coordinate(camera_depth_img,
            #                                                         top_left_pixel[0],
            #                                                         top_left_pixel[1], cam2arm)
            # marker_xyz_top_right = convert_pixel_to_arm_coordinate(camera_depth_img,
            #                                                         top_right_pixel[0],
            #                                                         top_right_pixel[1], cam2arm)
            # marker_xyz_bottom_right = convert_pixel_to_arm_coordinate(camera_depth_img,
            #                                                             bottom_right_pixel[0],
            #                                                             bottom_right_pixel[1],
            #                                                             cam2arm)
            # marker_xyz_bottom_left = convert_pixel_to_arm_coordinate(camera_depth_img,
            #                                                             bottom_left_pixel[0],
            #                                                             bottom_left_pixel[1],
            #                                                             cam2arm)
            #
            # # ground truth
            # half_marker_len = marker_length / 2
            # origin = np.array([0., 0., 0.])
            # top_left = np.array([-half_marker_len, half_marker_len, 0.])
            # top_right = np.array([half_marker_len, half_marker_len, 0.])
            # bottom_right = np.array([half_marker_len, -half_marker_len, 0.])
            # bottom_left = np.array([-half_marker_len, -half_marker_len, 0.])
            # ground_truth = [origin, top_left, top_right, bottom_right, bottom_left]
            # # for gt in ground_truth:
            # #     print(gt)
            #
            # # print('Middle marker xyz after optimisation: {}'.format(marker_xyz_middle))
            # # print('Top left marker xyz after optimisation: {}'.format(marker_xyz_top_left))
            # # print('Top Right marker xyz after optimisation: {}'.format(marker_xyz_top_right))
            # # print('Bottom Right marker xyz after optimisation: {}'.format(marker_xyz_bottom_right))
            # # print('Bottom Left marker xyz after optimisation: {}'.format(marker_xyz_bottom_left))
            # # print()
            # check_corner_frame_count += 1
            #
            # # TODO marker z's seem a bit wrong? Maybe it's the depth scale by 1-10cm off
            # # will only work if we are facing it directly? Or not, because the transformation should be correct
            # # assert marker_xyz_top_left[0] < marker_xyz_top_right[0] and \
            # #        marker_xyz_top_left[0]cv2. < marker_xyz_bottom_right[0] and \
            # #        marker_xyz_top_left[1] > marker_xyz_bottom_left[1] and \
            # #        marker_xyz_top_left[1] > marker_xyz_bottom_right[1]
            #
            # # TODO not doing the below anymore but would be good to check?
            # if marker_xyz_top_left is not None and marker_xyz_top_right is not None and marker_xyz_bottom_right is not None \
            #         and marker_xyz_bottom_left is not None and not (
            #         marker_xyz_top_left[0] < marker_xyz_top_right[0] and
            #         marker_xyz_top_left[0] < marker_xyz_bottom_right[0] and
            #         marker_xyz_top_left[1] > marker_xyz_bottom_left[1] and
            #         marker_xyz_top_left[1] > marker_xyz_bottom_right[1]):
            #
            #     if not marker_xyz_top_left[0] < marker_xyz_top_right[0]:
            #         marker_top_left_x_bigger_than_top_right += 1
            #     if not marker_xyz_top_left[0] < marker_xyz_bottom_right[0]:
            #         marker_top_left_x_bigger_than_bottom_right += 1
            #     if not marker_xyz_top_left[1] > marker_xyz_bottom_left[1]:
            #         marker_top_left_y_bigger_than_bottom_left += 1
            #     if not marker_xyz_top_left[1] > marker_xyz_bottom_right[1]:
            #         marker_top_left_y_bigger_than_bottom_right += 1
            #
            #     # print('Marker xyz position wrong')
            #     # print('marker_top_left_x_bigger_than_top_right:', marker_top_left_x_bigger_than_top_right)
            #     # print('marker_top_left_x_bigger_than_bottom_right:', marker_top_left_x_bigger_than_bottom_right)
            #     # print('marker_top_left_y_bigger_than_bottom_left:', marker_top_left_y_bigger_than_bottom_left)
            #     # print('marker_top_left_y_bigger_than_bottom_right:', marker_top_left_y_bigger_than_bottom_right)
            #     # print('frame_count: {}'.format(frame_count))
            #     # TODO one time they all followed each other exactly 3 numbers were the same out of 4. the 4th was 0
            #     # TODO what to do about z flip????? Does it affect x and y? Probably not or....
            #
            # # TODO could create my own drawAxis to test the depth and once that works everything will work right?
            #
            # cv2.circle(color_img, top_left_pixel, 4, color=(255, 0, 0))
            # cv2.circle(color_img, top_right_pixel, 4, color=(0, 255, 0))
            # cv2.circle(color_img, bottom_right_pixel, 4, color=(0, 0, 255))
            # cv2.circle(color_img, bottom_left_pixel, 4, color=(0, 0, 0))
            #

            if click_pix_coord is not None:
                cv2.circle(color_img, click_pix_coord, 5, (0, 0, 255), -1)
            if FK_gripper_tip_projected_to_image is not None:
                cv2.circle(color_img, FK_gripper_tip_projected_to_image, 5, (255, 0, 0), -1)
            if FK_wrist_projected_to_image is not None:
                cv2.circle(color_img, FK_wrist_projected_to_image, 5, (0, 255, 255), -1)
                cv2.line(color_img, FK_wrist_projected_to_image, FK_gripper_tip_projected_to_image, (0, 255, 255), thickness=3)
            if FK_elbow_projected_to_image is not None:
                cv2.circle(color_img, FK_elbow_projected_to_image, 5, (0, 0, 255), -1)
                cv2.line(color_img, FK_elbow_projected_to_image, FK_wrist_projected_to_image, (0, 0, 255), thickness=3)
            if FK_shoulder_projected_to_image is not None:
                cv2.circle(color_img, FK_shoulder_projected_to_image, 5, (0, 0, 0), -1)
                cv2.line(color_img, FK_shoulder_projected_to_image, FK_elbow_projected_to_image, (0, 0, 0), thickness=3)

            images = np.hstack((color_img, depth_colormap))
            # images = np.hstack((camera_color_img, depth_colormap))  # TODO to not have writing? Have better variable images

            cv2.imshow("image", images)
            k = cv2.waitKey(1)

            camera_color_img = cv2.cvtColor(camera_color_img, cv2.COLOR_BGR2RGB)  # for open3D

            # TODO put all into handle_keyboard_input()
            if k == ord('q'):
                cv2.destroyAllWindows()
                pipeline.stop()
                break


            if k == ord('a'):
                # activate estimate_pose and optimisation pipeline
                estimate_pose = not estimate_pose
                # reset all errors since I probably want to move the camera again
                lowest_error = 1000000
                lowest_optimised_error = 1000000
                print('estimate_pose set to: {}'.format(estimate_pose))


            if k == ord('s'):
                # print('Saving best optimised aruco cam2arm with error {}\n{}'.format(
                #             lowest_optimised_error, cam2arm))
                if cam2arm is not None:
                    saved_cam2arm = cam2arm
                    saved_arm2cam = arm2cam  # TODO arm2cam does not exist?
                    saved_rvec = rvec
                    saved_tvec = tvec
                    print('Saving aruco cam2arm {}\n'.format(saved_cam2arm))
                    print('Saving aruco arm2cam {}\n'.format(saved_arm2cam))
                    np.savetxt('data/latest_aruco_cam2arm.txt', cam2arm, delimiter=' ')

                    # TODO maybe for clicking i should only used saved cam2arm rather than it changing...


            if k == ord('p'):
                if curr_arm_xyz is not None:
                    x, y, z = curr_arm_xyz
                    # if z < 10:
                    #     print('Z was {} and below 10, setting to 10'.format(z))
                    #     z = 10
                    print('Z was {} setting to 10'.format(z))
                    z = 10
                    pre_grasp_z = 200
                    # TODO prepick and then pre grasp and then grasp and then. All in other process so we see camera?
                    # TODO how to organise all this code better? ROS? functions? bleh
                    # TODO create function for go_to_xyz here too
                    # TODO open gripper before
                    # TODO do chicken head dance with gripper in same xyz but wrist pitch changing
                    print('Going to pre-pick pose')
                    wrist_pitch = -42.0
                    dorna_url = 'http://localhost:8080/go_to_xyz'
                    # dorna_full_url = '{}?x={}&y={}&z={}'.format(dorna_url, x, y, pre_grasp_z)
                    dorna_full_url = '{}?x={}&y={}&z={}&wrist_pitch={}'.format(dorna_url, x, y, pre_grasp_z, wrist_pitch)
                    r = requests.get(url=dorna_full_url)
                    print('r.status_code: {}. r.text: {}'.format(r.status_code, r.text))
                    if r.status_code == 200 and r.text == 'success':
                        print('Sleeping before pre-grasp')
                        time.sleep(6)  # TODO how to avoid sleeps in the future? Ask if dorna is ready or loops or polling? or ros? or something?
                        # dorna_full_url = '{}?x={}&y={}&z={}&wrist_pitch={}'.format(dorna_url, x, y, 20, wrist_pitch)
                        dorna_full_url = '{}?x={}&y={}&z={}&wrist_pitch={}'.format(dorna_url, x, y, z, wrist_pitch)
                        r = requests.get(url=dorna_full_url)
                        print('r.status_code: {}. r.text: {}'.format(r.status_code, r.text))
                        if r.status_code == 200 and r.text == 'success':
                            print('Sleeping before closing gripper')
                            time.sleep(4)
                            dorna_grasp_full_url = 'http://localhost:8080/gripper?gripper_state=3'  # TODO different objects and do my conversion of object width to gripper close width 
                            r = requests.get(url=dorna_grasp_full_url)
                            print('r.status_code: {}. r.text: {}'.format(r.status_code, r.text))
                            if r.status_code == 200 and r.text == 'success':
                                print('Sleeping short before going up (after pick)')
                                time.sleep(1)
                                # dorna_full_url = '{}?x={}&y={}&z={}'.format(dorna_url, x, y, 150)
                                dorna_full_url = '{}?x={}&y={}&z={}&wrist_pitch={}'.format(dorna_url, x, y, 150, wrist_pitch)
                                r = requests.get(url=dorna_full_url)
                                print('r.status_code: {}. r.text: {}'.format(r.status_code, r.text))

                            # Optional or not, let go
                            dorna_grasp_full_url = 'http://localhost:8080/gripper?gripper_state=0'  # TODO could pass state for different objects and do my conversion of object width to gripper close width 
                            r = requests.get(url=dorna_grasp_full_url)
                            print('r.status_code: {}. r.text: {}'.format(r.status_code, r.text))
                else:
                    print('curr_arm_xyz is None')


            if k == ord('i'):
                # TODO comment here on what this key does
                if curr_arm_xyz is not None:
                    x, y, z = curr_arm_xyz
                    # z = 200  # pre pick
                    # z = 20
                    wrist_pitch = 0.0  # TODO this affects everything, understand how
                    wrist_pitch = -42.0  # TODO this affects everything
                    # TODO how to dynamically change wrist pitch for different things and choose one out of many? 
                    # TODO how to prevent all collisions?
                    # TODO aruco far away problem was to do with intrinsics I bet
                    fifth_IK_value = 0.0
                    xyz_pitch_roll = [x, y, z, wrist_pitch, fifth_IK_value]
                    joint_angles = i_k(xyz_pitch_roll)
                    print('joint_angles: ', joint_angles)

                    if joint_angles:
                        full_toolhead_fk, xyz_positions_of_all_joints = f_k(joint_angles)
                        print('full_toolhead_fk: ', full_toolhead_fk)

                        cam_pcd = get_full_pcd_from_rgbd(camera_color_img, camera_depth_img,
                                                pinhole_camera_intrinsic, visualise=False)

                        # only plot open3D arm when arm is in position. Otherwise if non-blocking
                        # TODO remove q1 and z offset now that aruco and solvePnP finds correct transform
                        # TODO never understood how this relates, if origin looks good but cam2arm is bad?
                        # transforming rgbd pointcloud using bad cam2arm means what? . What is the thing changing again?

                        full_arm_pcd, full_pcd_numpy = convert_cam_pcd_to_arm_pcd(cam_pcd, saved_cam2arm)

                        plot_open3d_Dorna(xyz_positions_of_all_joints, extra_geometry_elements=[full_arm_pcd, coordinate_frame_arm_frame, coordinate_frame_shoulder_height_arm_frame])

                    else:
                        print('IK returned none')
                else:
                    print('curr_arm_xyz is None')


            if k == ord('l'):
                # Use curr joint angles and show line mesh arm plotted over pointcloud arm after all transformations (some calculated here)

                joint_angles = get_joint_angles_from_dorna_flask()
                # joint_angles = [0, 0, 0, 0, 0]  # for when dorna is off # TODO do not forget
                print('joint_angles: ', joint_angles)

                full_toolhead_fk, xyz_positions_of_all_joints = f_k(joint_angles)
                print('full_toolhead_fk: ', full_toolhead_fk)

                ids_list = [l[0] for l in ids.tolist()]
                ids_list_with_index_key_tuple = [(idx, id_) for idx, id_ in enumerate(ids_list)]
                ids_list_sorted = sorted(ids_list_with_index_key_tuple, key=lambda x: x[1])
                image_points = []
                ids_list = [x[0] for x in ids.tolist()]
                for idx_of_marker, id_of_marker in ids_list_sorted:
                    image_points.append(corners[idx_of_marker].squeeze())

                num_ids_to_draw = 12
                id_1_rvec, id_1_tvec = None, None
                for list_idx, corner_id in enumerate(ids_list[0:num_ids_to_draw]):
                    if list_idx == 0:
                        rvec_aruco, tvec_aruco = all_rvec[list_idx, 0, :], all_tvec[list_idx, 0, :]
                    if corner_id == 1:
                        id_1_rvec, id_1_tvec = all_rvec[list_idx, 0, :], all_tvec[list_idx, 0, :]

                tvec, rvec = id_1_tvec, id_1_rvec  # so I can build object points from the ground up... 

                # TODO how to use depth here?

                if id_1_tvec is not None and id_1_rvec is not None:
                    spacing = marker_length + marker_separation
                    all_obj_points_found_from_id_1 = []
                    id_count = 1
                    for y_ in range(3):
                        for x_ in range(4):
                            if id_count in ids_list:
                                top_left = np.array([-half_marker_len + x_ * spacing, half_marker_len - y_ * spacing, 0.])
                                top_right = np.array([half_marker_len + x_ * spacing, half_marker_len - y_ * spacing, 0.])
                                bottom_right = np.array([half_marker_len + x_ * spacing, -half_marker_len - y_ * spacing, 0.])
                                bottom_left = np.array([-half_marker_len + x_ * spacing, -half_marker_len - y_ * spacing, 0.])
                                # below seems about right but then ruins the projections and I could just do a 2nd transformation to ground plane instead?
                                # top_left = np.array([-half_marker_len + x_ * spacing, half_marker_len - y_ * spacing, shoulder_height / 1000.0])
                                # top_right = np.array([half_marker_len + x_ * spacing, half_marker_len - y_ * spacing, shoulder_height / 1000.0])
                                # bottom_right = np.array([half_marker_len + x_ * spacing, -half_marker_len - y_ * spacing, shoulder_height / 1000.0])
                                # bottom_left = np.array([-half_marker_len + x_ * spacing, -half_marker_len - y_ * spacing, shoulder_height / 1000.0])

                                corners_3d_points = np.array([top_left, top_right, bottom_right, bottom_left])
                                all_obj_points_found_from_id_1.append(corners_3d_points)
                                imagePointsCorners, jacobian = cv2.projectPoints(corners_3d_points, rvec, tvec, camera_matrix, dist_coeffs)
                                # for idx, (x, y) in enumerate(imagePointsCorners.squeeze().tolist()):
                                #     cv2.circle(camera_color_img, (int(x), int(y)), 4, colors[idx], -1)

                            id_count += 1
                outval, rvec_pnp_opt, tvec_pnp_opt = cv2.solvePnP(np.concatenate(all_obj_points_found_from_id_1), np.concatenate(image_points), camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE)

                # TODO better understand insides of that function and have good descriptions of cam2arm vs arm2cam.
                cam2arm_opt, arm2cam_opt, _, _, _ = create_homogenous_transformations(tvec_pnp_opt, rvec_pnp_opt)

                marker_to_arm_transformation = np.identity(4)
                measured_y_distance_to_dorna_from_marker = -0.3  # with 12 big markers on plank the the left of dorna, rather than on shoulder stepper
                measured_y_distance_to_dorna_from_marker = -0.33
                # TODO why am i off by 2 or more cm? something slightly wrong with calculation here
                marker_to_arm_transformation[1, 3] = measured_y_distance_to_dorna_from_marker * 1000.0  # TODO in m or milimetre?
                arm2cam_opt = np.dot(arm2cam_opt, marker_to_arm_transformation)  # TODO rename all vars
                

                xyz_positions_of_all_joints = transform_dict_of_xyz(xyz_positions_of_all_joints, marker_to_arm_transformation)

                saved_cam2arm = cam2arm_opt
                cam2arm_opt_milimetres = np.copy(cam2arm_opt)
                cam2arm_opt_milimetres[:3, 3] *= 1000.0
                saved_arm2cam = arm2cam_opt

                cam_pcd = get_full_pcd_from_rgbd(camera_color_img, camera_depth_img,
                                        pinhole_camera_intrinsic, visualise=False)
                if saved_cam2arm is not None:
                    full_arm_pcd, full_pcd_numpy = convert_cam_pcd_to_arm_pcd(cam_pcd, saved_cam2arm)

                    # visualise camera coordinate frame
                    # TODO broken due to arm2cam milimetres something... fix. cam2arm_opt_milimetres should do it?
                    # aruco_coordinate_frame_cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0.0, 0.0, 0.0])
                    # aruco_coordinate_frame_cam_frame.transform(saved_arm2cam)
                    # origin_frame_cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0.0, 0.0, 0.0])
                    # mesh_box_cam = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=0.003)
                    # mesh_box_cam.transform(saved_arm2cam)
                    # elements_only_in_cam_frame = [cam_pcd, aruco_coordinate_frame_cam_frame, origin_frame_cam_frame, mesh_box_cam]
                    # o3d.visualization.draw_geometries(elements_only_in_cam_frame)

                    # visualise arm coordinate frame
                    cam_frame_in_arm_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                        size=25.0, origin=[0.0, 0.0, 0.0])
                    coordinate_frame_arm_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                        size=25.0, origin=[0.0, 0.0, 0.0])
                    coordinate_frame_shoulder_height_arm_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                        size=25.0, origin=[0.0, 0.0, shoulder_height])
                    cam_frame_in_arm_frame.transform(cam2arm_opt_milimetres)
                    coordinate_frame_arm_frame.transform(marker_to_arm_transformation)
                    coordinate_frame_shoulder_height_arm_frame.transform(marker_to_arm_transformation)

                    gripper_coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=25.0, origin=[0.0, 0.0, 0.0])
                    gripper_base_transform = get_gripper_base_transformation(joint_angles)
                    gripper_coordinate_frame.transform(gripper_base_transform)

                    mesh_box_arm_frame = o3d.geometry.TriangleMesh.create_box(width=100.0, height=100.0, depth=0.003)
                    mesh_box_arm_frame.transform(marker_to_arm_transformation)
                    mesh_box_arm_frame_shoulder_height_higher = o3d.geometry.TriangleMesh.create_box(width=100.0, height=100.0, depth=0.003)
                    mesh_box_arm_frame_shoulder_height_higher.transform(marker_to_arm_transformation)
                    mesh_box_arm_frame_shoulder_height_higher.translate(np.array([0, 0, shoulder_height]))
                    
                    # extra_elements = [full_arm_pcd, mesh_box_arm_frame, gripper_coordinate_frame, 
                    extra_elements = [full_arm_pcd, mesh_box_arm_frame, mesh_box_arm_frame_shoulder_height_higher, gripper_coordinate_frame, 
                                      coordinate_frame_arm_frame, coordinate_frame_shoulder_height_arm_frame, cam_frame_in_arm_frame]
                    plot_open3d_Dorna(xyz_positions_of_all_joints, extra_geometry_elements=extra_elements)

                    # TODO do proper handeye but put full big markers on gripper with cardboard. 
                    # TODO get click working now with big markers. 
                    # TODO should I transform all xyz_positions_of_all_joints
                    # TODO could I keep everything in metres until I send the final control command of dorna?
                    # TODO if I create a generic transformation from marker to dorna arm origin, then I can compose transformations to pick up objects?
                    # TODO the purpose of everything here. all I want to get to is my old hand eye aruco on shoulder calibration and then using that to try pick up objects close enough again, but this time with solvePnP and 12 markers, it should obviously work better.
                    # TODO why are we off in depth and coordinate frame is behind marker? problem is resolved when we get closer. Could try bigger markers too
                    # TODO is it possible the scaling is messing things up? it didn't before though. 
                    # TODO we seem to beginning arm from shoulder height, rather than ground height? why?
                    # TODO look into open3D interactive mode and change sliders of joints here or for ik and see how wrist pitch changes
                    # TODO look into camera settings so I can look down top down, side-ways and straight down barrel on arm to see how wrong transforms are
                    # TODO optimise on white sphere and track it
                    # TODO is there a way I could visual servo measure angles?
                    # TODO I was using speed 5000 (what does this mean in joint and xyz space? same thing, how fast can I go?)
                    # dist(np.array([148.166, -190.953, 440.97]), np.array([ 131.949075, -199.25261, 452.410165]))
                    # TODO if I'm specifying that the clicked point is the centre of the battery I could get an error metric from FK vs cam2arm click point
                else:
                    print('No saved cam2arm!')


            if k == ord('h'):  # save hand-eye calibration needed transforms
                # get and save calibration target transformation (target2cam) and gripper2base
                aruco_id_on_gripper = 4
                bgr_color_data = cv2.cvtColor(camera_color_img, cv2.COLOR_RGB2BGR)
                gray_data = cv2.cvtColor(bgr_color_data, cv2.COLOR_RGB2GRAY)
                corners, ids, rejectedImgPoints = aruco.detectMarkers(gray_data, aruco_dict, parameters=parameters)
                # frame_markers = aruco.drawDetectedMarkers(color_img, corners, ids)
                all_rvec, all_tvec, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)

                # TODO should I do open3D here too just to make sure the transform is good?
                found_correct_marker = False
                if aruco_id_on_gripper in ids:
                    gripper_aruco_index = [l[0] for l in ids.tolist()].index(aruco_id_on_gripper) 
                    rvec, tvec = all_rvec[gripper_aruco_index, 0, :], all_tvec[gripper_aruco_index, 0, :]
                    # aruco.drawAxis(color_img, camera_matrix, dist_coeffs, rvec, tvec, marker_length)
                    found_correct_marker = True
                else:
                    tvec, rvec = None, None

                if found_correct_marker and tvec is not None and rvec is not None:
                    # usual order: cam2arm, arm2cam
                    cam2target, target2cam, R_tc, R_ct, pos_camera = create_homogenous_transformations(tvec, rvec)
                    # target2cam, cam2target, R_tc, R_ct, pos_camera = create_homogenous_transformations(tvec, rvec)

                    assert(isRotationMatrix(R_tc))
                    assert(isRotationMatrix(R_ct))

                    fp = 'data/handeye/target2cam_{}.txt'.format(num_saved_handeye_transforms)
                    print('Saving target2cam at {} \n{}'.format(fp, target2cam))
                    np.savetxt(fp, target2cam, delimiter=' ')

                    # get and save gripper transformation (gripper2base)
                    joint_angles = get_joint_angles_from_dorna_flask()

                    # # the below is just for testing without running arm
                    # joint_angles = [0, 0, 0, 0, 0]
                    gripper_base_transform = get_gripper_base_transformation(joint_angles)
                    # TODO try get inverse (actual gripper2base) just to really confirm shit
                    fp = 'data/handeye/gripper2base_{}.txt'.format(num_saved_handeye_transforms)
                    print('Saving gripper2base at {} \n{}'.format(fp, gripper_base_transform))
                    np.savetxt(fp, gripper_base_transform, delimiter=' ')

                    num_saved_handeye_transforms += 1
                else:
                    print('No tvec or rvec!')


            if k == ord('c'):  # perform hand-eye calibration using saved transforms
                # TODO all of the below can probably go to handeye wrapper?
                handeye_data_dict = load_all_handeye_data()
                # plot_all_handeye_data(handeye_data_dict)
                handeye_calibrate_opencv(handeye_data_dict)

                # TODO why load from file again, why not just return from function?
                cam2arm = np.loadtxt('data/handeye/latest_cv2_cam2arm.txt', delimiter=' ')
                saved_cam2arm = cam2arm

                # TODO save image as well in folder
                # TODO what the hell am I doing, of course saved cam2arm is fucked up. The only way to is to use cam_pcd 
                cam_pcd = get_full_pcd_from_rgbd(camera_color_img, camera_depth_img, pinhole_camera_intrinsic, visualise=False)
                # full_arm_pcd, full_pcd_numpy = convert_cam_pcd_to_arm_pcd(cam_pcd, saved_cam2arm, in_milimetres=False)

                # plot_all_handeye_data(handeye_data_dict, cam_pcd=full_arm_pcd)
                plot_all_handeye_data(handeye_data_dict, cam_pcd=cam_pcd)

            frame_count += 1

        except ValueError as e:
            # TODO try except is important to not or safely crash realsense or?
            print('Error in main loop')
            print(e)
            print(traceback.format_exc())
            print('sys exc info')
            print(sys.exc_info()[2])

            # Stop streaming
            pipeline.stop()
            cv2.destroyAllWindows()