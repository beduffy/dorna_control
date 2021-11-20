from __future__ import print_function
import os
import sys
import argparse
import time
import math
import traceback

import pyrealsense2 as rs
from cv2 import aruco
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib as mpl
from scipy import optimize

from lib.vision import get_camera_coordinate, create_homogenous_transformations, convert_pixel_to_arm_coordinate


# helper functions
def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    # return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)  # todo doesn't work with x = 0 and x2 = 9.8e-17
    return abs(a - b) <= rel_tol

def dist(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

def rotationMatrixToEulerAngles(R):
    # Calculates rotation matrix to euler angles
    # The result is the same as MATLAB except the order
    # of the euler angles ( x and z are swapped ).

    def isRotationMatrix(R):
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype=R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6

    # print(R.shape)
    # if not isRotationMatrix(R):
    #     print('Not rotation matrix!')
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])  # todo understand this for life
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def get_rigid_transform_error(joined_input_array, cam_3d_coords):
    # todo make less global
    tvec_opt = joined_input_array[:3].copy()  # todo do I need to copy these?
    rvec_opt = joined_input_array[3:].copy()

    # R_ct = np.matrix(cv2.Rodrigues(rvec_opt)[0])  # todo confirm that rvec makes sense
    # # todo all geometric vision, slam, pose estimation is about consistency with yourself... fundamental.
    # # todo Can we turn everything into optimisation? study it more
    #
    # # todo replace this with homogenous transform function given tvec and rvec?
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

    # todo could do above with nice big distance function across all array dimensions. Did this in interactive_click_fabrik.py for side view
    # print('tvec: {}. rvec: {}. Error: {:.5f}'.format(tvec_opt, rvec_opt, error))
    return error


def optimise_transformation_to_origin(cam_3d_coords, init_tvec, init_rvec):  # todo first param isn't used....
    # todo remove first param
    # todo fix params and globals and make cleaner
    # todo add more "world points". With 3 markers we would have 3x5=15 points and should help optimisation find correct transform from a distance
    # todo instead of rvec axis-angle optimise quaternions or dual quaternions or something continuous

    # todo remove most of below
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
    # global init_rvec  # todo just for now

    print('Starting optimisation with init_tvec: {}. init_rvec: {}'.format(init_tvec, init_rvec))

    joined_input_array = np.hstack((init_tvec, init_rvec))

    # def zero_con(t):
    #     return t[0]
    # cons = [{'type': 'eq', 'fun': zero_con}]  # todo how do I specify the output of arm_coord z-axis to be 0 with constraints?

    # optim_result = optimize.minimize(get_rigid_transform_error, args=(init_tvec, init_rvec),  # todo use args?
    # optim_result = optimize.minimize(get_rigid_transform_error, joined_input_array,
    optim_result = optimize.minimize(get_rigid_transform_error, joined_input_array, args=(cam_3d_coords, ),
                                     options={'max_iter': 500, 'disp': True},
                                     method='Nelder-Mead')
                                     # method='BFGS')
                                     # method='Newton-CG')
                                     # method='L-BFGS-B')
                                     # method='SLSQP')  #constraints=cons
    # todo try other optimisers? Least squares? Should learn about all of them anyway!
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    # todo will more points help or make it harder??!?!
    # todo could do constrained optimsation of forcing z to 0!!!!!!

    tvec_opt = optim_result.x[:3]
    rvec_opt = optim_result.x[3:]
    print(optim_result.success)
    print(optim_result.status)
    print(optim_result.message)

    final_optimised_error = get_rigid_transform_error(np.hstack((tvec_opt, rvec_opt)), cam_3d_coords)

    return tvec_opt, rvec_opt, final_optimised_error


def click_callback(event, x, y, flags, param):
    global mouseX, mouseY
    global curr_arm_xyz, prev_arm_xyz
    if event == cv2.EVENT_LBUTTONUP:
        mouseX, mouseY = x, y

        arm_xyz = convert_pixel_to_arm_coordinate(camera_depth_img, mouseX, mouseY, cam2arm, verbose=True)
        arm_xyz = arm_xyz * 1000
        print('arm_xyz milimetres: ', arm_xyz)

    if event == cv2.EVENT_MBUTTONUP:
        # cv2.circle(camera_color_img, (x, y), 100, (255, 255, 0), -1)  # todo if I ever want this
        mouseX, mouseY = x, y

        curr_arm_xyz = convert_pixel_to_arm_coordinate(camera_depth_img, mouseX, mouseY, cam2arm, verbose=True)

        print('Distance to previous arm xyz 3D: {}. 2D distance: {}'.format(dist(curr_arm_xyz, prev_arm_xyz), dist(curr_arm_xyz[:2], prev_arm_xyz[:2])))
        prev_arm_xyz = curr_arm_xyz
        # TODO try except
        # IndexError: index 1258 is out of bounds for axis 0 with size 640


def estimate_cam2arm_on_frame(color_img, depth_img, estimate_pose=True, use_aruco=True):
    color_img = color_img.copy()
    depth_img = depth_img.copy()
    bgr_color_data = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
    gray_data = cv2.cvtColor(bgr_color_data, cv2.COLOR_RGB2GRAY)

    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray_data, aruco_dict,
                                                                parameters=parameters)
    frame_markers = aruco.drawDetectedMarkers(color_img, corners, ids)
    all_rvec, all_tvec, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)

    if not use_aruco:
        retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray_data, board,
                                                                cameraMatrix=camera_matrix,
                                                                distCoeffs=dist_coeffs)
        if charuco_ids is not None:
            # for i in range(charuco_ids.shape[0]):
            #     cv2.circle(color_img, tuple(charuco_corners[i][0]), 7, color=(255, 255, 0))
            #     cv2.circle(color_img, tuple(charuco_corners[i][0]), 7, color=(255, 255, 0))
            retval, board_rvec, board_tvec = cv2.aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, board, camera_matrix, dist_coeffs, all_rvec[0], all_tvec[0], useExtrinsicGuess=False)
            # retval, board_rvec, board_tvec = cv2.aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, board, camera_matrix, dist_coeffs)

            try:
                aruco.drawAxis(color_img, camera_matrix, dist_coeffs, board_rvec, board_tvec, 0.05)  # last param is axis length
            except Exception as e:
                print(e)
                tvec, rvec = None, None
            # tvec, rvec = np.swapaxes(board_tvec, 0, 1), np.swapaxes(board_rvec, 0, 1)  # why is this even necessary
            tvec, rvec = board_tvec.squeeze(), board_rvec.squeeze()


            # TODO use charuco and a single x offset. no need to move it off the board, use the boards width and height and this way it's actually easier
        else:
            tvec, rvec = None, None
    elif use_aruco:
        # TODO im not using depth!!!!!! 
        # TODO create new repo or use dorna control, start committing . copy over what is needed and clean all the code so i can think better

        if all_rvec is not None:
            # retval, board_rvec, board_tvec = cv2.aruco.estimatePoseBoard(corners, ids, board, camera_matrix, dist_coeffs, all_rvec, all_tvec)
            # aruco.drawAxis(color_img, camera_matrix, dist_coeffs, board_rvec, board_tvec, 0.05)  # last param is axis length

            # aruco.drawAxis(color_img, camera_matrix, dist_coeffs, all_rvec[0], all_tvec[0], marker_length / 2)  # in middle
            # aruco.drawAxis(color_img, camera_matrix, dist_coeffs, all_rvec[0], all_tvec[0], 0)  # jumps around
            
            found_correct_marker = False
            # print(ids)
            if id_on_shoulder_motor in ids:
                # TODO is this even correct?!?!?! since 1 index vs 0 index?!?!? ahh because I found correct index?
                shoulder_motor_marker_id = [l[0] for l in ids.tolist()].index(id_on_shoulder_motor) 
                rvec, tvec = all_rvec[shoulder_motor_marker_id, 0, :], all_tvec[shoulder_motor_marker_id, 0, :]  # get first marker
                found_correct_marker = True
            if found_correct_marker and rvec is not None and len(corners) == 4:  # TODO do not forget
            # if found_correct_marker and rvec is not None and len(corners) == 2:  # TODO do not forget
            # if found_correct_marker and rvec is not None and len(corners) == 1:  # TODO do not forget
                # aruco.drawAxis(color_img, camera_matrix, dist_coeffs, rvec, tvec, marker_length)

                # draw arm origin axis as well
                # roughly 96mm to arm and 41mm from there to centre of arm
                y_offset_for_id_1 = -0.13
                y_offset_for_id_3 = 0.127
                extra_y_offset_for_id_2 = -0.167
                extra_y_offset_for_id_4 = 0.173
                x_offset_for_id_2 = 0.2
                x_offset_for_id_4 = -0.2

                # found with depth camera clicking
                # y_offset_for_id_1 = -0.128
                # extra_y_offset_for_id_2 = -0.166
                # y_offset_for_id_3 = 0.1214
                # extra_y_offset_for_id_4 = 0.169
                
                imagePoints, jacobian = cv2.projectPoints(np.array([0.0, y_offset_for_id_1, 0.0]), rvec, tvec, camera_matrix, dist_coeffs)
                x, y = imagePoints.squeeze().tolist()
                cv2.circle(color_img, (int(x), int(y)), 5, (255, 255, 0), -1)
                
                half_marker_len = marker_length / 2
                top_left_first = np.array([-half_marker_len, half_marker_len + y_offset_for_id_1, 0.])
                top_right_first = np.array([half_marker_len, half_marker_len + y_offset_for_id_1, 0.])
                bottom_right_first = np.array([half_marker_len, -half_marker_len + y_offset_for_id_1, 0.])
                bottom_left_first = np.array([-half_marker_len, -half_marker_len + y_offset_for_id_1, 0.])

                top_left_second = np.array([-half_marker_len + x_offset_for_id_2, half_marker_len + extra_y_offset_for_id_2, 0.])
                top_right_second = np.array([half_marker_len + x_offset_for_id_2, half_marker_len + extra_y_offset_for_id_2, 0.])
                bottom_right_second = np.array([half_marker_len + x_offset_for_id_2, -half_marker_len + extra_y_offset_for_id_2, 0.])
                bottom_left_second = np.array([-half_marker_len + x_offset_for_id_2, -half_marker_len + extra_y_offset_for_id_2, 0.])

                top_left_third = np.array([-half_marker_len, half_marker_len + y_offset_for_id_3, 0.])
                top_right_third = np.array([half_marker_len, half_marker_len + y_offset_for_id_3, 0.])
                bottom_right_third = np.array([half_marker_len, -half_marker_len + y_offset_for_id_3, 0.])
                bottom_left_third = np.array([-half_marker_len, -half_marker_len + y_offset_for_id_3, 0.])

                top_left_fourth = np.array([-half_marker_len + x_offset_for_id_4, half_marker_len + extra_y_offset_for_id_4, 0.])
                top_right_fourth = np.array([half_marker_len + x_offset_for_id_4, half_marker_len + extra_y_offset_for_id_4, 0.])
                bottom_right_fourth = np.array([half_marker_len + x_offset_for_id_4, -half_marker_len + extra_y_offset_for_id_4, 0.])
                bottom_left_fourth = np.array([-half_marker_len + x_offset_for_id_4, -half_marker_len + extra_y_offset_for_id_4, 0.])
                
                # the below ends up being the same as corners[shoulder_motor_marker_id]. As it should be
                # top_left_first = np.array([-half_marker_len, half_marker_len, 0.])
                # top_right_first = np.array([half_marker_len, half_marker_len, 0.])
                # bottom_right_first = np.array([half_marker_len, -half_marker_len, 0.])
                # bottom_left_first = np.array([-half_marker_len, -half_marker_len, 0.])

                # TODO make sure order is correct if top left is always first and x is forward?!?!?!
                # corners_3d_points = np.array([top_left_first, top_right_first, bottom_right_first, bottom_left_first])
                # corners_3d_points = np.array([top_left_first, top_right_first, bottom_right_first, bottom_left_first, 
                #                               top_left_second, top_right_second, bottom_right_second, bottom_left_second])
                corners_3d_points = np.array([top_left_first, top_right_first, bottom_right_first, bottom_left_first, 
                                            top_left_second, top_right_second, bottom_right_second, bottom_left_second,
                                            top_left_third, top_right_third, bottom_right_third, bottom_left_third,
                                            top_left_fourth, top_right_fourth, bottom_right_fourth, bottom_left_fourth])
                imagePointsCorners, jacobian = cv2.projectPoints(corners_3d_points, rvec, tvec, camera_matrix, dist_coeffs)
                # for x, y in imagePointsCorners.squeeze().tolist():
                #     cv2.circle(color_img, (int(x), int(y)), 5, (0, 0, 255), -1)
                # x, y = imagePointsCorners.squeeze().tolist()
                # 
                # cv2.circle(color_img, (int(x), int(y)), 5, (0, 0, 255), -1)

                # TODO give initial guess. Did it help?
                # marker_indices = [shoulder_motor_marker_id, 4]
                # outval, rvec_arm, tvec_arm = cv2.solvePnP(corners_3d_points, corners[shoulder_motor_marker_id], camera_matrix, dist_coeffs)
                # outval, rvec_arm, tvec_arm = cv2.solvePnP(corners_3d_points, corners[shoulder_motor_marker_id], camera_matrix, dist_coeffs, rvec, tvec)
                # outval, rvec_arm, tvec_arm = cv2.solvePnP(corners_3d_points, corners[marker_indices], camera_matrix, dist_coeffs, rvec, tvec)
                # if there are only two, wrong could be wrong order
                corners_reordered = []
                ids_list = [l[0] for l in ids.tolist()]
                
                # for corder_id in [x[0] for x in ids.tolist()]:
                for corder_id in [1, 2, 3, 4]:
                # for corder_id in [x[0] - 1 for x in ids.tolist()]:
                    corder_index = ids_list.index(corder_id) 
                    # corners_reordered.append(corners[corder_id])
                    corners_reordered.append(corners[corder_index])
                    rvec_aruco, tvec_aruco = all_rvec[corder_index, 0, :], all_tvec[corder_index, 0, :]
                    aruco.drawAxis(color_img, camera_matrix, dist_coeffs, rvec_aruco, tvec_aruco, marker_length)
                
                # outval, rvec_arm, tvec_arm = cv2.solvePnP(corners_3d_points, np.hstack(corners_reordered).squeeze(), camera_matrix, dist_coeffs, rvec, tvec, useExtrinsicGuess=True)
                outval, rvec_arm, tvec_arm = cv2.solvePnP(corners_3d_points, np.hstack(corners_reordered).squeeze(), camera_matrix, dist_coeffs, rvec, tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_IPPE)
                # outval, rvec_arm, tvec_arm = cv2.solvePnP(corners_3d_points, np.hstack(corners_reordered).squeeze(), camera_matrix, dist_coeffs)
                # outval, rvec_arm, tvec_arm = cv2.solvePnP(corners_3d_points, np.hstack(corners_reordered).squeeze(), camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE)
                # outval, rvec_arm, tvec_arm = cv2.solvePnP(corners_3d_points, np.hstack(corners_reordered).squeeze(), camera_matrix, dist_coeffs, rvec, tvec, useExtrinsicGuess=True)
                # outval, rvec_arm, tvec_arm = cv2.solvePnP(corners_3d_points, np.hstack(corners), camera_matrix, dist_coeffs, rvec, tvec, useExtrinsicGuess=True)
                # outval, rvec_arm, tvec_arm = cv2.solvePnP(corners_3d_points, np.hstack(corners), camera_matrix, dist_coeffs, rvec, tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                # outval, rvec_arm, tvec_arm = cv2.solvePnP(corners_3d_points, corners[0], camera_matrix, dist_coeffs, rvec, tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                # outval, rvec_arm, tvec_arm = cv2.solvePnP(np.array([[0.0, -y_offset, 0.0]]), corners[shoulder_motor_marker_id], camera_matrix, dist_coeffs)

                # TODO could I just use a charuco board instead???
                # TODO bigger markers and how to minimise eyeball error? glue and pencil?

                # TODO to get to the bottom of my hand eye problem, I should also store cam2arm to aruco marker?
                rvec = rvec_arm.squeeze()
                tvec = tvec_arm.squeeze()

                imagePoints, jacobian = cv2.projectPoints(np.array([0.0, 0.0, 0.0]), rvec, tvec, camera_matrix, dist_coeffs)
                x, y = imagePoints.squeeze().tolist()
                cv2.circle(color_img, (int(x), int(y)), 5, (0, 0, 255), -1)
            else:
                print('Not doing solvePnP')
                tvec, rvec = None, None
            
            # todo ahhhhhhh more markers doesn't help the above. It'd be better for me to collect the marker positions of 4-10 markers and run into solvepnp
            # todo as this says: https://stackoverflow.com/questions/51709522/unstable-values-in-aruco-pose-estimation
            # todo https://github.com/opencv/opencv_contrib/blob/master/modules/aruco/src/aruco.cpp
            # https://stackoverflow.com/questions/51709522/unstable-values-in-aruco-pose-estimation
            # todo could use cube!!!
            # https://pypi.org/project/apriltag/

            # todo read this: https://www.learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
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

            # todo smart way to weight the aruco markers by their distance and fuse all the pose estimations.

            # if estimate_pose:  # further refine pose of specific marker by
            #     # todo does the above get better with more markers or not?!?!?!?! it doesn't because it's individual markers

            #     found_correct_marker = False
            #     # todo use initial guess as last all_rvec? This might help it be less jumpy?

            #     if ids is not None: # and (use_aruco and id_on_shoulder_motor in ids):
            #         if not use_aruco:
            #             # print(len(corners))
            #             # print(len(ids))
            #             retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray_data, board,
            #                                                                   cameraMatrix=camera_matrix,
            #                                                                   distCoeffs=dist_coeffs)
            #             if charuco_ids is not None:
            #                 for i in range(charuco_ids.shape[0]):
            #                     cv2.circle(color_img, tuple(charuco_corners[i][0]), 7, color=(255, 255, 0))
            #             # cv2.circle(color_img, charuco_corners[0][0], 7, color=(255, 255, 0))
            #         if use_aruco:
            #             # retval, board_rvec, board_tvec = cv2.aruco.estimatePoseBoard(corners, ids, board, camera_matrix, dist_coeffs, all_rvec, all_tvec)
            #             # retval, board_rvec, board_tvec = cv2.aruco.estimatePoseBoard(corners, ids, board, camera_matrix, dist_coeffs, None, None, useExtrinsicGuess=False)
            #             # img = cv2.aruco.drawPlanarBoard(board, (300, 300))  # board, outSize[, img[, marginSize[, borderBits]]]  # todo doesn't work, not showing things correctly?
            #             pass
            #         else:
            #             # if board_rvec is not None:
            #             #     retval, board_rvec, board_tvec = cv2.aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, board, camera_matrix, dist_coeffs, board_rvec, board_tvec, useExtrinsicGuess=True)
            #             # else:
            #             retval, board_rvec, board_tvec = cv2.aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, board, camera_matrix, dist_coeffs, all_rvec[0], all_tvec[0], useExtrinsicGuess=False)
            #             # retval, board_rvec, board_tvec = cv2.aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, board, camera_matrix, dist_coeffs)

            #         if not use_aruco:  # todo only charuco here
            #             try:
            #                 aruco.drawAxis(color_img, camera_matrix, dist_coeffs, board_rvec, board_tvec,
            #                                0.05)  # last param is axis length
            #             except Exception as e:
            #                 print(e)

            #         # retval, rvec, tvec = cv2.aruco.estimatePoseBoard
            #         # -- ret = [rvec, tvec, ?]
            #         # -- array of rotation and position of each marker in camera frame
            #         # -- rvec = [[rvec_1], [rvec_2], ...]    attitude of the marker respect to camera frame
            #         # -- tvec = [[tvec_1], [tvec_2], ...]    position of the marker in camera frame

            #         if use_aruco:
            #             if id_on_shoulder_motor in ids:
            #                 shoulder_motor_marker_id = [l[0] for l in ids.tolist()].index(id_on_shoulder_motor)
            #                 # todo calculate tvec distance between all adjacent marker middles. Should be 0.0033!!!!!!!!!!!!!!! optimise something to make it so?
            #                 # todo and maybe intrinsics too?

            #                 rvec, tvec = all_rvec[shoulder_motor_marker_id, 0, :], all_tvec[shoulder_motor_marker_id, 0, :]  # get first marker
            #                 found_correct_marker = True
            #         else:
            #             rvec, tvec = board_rvec.squeeze(), board_tvec.squeeze()

            #         # todo if no markers it crashes still
            #         if 'tvec' in locals():  # todo how to avoid this?
            #             cam2arm, arm2cam, R_tc, R_ct, pos_camera = create_homogenous_transformations(tvec, rvec)

            #             # -- Get the attitude in terms of euler 321 (Needs to be flipped first)
            #             roll_marker, pitch_marker, yaw_marker = rotationMatrixToEulerAngles(
            #                 R_flip * R_tc)

            #             if use_aruco:
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

            #                     # todo optimise all 4 corners and then if that still doesn't work all 4x12 corners and then charuco or whatever
            #                     # todo see how much aruco and charuco board error are
            #                     # todo swap to charuco and the charuco id 0 should be the corner!!!!!!!!!!!!!!!!!!!!!!!!!!
            #                     # todo is rounding bad since everyone talks about sub-pixel accuracy? how else would i index?

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
            #                             continue  # todo not a loop anymore. But returning will break everything
            #                             # return
            #                         # todo AttributeError: 'NoneType' object has no attribute 'reshape'
            #                         # todo probably because there is no depth at that part of the image. But it wasn't a problem before?

            #                     # todo should see how much my rigid body transform error changes with the same cam2arm!!!!
            #                     cam_coords = np.concatenate(cam_coords)

            #                     rigid_body_error_local = get_rigid_transform_error(joined_input_array, cam_coords)
            #                     print('Rigid body transform error: {}'.format(rigid_body_error))


            #             else:  # todo fix all below so charuco works as well
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

            #                             # todo how to enforce z is 0? for everything...
            #                             if len(corner_3d_positions) > 0:
            #                                 allowable_y_difference = 0.0008
            #                                 allowable_y_difference = 0.0012
            #                                 # todo TypeError: 'NoneType' object is not subscriptable happens if I cover board. Why?
            #                                 if abs(corner_3d_positions[-1][1] - corner_xyz_arm_coord[1]) > (charuco_square_length - allowable_y_difference):
            #                                     # print('Jumped row!!!')  # todo not detecting all. Now it is since i changed to 0.0012?
            #                                     jumped_row = True
            #                                 else:
            #                                     jumped_row = False

            #                                 dist_3d_to_last_point = dist(corner_xyz_arm_coord, corner_3d_positions[-1])
            #                                 dist_2d = dist(corner_xyz_arm_coord[:2], corner_3d_positions[-1][:2])
            #                                 # print('Dist 3D {:.4f}. Dist 2D {:.4f}. Absolute error 3D: {:.4f}. Absolute error 2D: {:.4f}'.format(dist_3d_to_last_point, dist_2d, abs(dist_3d_to_last_point - distance_between_adjacent_corners), abs(dist_2d - distance_between_adjacent_corners)))
            #                                 # todo this happened once: TypeError: unsupported operand type(s) for -: 'NoneType' and 'float'
            #                                 # if dist_2d < 0.05:  # to avoid adding cases when we swap row
            #                                 if not jumped_row:  # to avoid adding cases when we swap row
            #                                     all_dist_2ds.append(dist_2d)
            #                                     all_dist_3ds.append(dist_3d_to_last_point)
            #                                     if corner_3d_positions[-1][0] > corner_xyz_arm_coord[0]:
            #                                         pass
            #                                         # print('X to the right is smaller than the left!!!')  # todo is it really that bad to happen 1-10% of the time?

            #                             corner_3d_positions.append(corner_xyz_arm_coord)

            #                     # print('Mean 3D absolute error: {}. Mean 2D absolute error: {}'.format(sum(all_dist_3ds) / len(all_dist_3ds),
            #                     #                                         sum(all_dist_2ds) / len(all_dist_2ds)))

            if tvec is None or rvec is None:
                # print('tvec or vec is none')
                found_correct_marker = False

            if found_correct_marker:
                # is this id1 or not. nope it isn't.
                # tvec, rvec = all_tvec[0].squeeze(), all_rvec[0].squeeze()
                start_y = 30
                jump_amt = 30
                text_size = 0.6

                am2arm, arm2cam, R_tc, R_ct, pos_camera = create_homogenous_transformations(tvec, rvec)

                # -- Get the attitude in terms of euler 321 (Needs to be flipped first)
                roll_marker, pitch_marker, yaw_marker = rotationMatrixToEulerAngles(R_flip * R_tc)

                # -- Print the tag position in camera frame
                str_position = "MARKER Position x={:.5f}  y={:.5f}  z={:.5f}".format(tvec[0], tvec[1], tvec[2])
                cv2.putText(color_img, str_position, (0, start_y), font, text_size, (0, 255, 0), 2, cv2.LINE_AA)

                # -- Print the marker's attitude respect to camera frame
                str_attitude = "MARKER Attitude r={:.5f}  p={:.5f}  y={:.5f}".format(
                    math.degrees(roll_marker), math.degrees(pitch_marker),
                    math.degrees(yaw_marker))
                cv2.putText(color_img, str_attitude, (0, start_y + jump_amt * 1), font, text_size, (0, 255, 0), 2, cv2.LINE_AA)

                str_position = "CAMERA Position x={:.5f}  y={:.5f}  z={:.5f}".format(
                    pos_camera[0].item(), pos_camera[1].item(), pos_camera[2].item())
                cv2.putText(color_img, str_position, (0, start_y + jump_amt * 2), font, text_size, (0, 255, 0), 2, cv2.LINE_AA)

                # -- Get the attitude of the camera respect to the frame
                # roll_camera, pitch_camera, yaw_camera = rotationMatrixToEulerAngles(R_flip * R_tc)
                roll_camera, pitch_camera, yaw_camera = rotationMatrixToEulerAngles(R_flip * R_ct)  # todo no flip needed?
                str_attitude = "CAMERA Attitude r={:.5f}  p={:.5f}  y={:.5f}".format(
                    math.degrees(roll_camera), math.degrees(pitch_camera),
                    math.degrees(yaw_camera))
                cv2.putText(color_img, str_attitude, (0, start_y + jump_amt * 3), font, text_size, (0, 255, 0), 2, cv2.LINE_AA)

                # else:
                #     return -1, color_img, depth_img, None, None, None, None

                # return rigid_body_error_local, color_img, depth_img, pixel_positions_to_optimise, tvec, rvec, cam_coords
                # todo what to return? and to not make it ugly?
                # return tvec, rvec
        else:
            tvec, rvec = None, None

    return color_img, depth_img, tvec, rvec


if __name__ == '__main__':
    np.set_printoptions(precision=6, suppress=True)
    prev_arm_xyz = np.array([0., 0., 0.])

    #--- 180 deg rotation matrix around the x axis
    R_flip       = np.zeros((3, 3), dtype=np.float32)
    R_flip[0, 0] =  1.0
    R_flip[1, 1] = -1.0
    R_flip[2, 2] = -1.0

    #-- Font for the text in the image
    font = cv2.FONT_HERSHEY_PLAIN
    # todo seems very different to depth.intrinsics. Use depth intrinsics.....

    cam2arm = np.identity(4)
    # cam_coords = []

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    profile = pipeline.start(config)
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.holes_fill, 3)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
    color_intrin = color_frame.profile.as_video_stream_profile().intrinsics

    # calibration and marker params
    # marker_length = 0.0265
    # marker_length = 0.028
    marker_length = 0.0275
    use_aruco = False  # use charuco
    # use_aruco = True
    optimise_origin = False
    # optimise_origin = True
    # estimate_pose = True
    estimate_pose = False

    # todo When estimate_pose = False I could still visualise the marker finding but only optimise and save after key a is pressed

    # todo FileNotFoundError: [Errno 2] No such file or directory: 'data/best_aruco_cam2arm.txt'
    # todo how to ensure everything is run from root directory.... Absolute paths.

    # todo how to ensure marker/found frame is flat? the world is flattened I mean. Wrong assumption because it isn;t?
    # todo could find relative pose transformations between multiple markers and then use them to create absolute ground truth instead of using a ruler

    # todo most important: make it work from 70cm away. This is perfect. Do I need 2 extra markers?
    # todo for this I might need to try different optimisers or parameters. Also need to understand PnP better

    if use_aruco:
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        parameters = aruco.DetectorParameters_create()
        parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        # parameters.adaptiveThreshWinSizeMin = 3
        # parameters.adaptiveThreshWinSizeStep = 4  # todo test more and see if it makes worse/better
        board = cv2.aruco.GridBoard_create(3, 4, marker_length, 0.06, aruco_dict)  # marker_separation 0.06

        # fig = plt.figure()
        # nx = 4
        # ny = 3
        # for i in range(1, nx * ny + 1):
        #     ax = fig.add_subplot(ny, nx, i)
        #     img = aruco.drawMarker(aruco_dict, i, 700)
        #     plt.imshow(img, cmap=mpl.cm.gray, interpolation="nearest")
        #     ax.axis("off")

        # plt.savefig("markers.pdf")
        # plt.show()
    else:
        # Charuco
        parameters = aruco.DetectorParameters_create()
        parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        # charuco_marker_length = 0.019
        charuco_marker_length = 0.0145
        charuco_square_length = 0.028
        # todo maybe i can tune these more
        # todo find out how to do long distance pose estimation!!!

        # todo aruco con More susceptible to rotational ambiguity at medium to long ranges
        # https://stackoverflow.com/questions/52222327/improve-accuracy-of-pose-with-bigger-aruco-markers
        # https://stackoverflow.com/questions/51709522/unstable-values-in-aruco-pose-estimation

        # todo try apriltag_ros

        # board = cv2.aruco.CharucoBoard_create(7, 7, .025, .0125, aruco_dict)
        # board = cv2.aruco.CharucoBoard_create(7, 7, .025, .0125, dictionary)
        # img = board.draw((200 * 3, 200 * 3))
        board = cv2.aruco.CharucoBoard_create(7, 7, charuco_square_length, charuco_marker_length, aruco_dict)  # todo test new values

    # old calibration
    camera_matrix = np.array([[612.14801862, 0., 340.03640321],
                              [0., 611.29345062, 230.06928807],
                              [0., 0., 1.]])
    dist_coeffs = np.array(
        [1.80764862e-02, 1.09549436e+00, -3.38044260e-03, 4.04543459e-03, -4.26585263e+00])

    # directly from depth/color intrinsics from factory
    # camera_matrix = np.array([[depth_intrin.fx, 0., depth_intrin.ppx],
    #                           [0., depth_intrin.fy, depth_intrin.ppy],
    #                           [0., 0., 1.]])

    # camera_matrix = np.array([[color_intrin.fx, 0., color_intrin.ppx],
    #                           [0., color_intrin.fy, color_intrin.ppy],
    #                           [0., 0., 1.]])

    mouseX, mouseY = 0, 0
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', click_callback)

    check_corner_frame_count = 0
    frame_count = 1
    marker_top_left_x_bigger_than_top_right = 0
    marker_top_left_x_bigger_than_bottom_right = 0
    marker_top_left_y_bigger_than_bottom_left = 0
    marker_top_left_y_bigger_than_bottom_right = 0

    id_on_shoulder_motor = 1

    depth_sensor = profile.get_device().first_depth_sensor()

    # Using preset HighAccuracy for recording
    # depth_sensor.set_option(rs.option.visual_preset, Preset.HighAccuracy)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_scale = depth_sensor.get_depth_scale()

    align = rs.align(rs.stream.color)
    board_rvec = None

    # if estimate_pose:  # todo think about this
    # command = '/home/beduffy/anaconda/envs/py36/bin/python control/scripts/send_arm_to_home_position.py'
    # print('Running command: {}'.format(command))
    # os.system(command)
    # time.sleep(1)

    lowest_error = 1000000
    lowest_optimised_error = 1000000

    # wait for auto-exposure
    print('Running 10 frames to wait for auto-exposure')
    for i in range(10):
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

    print('Starting loop')
    while True:
        try:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            depth_frame = spatial.process(depth_frame)  # hole filling

            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
            color_intrin = color_frame.profile.as_video_stream_profile().intrinsics

            camera_depth_img = np.asanyarray(depth_frame.get_data())
            camera_color_img = np.asanyarray(color_frame.get_data())
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(camera_depth_img, alpha=0.03),
                                               cv2.COLORMAP_JET)  # todo why does it look so bad, add more contrast?

            # print(camera_depth_img.shape, camera_color_img.shape)

            # rigid_body_error, color_img, depth_img, \
            #     pixel_positions_to_optimise, \
            #     tvec, rvec, cam_coords = estimate_cam2arm_on_frame(camera_color_img, camera_depth_img, estimate_pose=estimate_pose)

            color_img, depth_img, tvec, rvec = estimate_cam2arm_on_frame(camera_color_img, camera_depth_img, estimate_pose=estimate_pose, use_aruco=use_aruco)

            # if pixel_positions_to_optimise:
            #     # middle_pixel = pixel_positions_to_optimise[0]  # todo first is a numpy array, the rest are lists....
            #     (middle_pixel, top_left_pixel, top_right_pixel,
            #      bottom_right_pixel, bottom_left_pixel) = pixel_positions_to_optimise

            #     hit_lowest_error = False
            #     if rigid_body_error < lowest_error:  # todo if optimise_origin works calculate lowest_error after or create 3rd cam2arm?
            #         lowest_error = rigid_body_error
            #         print('Saving best aruco cam2arm: \n{}'.format(cam2arm))
            #         print('Best error so far: {}'.format(rigid_body_error))
            #         np.savetxt('data/best_aruco_cam2arm.txt', cam2arm, delimiter=' ')
            #         hit_lowest_error = True

            #     # if optimise_origin: # todo do this better
            #     if hit_lowest_error:
            #         # todo making the assumption that the lowest error above will get the lowest possible error here!!!
            #         # todo why much better error when some markers are hidden? I'm closer? I don't think other markers help? prove it

            #         # todo do many tests!!!
            #         # todo since arm is in home position, we could plot in open3D right here!!!!!!!!
            #         # todo show pointcloud view with open3d (blocking vs non-blocking) to test how good all the frames are!!!!!!!!!!!!!!!!!!!!!!!
            #         # todo could do it on q key!! or every time here. Would help me visualise coordinate frame, pose and error


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
            # # todo marker z's seem a bit wrong? Maybe it's the depth scale by 1-10cm off
            # # will only work if we are facing it directly? Or not, because the transformation should be correct
            # # assert marker_xyz_top_left[0] < marker_xyz_top_right[0] and \
            # #        marker_xyz_top_left[0]cv2. < marker_xyz_bottom_right[0] and \
            # #        marker_xyz_top_left[1] > marker_xyz_bottom_left[1] and \
            # #        marker_xyz_top_left[1] > marker_xyz_bottom_right[1]
            #
            # # todo not doing the below anymore but would be good to check?
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
            #     # todo one time they all followed each other exactly 3 numbers were the same out of 4. the 4th was 0
            #     # todo what to do about z flip????? Does it affect x and y? Probably not or....
            #
            # # todo could create my own drawAxis to test the depth and once that works everything will work right?
            #
            # cv2.circle(color_img, top_left_pixel, 4, color=(255, 0, 0))
            # cv2.circle(color_img, top_right_pixel, 4, color=(0, 255, 0))
            # cv2.circle(color_img, bottom_right_pixel, 4, color=(0, 0, 255))
            # cv2.circle(color_img, bottom_left_pixel, 4, color=(0, 0, 0))
            #
            images = np.hstack((color_img, depth_colormap))
            # images = np.hstack((camera_color_img, depth_colormap))
            # images = camera_color_img

            cv2.imshow("image", images)
            k = cv2.waitKey(1)

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
                    print('Saving aruco cam2arm {}\n'.format(cam2arm))
                    np.savetxt('data/latest_aruco_cam2arm.txt', cam2arm, delimiter=' ')

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