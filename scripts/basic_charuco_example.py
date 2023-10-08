from __future__ import print_function
import sys
import math
import traceback
from glob import glob

try:
    # TODO remove try
    import open3d as o3d
    from skimage.measure import find_contours
except Exception as e:
    print(e)
    print('Tried to import open3d or skimage but not installed')
import pyrealsense2 as rs
from cv2 import aruco
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib as mpl
from scipy import optimize
from collections import deque

from lib.vision import isclose, dist, isRotationMatrix, rotationMatrixToEulerAngles, create_homogenous_transformations
from lib.vision_config import camera_matrix, dist_coeffs
from lib.realsense_helper import setup_start_realsense, realsense_get_frames, run_10_frames_to_wait_for_auto_exposure
from lib.aruco_helper import create_aruco_params, create_charuco_params, aruco_detect_draw_get_transforms, \
    show_matplotlib_all_aruco, show_matplotlib_all_charuco

# export PYTHONPATH=$PYTHONPATH:/home/ben/all_projects/dorna_control
# TODO Put MORE of these common things into some function/library?

if __name__ == '__main__':
    depth_intrin, color_intrin, depth_scale, pipeline, align, spatial = setup_start_realsense()

    board, parameters, aruco_dict, marker_length = create_charuco_params()
    # in this case marker_length is the smaller aruco marker

    camera_matrix = np.array([[color_intrin.fx, 0., color_intrin.ppx],
                            [0., color_intrin.fy, color_intrin.ppy],
                            [0., 0., 1.]])

    # TODO glue to back of notebook
    # TODO try bigger resolution but then use color intrinsics for camera_matrix. is distortion the same in that case? 
    # show_matplotlib_all_charuco(board)
    # TODO oh, should go bigger... charuco aruco length within square.

    # Opencv text params
    start_y = 30
    jump_amt = 30
    text_size = 1
    #--- 180 deg rotation matrix around the x axis
    R_flip       = np.zeros((3, 3), dtype=np.float32)
    R_flip[0, 0] =  1.0
    R_flip[1, 1] = -1.0
    R_flip[2, 2] = -1.0

    marker_pose_history = deque([], maxlen=100)

    #-- Font for the text in the image
    font = cv2.FONT_HERSHEY_PLAIN

    run_10_frames_to_wait_for_auto_exposure(pipeline, align)

    frame_count = 0
    print('Starting main loop')
    while True:
        try:
            color_frame, depth_frame = realsense_get_frames(pipeline, align, spatial)
            
            camera_depth_img = np.asanyarray(depth_frame.get_data())
            camera_color_img = np.asanyarray(color_frame.get_data())
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(camera_depth_img, alpha=0.03),
                                                cv2.COLORMAP_JET)  # todo why does it look so bad, add more contrast?
            
            bgr_color_data = cv2.cvtColor(camera_color_img, cv2.COLOR_RGB2BGR)
            gray_data = cv2.cvtColor(bgr_color_data, cv2.COLOR_RGB2GRAY)

            # corners, ids, all_rvec, all_tvec = aruco_detect_draw_get_transforms(gray_data, camera_color_img, aruco_dict, parameters, marker_length, camera_matrix, dist_coeffs)
            # ids, all_rvec, all_tvec = charuco_detect_draw_get_transforms(gray_data, camera_color_img, aruco_dict, parameters, marker_length, camera_matrix, dist_coeffs)


            # SUB PIXEL CORNER DETECTION CRITERION
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

            corners, ids, rejectedImgPoints = aruco.detectMarkers(gray_data, aruco_dict,
                                                                parameters=parameters)
            # frame_markers = aruco.drawDetectedMarkers(camera_color_img, corners, ids)  # TODO separate elsewhere? This function does too much?
            # TODO if there are two markers in image, it will not be more accurate right? or?
            all_rvec, all_tvec, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)

            print(ids)
            # import pdb;pdb.set_trace()
            if ids is not None and ids.shape[0] > 1:
                # TODO do more from below blog
                # SUB PIXEL DETECTION. From: https://mecaruco2.readthedocs.io/en/latest/notebooks_rst/Aruco/sandbox/ludovic/aruco_calibration_rotation.html
                for corner in corners:
                    cv2.cornerSubPix(gray_data, corner,
                                    winSize = (3,3),
                                    zeroZone = (-1,-1),
                                    criteria = criteria)

                # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
                # for corner in corners:
                #     cv2.cornerSubPix(gray, corner, winSize = (3,3), zeroZone = (-1,-1), criteria = criteria)
                
                retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray_data, board,
                                                                    cameraMatrix=camera_matrix,
                                                                    distCoeffs=dist_coeffs)
                if charuco_ids is not None:

                    # for i in range(charuco_ids.shape[0]):
                    #     cv2.circle(color_img, tuple(charuco_corners[i][0]), 7, color=(255, 255, 0))
                    #     cv2.circle(color_img, tuple(charuco_corners[i][0]), 7, color=(255, 255, 0))

                    cv2.aruco.drawDetectedCornersCharuco(camera_color_img, charuco_corners, charuco_ids, (255, 255, 0))
                    
                    # TODO how to pass in estimateParameters initial guess for extrinsic guess. 
                    retval, board_rvec, board_tvec = cv2.aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, board, 
                                                                                        camera_matrix, dist_coeffs, np.empty(1), np.empty(1))
                                                                                        # camera_matrix, dist_coeffs, 
                                                                                        # all_rvec[0], all_tvec[0], useExtrinsicGuess=True)  # TODO would be nice to use rvec 0 but not tvec hmm
                    # retval, board_rvec, board_tvec = cv2.aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, board, camera_matrix, dist_coeffs)
    # 
                    # aruco.drawAxis(camera_color_img, camera_matrix, dist_coeffs, board_rvec, board_tvec, 0.05)  # last param is axis length
                    try:
                        cv2.drawFrameAxes(camera_color_img, camera_matrix, dist_coeffs, board_rvec, board_tvec, marker_length)

                        tvec, rvec = board_tvec.squeeze(), board_rvec.squeeze()

                        print(ids)
                        ids_list = [l[0] for l in ids.tolist()]
                        for list_idx, corner_id in enumerate(ids_list):
                            rvec_aruco, tvec_aruco = all_rvec[list_idx, 0, :], all_tvec[list_idx, 0, :]
                            # TODO not resilient to blocking of the marker yet. cv2.error: OpenCV(4.6.0) /croot/opencv-suite_1676452025216/work/modules/calib3d/src/calibration.cpp:604: error: (-5:Bad argument) Rotation must be represented by 1x3 or 3x1 floating-point rotation vector, or 3x3 rotation matrix in function 'cvProjectPoints2Internal'
                            cv2.drawFrameAxes(camera_color_img, camera_matrix, dist_coeffs, rvec_aruco, tvec_aruco, marker_length)

                            # TODO I need charuco or aruco board and/or multiple markers... Use board beside arm and then use that as initial guess
                        
                            # TODO I need to look at a specific id... How will I change this in future? and make it more variable and stuff? Remember first id I see?
                            # if corner_id == 2:

                        # BOARD pose below!



                        # tvec, rvec = board_tvec, board_rvec  # TODO not needed due to squeeze?

                        # refactoring fail but a lesson in here. TODO
                        # cam2arm, arm2cam, roll_marker, pitch_marker, yaw_marker, roll_camera, pitch_camera, yaw_camera = get_transforms_and_euler_angles_to_marker(tvec, rvec)

                        cam2arm, arm2cam, R_tc, R_ct, pos_camera = create_homogenous_transformations(tvec, rvec)

                        # -- Get the attitude in terms of euler 321 (Needs to be flipped first)
                        roll_marker, pitch_marker, yaw_marker = rotationMatrixToEulerAngles(R_flip * R_tc)
                        # -- Get the attitude of the camera respect to the frame
                        roll_camera, pitch_camera, yaw_camera = rotationMatrixToEulerAngles(R_flip * R_ct)  # todo no flip needed?

                        # marker_pose_history.append((tvec[0], tvec[1], tvec[2], 
                        #                             math.degrees(roll_marker), math.degrees(pitch_marker), math.degrees(yaw_marker)))
                        # avg_6dof_pose = []
                        # for idx in range(6):
                        #     marker_pose_history_idx_val = np.mean([v[idx] for v in marker_pose_history]).item()
                        #     avg_6dof_pose.append(marker_pose_history_idx_val)

                        # TODO understand all of the below intuitively. 
                        # -- Print the tag position in camera frame
                        str_position = "CHARUCO Position x={:.5f}  y={:.5f}  z={:.5f}".format(tvec[0], tvec[1], tvec[2])
                        cv2.putText(camera_color_img, str_position, (0, start_y), font, text_size, (0, 255, 0), 2, cv2.LINE_AA)

                        # -- Print the marker's attitude respect to camera frame
                        str_attitude = "CHARUCO Attitude r={:.5f}  p={:.5f}  y={:.5f}".format(
                            math.degrees(roll_marker), math.degrees(pitch_marker),
                            math.degrees(yaw_marker))
                        cv2.putText(camera_color_img, str_attitude, (0, start_y + jump_amt * 1), font, text_size, (0, 255, 0), 2, cv2.LINE_AA)

                        str_position = "CAMERA Position x={:.5f}  y={:.5f}  z={:.5f}".format(
                            pos_camera[0].item(), pos_camera[1].item(), pos_camera[2].item())
                        cv2.putText(camera_color_img, str_position, (0, start_y + jump_amt * 2), font, text_size, (0, 255, 0), 2, cv2.LINE_AA)

                        str_attitude = "CAMERA Attitude r={:.5f}  p={:.5f}  y={:.5f}".format(
                            math.degrees(roll_camera), math.degrees(pitch_camera),
                            math.degrees(yaw_camera))
                        cv2.putText(camera_color_img, str_attitude, (0, start_y + jump_amt * 3), font, text_size, (0, 255, 0), 2, cv2.LINE_AA)

                        # str_marker_pose_avg = 'M pos: ({:.5f} {:.5f} {:.5f}), angles: ({:.5f} {:.5f} {:.5f})'.format(
                        #     avg_6dof_pose[0], avg_6dof_pose[1], avg_6dof_pose[2], avg_6dof_pose[3], avg_6dof_pose[4], avg_6dof_pose[5]
                        # )

                        # cv2.putText(camera_color_img, str_marker_pose_avg, (0, start_y + jump_amt * 4), font, text_size, (0, 255, 0), 2, cv2.LINE_AA)

                    except Exception as e:
                        print(e)
                        print('FAILURE. board_rvec')
                        print(board_rvec)
                    
            
            images = np.hstack((camera_color_img, depth_colormap))

            cv2.imshow("image", images)
            k = cv2.waitKey(1)

            if k == ord('q'):
                cv2.destroyAllWindows()
                pipeline.stop()
                break

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