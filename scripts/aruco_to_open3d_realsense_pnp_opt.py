from __future__ import print_function
import sys
import traceback
from glob import glob
import math

try:
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
from lib.vision import get_full_pcd_from_rgbd, convert_cam_pcd_to_arm_pcd
from lib.vision_config import camera_matrix, dist_coeffs, pinhole_camera_intrinsic
from lib.realsense_helper import setup_start_realsense, realsense_get_frames, run_10_frames_to_wait_for_auto_exposure, use_aruco_corners_and_realsense_for_3D_point
from lib.aruco_helper import create_aruco_params, aruco_detect_draw_get_transforms
from lib.dorna_kinematics import i_k, f_k
from lib.aruco_image_text import OpenCvArucoImageText

# export PYTHONPATH=$PYTHONPATH:/home/ben/all_projects/dorna_control

if __name__ == '__main__':

    depth_intrin, color_intrin, depth_scale, pipeline, align, spatial = setup_start_realsense()

    board, parameters, aruco_dict, marker_length = create_aruco_params()

    opencv_aruco_image_text = OpenCvArucoImageText()

    size = 0.1
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=size, origin=[0.0, 0.0, 0.0])

    cam2arm = np.identity(4)
    saved_cam2arm = cam2arm
    marker_pose_history = deque([], maxlen=100)  # TODO is this less noisy and accurate? And should be used?
    colors = ((255, 255, 255), (255, 0, 0), (0, 0, 0), (0, 0, 255))

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
            
            camera_color_img_orig = camera_color_img.copy()
            h, w = camera_color_img.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 1, (w,h))
            dst = cv2.undistort(camera_color_img, camera_matrix, dist_coeffs, None, newcameramtx)
            x, y, w, h = roi
            dst = dst[y:y+h, x:x+w]
            # https://stackoverflow.com/questions/73599224/aruco-markerdetector-undistorts-image
            # https://forum.opencv.org/t/aruco-marker-in-a-calibrated-image/7325
            # TODO how to use cropped image?
            # camera_color_img = dst

            # bgr_color_data = cv2.cvtColor(camera_color_img, cv2.COLOR_RGB2BGR)
            # bgr_color_data = cv2.cvtColor(camera_color_img.copy(), cv2.COLOR_RGB2BGR)
            bgr_color_data = cv2.cvtColor(dst, cv2.COLOR_RGB2BGR)
            gray_data = cv2.cvtColor(bgr_color_data, cv2.COLOR_RGB2GRAY)

            camera_color_img_debug = camera_color_img.copy()  # modified

            # corners, ids, all_rvec, all_tvec = aruco_detect_draw_get_transforms(gray_data, camera_color_img_debug, aruco_dict, parameters, marker_length, camera_matrix, dist_coeffs)
            corners, ids, rejectedImgPoints = aruco.detectMarkers(gray_data, aruco_dict,
                                                                parameters=parameters)
            # TODO Note DETECTmARKERS
            # The function does not correct lens distortion or takes it into account. It's recommended to undistort input image with corresponding camera model, if camera parameters are known
            frame_markers = aruco.drawDetectedMarkers(camera_color_img, corners, ids)  # TODO separate elsewhere? This function does too much?
            all_rvec, all_tvec, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)
            # TODO refine markers! https://docs.opencv.org/4.x/db/da9/tutorial_aruco_board_detection.html


            # only 11 markers... are they not matched correctly?
            # # obj_points, image_points = board.matchImagePoints(corners, ids)
            # # obj_points, image_points = cv2.aruco.getBoardObjectAndImagePoints(board, corners, ids)

            # # plt.scatter(obj_points[:, 0, 0], obj_points[:, 0, 1])
            # # plt.show()

            # actually gets 12 markers
            obj_points = board.objPoints  # in tuple format, hmm
            # for plotting it
            all_obj_points_x = []
            all_obj_points_y = []
            for marker in obj_points:
                for obj_corner in marker:
                    all_obj_points_x.append(obj_corner[0])
                    all_obj_points_y.append(obj_corner[1])
            # plt.scatter(all_obj_points_x, all_obj_points_y)
            # plt.show()

            

            # doesn't work
            # retval, board_rvec, board_tvec = cv2.aruco.estimatePoseBoard(corners, ids, board, camera_matrix, dist_coeffs, np.empty(1), np.empty(1))
            # if board_rvec is not None and board_rvec.shape[0] == 3:
            #     cam2arm_board, arm2cam_board, _, _, _ = create_homogenous_transformations(board_tvec, board_rvec)

            if ids is not None:
                # obj_points are in id 1-12 order
                # corners are in ids order. So just sort ids?
                # ids_list: [4, 3, 2, 1, 8, 7, 6, 5, 12, 11, 10, 9]
                # we want to get corner index 3 (id 1) first then index 2, etc
                ids_list = [l[0] for l in ids.tolist()]
                ids_list_with_index_key_tuple = [(idx, id_) for idx, id_ in enumerate(ids_list)]
                ids_list_sorted = sorted(ids_list_with_index_key_tuple, key=lambda x: x[1])

                image_points = []
                ids_list = [x[0] for x in ids.tolist()]
                for idx_of_marker, id_of_marker in ids_list_sorted:
                    image_points.append(corners[idx_of_marker].squeeze())
                
                # Re-projection error gives a good estimation of just how exact the found parameters are. 
                # mean_error = 0
                # for i in range(len(obj_points)):
                #     imgpoints2, _ = cv2.projectPoints(obj_points[i], all_rvec[i], all_tvec[i], camera_matrix, dist_coeffs)
                #     error = cv2.norm(image_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                #     mean_error += error
                # print("total error: {}".format(mean_error / len(obj_points)) )
                # TODO fix or just do my own reprojection error based on above. 
                '''
                InputArray, cv::InputArray, int, cv::InputArray)'
                > Input type mismatch (expected: '_src1.type() == _src2.type()'), where
                >     '_src1.type()' is 5 (CV_32FC1)
                > must be equal to
                >     '_src2.type()' is 13 (CV_32FC2)


                '''
                            
                # if image_points is not None:
                #     for square_corner_idx in range(image_points.shape[0]):
                #         x, y = image_points[square_corner_idx][0]
                #         cv2.circle(camera_color_img_debug, (int(x), int(y)), 4, colors[0], -1)
                # print(image_points.shape)

                print(ids_list)  # TODO why are they always in order now??!!?
                
                for list_idx, corner_id in enumerate(ids_list):
                    rvec_aruco, tvec_aruco = all_rvec[list_idx, 0, :], all_tvec[list_idx, 0, :]
                    cv2.drawFrameAxes(camera_color_img_debug, camera_matrix, dist_coeffs, rvec_aruco, tvec_aruco, marker_length)

                # below seems much more accurate at 60cm. How to use it?
                center = use_aruco_corners_and_realsense_for_3D_point(depth_frame, corners[list_idx], color_intrin)
                print('Center: {}'.format(center))  # TODO should print marker xyz here to make it easier to compare

                tvec, rvec = tvec_aruco, rvec_aruco  # for easier assignment if multiple markers later.

                half_marker_len = marker_length / 2
                # clockwise, top left has the usual aruco circle. My colours: white, blue, black, red
                top_left_first = np.array([-half_marker_len, half_marker_len, 0.])
                top_right_first = np.array([half_marker_len, half_marker_len, 0.])
                bottom_right_first = np.array([half_marker_len, -half_marker_len, 0.])
                bottom_left_first = np.array([-half_marker_len, -half_marker_len, 0.])

                corners_3d_points = np.array([top_left_first, top_right_first, bottom_right_first, bottom_left_first])

                # imagePointsCorners, jacobian = cv2.projectPoints(corners_3d_points, rvec, tvec, camera_matrix, dist_coeffs)
                # for idx, (x, y) in enumerate(imagePointsCorners.squeeze().tolist()):
                #     cv2.circle(camera_color_img_debug, (int(x), int(y)), 4, colors[idx], -1)
                
                # One validation that corners and corner_3d_points match: 
                # imagePointsCorners: [[274.0870951, 196.669700], [302.41902, 202.462821], [292.807514, 225.908426], [263.7825, 219.728288]]
                # corners: [[[274.1794 , 196.55408], [302.33575, 202.57216], [292.88254, 225.79572], [263.69882, 219.8472 ]]]
                # TODO since they are so close, it is crazy to use solvePnP? Explain why
                # import pdb;pdb.set_trace()

                # TODO even an aruco board with solve pnp would help here right?
                # TODO glue the aruco marker to see if that helps much. 

                cam2arm, arm2cam, R_tc, R_ct, pos_camera = create_homogenous_transformations(tvec, rvec)

                # -- Get the attitude in terms of euler 321 (Needs to be flipped first)
                roll_marker, pitch_marker, yaw_marker = rotationMatrixToEulerAngles(opencv_aruco_image_text.R_flip * R_tc)
                # -- Get the attitude of the camera respect to the frame
                roll_camera, pitch_camera, yaw_camera = rotationMatrixToEulerAngles(opencv_aruco_image_text.R_flip * R_ct)  # todo no flip needed?

                marker_pose_history.append((tvec[0], tvec[1], tvec[2], 
                                            math.degrees(roll_marker), math.degrees(pitch_marker), math.degrees(yaw_marker)))

                avg_6dof_pose = []
                for idx in range(6):
                    marker_pose_history_idx_val = np.mean([v[idx] for v in marker_pose_history]).item()
                    avg_6dof_pose.append(marker_pose_history_idx_val)

                opencv_aruco_image_text.put_marker_text(camera_color_img_debug, tvec, roll_marker, pitch_marker, yaw_marker)
                opencv_aruco_image_text.put_camera_text(camera_color_img_debug, pos_camera, roll_camera, pitch_camera, yaw_camera)
                opencv_aruco_image_text.put_avg_marker_text(camera_color_img_debug, avg_6dof_pose)

                saved_cam2arm = cam2arm
            
            images = np.hstack((camera_color_img_debug, depth_colormap))
            # images = np.hstack((camera_color_img_orig, depth_colormap))
            # images = np.hstack((camera_color_img, depth_colormap))
            # images = camera_color_img
            camera_color_img = cv2.cvtColor(camera_color_img, cv2.COLOR_BGR2RGB)  # for open3D

            cv2.imshow("image", images)
            cv2.imshow('undistorted', dst)
            k = cv2.waitKey(1)

            if k == ord('q'):
                cv2.destroyAllWindows()
                pipeline.stop()
                break

            # if k == ord(''):

            if k == ord('o'):
                cam_pcd = get_full_pcd_from_rgbd(camera_color_img, camera_depth_img,
                                            pinhole_camera_intrinsic, visualise=False)
                # full_arm_pcd, full_pcd_numpy = convert_cam_pcd_to_arm_pcd(cam_pcd, saved_cam2arm, 0.0)  # TODO do I need this or not?

                aruco_coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                                        size=size, origin=[0.0, 0.0, 0.0])
                # aruco_coordinate_frame.transform(cam2arm)  # didn't work. I lack understanding TODO
                aruco_coordinate_frame.transform(arm2cam)  # on the spot but rotated wrong...  ahh did it again and it looked good. it's aruco flaws... maybe this caused all my problems? I should use charuco board?

                mesh_box = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=0.003)
                # TODO translate inwards or not?
                mesh_box.transform(arm2cam)

                # TODO USE THE DEPTH CAMERA, THE GROUND PLANE, ALL OF THIS INFORMATION TO GET A BETTER ESTIMATE of the marker. And maybe then we are better. 

                # TODO write my understanding of all of the above.
                # everything is in camera coordinate frame
                # cam2arm means the transformation to bring points from camera frame to the aruco frame
                # arm2cam brings points from aruco frame to camera frame. 
                # TODO So then why does transforming coordinate frame at origin to camera frame make it go to correct place.
               
                # corners[0][0] += 10  # this proved they were different

                # outval, rvec_pnp_opt, tvec_pnp_opt = cv2.solvePnP(corners_3d_points, corners[0], camera_matrix, dist_coeffs)
                
                # full board
                # import pdb;pdb.set_trace()
                
                # https://forum.opencv.org/t/pose-estimation-tvec-values-jumping-inconsistently/12705/2
                # the above says use PNP iterative
                
                # outval, rvec_pnp_opt, tvec_pnp_opt = cv2.solvePnP(obj_points, image_points, camera_matrix, dist_coeffs)  # TODO probably not in correct order? check 
                outval, rvec_pnp_opt, tvec_pnp_opt = cv2.solvePnP(np.concatenate(obj_points), np.concatenate(image_points), camera_matrix, dist_coeffs)  # TODO probably not in correct order? check 
                # TODO draw big board tvec rvec? 
                
                # # outval, rvec_arm, tvec_arm = cv2.solvePnP(corners_3d_points, np.hstack(corners_reordered).squeeze(), camera_matrix, dist_coeffs, rvec, tvec, useExtrinsicGuess=True)
                # # outval, rvec_arm, tvec_arm = cv2.solvePnP(corners_3d_points, np.hstack(corners_reordered).squeeze(), camera_matrix, dist_coeffs, rvec, tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_IPPE)
                # outval, rvec_arm, tvec_arm = cv2.solvePnP(corners_3d_points, np.hstack(corners_reordered).squeeze(), camera_matrix, dist_coeffs)
                # # outval, rvec_arm, tvec_arm = cv2.solvePnP(corners_3d_points, np.hstack(corners_reordered).squeeze(), camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE)
                # # outval, rvec_arm, tvec_arm = cv2.solvePnP(corners_3d_points, np.hstack(corners_reordered).squeeze(), camera_matrix, dist_coeffs, rvec, tvec, useExtrinsicGuess=True)
                # # outval, rvec_arm, tvec_arm = cv2.solvePnP(corners_3d_points, np.hstack(corners), camera_matrix, dist_coeffs, rvec, tvec, useExtrinsicGuess=True)
                # outval, rvec_arm, tvec_arm = cv2.solvePnP(corners_3d_points, corners[shoulder_motor_marker_id], camera_matrix, dist_coeffs, rvec, tvec, useExtrinsicGuess=True)
                # # outval, rvec_arm, tvec_arm = cv2.solvePnP(corners_3d_points, np.hstack(corners), camera_matrix, dist_coeffs, rvec, tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                # # outval, rvec_arm, tvec_arm = cv2.solvePnP(corners_3d_points, corners[0], camera_matrix, dist_coeffs, rvec, tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                # # outval, rvec_arm, tvec_arm = cv2.solvePnP(np.array([[0.0, -y_offset, 0.0]]), corners[shoulder_motor_marker_id], camera_matrix, dist_coeffs)

                # TODO better understand insides of that function and have good descriptions of cam2arm vs arm2cam.
                _, arm2cam_opt, _, _, _ = create_homogenous_transformations(tvec_pnp_opt, rvec_pnp_opt)
                mesh_box_opt = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=0.003)
                mesh_box_opt.paint_uniform_color([1, 0.706, 0])
                mesh_box_opt.transform(arm2cam_opt)
                # mesh_box_opt.transform(arm2cam_board)

                list_of_geometry_elements = [cam_pcd, coordinate_frame, aruco_coordinate_frame, mesh_box, mesh_box_opt]
                o3d.visualization.draw_geometries(list_of_geometry_elements)

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