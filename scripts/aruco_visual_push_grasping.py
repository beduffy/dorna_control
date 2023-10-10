#!/usr/bin/env python
"""
Copied from visual push grasping and converted to use aruco markers instead of checkerboard
"""

import sys
import time

from cv2 import aruco
import pyrealsense2 as rs
# from rs import rs2_deproject_pixel_to_point
import matplotlib.pyplot as plt
import numpy as np
import cv2
# from real.camera import Camera
# from robot import Robot
from scipy import optimize
from mpl_toolkits.mplot3d import Axes3D

# User options (change me)
# --------------- Setup options ---------------
# tcp_host_ip = '100.127.7.223' # IP and port to robot arm as TCP client (UR5)
# tcp_port = 30002
# rtc_host_ip = '100.127.7.223' # IP and port to robot arm as real-time client (UR5)
# rtc_port = 30003
workspace_limits = np.asarray([[0.3, 0.748], [0.05, 0.4], [-0.2,
                                                           -0.1]])  # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
calib_grid_step = 0.05
checkerboard_offset_from_tool = [0, -0.13, 0.02]  # todo important
checkerboard_offset_from_tool = [0, 0, 0.0]  # todo important
tool_orientation = [-np.pi / 2, 0, 0]  # [0,-2.22,2.22] # [2.22,2.22,0]  #todo important!!

# intrinsics
width, height, fx, fy, ppx, ppy = (640.0, 480.0, 612.14801862, 611.29345062, 340.03640321,
                                   230.06928807)
print('workspace_limits: \n', workspace_limits)
# ---------------------------------------------


# Construct 3D calibration grid across workspace
gridspace_x = np.linspace(workspace_limits[0][0], workspace_limits[0][1],
                          1 + (workspace_limits[0][1] - workspace_limits[0][0]) / calib_grid_step)
gridspace_y = np.linspace(workspace_limits[1][0], workspace_limits[1][1],
                          1 + (workspace_limits[1][1] - workspace_limits[1][0]) / calib_grid_step)
gridspace_z = np.linspace(workspace_limits[2][0], workspace_limits[2][1],
                          1 + (workspace_limits[2][1] - workspace_limits[2][0]) / calib_grid_step)
calib_grid_x, calib_grid_y, calib_grid_z = np.meshgrid(gridspace_x, gridspace_y, gridspace_z)
num_calib_grid_pts = calib_grid_x.shape[0] * calib_grid_x.shape[1] * calib_grid_x.shape[2]
calib_grid_x.shape = (num_calib_grid_pts, 1)
calib_grid_y.shape = (num_calib_grid_pts, 1)
calib_grid_z.shape = (num_calib_grid_pts, 1)
calib_grid_pts = np.concatenate((calib_grid_x, calib_grid_y, calib_grid_z), axis=1)

measured_pts = []
observed_pts = []
observed_pix = []

# Move robot to home pose
# print('Connecting to robot...')
# robot = Robot(False, None, None, workspace_limits,
#               tcp_host_ip, tcp_port, rtc_host_ip, rtc_port,
#               False, None, None)
# robot.open_gripper()

# Slow down robot
# robot.joint_acc = 1.4
# robot.joint_vel = 1.05

# Make robot gripper point upwards
# robot.move_joints([-np.pi, -np.pi/2, np.pi/2, 0, np.pi/2, np.pi])
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# rs2_deproject_pixel_to_point

# todo click start and end of ruler and make sure 30cm  https://dev.intelrealsense.com/docs/rs-measure

# Move robot to each calibration point in workspace
print('Collecting data...')
num_checkerboards_found = 0

aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters_create()

for i in range(10):
    # to get lighting working
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

depth_units = 0.001

# for calib_pt_idx in range(num_calib_grid_pts):
for calib_pt_idx in range(100):
# for calib_pt_idx in range(1):
    # tool_position = calib_grid_pts[calib_pt_idx,:]
    tool_position = np.array([0., 0., 0.])
    # robot.move_to(tool_position, tool_orientation)  # todo need to measure IK errors to stop accumulation
    # todo first just move checkerboard and then raw_input() 'y' to say moving has stopped?
    # todo use something other than checkerboard?

    # time.sleep(1)

    # Find checkerboard center
    # checkerboard_size = (3,3)
    checkerboard_size = (4, 3)
    refine_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # camera_color_img, camera_depth_img = robot.get_camera_data()
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics  # both are not the same
    color_intrin = color_frame.profile.as_video_stream_profile().intrinsics

    camera_depth_img = np.asanyarray(depth_frame.get_data())
    camera_color_img = np.asanyarray(color_frame.get_data())
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(camera_depth_img, alpha=0.03),
                                       cv2.COLORMAP_JET)



    bgr_color_data = cv2.cvtColor(camera_color_img, cv2.COLOR_RGB2BGR)
    gray_data = cv2.cvtColor(bgr_color_data, cv2.COLOR_RGB2GRAY)
    # checkerboard_found, corners = cv2.findChessboardCorners(gray_data, checkerboard_size, None,
    #                                                         cv2.CALIB_CB_ADAPTIVE_THRESH)
    # checkerboard_found = False

    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray_data, aruco_dict, parameters=parameters)
    # frame_markers = aruco.drawDetectedMarkers(bgr_color_data.copy(), corners, ids)
    frame_markers = aruco.drawDetectedMarkers(camera_color_img.copy(), corners, ids)

    if ids is not None:
        checkerboard_found = True
        # plt.figure()
        # plt.imshow(frame_markers)
        for i in range(len(ids)):
            c = corners[i][0]
            cv2.circle(camera_color_img, (int(c[:, 0].mean()), int(c[:, 1].mean())), 3, color=(0,255,0))
        #     plt.plot([c[:, 0].mean()], [c[:, 1].mean()], "o", label="id={0}".format(ids[i]))
        #
        # # plt.show()

        marker_length = 0.0265
        camera_matrix = np.array([[612.14801862, 0., 340.03640321],
                                  [0., 611.29345062, 230.06928807],
                                  [0., 0., 1.]])
        # todo don't use the above. Use provided calibration. depth or color
        # dist_coeffs = np.array([[1.80764862e-02, 1.09549436e+00, -3.38044260e-03, 4.04543459e-03, -4.26585263e+00]])
        dist_coeffs = np.array(
            [1.80764862e-02, 1.09549436e+00, -3.38044260e-03, 4.04543459e-03, -4.26585263e+00])

        # rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix,
                                                        dist_coeffs)
        # aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec[0], tvec[0], 0.1)
        # aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec[0], tvec[0], 0.1)

        rotation_matrix, jacobian = cv2.Rodrigues(rvec[0])



        marker_pose_mat = np.zeros((4, 4))
        marker_pose_mat[:3, :3] = rotation_matrix
        marker_pose_mat[0, 3] = tvec[0][0][0]
        marker_pose_mat[1, 3] = tvec[0][0][1]
        marker_pose_mat[2, 3] = tvec[0][0][2]
        marker_pose_mat[3, 3] = 1.
        np.savetxt('data/camera_pose_marker_pose.txt', marker_pose_mat, delimiter=' ')
        # todo not every time?
        # todo what is this?


        # todo why does RMSE go to 0 but depth scale optimisation also? Because only 1 tool position?

        cam_pos = -np.matrix(rotation_matrix).T * np.matrix(tvec[0].T)
        print('Cam pos: ', cam_pos)


        for i in range(len(ids)):
            # aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec[i], tvec[i], 0.1)
            aruco.drawAxis(camera_color_img, camera_matrix, dist_coeffs, rvec[i], tvec[i], marker_length)
    else:
        checkerboard_found = False

    # Stack both images horizontally
    images = np.hstack((camera_color_img, depth_colormap))

    # Show images
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RealSense', images)
    cv2.waitKey(1)

    if checkerboard_found:
        # import pdb;pdb.set_trace()

        # corners_refined = cv2.cornerSubPix(gray_data, corners, (3, 3), (-1, -1), refine_criteria)

        # img = cv2.drawChessboardCorners(camera_color_img, checkerboard_size, corners_refined, checkerboard_found)
        # cv2.imshow('img', img)
        # cv2.waitKey(10)

        # Get observed checkerboard center 3D point in camera space
        # checkerboard_pix = np.round(corners_refined[4, 0, :]).astype(int)  # 5th corner
        c = corners[0][0]  # should only be one
        checkerboard_pix = np.array([int(round(c[:, 0].mean())), int(round(c[:, 1].mean()))])
        # todo one example was 659 so 65.9cm? when it was 40cm?
        checkerboard_z = camera_depth_img[checkerboard_pix[1]][checkerboard_pix[0]]
        # depth_pixel = [checkerboard_pix[1]], [checkerboard_pix[0]]
        # depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel_coordinate,
        #                                               depth_value_in_meter)  # todo??
        # checkerboard_x = np.multiply(checkerboard_pix[0]-robot.cam_intrinsics[0][2],checkerboard_z/robot.cam_intrinsics[0][0])
        # checkerboard_y = np.multiply(checkerboard_pix[1]-robot.cam_intrinsics[1][2],checkerboard_z/robot.cam_intrinsics[1][1])

        # assuming it's your regular 3x3 camera intrinsics
        checkerboard_x = np.multiply(checkerboard_pix[0] - ppx, checkerboard_z / fx)
        checkerboard_y = np.multiply(checkerboard_pix[1] - ppy, checkerboard_z / fy)

        if checkerboard_z == 0:
            print('Z value not found in depth image')
            continue
        else:
            print('Z value found: {}'.format(checkerboard_z))
            checkerboard_z *= depth_units  # todo why is my depth so bad here!?!!??!?!?!
            print('depth_frame.get_distance(checkerboard_pix[0], checkerboard_pix[1]): ', depth_frame.get_distance(checkerboard_pix[0], checkerboard_pix[1]))

        num_checkerboards_found += 1
        # Save calibration point and observed checkerboard center
        observed_pts.append([checkerboard_x, checkerboard_y, checkerboard_z])  # 3D point in camera coords
        # tool_position[2] += checkerboard_offset_from_tool
        tool_position = tool_position + checkerboard_offset_from_tool

        measured_pts.append(tool_position)  # 3D point in arm base coords
        observed_pix.append(checkerboard_pix)  # 2D pixel coordinate # 5th point

        # Draw and display the corners
        # vis = cv2.drawChessboardCorners(robot.camera.color_data, checkerboard_size, corners_refined, checkerboard_found)
        # draw 5th corner
        # vis = cv2.drawChessboardCorners(bgr_color_data, (1, 1), corners_refined[4, :, :],
        #                                 checkerboard_found)  # todo what is this? draw 1st row 1st col?
        # cv2.imwrite('data/aruco_images/%06d.png' % len(measured_pts), vis)
        # cv2.imshow('Calibration', vis)
        # cv2.waitKey(100)
        # cv2.waitKey(1000)
        # break

# Move robot back to home pose
# robot.go_home()

if num_checkerboards_found == 0:
    sys.exit('0 checkerboards found')
print('{} checkerboards found (including ones where z value was not 0'.format(num_checkerboards_found))
measured_pts = np.asarray(measured_pts)  # 3D points in arm base coords
observed_pts = np.asarray(observed_pts)  # 3D points in camera coords
observed_pix = np.asarray(observed_pix)  # 2D pixels coordinate # 5th point
world2camera = np.eye(4)


# Estimate rigid transform with SVD (from Nghia Ho)
def get_rigid_transform(A, B):
    # this explains it https://www.lucidar.me/en/mathematics/calculating-the-transformation-between-two-set-of-points/
    assert len(A) == len(B)
    N = A.shape[0]  # Total points
    # translation
    centroid_A = np.mean(A, axis=0)  # Centre of gravity (COG)
    centroid_B = np.mean(B, axis=0)
    AA = A - np.tile(centroid_A, (N, 1))  # Centre the points. Take COG away from all points
    BB = B - np.tile(centroid_B, (N, 1))
    # rotation
    H = np.dot(np.transpose(AA), BB)  # Dot is matrix multiplication for array
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0:  # Special reflection case
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)
    t = np.dot(-R, centroid_A.T) + centroid_B.T
    return R, t


def get_rigid_transform_error(z_scale):
# def get_rigid_transform_error():
    global measured_pts, observed_pts, observed_pix, world2camera, camera

    # Apply z offset and compute new observed points using camera intrinsics
    observed_z = observed_pts[:, 2:] * z_scale
    # observed_z = observed_pts[:, 2:]
    # observed_x = np.multiply(observed_pix[:,[0]]-robot.cam_intrinsics[0][2],observed_z/robot.cam_intrinsics[0][0])
    # observed_y = np.multiply(observed_pix[:,[1]]-robot.cam_intrinsics[1][2],observed_z/robot.cam_intrinsics[1][1])
    # todo are these color or depth intrinsics?
    observed_x = np.multiply(observed_pix[:, [0]] - ppx, observed_z / fx)
    observed_y = np.multiply(observed_pix[:, [1]] - ppy, observed_z / fy)
    new_observed_pts = np.concatenate((observed_x, observed_y, observed_z), axis=1)

    # Estimate rigid transform between measured points (arm coords) and new observed points (z-scale corrected 3D camera points)
    R, t = get_rigid_transform(np.asarray(measured_pts), np.asarray(new_observed_pts))
    t.shape = (3, 1)
    world2camera = np.concatenate((np.concatenate((R, t), axis=1), np.array([[0, 0, 0, 1]])),
                                  axis=0)

    # Compute rigid transform error
    registered_pts = np.dot(R, np.transpose(measured_pts)) + np.tile(t, (1, measured_pts.shape[0]))
    error = np.transpose(registered_pts) - new_observed_pts
    error = np.sum(np.multiply(error, error))
    rmse = np.sqrt(error / measured_pts.shape[0])
    print('error: {}. rmse: {}'.format(error, rmse))
    return rmse


# Optimize z scale w.r.t. rigid transform error
print('Calibrating...')
z_scale_init = 1
optim_result = optimize.minimize(get_rigid_transform_error, np.asarray(z_scale_init),
                                 method='Nelder-Mead')

# todo attempt to not optimize z
# optim_result = optimize.minimize(get_rigid_transform_error, #np.asarray(z_scale_init),
#                                  method='Nelder-Mead')
camera_depth_offset = optim_result.x

# Save camera optimized offset and camera pose
print('Saving ...')
np.savetxt('data/camera_depth_scale.txt', camera_depth_offset, delimiter=' ')
get_rigid_transform_error(camera_depth_offset)
camera_pose = np.linalg.inv(world2camera)
np.savetxt('data/camera_pose.txt', camera_pose, delimiter=' ')
print('Done.')

import pdb;pdb.set_trace()


# todo below
# DEBUG CODE -----------------------------------------------------------------------------------

sys.exit()
# todo understand all below and comment to explain it.
np.savetxt('data/measured_pts.txt', np.asarray(measured_pts), delimiter=' ')  # todo IndexError: too many indices for array
np.savetxt('data/observed_pts.txt', np.asarray(observed_pts), delimiter=' ')
np.savetxt('data/observed_pix.txt', np.asarray(observed_pix), delimiter=' ')
measured_pts = np.loadtxt('data/measured_pts.txt', delimiter=' ')
observed_pts = np.loadtxt('data/observed_pts.txt', delimiter=' ')
observed_pix = np.loadtxt('data/observed_pix.txt', delimiter=' ')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# plot all arm base coordinates
ax.scatter(measured_pts[:, 0], measured_pts[:, 1], measured_pts[:, 2], c='blue')

print('camera depth offset')
print(camera_depth_offset)
R, t = get_rigid_transform(np.asarray(measured_pts), np.asarray(observed_pts))
t.shape = (3, 1)
camera_pose = np.concatenate((np.concatenate((R, t), axis=1), np.array([[0, 0, 0, 1]])), axis=0)
camera2robot = np.linalg.inv(camera_pose)
t_observed_pts = np.transpose(
    np.dot(camera2robot[0:3, 0:3], np.transpose(observed_pts)) + np.tile(camera2robot[0:3, 3:], (
    1, observed_pts.shape[0])))

# plot all camera 3D coordinates
ax.scatter(t_observed_pts[:, 0], t_observed_pts[:, 1], t_observed_pts[:, 2], c='red')

new_observed_pts = observed_pts.copy()
new_observed_pts[:, 2] = new_observed_pts[:, 2] * camera_depth_offset[0]
R, t = get_rigid_transform(np.asarray(measured_pts), np.asarray(new_observed_pts))
t.shape = (3, 1)
camera_pose = np.concatenate((np.concatenate((R, t), axis=1), np.array([[0, 0, 0, 1]])), axis=0)
camera2robot = np.linalg.inv(camera_pose)
t_new_observed_pts = np.transpose(
    np.dot(camera2robot[0:3, 0:3], np.transpose(new_observed_pts)) + np.tile(camera2robot[0:3, 3:],
                                                                             (1,
                                                                              new_observed_pts.shape[
                                                                                  0])))
# todo plot whaT??
ax.scatter(t_new_observed_pts[:, 0], t_new_observed_pts[:, 1], t_new_observed_pts[:, 2], c='green')

plt.show()