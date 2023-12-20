from __future__ import print_function
import sys
import traceback
from glob import glob
import math
from collections import deque

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

from lib.vision import euler_yzx_to_axis_angle, rotationMatrixToEulerAngles, create_homogenous_transformations

mesh_box_opt = o3d.geometry.TriangleMesh.create_box(width=0.1, height=0.1, depth=0.003)
mesh_box_opt.paint_uniform_color([1, 0, 0])

size = 0.1
origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0.0, 0.0, 0.0])
coordinate_frame_1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0.3, 0.0, 0.0])
coordinate_frame_2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0.4, 0.0, 0.0])
coordinate_frame_3 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0.5, 0.0, 0.0])

camera_coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size * 1.3, origin=[0.0, 0.0, 0.0])

# rvec = np.array([1.0, 1.0, 1.0])
# z, x, y, angle = euler_yzx_to_axis_angle(np.pi / 2, 0, 0)  # pointing left in yaw, 45 deg. red.
# z, x, y, angle = euler_yzx_to_axis_angle(0, np.pi / 2, 0)  # pointing up in pitch 45 degrees. 
# z, x, y, angle = euler_yzx_to_axis_angle(0, 0, np.pi / 2)  # rolling right 45 degrees.

# I want camera to the right, pointing to the left 45 degrees left in yaw and 45 degrees down
z, x, y, angle = euler_yzx_to_axis_angle(np.pi / 2, -np.pi / 2, 0)
rvec = np.array([x, y, z])
camera_position_rel_to_origin = np.array([0.0, 0.3, 0.4])
tvec = camera_position_rel_to_origin   # TODO why does 0.3 bring it down but 
cam2_arm, arm2_cam, R_tc, R_ct, pos_camera = create_homogenous_transformations(tvec, rvec)
# camera_transformation = np.array(cam2_arm)
camera_transformation = np.array(arm2_cam)  # makes more intuitive sense, 0.3 tvec brings bigger coordinate frame forwards. 

camera_coordinate_frame.transform(camera_transformation)

#--- 180 deg rotation matrix around the x axis
R_flip       = np.zeros((3, 3), dtype=np.float32)
R_flip[0, 0] =  1.0
R_flip[1, 1] = -1.0
R_flip[2, 2] = -1.0

# TODO I need to understand conversion between all rotation types better
# TODO why are both below the same?
# -- Get the attitude in terms of euler 321 (Needs to be flipped first)
roll_marker, pitch_marker, yaw_marker = rotationMatrixToEulerAngles(R_flip * R_tc)
print('roll pitch yaw marker: ', roll_marker, pitch_marker, yaw_marker)
# -- Get the attitude of the camera respect to the frame
roll_camera, pitch_camera, yaw_camera = rotationMatrixToEulerAngles(R_flip * R_ct)  # todo no flip needed?
print('roll pitch yaw camera: ', roll_camera, pitch_camera, yaw_camera)



gripper_t = np.array([[0.3, 0.0, 0.0], [0.4, 0.0, 0.0], [0.5, 0.0, 0.0], [0.6, 0.0, 0.0]]) 
  
# hand_rotations = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 
#                             [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])  # non parallel motions make output be identity. ahhh
# hand_rotations = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.3], 
#                             [0.0, 0.0, 0.1], [0.0, 0.2, 0.0]]) 

# only rotation in roll since that won't ruin my simple translation. But they're still parallel? well only the x axis?
hand_rotations = []
rotations_in_roll = [0.1, -0.1, 0.2, 0]
for i in range(gripper_t.shape[0]):
    z, x, y, angle = euler_yzx_to_axis_angle(0, 0, rotations_in_roll[i])
    hand_rotations.append(np.array([x, y, z]))
hand_rotations = np.array(hand_rotations)

# TODO study spatial algebra. Find transformation between any two coordinate frames
# TODO nah, I create the transformation, so just create 3-6+ non-stupid transformations (translation is always easy), rotation matrices I'm finding above
# TODO well continuing. Easy to find all transformation from origin/arm base to gripper. a bit harder to find from camera to gripper. Can I just add both? probably yes! 

# # rotation matrix between the target and camera 
# R_target2cam = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [ 
#                         0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]) 
  
# # translation vector between the target and camera 
# t_target2cam = np.array([0.0, 0.0, 0.0, 0.0]) 

# camera_position_rel_to_origin = np.array([0.0, 0.3, 0.4])
t_target2cam = []
R_target2cam = []
print('camera_position_rel_to_origin: ', camera_position_rel_to_origin)
for idx in range(gripper_t.shape[0]):
    # print(idx, gripper_t[idx])
    diff_in_translation = camera_position_rel_to_origin - gripper_t[idx]
    # diff_in_translation = gripper_t[idx] - camera_position_rel_to_origin
    print(idx, gripper_t[idx])
    print(diff_in_translation)
    t_target2cam.append(diff_in_translation)
    R_target2cam.append(R_tc)

    # 0 [0.3 0.  0. ]
    # [-0.3  0.3  0.4]  # seems reasonable. We need to subtract 0.3 in x, move 0.3 to the right and 0.4 up in z
    # 1 [0.4 0.  0. ]
    # [-0.4  0.3  0.4]
    # 2 [0.5 0.  0. ]
    # [-0.5  0.3  0.4]


# https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#gaebfc1c9f7434196a374c382abf43439b
# InputArrayOfArrays 	R_gripper2base,
# InputArrayOfArrays 	t_gripper2base,
# InputArrayOfArrays 	R_target2cam,
# InputArrayOfArrays 	t_target2cam,

# R_gripper2base	Rotation part extracted from the homogeneous matrix that transforms a point expressed in the gripper frame to the robot base frame ( bTg). This is a vector (vector<Mat>) that contains the rotation, (3x3) rotation matrices or (3x1) rotation vectors, for all the transformations from gripper frame to robot base frame.
# t_gripper2base	Translation part extracted from the homogeneous matrix that transforms a point expressed in the gripper frame to the robot base frame ( bTg). This is a vector (vector<Mat>) that contains the (3x1) translation vectors for all the transformations from gripper frame to robot base frame.
# R_target2cam	Rotation part extracted from the homogeneous matrix that transforms a point expressed in the target frame to the camera frame ( cTt). This is a vector (vector<Mat>) that contains the rotation, (3x3) rotation matrices or (3x1) rotation vectors, for all the transformations from calibration target frame to camera frame.
# t_target2cam	Rotation part extracted from the homogeneous matrix that transforms a point expressed in the target frame to the camera frame ( cTt). This is a vector (vector<Mat>) that contains the (3x1) translation vectors for all the transformations from calibration target frame to camera frame.

# A minimum of 2 motions with non parallel rotation axes are necessary to determine the hand-eye transformation. 
# So at least 3 different poses are required, but it is strongly recommended to use many more poses.

# https://www.geeksforgeeks.org/calibratehandeye-python-opencv/

# transformation matrix 
# T, R = cv2.calibrateHandEye(gripper_t, eye_coords,  # TODO but docs say order of inputs is: Well first two could be both rotation and translation in 3vecs???
#                             R_target2cam, t_target2cam) 
# R, T = cv2.calibrateHandEye(hand_rotations, gripper_t, 
#                             hand_rotations, gripper_t)  # creates identity, which is good
R, T = cv2.calibrateHandEye(hand_rotations, gripper_t,   # probably done unless inverse TODO prove it. if gripper is 0.3 forward do I need to subtract 0.3 to get to robot base frame? but gripper frame is 0.3 forward
                            R_target2cam, t_target2cam)
print('\nR and T')
print(R)
print(T)

estimated_cam2_arm_transform = np.identity(4)
estimated_cam2_arm_transform[:3, :3] = R
estimated_cam2_arm_transform[0, 3] = T[0]
estimated_cam2_arm_transform[1, 3] = T[1]
estimated_cam2_arm_transform[2, 3] = T[2]

estimated_camera_coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size * 1.5, origin=[0.0, 0.0, 0.0])
estimated_camera_coordinate_frame.transform(estimated_cam2_arm_transform)


arm_position_coord_frames = [coordinate_frame_1, coordinate_frame_2, coordinate_frame_3]
list_of_geometry_elements = [origin_frame, camera_coordinate_frame, estimated_camera_coordinate_frame] + arm_position_coord_frames
o3d.visualization.draw_geometries(list_of_geometry_elements)


# TODO the big goal here would be to:
# 1. Using the 3 transformations from origin to the 3 coord frames of "fake arm" and then # TODO calculate simple. Identity rotation mat and translation 0.3, 0.4, 0.5
# 2. 3 transformations from camera to each of these coord frames. TODO how do I find these transformations. Translation is easy. Rotation is same as R_tc or R_ct? Can always prove I have things correct
# 3. Use opencv handeye function and then plot output transformation? 

'''
Can i do all of this in simulation without even using realsense? 
E.g. Put all objects and ground truth into simulation, visualise, then compute. 
Use Drake or Open3D? Well putting aruco into these is hard. But I don’t need to? I have all the transforms. I don’t even need to visualise depth cameras.
'''