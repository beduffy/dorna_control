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

from lib.vision import euler_yzx_to_axis_angle, rotationMatrixToEulerAngles, create_homogenous_transformations, get_inverse_homogenous_transform

def assert_condition_and_print(condition):
    print(condition)
    assert condition

# TODO create multiple transformations. Compose and join them together into one transformation 
# and easily understand transforming between two frames

size = 0.1
# origin frame
origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0.0, 0.0, 0.0])
origin_np = np.array([0.0, 0.0, 0.0, 1.0])

# first transformation into frame F1
translate = [0.3, 0, 0]
first_transformation = np.identity(4)
# first_transformation[:3, :3] = np.identity(3)  # no rotation
first_transformation[0, 3] = translate[0]
first_transformation[1, 3] = translate[1]
first_transformation[2, 3] = translate[2]
f1_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0.0, 0.0, 0.0])
f1_frame.transform(first_transformation)

first_transformation_inverse = get_inverse_homogenous_transform(first_transformation)

# import pdb;pdb.set_trace()

print(origin_np.shape)
print(first_transformation.shape)

print('First transformation, frame F1 (this brings points from origin frame to F1):')
print(first_transformation)
print('First transformation inverse,:')
print(first_transformation_inverse)

# testing numpy is close float check
condition = all(np.isclose(([-0.3, 0.0, 0.0]), np.array([-0.30000003, 0.00000000001, 0.00000000001])))
assert_condition_and_print(condition)

# import pdb;pdb.set_trace()
####### testing understanding of transformation and inverse transformation
# condition = all(np.isclose(np.dot(origin_np, first_transformation)[0:3], np.array([-0.3, 0.0, 0.0])))  # my first understanding was wrong
# Multiplying a point by homogenous matrix transformation brings a point from origin frame to F1 frame e.g. [0, 0, 0] -> [0.3, 0, 0]
# p_OF1_O meaning "position of F1 measured from origin expressed in origin frame"
p_OF1_O = np.dot(first_transformation, origin_np)[0:3]
condition = all(np.isclose(p_OF1_O, np.array([0.3, 0.0, 0.0])))
print('Transformed point (p_OF1_O): ', p_OF1_O)
assert_condition_and_print(condition)

# Multiplying a point by it's inverse transform brings a point from F1 frame to origin frame e.g. [0, 0, 0] -> [-0.3, 0, 0] 
# print(np.dot(origin_np, first_transformation_inverse)[0:3])  # omg, this gives [0, 0, 0] it has to be the other way around
# TODO wait, what does this mean? are we in the origin frame now or still in F1 frame? Wait, it's all in the notion:
# p_BA_C means "point/position of A measured from B expressed in C". So the below is:
# p_F1O_O meaning "position of origin measured from F1 expressed in origin frame" # TODO is it expressed in origin frame or not?
p_F1O_O = np.dot(first_transformation_inverse, origin_np)[0:3]
print('Transformed point (p_F1O_O): ', p_F1O_O)
condition = all(np.isclose(p_F1O_O, np.array([-0.3, 0.0, 0.0])))
assert_condition_and_print(condition)

condition = all(np.isclose(np.dot(first_transformation, np.array([0.3, 0.0, 0.0, 1.0]))[0:3], np.array([0.6, 0.0, 0.0])))
assert_condition_and_print(condition)

condition = all(np.isclose(np.dot(first_transformation, np.array([-0.3, 0.0, 0.0, 1.0]))[0:3], np.array([0.0, 0.0, 0.0])))
assert_condition_and_print(condition)

condition = all(np.isclose(np.dot(first_transformation_inverse, np.array([-0.3, 0.0, 0.0, 1.0]))[0:3], np.array([-0.6, 0.0, 0.0])))
assert_condition_and_print(condition)



list_of_geometry_elements = [origin_frame, f1_frame]
o3d.visualization.draw_geometries(list_of_geometry_elements)



sys.exit()
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

estimated_camera_coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size * 1.5, origin=[0.0, 0.0, 0.0])
# estimated_camera_coordinate_frame.transform(estimated_cam2_arm_transform)


arm_position_coord_frames = [coordinate_frame_1, coordinate_frame_2, coordinate_frame_3]
list_of_geometry_elements = [origin_frame, camera_coordinate_frame, estimated_camera_coordinate_frame] + arm_position_coord_frames
o3d.visualization.draw_geometries(list_of_geometry_elements)