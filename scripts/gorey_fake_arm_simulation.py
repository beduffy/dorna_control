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

from lib.vision import isclose, dist, isRotationMatrix, rotationMatrixToEulerAngles, create_homogenous_transformations


# https://stackoverflow.com/questions/58997792/how-do-you-convert-euler-angles-to-the-axis-angle-representation-in-python
def euler_yzx_to_axis_angle(z_e, x_e, y_e, normalize=True):
    # Assuming the angles are in radians.
    c1 = math.cos(z_e/2)
    s1 = math.sin(z_e/2)
    c2 = math.cos(x_e/2)
    s2 = math.sin(x_e/2)
    c3 = math.cos(y_e/2)
    s3 = math.sin(y_e/2)
    c1c2 = c1*c2
    s1s2 = s1*s2
    w = c1c2*c3 - s1s2*s3
    x = c1c2*s3 + s1s2*c3
    y = s1*c2*c3 + c1*s2*s3
    z = c1*s2*c3 - s1*c2*s3
    angle = 2 * math.acos(w)
    if normalize:
        norm = x*x+y*y+z*z
        if norm < 0.001:
            # when all euler angles are zero angle =0 so
            # we can set axis to anything to avoid divide by zero
            x = 1
            y = 0
            z = 0
        else:
            norm = math.sqrt(norm)
            x /= norm
            y /= norm
            z /= norm
    return z, x, y, angle
# print(euler_yzx_to_axis_angle(1.2, 1.5, 1.0))


mesh_box_opt = o3d.geometry.TriangleMesh.create_box(width=0.1, height=0.1, depth=0.003)
mesh_box_opt.paint_uniform_color([1, 0, 0])

size = 0.1
origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0.0, 0.0, 0.0])
coordinate_frame_1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0.3, 0.0, 0.0])
coordinate_frame_2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0.4, 0.0, 0.0])
coordinate_frame_3 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0.5, 0.0, 0.0])

camera_coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size * 1.3, origin=[0.0, 0.0, 0.0])

# TODO we can obviously go from rotation axis -> rotation matrix -> euler angles (two versions). But how to go backwards?

# rvec = np.array([1.0, 1.0, 1.0])
# z, x, y, angle = euler_yzx_to_axis_angle(np.pi / 2, 0, 0)  # pointing left in yaw, 45 deg. red.
# z, x, y, angle = euler_yzx_to_axis_angle(0, np.pi / 2, 0)  # pointing up in pitch 45 degrees. 
# z, x, y, angle = euler_yzx_to_axis_angle(0, 0, np.pi / 2)  # rolling right 45 degrees.

# I want camera to the right, pointing to the left 45 degrees left in yaw and 45 degrees down
z, x, y, angle = euler_yzx_to_axis_angle(np.pi / 2, -np.pi / 2, 0)
rvec = np.array([x, y, z])
tvec = np.array([0.0, 0.3, 0.4])   # TODO why does 0.3 bring it down but 
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

arm_position_coord_frames = [coordinate_frame_1, coordinate_frame_2, coordinate_frame_3]

list_of_geometry_elements = [origin_frame, camera_coordinate_frame] + arm_position_coord_frames
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