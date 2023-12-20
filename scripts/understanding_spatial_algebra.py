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
# https://manipulation.csail.mit.edu/pick.html


size = 0.1
# origin frame
origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0.0, 0.0, 0.0])
origin_np = np.array([0.0, 0.0, 0.0, 1.0])

# first homogenous transformation into frame F1
translate = [0.3, 0, 0]
first_transformation = np.identity(4)
# first_transformation[:3, :3] = np.identity(3)  # no rotation
first_transformation[0, 3] = translate[0]
first_transformation[1, 3] = translate[1]
first_transformation[2, 3] = translate[2]
first_transformation_inverse = get_inverse_homogenous_transform(first_transformation)
f1_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0.0, 0.0, 0.0])
f1_frame.transform(first_transformation)

# second transformation from F1 to F2
translate = [0, 0.3, 0]
second_transformation = np.identity(4)

from scipy.spatial.transform import Rotation
angles = [0, 0, np.pi / 2]  # TODO 2pi is full circle. pi 180, pi / 2 90
# https://stackoverflow.com/questions/54616049/converting-a-rotation-matrix-to-euler-angles-and-back-special-case
# ### first transform the matrix to euler angles
# r =  Rotation.from_matrix(rotation_matrix)
# angles = r.as_euler("zyx",degrees=True)

#### Modify the angles
# print(angles)
# angles[0] += 5


#### Then transform the new angles to rotation matrix again
r = Rotation.from_euler("xyz", angles, degrees=False)  # roll pitch yaw = xyz  # TODO why is matrix numbers looking very scientific and small?
new_rotation_matrix = r.as_matrix()

second_transformation[:3, :3] = new_rotation_matrix
second_transformation[0, 3] = translate[0]
second_transformation[1, 3] = translate[1]
second_transformation[2, 3] = translate[2]
second_transformation_inverse = get_inverse_homogenous_transform(second_transformation)
f2_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0.0, 0.0, 0.0])
joined_f1_f2_transformation = np.dot(first_transformation, second_transformation)
print('joined_f1_f2_transformation')
print(joined_f1_f2_transformation)
f2_frame.transform(joined_f1_f2_transformation)

inverse_joined_f1_f2_transformation = get_inverse_homogenous_transform(joined_f1_f2_transformation)


# TODO wait poses are not expressed in anything? but points are... But what am I dealing with?
# Russ: We do not use the "expressed in" frame subscript for pose; we always want the pose expressed in the reference frame.


# import pdb;pdb.set_trace()

print(origin_np.shape)
print(first_transformation.shape)

print('First transformation, frame F1 (this brings points from origin frame to F1 e.g. [0, 0, 0] -> [0.3, 0, 0]):')
print(first_transformation)
print('First transformation inverse (this brings points from origin frame to F1 e.g. [0, 0, 0] -> [-0.3, 0, 0]):')
print(first_transformation_inverse)
print('Second transformation from frame F1 to F2 (this brings points from F1 to F2 e.g. ):')
print(second_transformation)

# From Russ notes:
# Multiplication by a rotation can be used to change the "expressed in" frame:
# You might be surprised that a rotation alone is enough to change the expressed-in frame, but it's true. 
# The position of the expressed-in frame does not affect the relative position between two points.

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
# p_F1O_O meaning "position of origin measured from F1 expressed in origin frame" # TODO is it expressed in origin frame or not? I feel like it actually should be expressed if F1
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

# Should bring a point from origin to f2, points don't rotate so nothing complicated yet... 
condition = all(np.isclose(np.dot(joined_f1_f2_transformation, np.array([0.0, 0.0, 0.0, 1.0]))[0:3], np.array([0.3, 0.3, 0.0])))
assert_condition_and_print(condition)

# brings points from f2 to origin frame. Here's were things get funky. So we rotated 90 degrees left in yaw. So yeah:
# we have to go back -0.3 in x and then y is positive because we rotated!!!
p_F2O_O = np.dot(inverse_joined_f1_f2_transformation, np.array([0.0, 0.0, 0.0, 1.0]))[0:3]  # TODO expressed in F2
print(p_F2O_O)
condition = all(np.isclose(p_F2O_O, np.array([-0.3, 0.3, 0.0])))
assert_condition_and_print(condition)


list_of_geometry_elements = [origin_frame, f1_frame, f2_frame]
o3d.visualization.draw_geometries(list_of_geometry_elements)