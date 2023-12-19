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


mesh_box_opt = o3d.geometry.TriangleMesh.create_box(width=0.1, height=0.1, depth=0.003)
mesh_box_opt.paint_uniform_color([1, 0, 0])

size = 0.1
coordinate_frame_1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0.0, 0.0, 0.0])
coordinate_frame_2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0.3, 0.0, 0.0])
coordinate_frame_3 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0.4, 0.0, 0.0])
coordinate_frame_4 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0.5, 0.0, 0.0])

camera_coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size * 1.1, origin=[0.0, 0.0, 0.0])

arm_position_coord_frames = [coordinate_frame_1, coordinate_frame_2, coordinate_frame_3, coordinate_frame_4]

list_of_geometry_elements = [camera_coordinate_frame] + arm_position_coord_frames
o3d.visualization.draw_geometries(list_of_geometry_elements)