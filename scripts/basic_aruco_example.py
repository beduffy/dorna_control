from __future__ import print_function
import os
import sys
import argparse
import time
import math
import traceback
from glob import glob

try:
    import open3d as o3d
    from skimage.measure import find_contours
except Exception as e:
    print(e)
    print('Tried to import open3d or skimage but not installed')
import requests
import pyrealsense2 as rs
from cv2 import aruco
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib as mpl
from scipy import optimize

# TODO could I put these common things into some function/library?

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


marker_length = 0.0275
# marker_length = 0.0935  # big marker
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters_create()
parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
# parameters.adaptiveThreshWinSizeMin = 3
# parameters.adaptiveThreshWinSizeStep = 4  # todo test more and see if it makes worse/better
board = cv2.aruco.GridBoard_create(3, 4, marker_length, 0.06, aruco_dict)  # marker_separation 0.06

# old intrinsic calibration done by me
camera_matrix = np.array([[612.14801862, 0., 340.03640321],
                            [0., 611.29345062, 230.06928807],
                            [0., 0., 1.]])
dist_coeffs = np.array(
    [1.80764862e-02, 1.09549436e+00, -3.38044260e-03, 4.04543459e-03, -4.26585263e+00])


depth_sensor = profile.get_device().first_depth_sensor()

# Using preset HighAccuracy for recording
# depth_sensor.set_option(rs.option.visual_preset, Preset.HighAccuracy)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_scale = depth_sensor.get_depth_scale()

align = rs.align(rs.stream.color)
frame_count = 0

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
        
        color_img = camera_color_img  # stupid but for now
        bgr_color_data = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
        gray_data = cv2.cvtColor(bgr_color_data, cv2.COLOR_RGB2GRAY)

        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray_data, aruco_dict,
                                                                    parameters=parameters)
        frame_markers = aruco.drawDetectedMarkers(color_img, corners, ids)
        all_rvec, all_tvec, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)

        
        if ids is not None:
            print(ids)
            ids_list = [l[0] for l in ids.tolist()]
            for list_idx, corner_id in enumerate(ids_list):
                rvec_aruco, tvec_aruco = all_rvec[list_idx, 0, :], all_tvec[list_idx, 0, :]
                # rvec_aruco, tvec_aruco = all_rvec[corner_id, 0, :], all_tvec[corner_id, 0, :]
                # aruco.drawAxis(color_img, camera_matrix, dist_coeffs, rvec_aruco, tvec_aruco, marker_length)
                # https://stackoverflow.com/questions/72702953/attributeerror-module-cv2-aruco-has-no-attribute-drawframeaxes


        
        images = np.hstack((color_img, depth_colormap))
        # images = np.hstack((camera_color_img, depth_colormap))
        # images = camera_color_img

        cv2.imshow("image", images)
        k = cv2.waitKey(1)

        if k == ord('q'):
            cv2.destroyAllWindows()
            pipeline.stop()
            break

        camera_color_img = cv2.cvtColor(camera_color_img, cv2.COLOR_BGR2RGB)  # for open3D

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