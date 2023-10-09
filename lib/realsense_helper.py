import numpy as np
import pyrealsense2 as rs


def setup_start_realsense():
    # TODO create class
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    profile = pipeline.start(config)
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.holes_fill, 3)

    depth_sensor = profile.get_device().first_depth_sensor()
    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_scale = depth_sensor.get_depth_scale()

    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
    color_intrin = color_frame.profile.as_video_stream_profile().intrinsics

    depth_sensor = profile.get_device().first_depth_sensor()


    # TODO does this help aruco or not?
    if depth_sensor.supports(rs.option.emitter_enabled):
        print('Enabling emitter')
        # depth_sensor.set_option(rs.option.emitter_enabled, 1)
        print('Disabling emitter')
        depth_sensor.set_option(rs.option.emitter_enabled, 0)


    # Using preset HighAccuracy for recording
    # depth_sensor.set_option(rs.option.visual_preset, Preset.HighAccuracy)

    align = rs.align(rs.stream.color)

    return depth_intrin, color_intrin, depth_scale, pipeline, align, spatial


def run_10_frames_to_wait_for_auto_exposure(pipeline, align):
    # wait for auto-exposure
    print('Running 10 frames to wait for auto-exposure')
    for i in range(10):
        frames = pipeline.wait_for_frames()
        # TODO below 3 are not needed.
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()


def realsense_get_frames(pipeline, align, spatial):
    # TODO dupliated with above but didn't want to pass spatial to above. hmm
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    depth_frame = spatial.process(depth_frame)  # hole filling

    return color_frame, depth_frame


def use_aruco_corners_and_realsense_for_3D_point(depth_frame, corners_of_one_aruco, color_intrin):
    center = None
    center_point_2D = np.average(corners_of_one_aruco[0], axis=0)
    # import pdb;pdb.set_trace()
    depth = depth_frame.as_depth_frame().get_distance(center_point_2D[0], center_point_2D[1])
    center_point_3D = np.append(center_point_2D, depth)
    if depth != 0:
        # global center
        x = center_point_3D[0]
        y = center_point_3D[1]
        z = center_point_3D[2]
        ## see rs2 document: 
        ## https://github.com/IntelRealSense/librealsense/wiki/Projection-in-RealSense-SDK-2.0#point-coordinates
        ## and example: https://github.com/IntelRealSense/librealsense/wiki/Projection-in-RealSense-SDK-2.0#point-coordinates
        x, y, z = rs.rs2_deproject_pixel_to_point(color_intrin, [x, y], z)
        center = [x, y, z]

    return center