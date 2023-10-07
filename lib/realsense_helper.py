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

    # Using preset HighAccuracy for recording
    # depth_sensor.set_option(rs.option.visual_preset, Preset.HighAccuracy)

    align = rs.align(rs.stream.color)

    return depth_intrin, color_intrin, depth_scale, pipeline, align, spatial


def run_10_frames_to_wait_for_auto_exposure(pipeline, align):
    # wait for auto-exposure
    print('Running 10 frames to wait for auto-exposure')
    for i in range(10):
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()