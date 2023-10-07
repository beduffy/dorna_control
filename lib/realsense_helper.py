

def run_10_frames_to_wait_for_auto_exposure(pipeline, align):
    # wait for auto-exposure
    print('Running 10 frames to wait for auto-exposure')
    for i in range(10):
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()