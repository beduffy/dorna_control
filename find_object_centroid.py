import sys
import random
import time
import colorsys
import os
import traceback

import open3d as o3d
import cv2
import pyrealsense2 as rs
import torch  # if this is imported before realsense profile = pipeline.start(config) fails...
import torchvision
import numpy as np

from lib.vision import maskrcnn_detect, find_obj_centroid_from_pcd, \
    find_obj_centroid_from_mask, find_obj_centroid_from_maskrcnn, display_inlier_outlier, \
    find_obj_pose_from_pcd
from lib.vision_config import pinhole_camera_intrinsic, class_name_to_id, class_names
from lib.plot_funcs import display_instances


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).to(device)
    model.eval()

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

    print('Starting config')
    profile = pipeline.start(config)
    print('config was started')

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(depth_scale)
    align = rs.align(rs.stream.color)

    visualise_masks = False
    # visualise_masks = True
    # non_blocking_viz = True
    non_blocking_viz = False
    visualise_pcds = True
    # visualise_pcds = False

    frame_num = 0
    origin_frame_size = 0.03

    pcd = o3d.geometry.PointCloud()
    centroid_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.007)
    centroid_sphere.paint_uniform_color([1.0, 0.0, 0.0])
    centroid_sphere.compute_vertex_normals()

    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=origin_frame_size, origin=[0.0, 0.0, 0.0])

    if non_blocking_viz:
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        origin_x = 0  # only for proving/testing non-blocking

        # vis.add_geometry(pcd)
        vis.add_geometry(centroid_sphere)
        # vis.add_geometry(origin_frame)

    for i in range(10):  # for auto-exposure
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

    while True:
        try:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            camera_depth_img = np.asanyarray(depth_frame.get_data())
            camera_color_img = np.asanyarray(color_frame.get_data())
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(camera_depth_img, alpha=0.03),
                                               cv2.COLORMAP_JET)

            r, label_strings = maskrcnn_detect(camera_color_img, model, device, class_names)

            if visualise_masks:
                display_instances(camera_color_img, r['boxes'], r['masks'], r['labels'], class_names, r['scores'])
                # todo add param to only show target objects and raise score again

            # save image and then load
            # color_path = "latest_color_frame.jpg"
            # cv2.imwrite(color_path, camera_color_img)
            # depth_path = "latest_depth_frame.png"
            # # np.save(depth_path, camera_depth_img)
            # cv2.imwrite(depth_path, camera_depth_img.reshape(480, 640, 1))
            # source_color = o3d.io.read_image(color_path)
            # source_depth = o3d.io.read_image(depth_path)
            # todo test flat object to see if it is at 0.0 after cam2arm

            # objects we can pickup
            # import pdb;pdb.set_trace()
            output = find_obj_centroid_from_maskrcnn(camera_color_img, camera_depth_img, r, class_name_to_id=class_name_to_id, visualise_pcds=True)  # default is cup
            if output:
                center, pcd = output
            # most likely and soon
            # find_obj_centroid_from_maskrcnn(camera_color_img, camera_depth_img, r, label='bottle', visualise_pcds=True)
            # find_obj_centroid_from_maskrcnn(camera_color_img, camera_depth_img, r, label='plate', visualise_pcds=True)
            # center, pcd = find_obj_centroid_from_maskrcnn(camera_color_img, camera_depth_img, r, label='banana', visualise_pcds=True)
            # center, pcd = find_obj_centroid_from_maskrcnn(camera_color_img, camera_depth_img, r, label='apple', visualise_pcds=True)
            # center, pcd = find_obj_centroid_from_maskrcnn(camera_color_img, camera_depth_img, r, label='orange', class_name_to_id=class_name_to_id, visualise_pcds=True)
            # center, pcd = find_obj_centroid_from_maskrcnn(camera_color_img, camera_depth_img, r, label='orange', class_name_to_id=class_name_to_id, visualise_pcds=False)
            # find_obj_centroid(camera_color_img, camera_depth_img, r, label='cell phone', visualise=True)

            # obj_pose = find_obj_pose_from_pcd(pcd, visualise_pcds=True)
            # todo do obj pose in another file because I need to convert to arm coordinates!

            # likely to be possible
            # find_obj_centroid_from_maskrcnn(camera_color_img, camera_depth_img, r, label='fork', visualise_pcds=True)
            # find_obj_centroid_from_maskrcnn(camera_color_img, camera_depth_img, r, label='knife', visualise_pcds=True)
            # find_obj_centroid_from_maskrcnn(camera_color_img, camera_depth_img, r, label='spoon', visualise_pcds=True)
            # find_obj_centroid_from_maskrcnn(camera_color_img, camera_depth_img, r, label='wine glass', visualise_pcds=True)
            # find_obj_centroid_from_maskrcnn(camera_color_img, camera_depth_img, r, label='bowl', visualise_pcds=True)
            # find_obj_centroid_from_maskrcnn(camera_color_img, camera_depth_img, r, label='mouse', visualise_pcds=True)
            # find_obj_centroid_from_maskrcnn(camera_color_img, camera_depth_img, r, label='remote', visualise_pcds=True)
            # find_obj_centroid_from_maskrcnn(camera_color_img, camera_depth_img, r, label='toothbrush', visualise_pcds=True)

            # roughly possible
            # find_obj_centroid_from_maskrcnn(camera_color_img, camera_depth_img, r, label='keyboard', visualise_pcds=True)
            # find_obj_centroid_from_maskrcnn(camera_color_img, camera_depth_img, r, label='hat', visualise_pcds=True)
            # find_obj_centroid_from_maskrcnn(camera_color_img, camera_depth_img, r, label='book', visualise_pcds=True)
            # find_obj_centroid_from_maskrcnn(camera_color_img, camera_depth_img, r, label='handbag', visualise_pcds=True)
            # find_obj_centroid_from_maskrcnn(camera_color_img, camera_depth_img, r, label='vase', visualise_pcds=True)

            # can't pickup
            # find_obj_centroid_from_maskrcnn(camera_color_img, camera_depth_img, r, label='person', visualise_pcds=True)

            if visualise_pcds and output:
                # centroid_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.007)
                # centroid_sphere.compute_vertex_normals()  # probably helps lighting
                # centroid_sphere.paint_uniform_color([1.0, 0.0, 0.0])
                # centroid_sphere.translate(centroid_and_cov[0])  # todo which is better? looks like center
                centroid_sphere.translate(center, relative=False)  # todo only if blocking
                # centroid_sphere.translate(center, relative=True)  # todo this should work!! only if non blocking?

                if 'non_blocking_viz' in globals() and non_blocking_viz:
                    # todo non-blocking and watch how centroid changes throughout frames? why doesn't it work???

                    # vis.update_geometry(pcd)
                    # vis.update_geometry(centroid_sphere)
                    # vis.update_geometry(origin_frame)

                    # if frame_num % 100 == 0:  # todo it probably doesn't work because I create a new pcd each time, I should create one that is transparent
                    #     print('Recreating')
                    # vis.clear_geometries()
                    # vis.add_geometry(pcd)
                    # vis.add_geometry(centroid_sphere)
                    # vis.add_geometry(origin_frame)

                    vis.poll_events()
                    vis.update_renderer()

                    # vis.poll_events()
                    # vis.update_renderer()
                    # time.sleep(1)
                else:
                    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                        size=origin_frame_size, origin=[0.0, 0.0, 0.0])
                    o3d.visualization.draw_geometries([pcd, centroid_sphere, origin_frame])

            # todo matplot error fix bug X Error of failed request:  BadWindow (invalid Window parameter)
            # todo instead do non blocking matplotlib?

            camera_color_img = cv2.cvtColor(camera_color_img, cv2.COLOR_RGB2BGR)
            images = np.hstack((camera_color_img, depth_colormap))
            # cv2.imshow("object detection", image)
            cv2.imshow("object detection", images)
            k = cv2.waitKey(1)
            if k == ord('q'):
                cv2.destroyAllWindows()
                pipeline.stop()
                break

            # todo why does this work with camera control but within function it doesn't?
            # origin_x += 0.000001
            # origin_frame.translate(np.array([origin_x, 0.0, 0.0]))
            # vis.update_geometry(origin_frame)
            # vis.poll_events()
            # vis.update_renderer()

        # end of main loop
        except ValueError as e:
            print('Error in main loop')
            print(e)
            print(traceback.format_exc())
            print('sys exc info')
            print(sys.exc_info()[2])
            
            # Stop streaming
            pipeline.stop()
            cv2.destroyAllWindows()
            

    frame_num += 1
