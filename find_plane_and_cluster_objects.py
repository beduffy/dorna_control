import random
import time
import colorsys
import os

import open3d as o3d
import cv2
import pyrealsense2 as rs
import torch  # if this is imported before realsense profile = pipeline.start(config) fails...
import torchvision
import numpy as np
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib import patches, lines
from matplotlib.patches import Polygon

from lib.vision import get_full_pcd_from_rgbd, find_obj_centroid_from_pcd, \
    find_all_planes, find_all_clusters, find_all_objects_from_clusters
from lib.vision_config import class_names, class_name_to_id, pinhole_camera_intrinsic
from lib.extra import random_colors


o3d.utility.set_verbosity_level(o3d.utility.Debug)

# todo create 3D bounding boxes from pointclouds for free data? Learn objectness detection?
# todo how to turn plane into 2d image, just 2d bbox around it I suppose

# todo do top down view!!! and use collision detection or 2d Box

# todo learn how to project shapes along other shapes and planes and lines!!!
# https://www.euclideanspace.com/maths/geometry/elements/line/projections/index.htm

# todo pick bunch of points on ground to turn into polygon and then project/bump upwards? How to visualise it? Learn vertices and all that again?

# create script to specify cuboid width, height, length and xyz position. One for bin? Not needed?

def crop_pointcloud_with_polygon(pcd):
    # attempt to specify floor area for rubbish hunt
    # todo could crop floor with square crop and
    print("Demo for manual geometry cropping")
    print("1) Press 'Y' twice to align geometry with negative direction of y-axis")
    print("2) Press 'K' to lock screen and to switch to selection mode")
    print("3) Drag for rectangle selection,")
    print("   or use ctrl + left click for polygon selection")
    print("4) Press 'C' to get a selected geometry and to save it")
    print("5) Press 'F' to switch to freeview mode")
    # pcd = o3d.io.read_point_cloud("../../TestData/ICP/cloud_bin_0.pcd")
    o3d.visualization.draw_geometries_with_editing([pcd])  # todo polygon mode isn't working for some reason?


# print("Load a polygon volcqume and use it to crop the original point cloud")
# vol = o3d.visualization.read_selection_polygon_volume(
#     "../../TestData/Crop/cropped.json")
# chair = vol.crop_point_cloud(pcd)
# o3d.visualization.draw_geometries([chair])
# print("")

# todo better to just re-run the script and change the box position

# todo how to automatically find the below?
# {
# 	"axis_max" : 4.022921085357666,
# 	"axis_min" : -0.76341366767883301,
# 	"bounding_polygon" :
# 	[
# 		[ 2.6509309513852526, 0.0, 1.6834473132326844 ],
# 		[ 2.5786428246917148, 0.0, 1.6892074266735244 ],
# 		[ 2.4625790337552154, 0.0, 1.6665777078297999 ],
# 		[ 2.2228544982251655, 0.0, 1.6168160446813649 ],
# 		[ 2.166993206001413, 0.0, 1.6115495157201662 ],
# 		[ 2.1167895865303286, 0.0, 1.6257706054969348 ],
# 		[ 2.0634657721747383, 0.0, 1.623021658624539 ],
# 		[ 2.0568612343437236, 0.0, 1.5853892911207643 ],
# 		[ 2.1605399001237027, 0.0, 0.96228993255083017 ],
# 		[ 2.1956669387205228, 0.0, 0.95572746049785073 ],
# 		[ 2.2191318790575583, 0.0, 0.88734449982108754 ],
# 		[ 2.2484881847925919, 0.0, 0.87042807267013633 ],
# 		[ 2.6891234157295827, 0.0, 0.94140677988967603 ],
# 		[ 2.7328692490470647, 0.0, 0.98775740674840251 ],
# 		[ 2.7129337547575547, 0.0, 1.0398850034649203 ],
# 		[ 2.7592174072415405, 0.0, 1.0692940558509485 ],
# 		[ 2.7689216419453428, 0.0, 1.0953914441371593 ],
# 		[ 2.6851455625455669, 0.0, 1.6307334122162018 ],
# 		[ 2.6714776099981239, 0.0, 1.675524657088997 ],
# 		[ 2.6579576128816544, 0.0, 1.6819127849749496 ]
# 	],
# 	"class_name" : "SelectionPolygonVolume",
# 	"orthogonal_axis" : "Y",
# 	"version_major" : 1,
# 	"version_minor" : 0
# }


# todo just specify a little cube where objects can be....

# todo this whole script with find_object_centroid into scripts folder


if __name__ == '__main__':
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
    non_blocking_viz = True
    non_blocking_viz = False

    find_all_planes_and_cluster = False
    find_all_planes_and_cluster = True

    frame_num = 0
    origin_frame_size = 0.01

    # todo is the ground plane close to z = 0. Or probably not because camera isn't level? But could force it to be so?
    # todo i might turn the ground plane into a 2D image, do contours and find objects by holes?
    # todo confirm it's ground plane by actually knowing the height of the camera/arm...

    if non_blocking_viz:
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        pcd = o3d.geometry.PointCloud()
        centroid_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.007)
        centroid_sphere.paint_uniform_color([1.0, 0.0, 0.0])
        centroid_sphere.compute_vertex_normals()

        origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=origin_frame_size, origin=[0.0, 0.0, 0.0])

        origin_x = 0

        vis.add_geometry(pcd)
        vis.add_geometry(centroid_sphere)
        vis.add_geometry(origin_frame)

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

            if find_all_planes_and_cluster and frame_num > 5:
                pcd = get_full_pcd_from_rgbd(camera_color_img, camera_depth_img, pinhole_camera_intrinsic, visualise=False)

                # octree = o3d.geometry.Octree.convert_from_point_cloud(pcd, 0.01)  # todo why doesn't it work?
                # octree = o3d.geometry.Octree()
                # octree.convert_from_point_cloud(pcd, 0.01)  # todo no idea how to use it?
                # o3d.visualization.draw_geometries([octree])  # not for visualisation then?

                # voxel_grid = o3d.geometry.VoxelGrid.convert_from_point_cloud(pcd, 0.01)  # todo why doesn't it work?
                # voxel_grid = o3d.geometry.VoxelGrid()
                # voxel_grid.create_from_point_cloud(pcd, 0.01)  # todo no idea how to use it?
                # o3d.visualization.draw_geometries([voxel_grid])  # not for visualisation then?
                # todo crashes np.asarray(voxel_grid.voxels)
                # todo crashes voxel_grid.check_if_included(o3d.utility.Vector3dVector(np.array([1, 2, 3])))
                # todo params  voxel_grid.create_dense()
                # todo carving

                # todo
                # You could use np.where(grid == 1) on your grid to get the voxel indices,
                # where the grid data is 1. Concatenating those, you can then use
                # pcd = o3d.geometry.PointCloud(); pcd.points = o3d.utility.Vector3dVector(cat_indices)
                # to set the point coordinates.

                # pcd.estimate_normals()
                # # estimate radius for rolling ball
                # distances = pcd.compute_nearest_neighbor_distance()
                # avg_dist = np.mean(distances)
                # radius = 1.5 * avg_dist
                # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                #     pcd,
                #     o3d.utility.DoubleVector([radius, radius * 2]))
                #
                # trimesh = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles),
                #                           vertex_normals=np.asarray(mesh.vertex_normals))


                # c = crop_pointcloud_with_polygon(pcd)

                # todo remove other planes and stuff above them to then find it easy to find the rest?
                # find all planes
                # planes, outlier_clouds = find_all_planes(pcd, num_iterations_of_segmentation=3)
                planes, outlier_clouds = find_all_planes(pcd, num_iterations_of_segmentation=3, visualise_pcds=True)
                # find biggest plane
                # planes, outlier_cloud = find_all_planes(pcd, num_iterations_of_segmentation=1)
                # biggest_plane = planes[0]  # only 1 plane found

                # todo remove right stuff from outlier_clouds[0]
                cloud_without_ground = outlier_clouds[0]

                cluster_pcds = find_all_clusters(cloud_without_ground)
                # c = crop_pointcloud_with_polygon(pcd)
                object_pcds = find_all_objects_from_clusters(cluster_pcds)

                for obj_pcd in object_pcds:
                    center, inlier_pcd = find_obj_centroid_from_pcd(obj_pcd, visualise_centroid=True)

            camera_color_img = cv2.cvtColor(camera_color_img, cv2.COLOR_RGB2BGR)
            images = np.hstack((camera_color_img, depth_colormap))
            # cv2.imshow("object detection", image)q
            cv2.imshow("object detection", images)
            k = cv2.waitKey(1)
            if k == ord('q'):
                cv2.destroyAllWindows()
                pipeline.stop()
                break

            frame_num += 1

        # end of main loop
        except ValueError as e:
            print('Error in main loop')
            print(e)
            # Stop streaming
            pipeline.stop()
            cv2.destroyAllWindows()
