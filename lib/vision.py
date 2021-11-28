import math

# todo fix 5 of these imports so that devices that don't have matplotlib can actually work.... docker containers in particular
try:
    import matplotlib.pyplot as plt
except Exception as e:
    print(e)
    print('Tried to import matplotlib but not installed')

import numpy as np
import cv2
try:
    import open3d as o3d   # has to be imported before torch. todo. no torch and realsense are the problem
except Exception as e:
    print(e)
    print('Tried to import open3d but not installed')
import torch

# from control.lib.kinematics import isclose
from lib.extra import random_colors
from lib.vision_config import fx, fy, ppx, ppy

# todo to answer the question of what is more important or what we need more of (perception vs control) count how LOC, algorithms are in control vs vision after a few months

# helper functions
def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    # return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)  # todo doesn't work with x = 0 and x2 = 9.8e-17
    return abs(a - b) <= rel_tol


# ----------------------
# 3D Vision
# ----------------------

def create_homogenous_transformations(tvec_in, rvec_in):
    # todo add to vision library and compare with DRY duplicate
    # -- Obtain the rotation matrix tag->camera
    R_ct = np.matrix(cv2.Rodrigues(rvec_in)[0])  # todo confirm that rvec makes sense
    # R_tc = R_flip * R_ct.T
    R_tc = R_ct.T

    # From Wikipedia:
    # R, T are the extrinsic parameters which denote the coordinate system transformations from
    # 3D world coordinates to 3D camera coordinates. Equivalently, the extrinsic parameters define
    # the position of the camera center and the camera's heading in world coordinates.
    # T is the position of the origin of the world coordinate system expressed in coordinates of the
    # camera-centered coordinate system. T is often mistakenly considered the position of the camera.
    # The position, {\displaystyle C}C, of the camera expressed in world coordinates is
    #  C=-R^{-1}T=-R^{T}T}C=-R^{{-1}}T=-R^{T}T

    # tvec is the position of the marker in camera coordinates but the transformation from
    # camera to marker is then this:
    arm2cam_local = np.identity(4)
    arm2cam_local[:3, :3] = R_ct
    arm2cam_local[0, 3] = tvec_in[0]
    arm2cam_local[1, 3] = tvec_in[1]
    arm2cam_local[2, 3] = tvec_in[2]

    # -- Now get Position and attitude f the camera respect to the marker
    # pos_camera = -R_tc * np.matrix(tvec).T  # todo how could element-wise possibly work!?!?!?!?
    pos_camera = np.dot(-R_tc, np.matrix(tvec_in).T)

    # cam_position = np.dot(-arm2cam_rotation, tvec_arm2cam)  # .T   # arm2cam
    cam2arm_local = np.identity(4)
    cam2arm_local[:3, :3] = R_tc
    cam2arm_local[0, 3] = pos_camera[0]
    cam2arm_local[1, 3] = pos_camera[1]
    cam2arm_local[2, 3] = pos_camera[2]

    return cam2arm_local, arm2cam_local, R_tc, R_tc, pos_camera


def convert_pixel_to_arm_coordinate(camera_depth_img, pixel_x, pixel_y, cam2arm, verbose=False):
    # todo compare this function to the new cam to arm coord and write docstring so it's clear
    camera_coord = get_camera_coordinate(camera_depth_img, pixel_x, pixel_y, verbose=verbose)
    if camera_coord is not None:
        arm_coord = np.dot(cam2arm, camera_coord)

        if verbose:
            # print('MouseX: {}. MouseY: {}'.format(pixel_x, pixel_y))
            # print('tvec: {}'.format(tvec))
            # print('Camera click xyz coordinate: \n{}'.format(cam_coord))
            print('Arm coordinate: (cam2arm)  \n{}'.format(arm_coord))

        return arm_coord[:3]  # return arm coordinate xyz


def get_camera_coordinate(camera_depth_img, pixel_x, pixel_y, verbose=False):
    z_cam = get_depth_at_pixel(camera_depth_img, pixel_x, pixel_y)
    if isclose(z_cam, 0.0, 0.00001):
        if verbose:
            print('z_cam {} is close to 0'.format(z_cam))
        return None

    # intrinsics as global to library or parameter?

    # assuming it's your regular 3x3 camera intrinsics
    # x_cam = np.multiply(mouseX - ppx, z_cam / fx)  # todo using color intrinsics but depth hmm
    # y_cam = np.multiply(mouseY - ppy, z_cam / fy)
    x_cam = np.multiply(pixel_x - ppx, z_cam / fx)  # todo using color intrinsics but depth hmm
    y_cam = np.multiply(pixel_y - ppy, z_cam / fy)
    # above causes x_cam and y_cam to be much more accurate but z_cam is still 10cm off???
    # x_cam = np.multiply(mouseX - depth_intrin.ppx, z_cam / depth_intrin.fx)  # todo using color intrinsics but depth hmm
    # y_cam = np.multiply(mouseY - depth_intrin.ppy, z_cam / depth_intrin.fy)
    # x_cam = np.multiply(mouseX - color_intrin.ppx, z_cam / color_intrin.fx)
    # y_cam = np.multiply(mouseY - color_intrin.ppy, z_cam / color_intrin.fy)

    # todo not doing distortion and why doesn't the above match the equation at https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
    # todo instead of using depth, only use color somehow?

    cam_coord = np.array([x_cam, y_cam, z_cam, 1])

    return cam_coord


def get_depth_at_pixel(camera_depth_img, pixel_x, pixel_y, depth_scale=0.0010000000474974513):
    z_cam = camera_depth_img[pixel_y][pixel_x] * depth_scale
    # p = rs.rs2_deproject_pixel_to_point(depth_intrin, [mouseX, mouseY], z_cam)
    # p = rs.rs2_deproject_pixel_to_point(color_intrin, [pixel_x, pixel_y], z_cam)

    return z_cam


# ----------------------
# Direct pointcloud code
# ----------------------
# todo split into 3d vision and 2d vision files themself? Hard sometimes to understand difference?

def convert_cam_pcd_to_arm_pcd(cam_pcd, cam2arm, z_amount_to_add, q1=None):
    pcd_points = np.asarray(cam_pcd.points)
    pcd_numpy = convert_cam_to_arm_coordinates_array(pcd_points, cam2arm, z_amount_to_add, q1)
    transformed_pcd = o3d.geometry.PointCloud()
    transformed_pcd.points = o3d.utility.Vector3dVector(pcd_numpy[:, :3])
    transformed_pcd.colors = cam_pcd.colors

    return transformed_pcd, pcd_numpy


def convert_cam_to_arm_coordinates_array(camera_coordinates, cam2arm, z_amount_to_add, q1=None, xy_offset=12):
    """
    Converts matrix from camera coordinates to arm coordinates. Metre to milimetre conversion
    :param camera_coordinates: a numpy array (N, 3)
    :return:
    """

    # make homogenous by adding extra column of ones
    pcd_numpy_cam_coords = np.hstack((camera_coordinates, np.ones(
        (camera_coordinates.shape[0], 1), dtype=camera_coordinates.dtype)))
    # todo do I need top down dilation to make sure there are no gaps but could be risky too?
    # metre to milimetre conversion
    obj_pcd_numpy = (1000.0 * np.dot(cam2arm, pcd_numpy_cam_coords.T)).T

    # todo z is still off, tune with flat objects or with ground plane
    # print('Centroid arm coordinates (before z-offset fix): {}'.format(centroid_arm_coord))
    # print('Clipping z coordinate')
    # centroid_arm_coord[2] = np.clip(centroid_arm_coord[2], 50, 1000)
    # obj_pcd_numpy[:, 2] += z_amount_to_add

    # because there is an offset between shoulder marker and shoulder axis in x and y directions
    # todo in better english
    # if q1:
    #     if math.degrees(q1) > 90:
    #         q1 = math.radians(180 - math.degrees(q1))

    #     x_amount_to_add = np.cos(q1) * xy_offset
    #     y_amount_to_add = np.sin(q1) * xy_offset
    #     # obj_pcd_numpy[:, 0] += x_amount_to_add

    #     # todo looks good at 90 degrees but then looks like it stays in the centre after?

    #     # we're moving the pointcloud in the opposite direction  todo think about this more and explain better
    #     obj_pcd_numpy[:, 0] += x_amount_to_add   # todo which one!?!??!!?!?
    #     obj_pcd_numpy[:, 1] += y_amount_to_add  # todo left side quadrant is different? nah
    #     print('-=-=-=-==-=-=-=-=-+====++====')
    #     # obj_pcd_numpy[:, 1] -= y_amount_to_add   # todo -=??????
    #     # todo is the only way to test with real arm?
    #     # todo how do I even work out if this idea is a good idea or the right way of doing it?

    return obj_pcd_numpy


def convert_cam_to_arm_coordinate(cam_coord, cam2arm, z_amount_to_add, q1=None, xy_offset=12):
    # converts single point to arm coordinate
    # make camera coordinate homogeneous
    cam_coord_hom = np.array([cam_coord[0], cam_coord[1], cam_coord[2], 1])

    # transform to arm coordinates and convert from metres to milimetres
    arm_coord = 1000.0 * np.dot(cam2arm, cam_coord_hom)

    # because the aruco marker is on the shoulder motor but kinematics
    # treats the ground as z = 0, therefore add.... # todo explain better and
    # know why z is off in the first place
    # arm_coord[2] += z_amount_to_add

    # because there is an offset between shoulder marker and shoulder axis in x and y directions
    # if q1:
    #     if math.degrees(q1) > 90:
    #         q1 = math.radians(180 - math.degrees(q1))  # todo this is probably off!!!

    #     x_amount_to_add = np.cos(q1) * xy_offset
    #     y_amount_to_add = np.sin(q1) * xy_offset
    #     arm_coord[0] += x_amount_to_add  # todo was -=
    #     arm_coord[1] += y_amount_to_add  # todo -=??????

    return arm_coord[:3]


def find_obj_centroid_from_pcd(pcd, visualise_centroid=False, with_outlier_detection=True, display_inliers_outliers=False):
    """
    Also does outlier detection
    :param pcd:
    :return:
    """

    if with_outlier_detection:
        voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.005)

        # print("Statistical oulier removal")
        # todo make outlier removal better
        # cl, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        cl, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=40, std_ratio=1.0)
        # if not non_blocking_viz and visualise_pcds:  # todo will cause crash 'non_blocking_viz' in globals()
        if display_inliers_outliers:
            display_inlier_outlier(voxel_down_pcd, ind)

        inlier_cloud = voxel_down_pcd.select_down_sample(ind)

        # print("Radius oulier removal")  # todo this should work better but the other is best?
        # cl, ind = voxel_down_pcd.remove_radius_outlier(nb_points=16, radius=0.05)
        # display_inlier_outlier(voxel_down_pcd, ind)
    else:
        inlier_cloud = pcd

    centroid_and_cov = inlier_cloud.compute_mean_and_covariance()
    center = inlier_cloud.get_center()

    # print('Mean and covariance: {}'.format(centroid_and_cov))  # mean is same as centroid
    print('Centre of object (in camera coordinates) is at: {}'.format(center))

    if visualise_centroid:
        centroid_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.007)
        centroid_sphere.paint_uniform_color([1.0, 0.0, 0.0])
        centroid_sphere.compute_vertex_normals()
        centroid_sphere.translate(center, relative=False)
        o3d.visualization.draw_geometries([inlier_cloud, centroid_sphere])

    return center, inlier_cloud


def find_obj_centroid_from_mask(camera_color_img, camera_depth_img, mask):
    masked_obj_instance_color_img = camera_color_img.copy()
    masked_obj_instance_depth_img = camera_depth_img.copy()

    detached_numpy_mask = mask.cpu().detach().numpy()[0][0]

    # attempt at erosion
    detached_numpy_mask[detached_numpy_mask > 0.7] = 1
    detached_numpy_mask[detached_numpy_mask < 0.7] = 0
    # plt.imshow(detached_numpy_mask)
    # plt.show()

    # erosion so we remove bad mask points on edge of object to help outlier detection
    kernel = np.ones((5, 5), np.uint8)
    # todo make parameter and if 0 don't do it. Check if need to pass 0 or not call?
    detached_numpy_mask = cv2.erode(detached_numpy_mask, kernel,
                                    iterations=2)  # todo might be too much? add param for other objects e.g. forks

    # plt.imshow(detached_numpy_mask)
    # plt.show()
    # kernel = np.ones((5, 5), np.uint8)
    # detached_numpy_mask = cv2.erode(detached_numpy_mask, kernel, iterations=1)
    # plt.imshow(detached_numpy_mask)
    # plt.show()

    mask_bool = detached_numpy_mask > 0.7
    mask_xy_pixel_locs = mask_bool.nonzero()
    # masked_cup_instance_color_img = masked_cup_instance_color_img[mask_bool]
    masked_obj_instance_depth_img = masked_obj_instance_depth_img[mask_bool]

    # attempt with numpy and raw deprojection
    # observed_z = masked_obj_instance_depth_img * depth_scale
    observed_z = masked_obj_instance_depth_img * 0.0010000000474974513
    # observed_x = np.multiply(mask_xy_pixel_locs[0] - ppx, observed_z / fx)  # todo inverse
    # observed_y = np.multiply(mask_xy_pixel_locs[1] - ppy, observed_z / fy)
    observed_x = np.multiply(mask_xy_pixel_locs[1] - ppx, observed_z / fx)
    observed_y = np.multiply(mask_xy_pixel_locs[0] - ppy, observed_z / fy)

    pcd_numpy = np.concatenate(
        [observed_x.reshape(-1, 1), observed_y.reshape(-1, 1), observed_z.reshape(-1, 1)], axis=1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_numpy)  # todo find the original color info again?
    # pcd.colors = o3d.utility.Vector3dVector(masked_cup_instance_color_img[mask_xy_pixel_locs[1], mask_xy_pixel_locs[0]] / 255.0)
    pcd.colors = o3d.utility.Vector3dVector(
        masked_obj_instance_color_img[mask_xy_pixel_locs[0], mask_xy_pixel_locs[1]] / 255.0)

    # attempt with open3d
    # masked_obj_instance_depth_img[obj_instance_masks.cpu().detach().numpy()[0][0] < 0.7] = 100
    # attempt to create pcd from images and get cup out of o3d pcd
    # source_color = o3d.geometry.Image(masked_obj_instance_color_img)
    # source_depth = o3d.geometry.Image(masked_obj_instance_depth_img)
    # # todo if the order of is raster order (which it is) I could just remove a bunch of points
    # source_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    #     source_color, source_depth, depth_trunc=1000000.0, convert_rgb_to_intensity=False)
    # # o3d.visualization.draw_geometries([source_color])
    # # o3d.visualization.draw_geometries([source_depth])
    # pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    #         source_rgbd_image, pinhole_camera_intrinsic)
    # pcd_points = np.asarray(pcd.points)
    # pcd_colors = np.asarray(pcd.colors)

    center, inlier_pcd = find_obj_centroid_from_pcd(pcd)

    # return center, pcd  # todo do we ever want the original instead?
    return center, inlier_pcd


def find_obj_centroid_from_maskrcnn(camera_color_img, camera_depth_img, maskrcnn_results,
                                    class_name_to_id, label='cup', visualise_pcds=False,
                                    threshold_score=0.5):
    # global centroid_sphere, origin_frame, pcd  # todo won't work for when I call outside this file? works fine now but double check
    # global centroid_sphere, pcd

    label_id = class_name_to_id[label]

    # try to convert from numpy to open3d
    if label_id in [x.item() for x in maskrcnn_results['labels']]:  # todo could avoid all id stuff
        print(
            '\n------------\nFinding object centroid for label: {} \n------------\n'.format(label))

        # todo what if there are 2 cups/instances?!!? Loop through all or pick first!!
        obj_instance_bool = maskrcnn_results['labels'] == label_id  # cup
        obj_instance_masks = maskrcnn_results['masks'][obj_instance_bool]
        obj_instance_masks[obj_instance_masks > 0.7] = 1.0
        obj_instance_scores = maskrcnn_results['scores'][obj_instance_bool]
        # obj_instance_boxes = maskrcnn_results['boxes'][obj_instance_bool]
        # obj_instance_labels = maskrcnn_results['labels'][obj_instance_bool]

        # todo fix instance problem

        # if all(obj_instance_scores[i] > 0.5)  # todo only high score objects, otherwise it's just stupid. Actually detects egg cup as vase with higher probability...

        center, pcd = find_obj_centroid_from_mask(camera_color_img, camera_depth_img,
                                                  obj_instance_masks)

        return center, pcd


def find_obj_pose_from_pcd(pcd, visualise_pcds=False):
    pose = np.array([1.0, 1.0, 1.0])

    axis_box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(pcd.points)
    oriented_box = o3d.geometry.OrientedBoundingBox.create_from_points(pcd.points)
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.1, origin=[0.0, 0.0, 0.0])
    # todo could I use OrientedBoundingBox since it uses PCA and find the main axes of objects??? Yes
    # todo could setup a few oranges/aruco markers/flags to automatically create oriented box but could also just build and save?
    # todo ideal is just finding what objects are good enough to pick in arm's workspace, including on chassis

    # todo could always find surface plane and find the box height normal from this...
    # todo specify what I actually want: PCA on an objects shape relevant to ground/arm coordinate frame. Does .R give me this?
    # todo I need the ground plane to actually be the x-z plane. How to do this transformation? easy?
    # todo how to find the plane an object is on? Will be very useful to do this. We can assume it's the same as ground plane though
    # todo just use axis aligned box but with arm coordinate frame and not cam frame!!!

    axis_box.color = np.array([1.0, 0.0, 0.0])
    oriented_box.color = np.array([0.0, 1.0, 0.0])

    if visualise_pcds:
        # o3d.visualization.draw_geometries([pcd, axis_box])
        # o3d.visualization.draw_geometries([pcd, oriented_box])
        # o3d.visualization.draw_geometries([pcd, oriented_box, axis_box])
        o3d.visualization.draw_geometries([pcd, oriented_box, axis_box, coordinate_frame])

    # create most normal orientedbox and check what R is. E.g. use (0, 0, 0) and (1, 1, 1)
    # todo is the conclusion from below that orientedbox isn't good for finding pose of objects?
    pcd_test = o3d.geometry.PointCloud()
    pcd_test.points = o3d.utility.Vector3dVector(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0],
                                                           [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                                                           [0.0, 0.0, 1.0]]))
    oriented_box_test = o3d.geometry.OrientedBoundingBox.create_from_points(pcd_test.points)
    axis_box_test = o3d.geometry.AxisAlignedBoundingBox.create_from_points(pcd_test.points)

    axis_box_test.color = np.array([1.0, 0.0, 0.0])
    oriented_box_test.color = np.array([0.0, 1.0, 0.0])

    coordinate_frame_bigger = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.5, origin=[0.0, 0.0, 0.0])

    if visualise_pcds:
        o3d.visualization.draw_geometries([pcd_test, coordinate_frame_bigger, oriented_box_test, axis_box_test])

    return pose


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_down_sample(ind)
    outlier_cloud = cloud.select_down_sample(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])

    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


def get_full_pcd_from_rgbd(color_img, depth_img, pinhole_camera_intrinsic, visualise=False):
    # no need for [:, :, ::-1]. Changed realsense to rgb8  # [..., ::-1]
    source_color = o3d.geometry.Image(color_img)  # open3d Expects rgb but I convert back to bgr  # todo stop that.
    source_depth = o3d.geometry.Image(depth_img)

    source_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        source_color, source_depth, convert_rgb_to_intensity=False)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        source_rgbd_image, pinhole_camera_intrinsic)

    if visualise:
        o3d.visualization.draw_geometries([pcd])

    return pcd


def find_all_planes(pcd, num_iterations_of_segmentation=5, distance_threshold=0.01, ransac_n=100, num_iterations=100, visualise_pcds=False):
    """

    :param pcd:
    :param distance_threshold(float) – Max distance a point can be from the plane model,
            and still be considered an inlier.
    :param ransac_n: Number of initial points to be considered inliers in each iteration.
    :param num_iterations:  Number of iterations of ransac
    :param num_iterations_of_segmentation: set to 1 if you want to remove biggest plane
    :return: all_planes, outlier_clouds
    """
    print('Calling find_all_planes')
    # double distance_threshold = 0.01,
    # int ransac_n = 3,
    # It will find the plane with the greatest number of points in the cloud.
    # Therefore, if the cloud has several planes, the model can be rerun multiple
    # times until all planes are found by removing the inliers each time.
    # todo what is the stopping condition?

    colors = random_colors(num_iterations_of_segmentation)
    all_planes = []

    outlier_cloud = pcd
    outlier_clouds = []  # outlier after each plane is removed

    for i in range(num_iterations_of_segmentation):

        plane_params, plane_point_indices = outlier_cloud.segment_plane(distance_threshold=distance_threshold, ransac_n=ransac_n,
                                                              num_iterations=num_iterations)

        # todo check for failing condition and read open3d code and understand ransac better and choose better params

        inlier_cloud = outlier_cloud.select_down_sample(plane_point_indices)
        outlier_cloud = outlier_cloud.select_down_sample(plane_point_indices, invert=True)
        outlier_clouds.append(outlier_cloud)

        print("Iteration: {}. Showing plane inliers (with color: {})".format(i, colors[i]))
        inlier_cloud.paint_uniform_color(colors[i])

        all_planes.append(inlier_cloud)
        if visualise_pcds:
            # show all planes
            o3d.visualization.draw_geometries(all_planes)

            # after removing planes
            o3d.visualization.draw_geometries([outlier_cloud])

    # todo how to find the plane points underneath the objects. The full plane? Hole filling I suppose
    # todo Explore and tune how to find small objects (e.g. more resolution, no voxelisation, smaller distance treshold, don't remove as much of the table beside hills/objects)
    # todo understand open3D code and eventually segment cylinders, spheres, cubes, cuboids?
    # todo open3d code here: https://github.com/intel-isl/Open3D/pull/1287/files

    return all_planes, outlier_clouds  # 0th outlier cloud/plane is the biggest plane


def find_all_clusters(pcd, eps=0.01, voxelise=True, visualise_all_pcd=False):
    print('Finding all clusters, voxelising pointcloud')
    print('Num points: {}'.format(pcd.points))
    if voxelise:
        voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.005)  # todo too much?
        print('Num points after voxelising: {}'.format(voxel_down_pcd.points))
    else:
        voxel_down_pcd = pcd

    # todo can colour information make it even better? Can RGB weaker segmentation make it better?

    cmap = plt.get_cmap('Set2')
    # eps = 0.5
    # eps = 0.75
    # eps = 0.4  # todo what is this

    # todo remove floor and keep table. Could do this by remove points below plane!!! Easy

    clustering_output = voxel_down_pcd.cluster_dbscan(eps=eps, min_points=10, print_progress=True)
    labels = np.array(clustering_output)
    max_label = labels.max()
    print('%s has %d clusters' % ('Pointcloud', max_label + 1))

    old_colors = np.asarray(voxel_down_pcd.colors)  # todo not working....
    colors = cmap(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0  # todo transparent?
    unique, counts = np.unique(labels, return_counts=True)
    label_count = dict(zip(unique, counts))
    voxel_down_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    all_points = np.asarray(voxel_down_pcd.points)
    all_colors = np.asarray(voxel_down_pcd.colors)

    cluster_pcds = []

    if visualise_all_pcd:
        o3d.visualization.draw_geometries([voxel_down_pcd])

    for label_id in unique:
        cluster_pcd = o3d.geometry.PointCloud()
        cluster_pcd.points = o3d.utility.Vector3dVector(all_points[labels == label_id])
        # todo color in other pointcloud? seems off?
        # cluster_pcd.colors = o3d.utility.Vector3dVector(all_colors[labels == label_id])
        cluster_pcd.colors = o3d.utility.Vector3dVector(old_colors[labels == label_id])  # todo why doesn't this work?
        # cluster_pcd.colors = old_colors[labels == label_id]
        # todo get back original colors
        cluster_pcds.append(cluster_pcd)
        if visualise_all_pcd:
            o3d.visualization.draw_geometries([cluster_pcd])

    return cluster_pcds


def find_all_objects_from_clusters(cluster_pcds, lower_bound=100, upper_bound=1000):
    # todo don't include chassis and other big objects. How to choose (and maybe automatically) what is big for upper bound?
    # todo find other way to find objects... todo height of objects. width. whether it's touching the plane I care about. How much area of it is touching
    # todo classifier for objectness? What is object like? Or heuristic and geometry method
    # todo specify area on ground, polygon and pass in plane?
    # todo eventually specify bin hole polygon in 2d and 3d space and centre of that is the drop off point
    # todo could make user choose?

    likely_object_pcds = []

    for idx, c_pcd in enumerate(cluster_pcds):
        if len(c_pcd.points) < lower_bound:
            print('Cluster {} has too few points with {} points'.format(idx, len(c_pcd.points)))
            continue
        elif len(c_pcd.points) > upper_bound:
            print('Cluster {} has too many points with {} points'.format(idx, len(c_pcd.points)))
            continue

        print('Cluster {} with {} points is likely to be object'.format(idx, len(c_pcd.points)))
        likely_object_pcds.append(c_pcd)

    return likely_object_pcds

# ----------------------
# Object Detection
# ----------------------


def maskrcnn_detect(realsense_color_image, model, device, class_names):
    # Convert from numpy to torch, 0-1 range and right dimension order
    img = torch.from_numpy(realsense_color_image).float().to(device)
    img = img / 255.
    img = img.permute(2, 0, 1)
    # todo confirm we are running on RGB as maskrcnn expects and not BGR

    # run model
    predictions = model([img])
    r = predictions[0]

    label_strings = [class_names[x.item()] for x in r['labels']]
    print('Labels found by MaskRCNN: ', label_strings)

    return r, label_strings
