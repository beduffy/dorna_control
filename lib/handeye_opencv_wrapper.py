from __future__ import print_function
from glob import glob

import numpy as np
import cv2
import open3d as o3d

from lib.vision import get_inverse_homogenous_transform

# TODO should i separate the below stuff or not? to different files or functions?
from lib.dorna_kinematics import i_k, f_k
from lib.open3d_plot_dorna import plot_open3d_Dorna
from lib.vision import get_full_pcd_from_rgbd
from lib.vision import get_camera_coordinate, create_homogenous_transformations, convert_pixel_to_arm_coordinate, convert_cam_pcd_to_arm_pcd
from lib.vision_config import pinhole_camera_intrinsic
from lib.aruco_helper import create_aruco_params, aruco_detect_draw_get_transforms, calculate_pnp_12_markers, find_aruco_markers
from lib.aruco_image_text import OpenCvArucoImageText


# TODO plotting functions to another file?


def load_all_handeye_data(folder_name):
    # Open data/handeye folder and then load into dict
    gripper_transform_files = sorted(glob('data/{}/gripper2base*'.format(folder_name)))
    cam2target_files = sorted(glob('data/{}/target2cam*'.format(folder_name)))
    joint_angles_files = sorted(glob('data/{}/joint_angles*'.format(folder_name)))
    
    all_joint_angles = []
    for fp in joint_angles_files:
        joint_angles = np.loadtxt(fp, delimiter=' ')
        print('Loaded {} transform'.format(fp))
        all_joint_angles.append(joint_angles)

    all_gripper_rotation_mats = []
    all_gripper_tvecs = []
    all_gripper_rotation_mats_inverse = []
    all_gripper_tvecs_inverse = []
    for fp in gripper_transform_files:
        gripper_transform = np.loadtxt(fp, delimiter=' ')
        print('Loaded {} transform'.format(fp))
        gripper2base_rot = gripper_transform[:3, :3]
        all_gripper_rotation_mats.append(gripper2base_rot)

        # gripper2_base_tvec = gripper_transform[:3, 3] / 1000.0
        gripper2_base_tvec = gripper_transform[:3, 3]
        # all_gripper_tvecs.append(gripper_transform[:3, 3] * 1000.0)  # TODO remove if it makes no sense
        all_gripper_tvecs.append(gripper2_base_tvec)  # TODO milimetres or not!!!!! everything should be in metres... since that is what the camera is in 1000 milimetres / 1000 is 1m

        full_homo_gripper2base = np.identity(4)
        full_homo_gripper2base[:3, :3] = gripper2base_rot
        full_homo_gripper2base[0, 3] = gripper2_base_tvec[0]
        full_homo_gripper2base[1, 3] = gripper2_base_tvec[1]
        full_homo_gripper2base[2, 3] = gripper2_base_tvec[2]

        full_homo_base2gripper = get_inverse_homogenous_transform(full_homo_gripper2base)
        R_base2gripper_individual = full_homo_base2gripper[:3, :3]
        t_base2gripper_individual = full_homo_base2gripper[:3, 3]
        all_gripper_rotation_mats_inverse.append(R_base2gripper_individual)
        all_gripper_tvecs_inverse.append(t_base2gripper_individual)

    R_gripper2base = np.array(all_gripper_rotation_mats)
    t_gripper2base = np.array(all_gripper_tvecs)
    R_base2gripper = np.array(all_gripper_rotation_mats_inverse)
    t_base2gripper = np.array(all_gripper_tvecs_inverse)

    all_gripper2base_transforms = []
    for R, t in zip(all_gripper_rotation_mats, all_gripper_tvecs):
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        all_gripper2base_transforms.append(T)

    all_target2cam_rotation_mats = []
    all_target2cam_tvecs = []
    all_cam2target_rotation_mats = []
    all_cam2target_tvecs = []
    for fp in cam2target_files:
        cam2target_transform = np.loadtxt(fp, delimiter=' ')

        # Just testing but not right
        # target2cam = get_inverse_homogenous_transform(cam2target_transform)
        # cam2target_transform = target2cam

        print('Loaded {} transform'.format(fp))
        cam2target_rot = cam2target_transform[:3, :3]
        all_target2cam_rotation_mats.append(cam2target_rot)
        all_target2cam_tvecs.append(cam2target_transform[:3, 3])

        full_homo_target2cam = np.identity(4)
        full_homo_target2cam[:3, :3] = cam2target_rot
        full_homo_target2cam[0, 3] = cam2target_transform[:3, 3][0]
        full_homo_target2cam[1, 3] = cam2target_transform[:3, 3][1]
        full_homo_target2cam[2, 3] = cam2target_transform[:3, 3][2]

        full_homo_cam2target = get_inverse_homogenous_transform(full_homo_target2cam)
        R_cam2target_individual = full_homo_cam2target[:3, :3]
        t_cam2target_individual = full_homo_cam2target[:3, 3]
        all_cam2target_rotation_mats.append(R_cam2target_individual)
        all_cam2target_tvecs.append(t_cam2target_individual)

    R_target2cam = np.array(all_target2cam_rotation_mats)
    t_target2cam = np.array(all_target2cam_tvecs)
    R_cam2target = np.array(all_cam2target_rotation_mats)
    t_cam2target = np.array(all_cam2target_tvecs)

    # if images exist, load
    color_images = []
    color_img_files = sorted(glob('data/{}/color_img*.png'.format(folder_name)))
    for fp in color_img_files:
        img = cv2.imread(fp)
        color_images.append(img)

    depth_images = []
    depth_img_files = sorted(glob('data/{}/depth_img*.png'.format(folder_name)))
    for fp in depth_img_files:
        img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
        depth_images.append(img)

    handeye_data_dict = {
        'all_gripper_rotation_mats': all_gripper_rotation_mats,
        'all_gripper_tvecs': all_gripper_tvecs,
        'R_gripper2base': R_gripper2base,
        't_gripper2base': t_gripper2base,
        'R_base2gripper': R_base2gripper,
        't_base2gripper': t_base2gripper,
        'all_target2cam_rotation_mats': all_target2cam_rotation_mats,
        'all_target2cam_tvecs': all_target2cam_tvecs,
        'R_target2cam': R_target2cam,
        't_target2cam': t_target2cam,
        'R_cam2target': R_cam2target,
        't_cam2target': t_cam2target,
        'color_images': color_images,
        'depth_images': depth_images,
        'all_joint_angles': all_joint_angles
    }

    # creating further key-value pairs
    all_base2gripper_transforms = []
    for R, t in zip(handeye_data_dict['R_base2gripper'], handeye_data_dict['t_base2gripper']):
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        all_base2gripper_transforms.append(T)

    all_target2cam_transforms = []
    for R, t in zip(all_target2cam_rotation_mats, all_target2cam_tvecs):
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        all_target2cam_transforms.append(T)

    all_cam2target_transforms = []
    for R, t in zip(all_cam2target_rotation_mats, all_cam2target_tvecs):
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        all_cam2target_transforms.append(T)

    handeye_data_dict['all_gripper2base_transforms'] = all_gripper2base_transforms
    handeye_data_dict['all_base2gripper_transforms'] = all_base2gripper_transforms
    handeye_data_dict['all_target2cam_transforms'] = all_target2cam_transforms
    handeye_data_dict['all_cam2target_transforms'] = all_cam2target_transforms

    return handeye_data_dict


def test_transformations(handeye_data_dict):
    '''
        Tests:
        - R_gripper2base and t_gripper2base: The rotation and translation of the gripper (end-effector) in the base frame, i.e., the transformation from the base to the gripper.
        - R_target2cam and t_target2cam: The rotation and translation of the calibration target (e.g., ArUco marker) in the camera frame, i.e., the transformation from the camera to the target.
        - loop through all gripper2base and base2gripper and confirm the transfrom has its inverse correctly set by checking if output is identity
        - same for cam2target
        - ? TODO
    '''

    all_gripper2base_transforms = handeye_data_dict['all_gripper2base_transforms']
    all_base2gripper_transforms = handeye_data_dict['all_base2gripper_transforms']
    all_target2cam_transforms = handeye_data_dict['all_target2cam_transforms']
    all_cam2target_transforms = handeye_data_dict['all_cam2target_transforms']

    # gripper2base should bring base to gripper e.g. 0, 0, 0 to usual 0.4 metres in front
    # TODO not sure how to test but it does that since visualisation uses all_gripper2base_transforms


    # same with target2cam: transformation from the camera to the target. 0,0,0 to aruco target
    # TODO same, but visualisation works with target2cam

    identity_transform = np.eye(4)    
    # gripper2base and base2gripper should multiply to identity
    for idx, (transform, inverse_transform) in enumerate(zip(all_gripper2base_transforms, all_base2gripper_transforms)):
        combined_transform = transform @ inverse_transform
        # combined_transform = inverse_transform @ transform
        # Check if combined transform is close enough to identity
        if np.allclose(combined_transform, identity_transform, atol=1e-5):
            # print("Combined transform is close enough to identity")
            pass
        else:
            print(combined_transform)
            print("\n\n\nCombined transform is not close enough to identity, AHHHHHHH!!!!!!\n\n\n")
            sys.exit()
    
    # same with cam2target and target2cam
    for idx, (transform, inverse_transform) in enumerate(zip(all_target2cam_transforms, all_cam2target_transforms)):
        combined_transform = transform @ inverse_transform
        # combined_transform = inverse_transform @ transform
        # Check if combined transform is close enough to identity
        if np.allclose(combined_transform, identity_transform, atol=1e-5):
            # print("Combined transform is close enough to identity")
            pass
        else:
            print(combined_transform)
            print("\n\n\nCombined transform is not close enough to identity, AHHHHHHH!!!!!!\n\n\n")
            sys.exit()



    # TODO calibrate camera intrinsics should use full aruco pipeline
    # TODO get mean projection and see what my pixel reprojection error is on all 12 corners or other position

    opencv_aruco_image_text = OpenCvArucoImageText()
    board, parameters, aruco_dict, marker_length = create_aruco_params()
    marker_separation = 0.0065
    ids, corners, all_rvec, all_tvec = None, None, None, None  # TODO global remove
    id_on_shoulder_motor = 1


    color_images = handeye_data_dict['color_images']
    depth_images = handeye_data_dict['depth_images']

    first_color_image = color_images[0]
    first_depth_image = depth_images[0]

    camera_color_img_debug = first_color_image.copy()
    color_img, tvec, rvec, ids, corners, all_rvec, all_tvec = find_aruco_markers(first_color_image, aruco_dict, parameters, marker_length, id_on_shoulder_motor, opencv_aruco_image_text, camera_color_img_debug)

    cam2arm_opt, arm2cam_opt, input_obj_points_concat, input_img_points_concat = calculate_pnp_12_markers(corners, ids, all_rvec, all_tvec, marker_length=marker_length, marker_separation=marker_separation)

    output = calibrate_camera_intrinsics(color_images)
    # output = calibrate_camera_intrinsics(color_images[:5])
    print('Camera intrinsics: \n{}'.format(output))

    cv2.imshow('Camera Color Image Debug', camera_color_img_debug)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def calibrate_camera_intrinsics(images):
    """
    Calibrate camera intrinsics using ArUco markers detected in a list of images.
    
    Args:
        images: List of color images containing ArUco markers
        
    Returns:
        camera_matrix: 3x3 camera intrinsic matrix
        dist_coeffs: Distortion coefficients
        mean_error: Mean reprojection error in pixels
        errors: List of reprojection errors for each point
    """
    # Create ArUco detector objects
    opencv_aruco_image_text = OpenCvArucoImageText()
    board, parameters, aruco_dict, marker_length = create_aruco_params()
    marker_separation = 0.0065
    
    # Collect object and image points from all images
    objpoints = []  # 3D points in marker coordinate system
    imgpoints = []  # 2D points in image plane
    
    for idx, img in enumerate(images):
        print('Image no. ', idx)
        # Use existing ArUco detection pipeline
        camera_color_img_debug = img.copy()
        color_img, tvec, rvec, ids, corners, all_rvec, all_tvec = find_aruco_markers(
            img, aruco_dict, parameters, marker_length, 
            id_on_shoulder_motor=1, 
            opencv_aruco_image_text=opencv_aruco_image_text,
            camera_color_img_debug=camera_color_img_debug
        )
        
        cam2arm_opt, arm2cam_opt, input_obj_points_concat, input_img_points_concat = calculate_pnp_12_markers(corners, ids, all_rvec, all_tvec, marker_length=marker_length, marker_separation=marker_separation)

        # Reshape and convert types to match expected format
        obj_points = input_obj_points_concat.reshape(-1, 1, 3).astype(np.float32)
        img_points = input_img_points_concat.reshape(-1, 1, 2).astype(np.float32)
        
        objpoints.append(obj_points)
        imgpoints.append(img_points)

    print('Finished loading all object and image points. Running calibration')
    # Get image dimensions from first image
    height, width = images[0].shape[:2]
    
    # Initial camera matrix guess
    camera_matrix = np.array([
        [width, 0, width/2],
        [0, height, height/2],
        [0, 0, 1]
    ], dtype=np.float32)
    dist_coeffs = np.zeros(5)

    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera( objpoints, imgpoints, (width, height), None, None)

    # Calculate reprojection error
    mean_error = 0
    errors = []
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        errors.append(error)
        mean_error += error

    mean_error = mean_error/len(objpoints)
    print("Mean reprojection error: {} pixels".format(mean_error))
    print("Individual errors: {}".format(errors))

    return mtx, dist, mean_error, errors


def plot_all_handeye_data(handeye_data_dict, eye_in_hand=False):
    all_gripper_rotation_mats = handeye_data_dict['all_gripper_rotation_mats']
    all_gripper_tvecs = handeye_data_dict['all_gripper_tvecs']
    all_gripper2base_transforms = handeye_data_dict['all_gripper2base_transforms']
    R_gripper2base = handeye_data_dict['R_gripper2base']
    t_gripper2base = handeye_data_dict['t_gripper2base']
    R_base2gripper = handeye_data_dict['R_base2gripper']
    t_base2gripper = handeye_data_dict['t_base2gripper']
    all_target2cam_rotation_mats = handeye_data_dict['all_target2cam_rotation_mats']
    all_target2cam_tvecs = handeye_data_dict['all_target2cam_tvecs']
    all_target2cam_transforms = handeye_data_dict['all_target2cam_transforms']
    R_target2cam = handeye_data_dict['R_target2cam']
    t_target2cam = handeye_data_dict['t_target2cam']
    saved_cam2arm = handeye_data_dict['saved_cam2arm']
    all_joint_angles = handeye_data_dict['all_joint_angles']
    color_images = handeye_data_dict['color_images']
    depth_images = handeye_data_dict['depth_images']

    saved_cam2arm = handeye_data_dict['saved_cam2arm']  # assuming handeye_calibrate_opencv has been called

    # use color and depth images to create point cloud from first color + depth pair
    if handeye_data_dict['color_images']:
        camera_color_img = handeye_data_dict['color_images'][0]
        camera_depth_img = handeye_data_dict['depth_images'][0]
    cam_pcd_first_image_pair = get_full_pcd_from_rgbd(camera_color_img, camera_depth_img, pinhole_camera_intrinsic, visualise=False)
    # full_arm_pcd, full_pcd_numpy = convert_cam_pcd_to_arm_pcd(cam_pcd_first_image_pair, saved_cam2arm, in_milimetres=False)

    plot_one_arm_gripper_camera_frame_eye_in_hand(all_gripper2base_transforms, all_joint_angles, gripper2cam=saved_cam2arm)
    
    plot_arm_gripper_frames(all_gripper2base_transforms, all_joint_angles, plot_camera_on_gripper_if_eye_in_hand=eye_in_hand, gripper2cam=saved_cam2arm)

    plot_aruco_frames_in_camera_frame(all_target2cam_transforms, cam_pcd_first_image_pair)

    plot_blah(handeye_data_dict, cam_pcd_first_image_pair, saved_cam2arm)


def plot_blah(handeye_data_dict, cam_pcd_first_image_pair, saved_cam2arm):
    '''
    Below I am visualising origin (in camera coordinates) and the arm frame.
    And pointcloud from camera transformed to arm frame... but that does not make sense?
    trying a better explanation:

    plotting in camera frame:
    - origin frame
    - transformed frame by cam2arm (does not make sense in eye in hand)
    - gripper frames first transformed by gripper2base and combined with cam2arm
    - aruco frames but we are already in camera frame

    

    # TODO if eye-in-hand and i manually measure it, what visualisation will show it working or not? or show problems?
    # TODO clear english of what I want here... 
    '''

    all_gripper2base_transforms = handeye_data_dict['all_gripper2base_transforms']
    all_target2cam_transforms = handeye_data_dict['all_target2cam_transforms']

    frame_size = 0.1
    sphere_size = 0.01
    # Create a red sphere at the origin frame for clear identification
    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size, origin=[0.0, 0.0, 0.0])
    origin_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_size, resolution=20)
    origin_sphere.paint_uniform_color([1, 0, 0])  # Red

    # Create a green sphere at the transformed frame for clear identification
    transformed_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size, origin=[0.0, 0.0, 0.0])
    transformed_frame.transform(saved_cam2arm)
    transformed_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_size, resolution=20)
    transformed_sphere.paint_uniform_color([0, 1, 0])  # Green
    transformed_sphere.transform(saved_cam2arm)  # TODO what am i doing here. I assume transforming origin in camera frame, brings us to arm frame so this coordinate frame should be in base of arm

    geometry_to_plot = []
    # given transformed frame, now i can also plot all gripper transformations after saved_cam2_arm to see where that frame is
    for idx, gripper2base_transform in enumerate(all_gripper2base_transforms):
        # Create coordinate frame for each gripper transform
        gripper_coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                                        size=frame_size, origin=[0.0, 0.0, 0.0])
        # combined_transform = np.dot(saved_cam2arm, homo_transform)
        combined_transform = gripper2base_transform @ saved_cam2arm
        gripper_coordinate_frame.transform(combined_transform)
        # gripper_coordinate_frame.transform(saved_cam2arm)
        # gripper_coordinate_frame.transform(homo_transform)

        # Create a sphere for each gripper transform
        gripper_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_size)
        gripper_sphere.paint_uniform_color([0, 0, 1])  # Blue for distinction
        # gripper_sphere.transform(saved_cam2arm)
        # gripper_sphere.transform(gripper2base_transform)
        gripper_sphere.transform(combined_transform)

        # Add the created geometries to the list for plotting
        geometry_to_plot.append(gripper_sphere)
        geometry_to_plot.append(gripper_coordinate_frame)


    for idx, target2cam_transform in enumerate(all_target2cam_transforms):
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                                        size=frame_size, origin=[0.0, 0.0, 0.0])
        coordinate_frame.transform(target2cam_transform)

        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_size)
        sphere.transform(target2cam_transform)
        geometry_to_plot.append(sphere)
        geometry_to_plot.append(coordinate_frame)

        # # Adding text to the plot for better identification
        # text_position = np.array(homo_transform)[0:3, 3] + np.array([0, 0, sphere_size * 2])  # Positioning text above the sphere
        # text = f"Frame {idx}"
        # text_3d = o3d.geometry.Text3D(text, position=text_position, font_size=10, density=1, font_path="OpenSans-Regular.ttf")
        # geometry_to_plot.append(text_3d)

    print('Visualising origin, transformed frame and spheres and coordinate frames')  # TODO what are we doing here?
    # TODO rename transformed frame and understand which frame is which frame
    # list_of_geometry_elements = [origin_frame, transformed_frame, origin_sphere, transformed_sphere] + geometry_to_plot
    # list_of_geometry_elements = [full_arm_pcd, origin_frame, transformed_frame, origin_sphere, transformed_sphere] + geometry_to_plot
    list_of_geometry_elements = [cam_pcd_first_image_pair, origin_frame, transformed_frame, origin_sphere, transformed_sphere] + geometry_to_plot
    # list_of_geometry_elements = [origin_frame_transformed_from_camera_frame, camera_coordinate_frame] + arm_position_coord_frames
    o3d.visualization.draw_geometries(list_of_geometry_elements)


def plot_arm_gripper_frames(all_gripper2base_transforms, all_joint_angles, plot_camera_on_gripper_if_eye_in_hand=False, gripper2cam=None):
    # TODO clean entire function, comments etc, hard to think about
    origin_arm_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0.0, 0.0, 0.0])

    sphere_size = 0.01
    geometry_to_plot = []
    geometry_to_plot.append(origin_arm_frame)
    for idx, gripper_transform in enumerate(all_gripper2base_transforms):
    # for idx, homo_transform in enumerate(all_base2gripper_transforms):  # TODO why does this look so weird. I don't fully understand enough here
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                                        size=0.1, origin=[0.0, 0.0, 0.0])
        coordinate_frame.transform(gripper_transform)

        gripper_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_size)
        gripper_sphere.paint_uniform_color([0, 0, 1])
        gripper_sphere.transform(gripper_transform)

        geometry_to_plot.append(gripper_sphere)
        geometry_to_plot.append(coordinate_frame)

        # if plot_camera_on_gripper_if_eye_in_hand and gripper2cam is not None:
        #     combined_transform_from_arm_to_gripper_to_camera = gripper_transform @ gripper2cam
        #     camera_on_gripper_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        #                                 size=0.1, origin=[0.0, 0.0, 0.0])
        #     camera_on_gripper_frame.transform(combined_transform_from_arm_to_gripper_to_camera)
        #     geometry_to_plot.append(camera_on_gripper_frame)

    # plot arms too
    shoulder_height_in_mm = 206.01940000000002 / 1000.0
    coordinate_frame_shoulder_height_arm_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.3, origin=[0.0, 0.0, shoulder_height_in_mm])
    for idx, joint_angles in enumerate(all_joint_angles):
        joint_angles = joint_angles.tolist()
        
        full_toolhead_fk, xyz_positions_of_all_joints = f_k(joint_angles)
        print('full_toolhead_fk (in metres): ', full_toolhead_fk)

        arm_plot_geometry = plot_open3d_Dorna(xyz_positions_of_all_joints, 
                          extra_geometry_elements=[coordinate_frame_shoulder_height_arm_frame],
                          do_plot=False)

        geometry_to_plot.extend(arm_plot_geometry)


    print('Visualising gripper frames + arm origin frame in arm frame')
    o3d.visualization.draw_geometries(geometry_to_plot)


def plot_one_arm_gripper_camera_frame_eye_in_hand(all_gripper2base_transforms, all_joint_angles, gripper2cam):
    origin_arm_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0.0, 0.0, 0.0])

    sphere_size = 0.01
    geometry_to_plot = []
    geometry_to_plot.append(origin_arm_frame)
    for idx, gripper_transform in enumerate(all_gripper2base_transforms[:1]):
    # for idx, homo_transform in enumerate(all_base2gripper_transforms):  # TODO why does this look so weird. I don't fully understand enough here
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                                        size=0.1, origin=[0.0, 0.0, 0.0])
        coordinate_frame.transform(gripper_transform)

        gripper_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_size)
        gripper_sphere.paint_uniform_color([0, 0, 1])
        gripper_sphere.transform(gripper_transform)

        geometry_to_plot.append(gripper_sphere)
        geometry_to_plot.append(coordinate_frame)

        combined_transform_from_arm_to_gripper_to_camera = gripper_transform @ gripper2cam
        camera_on_gripper_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                                    size=0.1, origin=[0.0, 0.0, 0.0])
        camera_on_gripper_frame.transform(combined_transform_from_arm_to_gripper_to_camera)
        geometry_to_plot.append(camera_on_gripper_frame)

    # plot arms too
    shoulder_height_in_mm = 206.01940000000002 / 1000.0
    coordinate_frame_shoulder_height_arm_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.3, origin=[0.0, 0.0, shoulder_height_in_mm])
    for idx, joint_angles in enumerate(all_joint_angles[:1]):
        joint_angles = joint_angles.tolist()
        
        full_toolhead_fk, xyz_positions_of_all_joints = f_k(joint_angles)
        print('full_toolhead_fk (in metres): ', full_toolhead_fk)

        arm_plot_geometry = plot_open3d_Dorna(xyz_positions_of_all_joints, 
                          extra_geometry_elements=[coordinate_frame_shoulder_height_arm_frame],
                          do_plot=False)

        geometry_to_plot.extend(arm_plot_geometry)


    print('Visualising gripper frames + arm origin frame in arm frame')
    o3d.visualization.draw_geometries(geometry_to_plot)


def plot_aruco_frames_in_camera_frame(all_target2cam_transforms, cam_pcd):
    # TODO do I want to visualise how the aruco coordinate frame looks for each image? Give option and loop through all? It would prove less distortion effects?
    # TODO do the below for gripper poses as well, they should perfectly align rotation-wise to aruco poses
    # TODO ideally I'd visualise a frustum. matplotlib?
    # TODO draw mini plane of all arucos rather than coordinate frames. https://github.com/isl-org/Open3D/issues/3618
    
    geometry_to_plot = []
    origin_cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0.0, 0.0, 0.0])
    geometry_to_plot.append(origin_cam_frame)

    for idx, target2cam_transform in enumerate(all_target2cam_transforms):
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                                        # size=0.1, origin=[cam2target_tvec[0], cam2target_tvec[1], cam2target_tvec[2]])
                                        size=0.1, origin=[0.0, 0.0, 0.0])
        # coordinate_frame.rotate(all_target2cam_rotation_mats[idx])
        coordinate_frame.transform(target2cam_transform)

        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        # TODO WHY is the sphere always a bit higher than the origin of the coordinate frame?
        # sphere.translate([cam2target_tvec[0], cam2target_tvec[1], cam2target_tvec[2]])
        sphere.transform(target2cam_transform)
        geometry_to_plot.append(sphere)
        geometry_to_plot.append(coordinate_frame)
        # TODO need better way to visualise? images? pointclouds? Origin should be different or bigger and point x outward?

    if cam_pcd is not None:
        geometry_to_plot.append(cam_pcd)
    print('Visualising camera origin and aruco frames in camera frame (with first cam_pcd)')
    o3d.visualization.draw_geometries(geometry_to_plot)



def handeye_calibrate_opencv(handeye_data_dict, folder_name, eye_in_hand=True):
    '''
        Modifies handeye_data_dict with saved_cam2arm 
    '''

    # TODO double check all gripper2base and etc
    # all_gripper_rotation_mats = handeye_data_dict['all_gripper_rotation_mats']
    # all_gripper_tvecs = handeye_data_dict['all_gripper_tvecs']
    R_gripper2base = handeye_data_dict['R_gripper2base']
    t_gripper2base = handeye_data_dict['t_gripper2base']
    R_base2gripper = handeye_data_dict['R_base2gripper']
    t_base2gripper = handeye_data_dict['t_base2gripper']
    # all_target2cam_rotation_mats = handeye_data_dict['all_target2cam_rotation_mats']
    # all_target2cam_tvecs = handeye_data_dict['all_target2cam_tvecs']
    R_target2cam = handeye_data_dict['R_target2cam']
    t_target2cam = handeye_data_dict['t_target2cam']
    R_cam2target = handeye_data_dict['R_cam2target']
    t_cam2target = handeye_data_dict['t_cam2target']

    method = cv2.CALIB_HAND_EYE_TSAI  # default
    method = cv2.CALIB_HAND_EYE_DANIILIDIS  # tried both, they both work
    
    # eye-in-hand (according to default opencv2 params and weak documentation. "inputting the suitable transformations to the function" for eye-to-hand)
    # first formula has b_T_c for X so that's what comes out of function. It expects b_T_g and c_T_t so gripper2base and target2cam

    # R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, method=method)
    # R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(R_gripper2base=R_gripper2base, 
    #                                                     t_gripper2base=t_gripper2base, 
    #                                                     R_target2cam=R_target2cam, 
    #                                                     t_target2cam=t_target2cam, method=method)

    # eye-to-hand
    # second formula has b_T_c for X so camera2base actually is what comes out of function. It expects g_T_b (base2gripper) and c_T_t (target2cam)
    # R_camera2base, t_camera2base = cv2.calibrateHandEye(R_base2gripper, t_base2gripper, R_target2cam, t_target2cam)

    # outputs
    # for eye-in-hand, output should be  ->	R_cam2gripper, t_cam2gripper  
    # but for eye to hand: R_cam2base_est, t_cam2base_est from unit tests in opencv


    if eye_in_hand:
        # eye-in-hand
        # TODO outputs below are actually R_cam2gripper, t_cam2gripper
        R_cam2base_est, t_cam2base_est = cv2.calibrateHandEye(R_gripper2base, t_gripper2base,
                                                            #   R_cam2target, t_cam2target, method=method)
                                                              R_target2cam, t_target2cam, method=method)  # this was so much better to above? and also more correct according to transforms
    else:
        # eye-to-hand
        R_cam2base_est, t_cam2base_est = cv2.calibrateHandEye(R_base2gripper, t_base2gripper,
                                                              R_target2cam, t_target2cam, method=method)

    full_homo_RT = np.identity(4)
    full_homo_RT[:3, :3] = R_cam2base_est
    full_homo_RT[:3, 3] = t_cam2base_est.T

    print('Saving handeye (cv2) transform \n{}'.format(full_homo_RT))
    np.savetxt('data/{}/latest_cv2_cam2arm.txt'.format(folder_name), full_homo_RT, delimiter=' ')
    handeye_data_dict['saved_cam2arm'] = full_homo_RT

    # # pos_camera = np.dot(-R_cam2gripper, np.matrix(t_cam2gripper).T)
    # # TODO why does pos_camera seem to have the better position and similarity to my old cam2arms?
    # pos_camera = np.dot(-R_cam2gripper, np.matrix(t_cam2gripper))
    # cam2arm_local = np.identity(4)
    # cam2arm_local[:3, :3] = R_cam2gripper.T
    # # cam2arm_local[:3, :3] = R_cam2gripper
    # cam2arm_local[0, 3] = pos_camera[0]
    # cam2arm_local[1, 3] = pos_camera[1]
    # cam2arm_local[2, 3] = pos_camera[2]
    # # print('cam2arm inverse:\n{}'.format(cam2arm_local))
    # print('handeye (cv2) cam2arm inverse \n{}'.format(cam2arm_local))
    # # print('Saving handeye (cv2) cam2arm inverse \n{}'.format(cam2arm_local))
    # # np.savetxt('data/{}/latest_cv2_cam2arm.txt'.format(folder_name), cam2arm_local, delimiter=' ')


# Eduardo's code from here: https://forum.opencv.org/t/eye-to-hand-calibration/5690/10
# TODO maybe it's nice to just have a param for eye-to-hand like that so I can keep everything pretty similar?
def calibrate_eye_hand(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, eye_to_hand=True):
    # the only that changes is gripper2base vs base2gripper
    if eye_to_hand:
        # change coordinates from gripper2base to base2gripper
        R_base2gripper, t_base2gripper = [], []
        for R, t in zip(R_gripper2base, t_gripper2base):
            R_b2g = R.T
            t_b2g = -R_b2g @ t
            R_base2gripper.append(R_b2g)
            t_base2gripper.append(t_b2g)
        
        # change parameters values
        R_gripper2base = R_base2gripper
        t_gripper2base = t_base2gripper

    # calibrate
    R, t = cv2.calibrateHandEye(
        R_gripper2base=R_gripper2base,
        t_gripper2base=t_gripper2base,
        R_target2cam=R_target2cam,
        t_target2cam=t_target2cam,
    )

    return R, t

if __name__ == '__main__':
    handeye_calibrate_opencv()



'''
Possible things that are wrong:
- base2gripper vs gripper2base
- something wrong in get_gripper_base_transformation()? but gripper tip transform looks good... what if it looks good but rotation convention is wrong?
- something wrong with aruco detection or transform? aruco range, maybe I should have board? So less noise? aruco rotation shouldn't matter
- something wrong with hand-eye calculation? What if it expects lists or joined vectors or NxM array?
- try different handeye methods, try more data (they say you need 3+)
- normal typo bug somewhere? I've been triple checking
- this is non-linear optimisation and no ability to give initial guess?
- dorna negative joint angles, but gripper tip transform looks good...
- intrinsics, but it should at least give reasonable results with bad intrinsics like we got before with aruco transforms?
- arm in mm vs metres. Does distance affect rotation? No it does not. That is, wrist pitch, roll and base_yaw are the same if the arm's joints are a million metres long

Unlikely/confirmed:
- target2cam vs cam2target

What I can do about it
- since camera is static, visualise all marker poses!!!!!!!!!!!!!!!!!! better than what I did.
- How to visualise and confirm all gripper poses? If everything is correct the marker poses could be ICP'd to the gripper poses but that's the whole point right?
- Just try inverting gripper2base to see what happens
- compare with id1 aruco strategy and see what part of translation and more specifically rotation is wrong!!!!
- how to visualise and double check transformations better. I don't understand the opencv spatial algebra direction. 
- visualise arm coordinate origin and camera coordinate zero? I already am doing arm
- confirm target2cam and base2gripper separately
- use rvec instead of rotation matrix?
- read internet:
    - https://stackoverflow.com/questions/57008332/opencv-wrong-result-in-calibratehandeye-function
    - https://www.reddit.com/r/opencv/comments/n46qaf/question_hand_eye_calibration_bad_results/
    - https://code.ihub.org.cn/projects/729/repository/revisions/master/entry/modules/calib3d/test/test_calibration_hand_eye.cpp
    - original PR https://github.com/opencv/opencv/pull/13880 and matlab code copied http://lazax.com/www.cs.columbia.edu/~laza/html/Stewart/matlab/handEye.m
    - http://campar.in.tum.de/Chair/HandEyeCalibration
    - https://forum.opencv.org/t/hand-eye-calibration/1880
    - very good https://visp-doc.inria.fr/doxygen/visp-daily/tutorial-calibration-extrinsic.html TODO find  hand_eye_calibration_show_extrinsics.py
    - https://programming.vip/docs/5ee2e219e75f3.html
    - https://github.com/IFL-CAMP/easy_handeye/blob/master/docs/troubleshooting.md
    - https://forum.opencv.org/t/eye-to-hand-calibration/5690/2 this is good. Rotation matrices are dangerous but not ambiguous
    - totally different way: https://www.codetd.com/en/article/12950073
    - simulator thing https://pythonrepo.com/repo/caijunhao-calibration-python-computer-vision
    - https://blog.zivid.com/the-practical-guide-to-3d-hand-eye-calibration-with-zivid-one
- other packages. read how other packages do it
    - try UR5 calibration thing (why does it need offset?)
    - Try handical
    - try easy handeye and just publish all frames (base_link doesn't move, ee_link can just be 6 or 7 vector transform from keyboard_control.py, /optical_base_frame will probably need realsense-ros, /optical_target will need aruco shit): https://github.com/IFL-CAMP/easy_handeye
    - https://github.com/crigroup/handeye/blob/master/src/handeye/calibrator.py
'''

# TODO might need base2gripper, inverse of above actually
# TODO to how many degrees are my joint angles actually close to 0? what if 0 rounding so it is 0.001 but im doing numpy suppress
# TODO target2cam or cam2target. Ahh opencv param names according to eye-in-hand vs eye-to-hand might change
# TODO arm2cam or cam2arm? should get to the bottom of this forever. camera coordinate in arm coordinates and the transform is the same?
# TODO save pic or not? Save reprojection error or ambiguity or something?
# TODO eventually put realsense in hand as well and do eye-in-hand. And multiple realsenses (maybe swap to handical or other? or do each one individually?)
# TODO maybe just measure with a ruler and see what happens with the transform and stuff and see how it works

