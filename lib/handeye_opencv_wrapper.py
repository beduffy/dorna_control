from __future__ import print_function
from glob import glob
import sys

import numpy as np
import cv2
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

# TODO should i separate the below stuff or not? to different files or functions?
from lib.vision import calculate_reprojection_error, get_inverse_homogenous_transform
from lib.vision_config import camera_matrix, dist_coeffs
from lib.aruco_helper import create_aruco_params, aruco_detect_draw_get_transforms, calculate_pnp_12_markers, find_aruco_markers
from lib.aruco_image_text import OpenCvArucoImageText
from lib.handeye_plotting import plot_every_cam_pcd_and_aruco_marker

# TODO I used to calculate inverse cam2arm but transformation between two transforms should be the same?
# TODO what if gripper pose x isn't suppose to be forward? Would that change anything?
# TODO another massive source of innacuracy is my kinematic calibration............. what if imade the toolhead to be 0? still rotation is a problem. less translational error though in pitch but not wrist roll
# TODO email dorna people on why this (the above todo) happens
# TODO read all handeye low level code and the papers accompanying them: https://github.com/opencv/opencv/blob/4.x/modules/calib3d/src/calibration_handeye.cpp


def load_all_handeye_data(folder_name):
    # Open data/handeye folder and then load all data into handeye_data_dict
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

        gripper2_base_tvec = gripper_transform[:3, 3]
        all_gripper_tvecs.append(gripper2_base_tvec)

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

    # creating further key-value pairs
    all_base2gripper_transforms = []
    for R, t in zip(R_base2gripper, t_base2gripper):
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

    handeye_data_dict = {
        # 'all_gripper_rotation_mats': all_gripper_rotation_mats,
        # 'all_gripper_tvecs': all_gripper_tvecs,
        'R_gripper2base': R_gripper2base,
        't_gripper2base': t_gripper2base,
        'R_base2gripper': R_base2gripper,
        't_base2gripper': t_base2gripper,
        # 'all_target2cam_rotation_mats': all_target2cam_rotation_mats,
        # 'all_target2cam_tvecs': all_target2cam_tvecs,
        'R_target2cam': R_target2cam,
        't_target2cam': t_target2cam,
        'R_cam2target': R_cam2target,
        't_cam2target': t_cam2target,
        'all_gripper2base_transforms': all_gripper2base_transforms,
        'all_base2gripper_transforms': all_base2gripper_transforms,
        'all_target2cam_transforms': all_target2cam_transforms,
        'all_cam2target_transforms': all_cam2target_transforms,
        'color_images': color_images,
        'depth_images': depth_images,
        'all_joint_angles': all_joint_angles
    }

    return handeye_data_dict


def test_transformations(handeye_data_dict):
    '''
        Tests:
        - R_gripper2base and t_gripper2base: The rotation and translation of the gripper (end-effector) in the base frame, i.e., the transformation from the base to the gripper.
        - R_target2cam and t_target2cam: The rotation and translation of the calibration target (e.g., ArUco marker) in the camera frame, i.e., the transformation from the camera to the target.
        - loop through all gripper2base and base2gripper and confirm the transfrom has its inverse correctly set by checking if output is identity
        - same for cam2target
        - camera calibration intrinsics and reconstruction error before and after
        - recalculate aruco pipeline with undistorted images
        - visualise some images
    '''

    R_gripper2base = handeye_data_dict['R_gripper2base']
    t_gripper2base = handeye_data_dict['t_gripper2base']
    R_base2gripper = handeye_data_dict['R_base2gripper']
    t_base2gripper = handeye_data_dict['t_base2gripper']
    R_target2cam = handeye_data_dict['R_target2cam']
    t_target2cam = handeye_data_dict['t_target2cam']
    R_cam2target = handeye_data_dict['R_cam2target']
    t_cam2target = handeye_data_dict['t_cam2target']
    all_gripper2base_transforms = handeye_data_dict['all_gripper2base_transforms']
    all_base2gripper_transforms = handeye_data_dict['all_base2gripper_transforms']
    all_target2cam_transforms = handeye_data_dict['all_target2cam_transforms']
    all_cam2target_transforms = handeye_data_dict['all_cam2target_transforms']

    # verify base2gripper and gripper2base mean what I think they mean
    # gripper2base should bring base to gripper e.g. 0, 0, 0 to usual 0.4 metres in front
    # TODO not sure how to test but it does that since visualisation uses all_gripper2base_transforms
    # same with target2cam: transformation from the camera to the target. 0,0,0 to aruco target
    # TODO same, but visualisation works with target2cam
    first_gripper2base_transform = all_gripper2base_transforms[0]
    first_base2gripper_transform = all_base2gripper_transforms[0]
    print('t_base2gripper[0] (since x forwards should show more positive t): {}'.format(t_base2gripper[0]))
    print('t_gripper2base[0] (since x forwards should show more negative t to get back from gripper): {}'.format(t_gripper2base[0]))
    # I thought language of bla2foo would bring points in bla to foo, but i've been bitten before e.g. this says the opposite, oh wait it doesn't:
    # R_gripper2base	Rotation part extracted from the homogeneous matrix that transforms a point expressed in the gripper frame to the robot base frame
    # assert(first_base2gripper_transform @ np.array([0.0, 0.0, 0.0, 1.0]))
    # assert(first_gripper2base_transform @ np.array([0.0, 0.0, 0.0, 1.0]))
    # TODO first_base2gripper_transform is showing negative 4th column translation, is it inverted? seems so

    # TODO verifying target2cam vs cam2target might be easy since z is always forward in one: probably cam2target
    # TODO make sure RGB and not BGR?

    # Verify input transforms - add debug prints
    # print("Sample gripper2base transform:\n", np.vstack((R_gripper2base[0], t_gripper2base[0].reshape(1,3))))
    # print("Sample target2cam transform:\n", np.vstack((R_target2cam[0], t_target2cam[0].reshape(1,3))))

    # Verify units are in meters
    # gripper2_base_tvec = t_gripper2base[:3, 3]
    print('t_gripper2base[0]: ', t_gripper2base[0])
    if np.any(abs(t_gripper2base[0]) > 10):  # Assuming arm can't be >10m
        print("Warning: Large translation values detected. Check units.")
    else:
        print('gripper transforms seem to be in metres since no large units above 10 (which would indicate mms)')

    check_pose_distribution(R_gripper2base, t_gripper2base)

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

    color_images = handeye_data_dict['color_images']
    depth_images = handeye_data_dict['depth_images']

    # camera intrinsic
    # mtx, dist, mean_error, errors = calibrate_camera_intrinsics(color_images, camera_matrix, dist_coeffs)
    # # mtx, dist, mean_error, errors = calibrate_camera_intrinsics(color_images[:5])
    # # TODO print better so i can copy to vision_config
    # print('Camera intrinsics: \n camera intrinsics: {}\ndist_coeffs: {}\n mean_error: {}'.format(mtx, dist, mean_error))

    # check 1 image aruco visualisation and errors testing
    opencv_aruco_image_text = OpenCvArucoImageText()
    board, parameters, aruco_dict, marker_length = create_aruco_params()
    marker_separation = 0.0065
    ids, corners, all_rvec, all_tvec = None, None, None, None  # TODO global remove
    id_on_shoulder_motor = 1

    first_color_image = color_images[0]
    camera_color_img_debug = first_color_image.copy()
    color_img, tvec, rvec, ids, corners, all_rvec, all_tvec = find_aruco_markers(first_color_image, aruco_dict, parameters, marker_length, id_on_shoulder_motor, opencv_aruco_image_text, camera_color_img_debug)
    cam2target_opt, target2cam_opt, tvec_pnp_opt, rvec_pnp_opt, input_obj_points_concat, input_img_points_concat = calculate_pnp_12_markers(corners, ids, all_rvec, all_tvec, marker_length=marker_length, marker_separation=marker_separation)

    # plot_every_cam_pcd_and_aruco_marker(color_images, depth_images, handeye_data_dict['all_target2cam_transforms'])

    # cv2.imshow('Camera Color Image Debug', camera_color_img_debug)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # TODO draw all aruco axes to find instabilities, use different library or? but before that just use open3d
    # TODO does aruco text below have instabilites from 1 marker or all 12?
    # TODO loop through all pointclouds and aruco transform and then comment function, just a sanity check
    # TODO just realised first cam_pcd in october 30 dataset does not have aruco on board!!! 

    # # Undistort the first image using camera matrix and distortion coefficients
    # h, w = first_color_image.shape[:2]
    # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 1, (w,h))
    # undistorted_img = cv2.undistort(first_color_image, camera_matrix, dist_coeffs, None, newcameramtx)
    # # Crop the image based on ROI
    # x, y, w, h = roi
    # undistorted_img = undistorted_img[y:y+h, x:x+w]
    # # Display original and undistorted images side by side
    # # comparison = np.hstack((first_color_image, undistorted_img))
    # # cv2.imshow('Original vs Undistorted', comparison)
    # # cv2.imshow('Original vs Undistorted', undistorted_img)
    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows()


    # print('Before undistortion, first cam_pcd and aruco marker')
    # plot_every_cam_pcd_and_aruco_marker(color_images[:1], depth_images[:1], handeye_data_dict['all_target2cam_transforms'])
    # recalculate_transforms_with_undistorted_images(color_images, handeye_data_dict)
    # print('After undistortion and recalculating all images, first cam_pcd and aruco marker')

    # # TODO why does undistortion show markers in wrong places?!?!? 
    # plot_every_cam_pcd_and_aruco_marker(color_images[:1], depth_images[:1], handeye_data_dict['all_target2cam_transforms'])

    # Show all, too slow
    # plot_every_cam_pcd_and_aruco_marker(color_images, depth_images, handeye_data_dict['all_target2cam_transforms'])


def recalculate_transforms_with_undistorted_images(color_images, handeye_data_dict):
    # TODO could recalculate all opencv transforms, and with undistortion TODO make sure RGB and not BGR?
    # store back into handeye_data_dict. recalculate all_target2cam_transforms, all_cam2target_transforms
    # Recalculate transforms using undistorted images
    opencv_aruco_image_text = OpenCvArucoImageText()
    board, parameters, aruco_dict, marker_length = create_aruco_params()
    marker_separation = 0.0065
    id_on_shoulder_motor = 1

    new_target2cam_transforms = []
    new_cam2target_transforms = []
    new_target2cam_rotation_mats = []
    new_target2cam_tvecs = []
    new_cam2target_rotation_mats = []
    new_cam2target_tvecs = []

    # img = color_images[0]
    # h, w = img.shape[:2]
    # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 1, (w,h))
    # # Adjust principal point in the new camera matrix after cropping
    # newcameramtx[0, 2] -= x  # cx adjustment
    # newcameramtx[1, 2] -= y  # cy adjustment
    # # Crop the image based on ROI
    # x, y, w, h = roi

    for img in color_images:
        # Undistort image
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 1, (w,h))
        # Crop the image based on ROI
        x, y, w, h = roi
        # Adjust principal point in the new camera matrix after cropping
        newcameramtx[0, 2] -= x  # cx adjustment
        newcameramtx[1, 2] -= y  # cy adjustment
        # print(newcameramtx)
        
        undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs, None, newcameramtx)
        
        # TODO does cropping ruin focal intrinsics?
        # TODO change principal point
        undistorted_img = undistorted_img[y:y+h, x:x+w]

        # Find ArUco markers in undistorted image
        camera_color_img_debug = undistorted_img.copy()
        _, tvec, rvec, ids, corners, all_rvec, all_tvec = find_aruco_markers(
            undistorted_img, aruco_dict, parameters, marker_length, 
            id_on_shoulder_motor, opencv_aruco_image_text, camera_color_img_debug, newcameramtx
        )
        
        # TODO ahh running through images there are images that don't move which explains 0 gripper euclidean dist? need to add throttle to a key?
        # TODO also seems like it finds it harder to find some markers? that was before using newcameramtx but check again!
        # cv2.imshow('Camera Color Image Debug undistort', camera_color_img_debug)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # TODO very strange bug in here that only appeared recently. id1 missing
        try:
            # Calculate transforms using undistorted image
            cam2arm_opt, arm2cam_opt, tvec_pnp_opt, rvec_pnp_opt, _, _ = calculate_pnp_12_markers(
                corners, ids, all_rvec, all_tvec, marker_length=marker_length, 
                marker_separation=marker_separation
            )
        except Exception as e:
            cv2.imshow('ERROR', undistorted_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            cv2.imshow('undistort error aruco', camera_color_img_debug)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        target2cam_transform = arm2cam_opt  # TODO same thing i did in handeye, pretty similar result to below

        # Convert to rotation matrix
        # target2cam_rot, _ = cv2.Rodrigues(rvec_pnp_opt)
        target2cam_rot = target2cam_transform[:3, :3]
        
        # # Create transforms
        # target2cam_transform = np.eye(4)
        # target2cam_transform[:3, :3] = target2cam_rot
        # target2cam_transform[:3, 3] = tvec_pnp_opt.reshape(3)
        
        cam2target_transform = get_inverse_homogenous_transform(target2cam_transform)
        
        # Store new transforms and components
        new_target2cam_transforms.append(target2cam_transform)
        new_cam2target_transforms.append(cam2target_transform)
        new_target2cam_rotation_mats.append(target2cam_rot)
        new_target2cam_tvecs.append(tvec_pnp_opt.reshape(3))
        new_cam2target_rotation_mats.append(cam2target_transform[:3, :3])
        new_cam2target_tvecs.append(cam2target_transform[:3, 3])

    # TODO double check we have all the important ones
    # Update handeye_data_dict with new transforms
    handeye_data_dict.update({
        'all_target2cam_transforms': new_target2cam_transforms,
        'all_cam2target_transforms': new_cam2target_transforms,
        'R_target2cam': np.array(new_target2cam_rotation_mats),
        't_target2cam': np.array(new_target2cam_tvecs),
        'R_cam2target': np.array(new_cam2target_rotation_mats),
        't_cam2target': np.array(new_cam2target_tvecs)
    })


def verify_calibration(handeye_data_dict, R_cam2gripper, t_cam2gripper):
    """Verify calibration quality using AX=XB equation
        The Equation AX = XB means:
        gripper2base * cam2gripper = cam2gripper * target2cam

        Going from target to base can be done two equivalent ways:
        1. target -> camera -> gripper -> base
        2. target -> camera -> gripper -> base TODO this is wrong

        # TODO could I build my own optimisation, beginning with ruler, then do translation and then rotation
    """

    print('Verifying calibration with AX = XB')
    all_rotation_errors = []
    all_translation_errors = []
    ## TODO less printing
    for i in range(len(handeye_data_dict['R_gripper2base'])):
        # Build transforms
        X = np.eye(4)
        X[:3,:3] = R_cam2gripper
        X[:3,3] = t_cam2gripper.ravel()
        
        A = np.eye(4)
        # A[:3,:3] = handeye_data_dict['R_gripper2base'][i]
        # A[:3,3] = handeye_data_dict['t_gripper2base'][i]
        A[:3,:3] = handeye_data_dict['R_base2gripper'][i]  # TODO since inverted. just fix it at the core
        A[:3,3] = handeye_data_dict['t_base2gripper'][i]
        
        B = np.eye(4)
        B[:3,:3] = handeye_data_dict['R_target2cam'][i]
        B[:3,3] = handeye_data_dict['t_target2cam'][i]
        
        # AX should equal XB
        AX = A @ X
        XB = X @ B

        # TODO why norm? difference between AX vs XB TODO understand entire function more since it is in essense what we are optimising for
        matrix_subtract = AX - XB
        # print(matrix_subtract)
        full_error = np.linalg.norm(matrix_subtract)
        # You might want to separate rotation and translation errors
        rotation_error = np.linalg.norm(matrix_subtract[:3, :3])
        translation_error = np.linalg.norm(matrix_subtract[:3, 3])

        all_rotation_errors.append(rotation_error)
        all_translation_errors.append(translation_error)
        # TODO sum all errors and optimise myself
        
        # print(f"Transform pair {i} subtract:\n {matrix_subtract}")
        # print(f"Transform pair {i} error: {full_error:.3f}")
        # print(f"Transform pair {i} rotation_error: {rotation_error:.3f}")  # TODO in radians right or in rotational matrix values?
        # print(f"Transform pair {i} translation_error: {translation_error:.3f}")

    print(f"Rotation error. Min: {min(all_rotation_errors):.3f} Max: {max(all_rotation_errors):.3f} Avg: {sum(all_rotation_errors) / len(all_rotation_errors):.3f}")
    print(f"Translation error. Min {min(all_translation_errors):.3f} Max: {max(all_translation_errors):.3f}. Avg: {sum(all_translation_errors) / len(all_translation_errors):.3f}")



def optimize_cam2gripper_transform(handeye_data_dict, R_cam2gripper_manual, t_cam2gripper_manual):
    def error_function(params):
        # Extract rotation and translation from params
        # rx, ry, rz, tx, ty, tz = params
        # R = Rotation.from_euler('xyz', [rx, ry, rz]).as_matrix()
        # First 9 params are rotation matrix elements (row by row)
        R = np.array(params[:9]).reshape(3,3)
        # Last 3 params are translation
        t = np.array(params[9:])
        
        # Build transform
        X = np.eye(4)
        X[:3, :3] = R
        X[:3, 3] = t
        
        total_error = 0
        for i in range(len(handeye_data_dict['R_gripper2base'])):
            # A = build_transform(handeye_data_dict['R_gripper2base'][i], 
            #                   handeye_data_dict['t_gripper2base'][i])
            # B = build_transform(handeye_data_dict['R_target2cam'][i], 
            #                   handeye_data_dict['t_target2cam'][i])
            
            A = np.eye(4)
            # A[:3,:3] = handeye_data_dict['R_gripper2base'][i]
            # A[:3,3] = handeye_data_dict['t_gripper2base'][i]
            A[:3,:3] = handeye_data_dict['R_base2gripper'][i]  # TODO since inverted. just fix it at the core
            A[:3,3] = handeye_data_dict['t_base2gripper'][i]
            
            B = np.eye(4)
            B[:3,:3] = handeye_data_dict['R_target2cam'][i]
            B[:3,3] = handeye_data_dict['t_target2cam'][i]
            
            # Calculate AX-XB error
            matrix_diff = A @ X - X @ B
            
            # Separate rotation and translation errors
            rotation_error = np.linalg.norm(matrix_diff[:3, :3])
            translation_error = np.linalg.norm(matrix_diff[:3, 3])
            
            total_error += rotation_error + translation_error
            
        # Add penalty for non-orthogonal rotation matrix
        orthogonality_error = np.linalg.norm(R @ R.T - np.eye(3))
        det_error = abs(np.linalg.det(R) - 1.0)
        
        return total_error + 10.0 * (orthogonality_error + det_error)  # Weight the orthogonality constraint
    
    # Initial params: flatten R and concatenate with t
    initial_params = np.concatenate([R_cam2gripper_manual.flatten(), t_cam2gripper_manual])
    
    # Add bounds to keep the optimization reasonable
    # bounds = [(-1, 1)] * 9 + [(-0.2, 0.2)] * 3  # rotation matrix elements between -1,1, translations ±20cm
    bounds = [(-1, 1)] * 9 + [(-0.1, 0.1)] * 3  # rotation matrix elements between -1,1, translations ±20cm
    
    # Optimize
    # result = minimize(error_function, initial_params, 
    #                  method='SLSQP',  # Changed to SLSQP to handle bounds
    #                  bounds=bounds)
    
    result = minimize(error_function, initial_params, 
                     method='Nelder-Mead')
    
    if not result.success:
        print("Warning: Optimization did not converge!")
        print("Message:", result.message)
    
    # Extract and return R and t separately
    R_optimized = result.x[:9].reshape(3,3)
    t_optimized = result.x[9:]
    
    # Project rotation matrix to closest orthogonal matrix
    U, _, Vh = np.linalg.svd(R_optimized)
    R_optimized = U @ Vh
    
    return R_optimized, t_optimized


def optimize_cam2gripper_transform_individual(handeye_data_dict, R_cam2gripper_manual, t_cam2gripper_manual):
    def translation_error_function(t_params):
        # Use fixed rotation
        R = R_cam2gripper_manual
        t = np.array(t_params)
        
        # Build transform
        X = np.eye(4)
        X[:3, :3] = R
        X[:3, 3] = t
        
        total_error = 0
        for i in range(len(handeye_data_dict['R_gripper2base'])):
            A = np.eye(4)
            A[:3, :3] = handeye_data_dict['R_base2gripper'][i]
            A[:3, 3] = handeye_data_dict['t_base2gripper'][i]
            
            B = np.eye(4)
            B[:3, :3] = handeye_data_dict['R_target2cam'][i]
            B[:3, 3] = handeye_data_dict['t_target2cam'][i]
            
            # Calculate AX-XB error
            matrix_diff = A @ X - X @ B
            translation_error = np.linalg.norm(matrix_diff[:3, 3])
            total_error += translation_error
        
        return total_error
    
    def rotation_error_function(r_params):
        # Use fixed translation
        R = np.array(r_params).reshape(3, 3)
        t = t_cam2gripper_manual
        
        # Build transform
        X = np.eye(4)
        X[:3, :3] = R
        X[:3, 3] = t
        
        total_error = 0
        for i in range(len(handeye_data_dict['R_gripper2base'])):
            A = np.eye(4)
            A[:3, :3] = handeye_data_dict['R_base2gripper'][i]
            A[:3, 3] = handeye_data_dict['t_base2gripper'][i]
            
            B = np.eye(4)
            B[:3, :3] = handeye_data_dict['R_target2cam'][i]
            B[:3, 3] = handeye_data_dict['t_target2cam'][i]
            
            # Calculate AX-XB error
            matrix_diff = A @ X - X @ B
            rotation_error = np.linalg.norm(matrix_diff[:3, :3])
            total_error += rotation_error
        
        # Add penalty for non-orthogonal rotation matrix
        orthogonality_error = np.linalg.norm(R @ R.T - np.eye(3))
        det_error = abs(np.linalg.det(R) - 1.0)
        
        return total_error + 10.0 * (orthogonality_error + det_error)
    
    # Optimize translation first
    translation_bounds = [(-0.1, 0.1)] * 3  # Adjust bounds based on expected translation
    translation_result = minimize(translation_error_function, t_cam2gripper_manual, method='SLSQP', bounds=translation_bounds)
    t_optimized = translation_result.x
    
    # Optimize rotation next
    initial_rotation_params = R_cam2gripper_manual.flatten()
    rotation_bounds = [(-1, 1)] * 9
    rotation_result = minimize(rotation_error_function, initial_rotation_params, method='SLSQP', bounds=rotation_bounds)
    R_optimized = rotation_result.x.reshape(3, 3)
    
    # Project rotation matrix to closest orthogonal matrix
    U, _, Vh = np.linalg.svd(R_optimized)
    R_optimized = U @ Vh
    
    return R_optimized, t_optimized


def verify_transform_chain(handeye_data_dict, saved_cam2gripper):
    """Verify each transform in the calibration chain makes physical sense."""
    
    # 1. Verify gripper2base transforms
    R_cam2target = handeye_data_dict['R_cam2target']
    t_cam2target = handeye_data_dict['t_cam2target']
    R_base2gripper = handeye_data_dict['R_base2gripper']
    t_base2gripper = handeye_data_dict['t_base2gripper']
    R_gripper2base = handeye_data_dict['R_gripper2base']
    t_gripper2base = handeye_data_dict['t_gripper2base']
    
    print("\n=== Gripper to Base Transform Verification ===")
    print("First gripper position in base frame:", t_gripper2base[0])
    print("Expected: Roughly 0.3-0.4m in x forward, minimal y/z if centered")
    
    # TODO fix below
    # Check if transforms are in expected ranges
    max_expected_gripper_distance = 0.5  # 50cm - adjust based on your robot
    if np.any(np.abs(t_gripper2base) > max_expected_gripper_distance):
        print("WARNING: Gripper positions seem too far from base!")
    
    # 2. Verify target2cam transforms
    R_target2cam = handeye_data_dict['R_target2cam']
    t_target2cam = handeye_data_dict['t_target2cam']
    
    print("\n=== Target to Camera Transform Verification ===")
    print("First marker position in camera frame:", t_target2cam[0])
    print("Expected: Z should be positive (marker in front of camera)")
    print("Expected: X,Y should be centered if marker is centered in image")

    # when arm moves, pointcloud should move with it. gripper transformation from pose 1 to 2 is same as camera transformation from pose 1 to 2
    # first just check transformation
    # TODO should still understand correct and wrong choices below 
    # print('\nt_gripper2base[0] and [1]') 
    # print(t_gripper2base[0], t_gripper2base[1])
    # distance_gripper = np.linalg.norm(t_gripper2base[0] - t_gripper2base[1])
    # print('euclidean distance between both gripper positions: {:.3f}'.format(distance_gripper))

    # print('\nt_base2gripper[0] and [1]') 
    # print(t_base2gripper[0], t_base2gripper[1])
    # distance_gripper = np.linalg.norm(t_base2gripper[0] - t_base2gripper[1])  # TODO euclidean distance is 0?!?!?!
    # print('euclidean distance between both gripper positions: {:.3f}'.format(distance_gripper))

    # print('\nt_base2gripper[1] and [2]') 
    # print(t_base2gripper[1], t_base2gripper[2])
    # distance_gripper = np.linalg.norm(t_base2gripper[1] - t_base2gripper[2])
    # print('euclidean distance between both gripper positions: {:.3f}'.format(distance_gripper))

    # print('\nt_target2cam[0] and [1]')  # TODO is it cam2target? the camera is moving but the target isn't
    # print(t_target2cam[0], t_target2cam[1])
    # distance_target2cam = np.linalg.norm(t_target2cam[0] - t_target2cam[1])
    # print('euclidean distance between both target2cam positions: {:.3f}'.format(distance_target2cam))

    # print('\nt_cam2target[0] and [1]')  # TODO is it cam2target? the camera is moving but the target isn't
    # print(t_cam2target[0], t_cam2target[1])
    # distance_cam2target = np.linalg.norm(t_cam2target[0] - t_cam2target[1])
    # print('euclidean distance between both cam2target positions: {:.3f}'.format(distance_cam2target))


    # print('\nt_gripper2base[1] and [2]') 
    # print(t_gripper2base[1], t_gripper2base[2])
    # distance_gripper = np.linalg.norm(t_gripper2base[1] - t_gripper2base[2])
    # print('euclidean distance between both gripper positions: {:.3f}'.format(distance_gripper))

    # print('\nt_cam2target[1] and [2]')  # TODO is it cam2target? the camera is moving but the target isn't
    # print(t_cam2target[1], t_cam2target[2])
    # distance_cam2target = np.linalg.norm(t_cam2target[1] - t_cam2target[2])
    # print('euclidean distance between both cam2target positions: {:.3f}'.format(distance_cam2target))

    # nice the below outputs:
    '''
    From 0 to 1: gripper dist: 0.060. euclidean_dist: 0.051
    From 1 to 2: gripper dist: 0.175. euclidean_dist: 0.208
    From 2 to 3: gripper dist: 0.035. euclidean_dist: 0.035
    From 3 to 4: gripper dist: 0.037. euclidean_dist: 0.041
    From 4 to 5: gripper dist: 0.099. euclidean_dist: 0.090
    From 5 to 6: gripper dist: 0.035. euclidean_dist: 0.036
    From 6 to 7: gripper dist: 0.034. euclidean_dist: 0.011
    '''
    # TODO of course the realsense is at an angle so wrist roll/pitch angles change the distance but never too far
    # TODO deeply understand why cam2target and not target2cam
    # e.g. when arm moves, pointcloud should move with it. gripper transformation from pose 1 to 2 is similar enough to camera transformation (camera can rotate with wrist pitch/roll)
    # TODO but first first few angles I did not rotate/touch wrist. There's of course aruco error too 
    # TODO From 16 to 17: gripper dist: 0.000. camera dist: 0.028
    # From 17 to 18: gripper dist: 0.000. camera dist: 0.041
    # print('\nConsecutive euclidean distances for gripper2base and cam2target')
    # for idx in range(t_gripper2base.shape[0] - 1):
    #     distance_gripper = np.linalg.norm(t_gripper2base[idx] - t_gripper2base[idx + 1])
    #     distance_cam2target = np.linalg.norm(t_cam2target[idx] - t_cam2target[idx + 1])
    #     print('From {} to {}: gripper dist: {:.3f}. camera dist: {:.3f}'.format(idx, idx + 1, distance_gripper, distance_cam2target))
    
    # # TODO also the meaning of gripper2base is strange since base is fixed
    # # TODO this sometimes returns bigger differences, sometimes very close and then sometimes 0 distance, how is that possible?
    # '''
    # From 12 to 13: gripper dist: 0.198. camera dist: 0.315
    # From 13 to 14: gripper dist: 0.035. camera dist: 0.034
    # From 14 to 15: gripper dist: 0.035. camera dist: 0.035
    # From 15 to 16: gripper dist: 0.035. camera dist: 0.038
    # From 16 to 17: gripper dist: 0.267. camera dist: 0.063
    # From 17 to 18: gripper dist: 0.516. camera dist: 0.116
    # From 18 to 19: gripper dist: 0.000. camera dist: 0.191
    # '''
    # print('\nConsecutive euclidean distances for t_base2gripper and target2cam')
    # for idx in range(t_base2gripper.shape[0] - 1):
    #     distance_gripper = np.linalg.norm(t_base2gripper[idx] - t_base2gripper[idx + 1])
    #     distance_target2cam = np.linalg.norm(t_target2cam[idx] - t_target2cam[idx + 1])
    #     print('From {} to {}: gripper dist: {:.3f}. camera dist: {:.3f}'.format(idx, idx + 1, distance_gripper, distance_target2cam))

    # Check if marker is in front of camera
    if np.any(t_target2cam[:, 2] < 0):
        # TODO deeply understand why its target to cam here
        msg = "WARNING: Some markers appear behind camera! Check target2cam transforms"
        print(msg)
        sys.exit(msg)
    
    # 3. Verify final cam2gripper transform
    print("\n=== Camera to Gripper Transform Verification ===")
    R_cam2gripper = saved_cam2gripper[:3, :3]
    t_cam2gripper = saved_cam2gripper[:3, 3]
    
    print("Final camera position in gripper frame:", t_cam2gripper)
    print("Expected: Should match physical mounting distances (~10cm offsets)")
    
    # Check if transform seems reasonable
    max_expected_camera_offset = 0.2  # 20cm - adjust based on your setup
    if np.any(np.abs(t_cam2gripper) > max_expected_camera_offset):
        print("WARNING: Camera offset from gripper seems too large!")
    
    # 4. Verify rotation matrices are valid
    def verify_rotation_matrix(R, name):
        # Check orthogonality
        # if not np.allclose(np.dot(R, R.T), np.eye(3), atol=1e-6):
        if not np.allclose(np.dot(R, R.T), np.eye(3), atol=1e-1):
            print(f"WARNING: {name} is not orthogonal!")
            print('name: {}. R: {}'.format(name, R))
            print('np.dot(R, R.T): ', np.dot(R, R.T))
            msg = 'EXITED ROTATION orthogonality failed'
            print(msg)
            sys.exit(msg)
        # Check determinant
        if not np.allclose(np.linalg.det(R), 1.0, atol=1e-6):
            print(f"WARNING: {name} determinant is not 1!")
            print('name: {}. R: {}'.format(name, R))
            print('np.linalg.det(R): {:.7f}'.format(np.linalg.det(R)))
            # import pdb;pdb.set_trace()
            sys.exit('EXITED ROTATION determinant failed')

    
    verify_rotation_matrix(R_gripper2base[0], "First gripper2base rotation")
    verify_rotation_matrix(R_target2cam[0], "First target2cam rotation")
    verify_rotation_matrix(R_cam2gripper, "Final cam2gripper rotation")
    
    # TODO bring other function call here?
    # 5. Verify AX=XB equation
    # print("\n=== AX=XB Equation Verification ===")
    # for i in range(len(R_gripper2base)):
    #     # Build transforms
    #     A = np.eye(4)
    #     A[:3, :3] = R_gripper2base[i]
    #     A[:3, 3] = t_gripper2base[i]
        
    #     X = saved_cam2gripper
        
    #     B = np.eye(4)
    #     B[:3, :3] = R_target2cam[i]
    #     B[:3, 3] = t_target2cam[i]
        
    #     # Compare AX and XB
    #     AX = np.dot(A, X)
    #     XB = np.dot(X, B)
    #     error = np.linalg.norm(AX - XB)
    #     if error > 0.1:  # adjust threshold as needed
    #         print(f"Large AX=XB error at pose {i}: {error}")
    
    return True


def check_pose_distribution(R_gripper2base, t_gripper2base):
    """Check if calibration poses are well-distributed"""
    # Convert rotations to euler angles
    euler_angles = [cv2.Rodrigues(R)[0].ravel() for R in R_gripper2base]
    euler_angles = np.array(euler_angles)
    
    # Check rotation variation
    angle_range = np.ptp(euler_angles, axis=0)
    print("Rotation ranges (rad):", angle_range)
    
    # Check translation variation
    trans_range = np.ptp(t_gripper2base, axis=0)
    print("Translation ranges (m):", trans_range)
    
    min_angle_range = 0.5
    min_translation_range = 0.1
    if np.any(angle_range < min_angle_range):  # Less than ~30 degrees
        # TODO Rotation ranges (rad): [1.41097524 0.34964032 0.46533604]. TWO WERE BElow 0.5, hmm
        print("Warning: Limited rotation variation, less than: {}".format(min_angle_range))
    if np.any(trans_range < min_translation_range):  # Less than 10cm
        print("Warning: Limited translation variation, less than: {}".format(min_translation_range))


def calibrate_camera_intrinsics(images, camera_matrix, dist_coeffs):
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
    
    before_calibration_tvecs = []
    before_calibration_rvecs = []

    for idx, img in enumerate(images):
        # print('Running aruco + solvepnp on image no. ', idx)
        # Use existing ArUco detection pipeline
        camera_color_img_debug = img.copy()
        color_img, tvec, rvec, ids, corners, all_rvec, all_tvec = find_aruco_markers(
            img, aruco_dict, parameters, marker_length, 
            id_on_shoulder_motor=1, 
            opencv_aruco_image_text=opencv_aruco_image_text,
            camera_color_img_debug=camera_color_img_debug
        )
        
        cam2arm_opt, arm2cam_opt, tvec_pnp_opt, rvec_pnp_opt, input_obj_points_concat, input_img_points_concat = calculate_pnp_12_markers(corners, ids, all_rvec, all_tvec, marker_length=marker_length, marker_separation=marker_separation)

        before_calibration_tvecs.append(tvec_pnp_opt)
        before_calibration_rvecs.append(rvec_pnp_opt)

        # Reshape and convert types to match expected format
        obj_points = input_obj_points_concat.reshape(-1, 1, 3).astype(np.float32)
        img_points = input_img_points_concat.reshape(-1, 1, 2).astype(np.float32)
        
        objpoints.append(obj_points)
        imgpoints.append(img_points)

    print('Finished loading all object and image points. Running intrinsic calibration')
    # Get image dimensions from first image
    height, width = images[0].shape[:2]

    # TODO using my own solvepnp for finding necessary rvecs and tvecs but isn't this another source of error compared to what calibrateCamera does? But...
    mean_error, errors = calculate_reprojection_error(objpoints, imgpoints, before_calibration_rvecs, before_calibration_tvecs, camera_matrix, dist_coeffs)
    print('before calibration:')
    print("Mean reprojection error: {} pixels".format(mean_error))
    # print("Individual errors: {}".format(errors))

    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, 
        imgpoints, 
        (width, height),
        cameraMatrix=camera_matrix,  # Add initial guess
        distCoeffs=dist_coeffs,      # Add initial guess
        flags=cv2.CALIB_USE_INTRINSIC_GUESS  # Add flag to use initial guess
    )

    mean_error, errors = calculate_reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist)
    print('after calibration:')
    print("Mean reprojection error: {} pixels".format(mean_error))
    print("Individual errors: {}".format(errors))

    return mtx, dist, mean_error, errors


def handeye_calibrate_opencv(handeye_data_dict, folder_name, eye_in_hand=True):
    '''
        Modifies handeye_data_dict with saved_cam2gripper 
    '''

    R_gripper2base = handeye_data_dict['R_gripper2base']
    t_gripper2base = handeye_data_dict['t_gripper2base']
    R_base2gripper = handeye_data_dict['R_base2gripper']
    t_base2gripper = handeye_data_dict['t_base2gripper']
    R_target2cam = handeye_data_dict['R_target2cam']
    t_target2cam = handeye_data_dict['t_target2cam']
    R_cam2target = handeye_data_dict['R_cam2target']
    t_cam2target = handeye_data_dict['t_cam2target']

    method = cv2.CALIB_HAND_EYE_TSAI  # default
    # method = cv2.CALIB_HAND_EYE_DANIILIDIS  # tried both, they both work in simulation
    
    # eye-in-hand (according to default opencv2 params and weak documentation. "inputting the suitable transformations to the function" for eye-to-hand)
    # first formula has b_T_c for X so that's what comes out of function. It expects b_T_g and c_T_t so gripper2base and target2cam, read transforms backwards
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

    full_homo_RT = np.identity(4)
    if eye_in_hand:
        # eye-in-hand
        # R_cam2base_est, t_cam2base_est = cv2.calibrateHandEye(R_gripper2base, t_gripper2base,
        # R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(R_gripper2base, t_gripper2base,
        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(R_base2gripper, t_base2gripper,
                                                            #   R_cam2target, t_cam2target, method=method)
                                                              R_target2cam, t_target2cam, method=method)  # this was so much better to above? and also more correct according to transforms
        full_homo_RT[:3, :3] = R_cam2gripper
        full_homo_RT[:3, 3] = t_cam2gripper.T
    else:
        # eye-to-hand
        R_cam2base_est, t_cam2base_est = cv2.calibrateHandEye(R_base2gripper, t_base2gripper,
                                                              R_target2cam, t_target2cam, method=method)

        full_homo_RT[:3, :3] = R_cam2base_est
        full_homo_RT[:3, 3] = t_cam2base_est.T

    print('Saving handeye (cv2) transform \n{}'.format(full_homo_RT))
    np.savetxt('data/{}/latest_cv2_cam2gripper.txt'.format(folder_name), full_homo_RT, delimiter=' ')
    handeye_data_dict['saved_cam2gripper'] = full_homo_RT  # TODO gotta be careful with naming
    # Invert to get cam2gripper
    gripper2cam = np.linalg.inv(full_homo_RT)

    handeye_data_dict['saved_cam2gripper'] = full_homo_RT
    handeye_data_dict['saved_gripper2cam'] = gripper2cam

    if not eye_in_hand:
        sys.exit('NOT IMPLEMENTED YET')
        assert()


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
# Eduardo's code from here: https://forum.opencv.org/t/eye-to-hand-calibration/5690/10
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