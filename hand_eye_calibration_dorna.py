from __future__ import print_function
import os
import sys
import argparse
import time
import math
import traceback
from glob import glob
from datetime import datetime

# this SO solved my scipy problem
# https://askubuntu.com/questions/1393285/how-to-install-glibcxx-3-4-29-on-ubuntu-20-04
import open3d as o3d
import requests
import pyrealsense2 as rs
from cv2 import aruco
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib as mpl
from scipy import optimize

from lib.vision import get_full_pcd_from_rgbd
from lib.vision import get_camera_coordinate, create_homogenous_transformations, convert_pixel_to_arm_coordinate, convert_cam_pcd_to_arm_pcd
from lib.vision import isclose, dist, isRotationMatrix, rotationMatrixToEulerAngles
from lib.vision_config import pinhole_camera_intrinsic
from lib.vision_config import camera_matrix, dist_coeffs
from lib.realsense_helper import setup_start_realsense, realsense_get_frames, run_10_frames_to_wait_for_auto_exposure
from lib.aruco_helper import create_aruco_params, aruco_detect_draw_get_transforms, calculate_pnp_12_markers
from lib.handeye_opencv_wrapper import handeye_calibrate_opencv, load_all_handeye_data, plot_all_handeye_data
from lib.dorna_kinematics import i_k, f_k
from lib.open3d_plot_dorna import plot_open3d_Dorna
from lib.aruco_image_text import OpenCvArucoImageText


# TODO seems very different to depth.intrinsics. Use depth intrinsics.....
# TODO .translate is easy for FK, what I've been doing, then the only other angles that matter is wrist pitch, wrist roll and base? pitch, roll and yaw I suppose
# TODO so I need to find the 4x4 matrix which will make .transform of coordinate frame onto gripper. And it's gripper to base or base 2 gripper?
# TODO once I have 4x4, I actually only need 3x1 or 3x3 rvec and 3x1 tvec
# TODO tvec and rvec come straight from aruco
# TODO why do we get cam2gripper and not cam2base though? But it's the gripper expressed in base coordinates soooooo base?
# TODO how to ensure everything is run from root directory.... Absolute paths.
# TODO how to ensure marker/found frame is flat? the world is flattened I mean. Wrong assumption because it isn;t?
# TODO could find relative pose transformations between multiple markers and then use them to create absolute ground truth instead of using a ruler
# TODO stop indenting so much, fix it with classes and stuff
# TODO im not using depth for 3D or 2D points!!!!!!?
# TODO how to minimise eyeball error? glue and pencil?
# TODO why do some angles not work well? 
# TODO https://github.com/opencv/opencv_contrib/blob/master/modules/aruco/src/aruco.cpp
# TODO could use cube!!!
# TODO read this: https://www.learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
# TODO Feb 24th 2024 update:
'''
What do I want?
To pick up any object to a high accuracy (>65% success rate), even if I'm clicking to choose the object. 

Where are we?:
Option A: Big board of markers with solvePnP worked but still there is eyeball error of placement of board and 
how far it is from dorna center AND eyeball error of dorna arm's wrist rotation (and other joints). 
BUT I did get close to picking up batteries with WORSE before so it 
could be a nice quick win to pick something up again. Just feels a bit hacky since my aruco is loosely placed. 
AND then optimise the last few milimetres with some optimisation code?

Option B: Alternatively, I've successfully understood a bit more spatial algebra and made my fake gorey arm simulation 
(with perfect transforms) work. All I have to do to make it work in the real world is debug correctly and have 
clean code so I understand every input to everywhere. E.g. maybe use cardboard of 12 big aruco markers on gripper, 
save these transforms WITH also the RGBD images, save everything to specific folder which can be retested 
without an arm and then visualise outputted cam2arm within rgbd pointcloud. I need to save SolvePnP's transform on key press 
rather than only one marker. In my gorey fake arm simulation, I had the order wrong in cam2target, didn't need inverse. 

Ok going for this option B right now at least. 
'''




def get_joint_angles_from_dorna_flask():
    r = requests.get('http://localhost:8081/get_xyz_joint')
    robot_data = r.json()
    joint_angles = robot_data['robot_joint_angles']

    return joint_angles


def transform_dict_of_xyz(xyz_dict, transform):
    # transforming below dict of lists with transform
    # xyz_positions_of_all_joints = {'shoulder': [shoulder_x, shoulder_y, shoulder_z], 
    #                                    'elbow': [elbow_x, elbow_y, elbow_z], 
    #                                    'wrist': [wrist_x, wrist_y, wrist_z], 
    #                                    'toolhead': xyz_toolhead_pos}

    new_dict = {}

    for key in xyz_dict.keys():
        arr = np.ones(4)  # just to make the last element a 1
        arr[:3] = xyz_dict[key]
        transformed_arr = np.dot(transform, arr)
        new_dict[key] = [transformed_arr[0], transformed_arr[1], transformed_arr[2]]

    return new_dict


def get_gripper_base_transformation(joint_angles):
    full_toolhead_fk, xyz_positions_of_all_joints = f_k(joint_angles)

    # TODO are things weird because of dorna's weird angle coordinate system?
    # because dorna's j1 is measure relative to ground plane, j2 is then relative to j1, j3 relative to j2. But also inverse direction right?
    
    joint_angles_copy = joint_angles.copy()
    joint_angles_copy[3] = -joint_angles_copy[3]
    joint_angles_copy[2] = -joint_angles_copy[2]
    joint_angles_copy[1] = -joint_angles_copy[1]

    joint_angles_rad = [math.radians(j) for j in joint_angles_copy]

    gripper_base_transform = np.identity(4)
    # arm2cam_local[:3, :3] = R_ct
    # TODO is the below in metres? yes but in get_handeye data we convert to metres
    gripper_base_transform[0, 3] = full_toolhead_fk[0]
    gripper_base_transform[1, 3] = full_toolhead_fk[1]
    gripper_base_transform[2, 3] = full_toolhead_fk[2]

    # TODO, wait it's the toolhead bottom which rotates, not the gripper tip, does this affect anything?
    wrist_pitch = np.sum(joint_angles_rad[1:4])
    wrist_roll = joint_angles_rad[4]
    base_yaw = joint_angles_rad[0]  # and the only way we can yaw
    # rot_mat = o3d.geometry.get_rotation_matrix_from_xyz(np.array([wrist_roll, wrist_pitch, base_yaw]))
    # rot_mat = o3d.geometry.get_rotation_matrix_from_zyx(np.array([wrist_roll, wrist_pitch, base_yaw]))
    rot_mat = o3d.geometry.get_rotation_matrix_from_zyx(np.array([base_yaw, wrist_pitch, wrist_roll]))
    # rot_mat = o3d.geometry.get_rotation_matrix_from_zyx(np.array([wrist_pitch, base_yaw, wrist_roll]))
    gripper_base_transform[:3, :3] = rot_mat

    return gripper_base_transform


def click_callback(event, x, y, flags, param):
    global mouseX, mouseY
    global curr_arm_xyz, prev_arm_xyz, click_pix_coord
    global FK_gripper_tip_projected_to_image, FK_wrist_projected_to_image, FK_elbow_projected_to_image, FK_shoulder_projected_to_image
    if event == cv2.EVENT_LBUTTONUP:
        mouseX, mouseY = x, y

        click_pix_coord = (int(round(mouseX)), int(round(mouseY)))

        if saved_cam2arm is not None:


            # TODO rename curr_arm_xyz to pick_xyz_cam_to_arm. Or did I call it that because I was comparing FK to deprojected something


            curr_arm_xyz = convert_pixel_to_arm_coordinate(camera_depth_img, mouseX, mouseY, saved_cam2arm, verbose=True)
            if curr_arm_xyz is not None:
                curr_arm_xyz = curr_arm_xyz * 1000

                camera_coord = get_camera_coordinate(camera_depth_img, mouseX, mouseY, verbose=True)
                print('camera_coord using depth and intrinsics:', camera_coord)
                print('saved tvec:', saved_tvec)  # TODO how different is saved_tvec to inverse saved
                print('hand-eye click in arm coords: ', [x for x in curr_arm_xyz])

                # TODO why am i using saved tvec etc?

                # joint_angles = get_joint_angles_from_dorna_flask()
                joint_angles = [0, 0, 0, 0, 0]  # TODO do not forget or add variable if dorna is running + homed or not

                # TODO remove prints for f_k or have param
                full_toolhead_fk, xyz_positions_of_all_joints = f_k(joint_angles)
                dist_btwn_FK_and_cam2arm_transformed_click = dist(full_toolhead_fk[:3], curr_arm_xyz)

                # TODO round all prints below, too hard to read
                print('FK xyz: ', full_toolhead_fk[:3])
                print('dist_btwn_FK_and_cam2arm_transformed_click: {}'.format(dist_btwn_FK_and_cam2arm_transformed_click))
                
                # TODO do I not need arm2cam? or in other words why the hell did not doing it make such a good result? does projectPoints do its own thing? yes
                gripper_tip_fk_metric = np.array(full_toolhead_fk[:3]) / 1000
                # print('gripper_tip_fk_metric: ', gripper_tip_fk_metric)
                imagePointsFK, jacobian = cv2.projectPoints(gripper_tip_fk_metric, saved_rvec, saved_tvec, camera_matrix, dist_coeffs)
                x_FK_pix, y_FK_pix = imagePointsFK.squeeze().tolist()
                FK_gripper_tip_projected_to_image = (int(round(x_FK_pix)), int(round(y_FK_pix)))
                print('click_pix_coord: {}, FK_gripper_tip_projected_to_image: {}'.format(click_pix_coord, FK_gripper_tip_projected_to_image))

                wrist_fk_metric = np.array(xyz_positions_of_all_joints['wrist']) / 1000
                imagePointsFK, jacobian = cv2.projectPoints(wrist_fk_metric, saved_rvec, saved_tvec, camera_matrix, dist_coeffs)
                x_FK_pix, y_FK_pix = imagePointsFK.squeeze().tolist()
                FK_wrist_projected_to_image = (int(round(x_FK_pix)), int(round(y_FK_pix)))

                elbow_fk_metric = np.array(xyz_positions_of_all_joints['elbow']) / 1000
                imagePointsFK, jacobian = cv2.projectPoints(elbow_fk_metric, saved_rvec, saved_tvec, camera_matrix, dist_coeffs)
                x_FK_pix, y_FK_pix = imagePointsFK.squeeze().tolist()
                FK_elbow_projected_to_image = (int(round(x_FK_pix)), int(round(y_FK_pix)))

                shoulder_fk_metric = np.array(xyz_positions_of_all_joints['shoulder']) / 1000
                imagePointsFK, jacobian = cv2.projectPoints(shoulder_fk_metric, saved_rvec, saved_tvec, camera_matrix, dist_coeffs)
                x_FK_pix, y_FK_pix = imagePointsFK.squeeze().tolist()
                FK_shoulder_projected_to_image = (int(round(x_FK_pix)), int(round(y_FK_pix)))
            else:
                print('curr_arm_xyz is None')
        else:
            print('no saved_cam2arm. Needed for click')

    # if event == cv2.EVENT_MBUTTONUP:
    #     # cv2.circle(camera_color_img, (x, y), 100, (255, 255, 0), -1)  # TODO if I ever want this
    #     mouseX, mouseY = x, y

    #     curr_arm_xyz = convert_pixel_to_arm_coordinate(camera_depth_img, mouseX, mouseY, cam2arm, verbose=True)

    #     print('Distance to previous arm xyz 3D: {}. 2D distance: {}'.format(dist(curr_arm_xyz, prev_arm_xyz), dist(curr_arm_xyz[:2], prev_arm_xyz[:2])))
    #     prev_arm_xyz = curr_arm_xyz
    #     # TODO try except
    #     # IndexError: index 1258 is out of bounds for axis 0 with size 640


def find_aruco_markers(color_img, depth_img):
    # global ids, corners, all_rvec, all_tvec

    color_img = color_img.copy()
    depth_img = depth_img.copy()
    bgr_color_data = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
    gray_data = cv2.cvtColor(bgr_color_data, cv2.COLOR_RGB2GRAY)

    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray_data, aruco_dict,
                                                                parameters=parameters)
    frame_markers = aruco.drawDetectedMarkers(color_img, corners, ids)
    all_rvec, all_tvec, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)

    if all_rvec is not None:
        ids_list = [l[0] for l in ids.tolist()]

        if len(ids_list) >= 1:
            # TODO am i using the below?
            for corner_id in [1, 2, 3, 4]:
                # TODO what am I doing below?
                # for corner_id in [4]:  # TODO make it not crash if other aruco etc!!!
                if corner_id in ids_list:
                    corner_index = ids_list.index(corner_id) 
                    # rvec_aruco, tvec_aruco = all_rvec[corner_index, 0, :], all_tvec[corner_index, 0, :]
                    # aruco.drawAxis(color_img, camera_matrix, dist_coeffs, rvec_aruco, tvec_aruco, marker_length)

        found_correct_marker = False
        if id_on_shoulder_motor in ids:
            # TODO is this even correct?!?!?! since 1 index vs 0 index?!?!? ahh because I found correct index?
            shoulder_motor_marker_id = [l[0] for l in ids.tolist()].index(id_on_shoulder_motor) 
            rvec, tvec = all_rvec[shoulder_motor_marker_id, 0, :], all_tvec[shoulder_motor_marker_id, 0, :]  # get first marker
            found_correct_marker = True
        else:
            print('Did not find shoulder marker, {}'.format(ids_list))
            tvec, rvec = None, None
            # pass

        if found_correct_marker:  # TODO jan 2023 does this make any sense anymore if im using 12 markers?
            # tvec, rvec = all_tvec[0].squeeze(), all_rvec[0].squeeze()

            # TODO typo below??? WILL IT BREAK THINGS and overwrite cam2arm or not?
            # TODO the below repeats from basic_aruco_example.py
            cam2arm, arm2cam, R_tc, R_ct, pos_camera = create_homogenous_transformations(tvec, rvec)
            # cam2arm, arm2cam, R_tc, R_ct, pos_camera = create_homogenous_transformations(tvec, rvec)

            # -- Get the attitude in terms of euler 321 (Needs to be flipped first)
            roll_marker, pitch_marker, yaw_marker = rotationMatrixToEulerAngles(opencv_aruco_image_text.R_flip * R_tc)
            # -- Get the attitude of the camera respect to the frame
            roll_camera, pitch_camera, yaw_camera = rotationMatrixToEulerAngles(opencv_aruco_image_text.R_flip * R_ct)  # todo no flip needed?

            opencv_aruco_image_text.put_marker_text(camera_color_img_debug, tvec, roll_marker, pitch_marker, yaw_marker)
            opencv_aruco_image_text.put_camera_text(camera_color_img_debug, pos_camera, roll_camera, pitch_camera, yaw_camera)
            # opencv_aruco_image_text.put_avg_marker_text(camera_color_img, avg_6dof_pose)
    else:
        tvec, rvec = None, None

    return color_img, depth_img, tvec, rvec, ids, corners, all_rvec, all_tvec


if __name__ == '__main__':
    np.set_printoptions(precision=6, suppress=True)
    opencv_aruco_image_text = OpenCvArucoImageText()

    handeye_data_folder_name = datetime.now().strftime('%d_%m_%Y_%H_%M_%S')
    print('handeye_data_folder_name for all saved transforms (not created yet): ', handeye_data_folder_name)

    # get_joint_angles_from_dorna_flask()  # TODO if failed then changed all further calls to hardcoded joint values and make it very clear no arm connected?

    prev_arm_xyz = np.array([0., 0., 0.])
    cam2arm = np.identity(4)
    saved_cam2arm = None
    saved_arm2cam = None
    curr_arm_xyz = None
    # cam_coords = []
    click_pix_coord = None
    FK_gripper_tip_projected_to_image = None
    FK_wrist_projected_to_image = None
    FK_elbow_projected_to_image = None
    FK_shoulder_projected_to_image = None
    saved_rvec = None
    saved_tvec = None

    # trying to put text into open3D
    # label_3D = o3d.visualization.gui.Label3D('hey', np.array([0, 0, 0]))
    # o3d.visualization.draw_geometries([label_3D])
    # https://stackoverflow.com/questions/71959737/display-text-on-point-cloud-vertex-in-pyrender-open3d

    depth_intrin, color_intrin, depth_scale, pipeline, align, spatial = setup_start_realsense()

    # calibration and marker params
    shoulder_height = 206.01940000000002

    board, parameters, aruco_dict, marker_length = create_aruco_params()
    half_marker_len = marker_length / 2
    colors = ((255, 255, 255), (255, 0, 0), (0, 0, 0), (0, 0, 255))
    marker_separation = 0.0065
    ids, corners, all_rvec, all_tvec = None, None, None, None  # TODO global remove

    mouseX, mouseY = 0, 0
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', click_callback)

    check_corner_frame_count = 0
    frame_count = 1
    num_saved_handeye_transforms = 0

    id_on_shoulder_motor = 1

    # if arm_properly_calibrated_homed:  # TODO think about this. would be nice 
    # command = '/home/beduffy/anaconda/envs/py36/bin/python control/scripts/send_arm_to_home_position.py'
    # print('Running command: {}'.format(command))
    # os.system(command)
    # time.sleep(1)

    run_10_frames_to_wait_for_auto_exposure(pipeline, align)

    print('Starting main loop')
    while True:
        try:
            color_frame, depth_frame = realsense_get_frames(pipeline, align, spatial)

            camera_depth_img = np.asanyarray(depth_frame.get_data())
            camera_color_img = np.asanyarray(color_frame.get_data())  # TODO rename to input image and then other image with circles call debug
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(camera_depth_img, alpha=0.03),
                                               cv2.COLORMAP_JET)  # TODO why does it look so bad, add more contrast?

            camera_color_img_debug = camera_color_img.copy()
            color_img, depth_img, tvec, rvec, ids, corners, all_rvec, all_tvec = find_aruco_markers(camera_color_img, camera_depth_img)

            if tvec is not None and rvec is not None:
                cam2arm, arm2cam, R_tc, R_ct, pos_camera = create_homogenous_transformations(tvec, rvec)

            if click_pix_coord is not None:
                cv2.circle(color_img, click_pix_coord, 5, (0, 0, 255), -1)
            if FK_gripper_tip_projected_to_image is not None:
                cv2.circle(color_img, FK_gripper_tip_projected_to_image, 5, (255, 0, 0), -1)
            if FK_wrist_projected_to_image is not None:
                cv2.circle(color_img, FK_wrist_projected_to_image, 5, (0, 255, 255), -1)
                cv2.line(color_img, FK_wrist_projected_to_image, FK_gripper_tip_projected_to_image, (0, 255, 255), thickness=3)
            if FK_elbow_projected_to_image is not None:
                cv2.circle(color_img, FK_elbow_projected_to_image, 5, (0, 0, 255), -1)
                cv2.line(color_img, FK_elbow_projected_to_image, FK_wrist_projected_to_image, (0, 0, 255), thickness=3)
            if FK_shoulder_projected_to_image is not None:
                cv2.circle(color_img, FK_shoulder_projected_to_image, 5, (0, 0, 0), -1)
                cv2.line(color_img, FK_shoulder_projected_to_image, FK_elbow_projected_to_image, (0, 0, 0), thickness=3)

            images = np.hstack((camera_color_img_debug, depth_colormap))  # TODO to not have writing? Have better variable images

            cv2.imshow("image", images)
            k = cv2.waitKey(1)

            camera_color_img = cv2.cvtColor(camera_color_img, cv2.COLOR_BGR2RGB)  # for open3D

            # TODO put all into handle_keyboard_input()
            if k == ord('q'):
                cv2.destroyAllWindows()
                pipeline.stop()
                break

            # Saves current cam2arm, arm2cam, rvec, tvec from chosen marker ID # TODO but that doesn't make too much sense
            if k == ord('s'):
                # TODO can be nice to click things online but I can do offline. to remove all of this or not?
                if cam2arm is not None:
                    saved_cam2arm = cam2arm 
                    saved_arm2cam = arm2cam  # TODO arm2cam does not exist?
                    saved_rvec = rvec
                    saved_tvec = tvec
                    print('Saving aruco cam2arm {}\n'.format(saved_cam2arm))
                    print('Saving aruco arm2cam {}\n'.format(saved_arm2cam))
                    np.savetxt('data/latest_aruco_cam2arm.txt', cam2arm, delimiter=' ')

                    # TODO maybe for clicking i should only used saved cam2arm rather than it changing...


            # Go to saved xyz position from click, but first pre-grasp pose right above.
            if k == ord('p'):
                if curr_arm_xyz is not None:
                    x, y, z = curr_arm_xyz
                    # if z < 10:
                    #     print('Z was {} and below 10, setting to 10'.format(z))
                    #     z = 10
                    print('Z was {} setting to 10'.format(z))
                    z = 10
                    pre_grasp_z = 200
                    # TODO prepick and then pre grasp and then grasp and then. All in other process so we see camera?
                    # TODO how to organise all this code better? ROS? functions? bleh
                    # TODO create function for go_to_xyz here too
                    # TODO open gripper before
                    # TODO do chicken head dance with gripper in same xyz but wrist pitch changing
                    print('Going to pre-pick pose')
                    wrist_pitch = -42.0
                    dorna_url = 'http://localhost:8081/go_to_xyz'
                    # dorna_full_url = '{}?x={}&y={}&z={}'.format(dorna_url, x, y, pre_grasp_z)
                    dorna_full_url = '{}?x={}&y={}&z={}&wrist_pitch={}'.format(dorna_url, x, y, pre_grasp_z, wrist_pitch)
                    r = requests.get(url=dorna_full_url)
                    print('r.status_code: {}. r.text: {}'.format(r.status_code, r.text))
                    if r.status_code == 200 and r.text == 'success':
                        print('Sleeping before pre-grasp')
                        time.sleep(6)  # TODO how to avoid sleeps in the future? Ask if dorna is ready or loops or polling? or ros? or something?
                        # dorna_full_url = '{}?x={}&y={}&z={}&wrist_pitch={}'.format(dorna_url, x, y, 20, wrist_pitch)
                        dorna_full_url = '{}?x={}&y={}&z={}&wrist_pitch={}'.format(dorna_url, x, y, z, wrist_pitch)
                        r = requests.get(url=dorna_full_url)
                        print('r.status_code: {}. r.text: {}'.format(r.status_code, r.text))
                        if r.status_code == 200 and r.text == 'success':
                            print('Sleeping before closing gripper')
                            time.sleep(4)
                            dorna_grasp_full_url = 'http://localhost:8081/gripper?gripper_state=3'  # TODO different objects and do my conversion of object width to gripper close width 
                            r = requests.get(url=dorna_grasp_full_url)
                            print('r.status_code: {}. r.text: {}'.format(r.status_code, r.text))
                            if r.status_code == 200 and r.text == 'success':
                                print('Sleeping short before going up (after pick)')
                                time.sleep(1)
                                # dorna_full_url = '{}?x={}&y={}&z={}'.format(dorna_url, x, y, 150)
                                dorna_full_url = '{}?x={}&y={}&z={}&wrist_pitch={}'.format(dorna_url, x, y, 150, wrist_pitch)
                                r = requests.get(url=dorna_full_url)
                                print('r.status_code: {}. r.text: {}'.format(r.status_code, r.text))

                            # Optional or not, let go
                            dorna_grasp_full_url = 'http://localhost:8081/gripper?gripper_state=0'  # TODO could pass state for different objects and do my conversion of object width to gripper close width 
                            r = requests.get(url=dorna_grasp_full_url)
                            print('r.status_code: {}. r.text: {}'.format(r.status_code, r.text))
                else:
                    print('curr_arm_xyz is None')


            # Visualise how dorna stick/line mesh arm in pointcloud would look like picking up chosen click position
            if k == ord('i'):
                if curr_arm_xyz is not None:
                    x, y, z = curr_arm_xyz
                    # z = 200  # pre pick
                    wrist_pitch = 0.0  # TODO this affects everything, understand how
                    wrist_pitch = -42.0  # TODO this affects everything
                    # TODO how to dynamically change wrist pitch for different things and choose one out of many? 
                    # TODO how to prevent all collisions?
                    # TODO aruco far away problem was to do with intrinsics I bet
                    fifth_IK_value = 0.0
                    xyz_pitch_roll = [x, y, z, wrist_pitch, fifth_IK_value]
                    joint_angles = i_k(xyz_pitch_roll)
                    print('joint_angles: ', joint_angles)

                    if joint_angles:
                        full_toolhead_fk, xyz_positions_of_all_joints = f_k(joint_angles)
                        print('full_toolhead_fk: ', full_toolhead_fk)

                        cam_pcd = get_full_pcd_from_rgbd(camera_color_img, camera_depth_img,
                                                pinhole_camera_intrinsic, visualise=False)

                        # only plot open3D arm when arm is in position. Otherwise if non-blocking
                        # TODO remove q1 and z offset now that aruco and solvePnP finds correct transform
                        # TODO never understood how this relates, if origin looks good but cam2arm is bad?
                        # transforming rgbd pointcloud using bad cam2arm means what? . What is the thing changing again?

                        full_arm_pcd, full_pcd_numpy = convert_cam_pcd_to_arm_pcd(cam_pcd, saved_cam2arm)

                        plot_open3d_Dorna(xyz_positions_of_all_joints, extra_geometry_elements=[full_arm_pcd, coordinate_frame_arm_frame, coordinate_frame_shoulder_height_arm_frame])

                    else:
                        print('IK returned none')
                else:
                    print('curr_arm_xyz is None')


            if k == ord('l'):
                # TODO verify what l does and write here
                # TODO this is wrong description: Use curr joint angles and show line mesh arm plotted over pointcloud arm after all transformations (some calculated here)

                # What I want 'l' to do: 
                # actually don't know...

                # What 'l' changed to in January 2024:
                # Running solvePnP on obj+img points, visualising most accurate transforms, 
                # AND OVERWRITING saved_cam2arm for then running with 'i' key AND later 'p' key if it worked...

                # what if 'l' is option A and 'h' is option B?

                # joint_angles = get_joint_angles_from_dorna_flask()
                joint_angles = [0, 0, 0, 0, 0]  # for when dorna is off # TODO do not forget
                print('joint_angles: ', joint_angles)

                full_toolhead_fk, xyz_positions_of_all_joints = f_k(joint_angles)
                print('full_toolhead_fk: ', full_toolhead_fk)

                # TODO first what am i doing below? first idea and then functions
                # Calculate image points of 12 aruco markers
                
                cam2arm_opt, arm2cam_opt = calculate_pnp_12_markers(corners, ids, all_rvec, all_tvec, marker_length=marker_length, marker_separation=marker_separation)


                # TODO I might want to switch between option A and B. 
                # Because I had original 12 markers to the left of the shoulder (option A mentioned above) 
                # I wanted to further compose a transform, from marker to center of dorna. 
                marker_to_arm_transformation = np.identity(4)
                measured_y_distance_to_dorna_from_marker = -0.3  # with 12 big markers on plank the the left of dorna, rather than on shoulder stepper
                measured_y_distance_to_dorna_from_marker = -0.33
                # TODO why am i off by 2 or more cm? something slightly wrong with calculation here
                marker_to_arm_transformation[1, 3] = measured_y_distance_to_dorna_from_marker * 1000.0  # TODO in m or milimetre?
                arm2cam_opt = np.dot(arm2cam_opt, marker_to_arm_transformation)  # TODO rename all vars
                

                # TODO what does the below do? 
                xyz_positions_of_all_joints = transform_dict_of_xyz(xyz_positions_of_all_joints, marker_to_arm_transformation)

                saved_cam2arm = cam2arm_opt
                cam2arm_opt_milimetres = np.copy(cam2arm_opt)
                cam2arm_opt_milimetres[:3, 3] *= 1000.0
                saved_arm2cam = arm2cam_opt  # TODO wtf, double overwrite. Is this more correct? below im using milimetres.

                cam_pcd = get_full_pcd_from_rgbd(camera_color_img, camera_depth_img,
                                        pinhole_camera_intrinsic, visualise=False)
                if saved_cam2arm is not None:
                    full_arm_pcd, full_pcd_numpy = convert_cam_pcd_to_arm_pcd(cam_pcd, saved_cam2arm)

                    # visualise camera coordinate frame
                    # TODO broken due to arm2cam milimetres something... fix. cam2arm_opt_milimetres should do it?
                    # aruco_coordinate_frame_cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0.0, 0.0, 0.0])
                    # aruco_coordinate_frame_cam_frame.transform(saved_arm2cam)
                    # origin_frame_cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0.0, 0.0, 0.0])
                    # mesh_box_cam = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=0.003)
                    # mesh_box_cam.transform(saved_arm2cam)
                    # elements_only_in_cam_frame = [cam_pcd, aruco_coordinate_frame_cam_frame, origin_frame_cam_frame, mesh_box_cam]
                    # o3d.visualization.draw_geometries(elements_only_in_cam_frame)

                    # visualise arm coordinate frame
                    cam_frame_in_arm_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                        size=25.0, origin=[0.0, 0.0, 0.0])
                    coordinate_frame_arm_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                        size=25.0, origin=[0.0, 0.0, 0.0])
                    coordinate_frame_shoulder_height_arm_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                        size=25.0, origin=[0.0, 0.0, shoulder_height])
                    cam_frame_in_arm_frame.transform(cam2arm_opt_milimetres)
                    coordinate_frame_arm_frame.transform(marker_to_arm_transformation)
                    coordinate_frame_shoulder_height_arm_frame.transform(marker_to_arm_transformation)

                    gripper_coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=25.0, origin=[0.0, 0.0, 0.0])
                    gripper_base_transform = get_gripper_base_transformation(joint_angles)
                    gripper_coordinate_frame.transform(gripper_base_transform)

                    mesh_box_arm_frame = o3d.geometry.TriangleMesh.create_box(width=100.0, height=100.0, depth=0.003)
                    mesh_box_arm_frame.transform(marker_to_arm_transformation)
                    mesh_box_arm_frame_shoulder_height_higher = o3d.geometry.TriangleMesh.create_box(width=100.0, height=100.0, depth=0.003)
                    mesh_box_arm_frame_shoulder_height_higher.transform(marker_to_arm_transformation)
                    mesh_box_arm_frame_shoulder_height_higher.translate(np.array([0, 0, shoulder_height]))
                    
                    # extra_elements = [full_arm_pcd, mesh_box_arm_frame, gripper_coordinate_frame, 
                    extra_elements = [full_arm_pcd, mesh_box_arm_frame, mesh_box_arm_frame_shoulder_height_higher, gripper_coordinate_frame, 
                                      coordinate_frame_arm_frame, coordinate_frame_shoulder_height_arm_frame, cam_frame_in_arm_frame]
                    plot_open3d_Dorna(xyz_positions_of_all_joints, extra_geometry_elements=extra_elements)

                    # TODO get click working now with big markers. 
                    # TODO should I transform all xyz_positions_of_all_joints
                    # TODO could I keep everything in metres until I send the final control command of dorna?
                    # TODO if I create a generic transformation from marker to dorna arm origin, then I can compose transformations to pick up objects?
                    # TODO the purpose of everything here. all I want to get to is my old hand eye aruco on shoulder calibration and then using that to try pick up objects close enough again, but this time with solvePnP and 12 markers, it should obviously work better.
                    # TODO why are we off in depth and coordinate frame is behind marker? problem is resolved when we get closer. Could try bigger markers too
                    # TODO is it possible the scaling is messing things up? it didn't before though. 
                    # TODO we seem to beginning arm from shoulder height, rather than ground height? why?
                    # TODO look into open3D interactive mode and change sliders of joints here or for ik and see how wrist pitch changes
                    # TODO look into camera settings so I can look down top down, side-ways and straight down barrel on arm to see how wrong transforms are
                    # TODO optimise on white sphere and track it
                    # TODO is there a way I could visual servo measure angles?
                    # TODO I was using speed 5000 (what does this mean in joint and xyz space? same thing, how fast can I go?)
                    # dist(np.array([148.166, -190.953, 440.97]), np.array([ 131.949075, -199.25261, 452.410165]))
                    # TODO if I'm specifying that the clicked point is the centre of the battery I could get an error metric from FK vs cam2arm click point
                else:
                    print('No saved cam2arm!')


            # get and save calibration target transformation (target2cam) and gripper2base
            if k == ord('h'):  # save hand-eye calibration needed transforms

                # What I want 'h' to do:
                # Calculate, visualise and save cam2target and gripper2base to new folder (e.g. handeye_24_02_2024_HH_MM_SS). 
                # But the new idea will be to use SolvePnP with 12 markers. 
                # AND Also save the RGBD images so I can visualise each transform individually but also in one pointcloud assuming camera does not move.
                # later make it easy to test, visualise on any folder. In my gorey fake arm simulation, I had the order wrong in cam2target, didn't need inverse. 
                
                # TODO should I do open3D here too just to make sure the transform is good? Meh, I can analyse later in my test script. Remove this TODO
                # TODO do I Want to keep the code to make it work on 1 marker?
                # TODO how will i rotate the gripper with large 12 cardboard markers on the gripper? Only add after homing?, first see if it is a problem
                # TODO only if we see markers or 12 markers?
                # if tvec is not None and rvec is not None:
                cam2arm_opt, arm2cam_opt = calculate_pnp_12_markers(corners, ids, all_rvec, all_tvec, marker_length=marker_length, marker_separation=marker_separation)

                # aruco_id_on_gripper = 4
                # bgr_color_data = cv2.cvtColor(camera_color_img, cv2.COLOR_RGB2BGR)
                # gray_data = cv2.cvtColor(bgr_color_data, cv2.COLOR_RGB2GRAY)
                # corners, ids, rejectedImgPoints = aruco.detectMarkers(gray_data, aruco_dict, parameters=parameters)
                # # frame_markers = aruco.drawDetectedMarkers(color_img, corners, ids)
                # all_rvec, all_tvec, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)
                # found_correct_marker = False
                # if aruco_id_on_gripper in ids:
                #     gripper_aruco_index = [l[0] for l in ids.tolist()].index(aruco_id_on_gripper) 
                #     rvec, tvec = all_rvec[gripper_aruco_index, 0, :], all_tvec[gripper_aruco_index, 0, :]
                #     # aruco.drawAxis(color_img, camera_matrix, dist_coeffs, rvec, tvec, marker_length)
                #     found_correct_marker = True
                # else:
                #     tvec, rvec = None, None

                # cam2target, target2cam, R_tc, R_ct, pos_camera = create_homogenous_transformations(tvec, rvec)

                # assert(isRotationMatrix(R_tc))
                # assert(isRotationMatrix(R_ct))

                # TODO better variable to avoid format repetition
                full_handeye_folder_path = 'data/{}'.format(handeye_data_folder_name)
                if not os.path.exists(full_handeye_folder_path):
                     os.makedirs(full_handeye_folder_path)
                
                target2cam = arm2cam_opt  # TODO debatable and make sure! TODO and horribly confusing fix it
                fp = '{}/target2cam_{}.txt'.format(full_handeye_folder_path, num_saved_handeye_transforms)
                print('Saving target2cam at {} \n{}'.format(fp, target2cam))
                np.savetxt(fp, target2cam, delimiter=' ')

                # get and save gripper transformation (gripper2base)
                joint_angles = get_joint_angles_from_dorna_flask()  # TODO do not forget
                # the below is just for testing without running arm
                # joint_angles = [0, 0, 0, 0, 0]
                gripper_base_transform = get_gripper_base_transformation(joint_angles)  # TODO is this in mm? yes

                # TODO visualise full fk of these newly saved joint angles
                fp = '{}/joint_angles_{}.txt'.format(full_handeye_folder_path, num_saved_handeye_transforms)
                print('Saving joint angles at {} \n{}'.format(fp, joint_angles))
                np.savetxt(fp, joint_angles, delimiter=' ')
                # TODO try get inverse (actual gripper2base) just to really confirm shit
                fp = '{}/gripper2base_{}.txt'.format(full_handeye_folder_path, num_saved_handeye_transforms)
                print('Saving gripper2base at {} \n{}'.format(fp, gripper_base_transform))
                np.savetxt(fp, gripper_base_transform, delimiter=' ')

                # save images so we can analyse and understand transforms better later
                # TODO stop using camera_color_img for aruco text!!!!!
                fp = '{}/color_img_{}.png'.format(full_handeye_folder_path, num_saved_handeye_transforms)
                cv2.imwrite(fp, camera_color_img)
                fp = '{}/depth_img_{}.png'.format(full_handeye_folder_path, num_saved_handeye_transforms)
                cv2.imwrite(fp, camera_depth_img)

                num_saved_handeye_transforms += 1
                # else:
                #     print('No tvec or rvec!')


            if k == ord('c'):  # perform hand-eye calibration using saved transforms
                # TODO all of the below can probably go to handeye wrapper?
                handeye_data_dict = load_all_handeye_data(folder_name=handeye_data_folder_name)
                # plot_all_handeye_data(handeye_data_dict)
                handeye_calibrate_opencv(handeye_data_dict, handeye_data_folder_name)

                # TODO why load from file again, why not just return from function?
                cam2arm = np.loadtxt('data/{}/latest_cv2_cam2arm.txt'.format(handeye_data_folder_name), delimiter=' ')
                saved_cam2arm = cam2arm
                
                # TODO what the hell am I doing, of course saved cam2arm is fucked up. The only way to is to use cam_pcd 
                cam_pcd = get_full_pcd_from_rgbd(camera_color_img, camera_depth_img, pinhole_camera_intrinsic, visualise=False)
                # full_arm_pcd, full_pcd_numpy = convert_cam_pcd_to_arm_pcd(cam_pcd, saved_cam2arm, in_milimetres=False)


                # plot_all_handeye_data(handeye_data_dict, cam_pcd=full_arm_pcd)
                plot_all_handeye_data(handeye_data_dict, cam_pcd=cam_pcd)

            frame_count += 1

        except ValueError as e:
            # TODO try except is important to not or safely crash realsense or?
            print('Error in main loop')
            print(e)
            print(traceback.format_exc())

            pipeline.stop()
            cv2.destroyAllWindows()