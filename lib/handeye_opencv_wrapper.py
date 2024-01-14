from __future__ import print_function
from glob import glob

import numpy as np
import cv2
import open3d as o3d

from lib.vision import get_inverse_homogenous_transform


def load_all_handeye_data():
    # Open data/handeye folder and then load into dict
    gripper_transform_files = sorted(glob('data/handeye/gripper2base*'))
    cam2target_files = sorted(glob('data/handeye/target2cam*'))

    all_gripper_rotation_mats = []
    all_gripper_tvecs = []
    for fp in gripper_transform_files:
        gripper_transform = np.loadtxt(fp, delimiter=' ')
        print('Loaded {} transform'.format(fp))
        gripper2base_rot = gripper_transform[:3, :3]
        all_gripper_rotation_mats.append(gripper2base_rot)
        # all_gripper_tvecs.append(gripper_transform[:3, 3] * 1000.0)  # TODO remove if it makes no sense
        all_gripper_tvecs.append(gripper_transform[:3, 3] / 1000.0)
    R_gripper2base = np.array(all_gripper_rotation_mats)
    t_gripper2base = np.array(all_gripper_tvecs)

    all_target2cam_rotation_mats = []
    all_target2cam_tvecs = []
    for fp in cam2target_files:
        cam2target_transform = np.loadtxt(fp, delimiter=' ')

        # Just testing but not right
        # target2cam = get_inverse_homogenous_transform(cam2target_transform)
        # cam2target_transform = target2cam

        print('Loaded {} transform'.format(fp))
        cam2target_rot = cam2target_transform[:3, :3]
        all_target2cam_rotation_mats.append(cam2target_rot)
        all_target2cam_tvecs.append(cam2target_transform[:3, 3])
        # all_target2cam_tvecs.append(cam2target_transform[:3, 3] / 1000.0)  # TODO or multiply gripper coord by 1000?
    R_target2cam = np.array(all_target2cam_rotation_mats)
    t_target2cam = np.array(all_target2cam_tvecs)

    handeye_data_dict = {
        'all_gripper_rotation_mats': all_gripper_rotation_mats,
        'all_gripper_tvecs': all_gripper_tvecs,
        'R_gripper2base': R_gripper2base,
        't_gripper2base': t_gripper2base,
        'all_target2cam_rotation_mats': all_target2cam_rotation_mats,
        'all_target2cam_tvecs': all_target2cam_tvecs,
        'R_target2cam': R_target2cam,
        't_target2cam': t_target2cam,
    }
    return handeye_data_dict


def plot_all_handeye_data(handeye_data_dict, cam_pcd=None):
    all_gripper_rotation_mats = handeye_data_dict['all_gripper_rotation_mats']
    all_gripper_tvecs = handeye_data_dict['all_gripper_tvecs']
    R_gripper2base = handeye_data_dict['R_gripper2base']
    t_gripper2base = handeye_data_dict['t_gripper2base']
    all_target2cam_rotation_mats = handeye_data_dict['all_target2cam_rotation_mats']
    all_target2cam_tvecs = handeye_data_dict['all_target2cam_tvecs']
    R_target2cam = handeye_data_dict['R_target2cam']
    t_target2cam = handeye_data_dict['t_target2cam']

    geometry_to_plot = []
    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                                        size=0.2, origin=[0.0, 0.0, 0.0])
    geometry_to_plot.append(origin_frame)
    for idx, base2gripper_tvec in enumerate(all_gripper_tvecs):
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                                        size=0.1, origin=[base2gripper_tvec[0], base2gripper_tvec[1], base2gripper_tvec[2]])  # TODO shouldn't be doing this, should be using from transform probably?
        coordinate_frame.rotate(all_gripper_rotation_mats[idx])

        # sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        # sphere.translate([cam2target_tvec[0], cam2target_tvec[1], cam2target_tvec[2]])
        # geometry_to_plot.append(sphere)
        geometry_to_plot.append(coordinate_frame)

    print('Visualising origin and gripper frames in arm frame')
    o3d.visualization.draw_geometries(geometry_to_plot)

    # TODO do the below for gripper poses as well, they should perfectly align rotation-wise to aruco poses
    geometry_to_plot = []
    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                                        size=0.2, origin=[0.0, 0.0, 0.0])
    # TODO ideally I'd visualise a frustum. matplotlib?
    # TODO draw mini plane of all arucos rather than coordinate frames. https://github.com/isl-org/Open3D/issues/3618
    geometry_to_plot.append(origin_frame)
    for idx, cam2target_tvec in enumerate(all_target2cam_tvecs):
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                                        size=0.1, origin=[cam2target_tvec[0], cam2target_tvec[1], cam2target_tvec[2]])
        coordinate_frame.rotate(all_target2cam_rotation_mats[idx])

        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        # TODO WHY is the sphere always a bit higher than the origin of the coordinate frame?
        sphere.translate([cam2target_tvec[0], cam2target_tvec[1], cam2target_tvec[2]])
        geometry_to_plot.append(sphere)
        geometry_to_plot.append(coordinate_frame)
        # TODO need better way to visualise? images? pointclouds? Origin should be different or bigger and point x outward?

    if cam_pcd is not None:
        geometry_to_plot.append(cam_pcd)
    print('Visualising camera origin and aruco frames in arm frame')
    o3d.visualization.draw_geometries(geometry_to_plot)


def handeye_calibrate_opencv(handeye_data_dict):
    # all_gripper_rotation_mats = handeye_data_dict['all_gripper_rotation_mats']
    # all_gripper_tvecs = handeye_data_dict['all_gripper_tvecs']
    R_gripper2base = handeye_data_dict['R_gripper2base']
    t_gripper2base = handeye_data_dict['t_gripper2base']
    # all_target2cam_rotation_mats = handeye_data_dict['all_target2cam_rotation_mats']
    # all_target2cam_tvecs = handeye_data_dict['all_target2cam_tvecs']
    R_target2cam = handeye_data_dict['R_target2cam']
    t_target2cam = handeye_data_dict['t_target2cam']

    method = cv2.CALIB_HAND_EYE_DANIILIDIS
    # TODO try others
    # method = 
    # eye-in-hand (according to default opencv2 params and weak documentation. "inputting the suitable transformations to the function" for eye-to-hand)
    # first formula has b_T_c for X so that's what comes out of function. It expects b_T_g and c_T_t so gripper2base and target2cam
    # R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, method=method)
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(R_gripper2base=R_gripper2base, 
                                                        t_gripper2base=t_gripper2base, 
                                                        R_target2cam=R_target2cam, 
                                                        t_target2cam=t_target2cam, method=method)
    # eye-to-hand
    # second formula has b_T_c for X so camera2base actually is what comes out of function. It expects g_T_b (base2gripper) and c_T_t (target2cam)
    # R_camera2base, t_camera2base = cv2.calibrateHandEye(R_base2gripper, t_base2gripper, R_target2cam, t_target2cam)
    # TODO could rename nothing and just change inputs to be right, of course I could. 

    cam2arm = np.identity(4)
    cam2arm[:3, :3] = R_cam2gripper
    cam2arm[:3, 3] = t_cam2gripper.squeeze()
    print('Saving handeye (cv2) cam2arm \n{}'.format(cam2arm))
    np.savetxt('data/handeye/latest_cv2_cam2arm.txt', cam2arm, delimiter=' ')

    # pos_camera = np.dot(-R_cam2gripper, np.matrix(t_cam2gripper).T)
    # TODO why does pos_camera seem to have the better position and similarity to my old cam2arms?
    pos_camera = np.dot(-R_cam2gripper, np.matrix(t_cam2gripper))
    cam2arm_local = np.identity(4)
    cam2arm_local[:3, :3] = R_cam2gripper.T
    # cam2arm_local[:3, :3] = R_cam2gripper
    cam2arm_local[0, 3] = pos_camera[0]
    cam2arm_local[1, 3] = pos_camera[1]
    cam2arm_local[2, 3] = pos_camera[2]
    # print('cam2arm inverse:\n{}'.format(cam2arm_local))
    print('handeye (cv2) cam2arm inverse \n{}'.format(cam2arm_local))
    # print('Saving handeye (cv2) cam2arm inverse \n{}'.format(cam2arm_local))
    # np.savetxt('data/handeye/latest_cv2_cam2arm.txt', cam2arm_local, delimiter=' ')


# Eduardo's code from here: https://forum.opencv.org/t/eye-to-hand-calibration/5690/10
# TODO maybe it's nice to just have a param for eye-to-hand like that so I can keep everything pretty similar?
def calibrate_eye_hand(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, eye_to_hand=True):
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
# TODO target2cam or cam2target. Ahh opencv param names according to eye-in-hand vs eye-to-hand might change
# TODO arm2cam or cam2arm? should get to the bottom of this forever. camera coordinate in arm coordinates and the transform is the same?

# TODO save pic or not? Save reprojection error or ambiguity or something?
# TODO would be nice to plot all poses or coordinate frames or something
# TODO how to avoid aruco error at range? Bigger? Board? Hold a checkerboard?
# TODO run c key everytime here after 2? if it doesn't take too long, run it every time here?
# TODO eventually put realsense in hand as well and do eye-in-hand. And multiple realsenses (maybe swap to handical or other? or do each one individually?)