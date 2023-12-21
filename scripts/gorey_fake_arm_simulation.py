from __future__ import print_function
import sys
import traceback
from glob import glob
import math
from collections import deque

try:
    import open3d as o3d
    from skimage.measure import find_contours
except Exception as e:
    print(e)
    print('Tried to import open3d or skimage but not installed')
import pyrealsense2 as rs
from cv2 import aruco
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib as mpl
from scipy import optimize
from scipy.spatial.transform import Rotation

from lib.vision import euler_yzx_to_axis_angle, rotationMatrixToEulerAngles, create_homogenous_transformations, get_inverse_homogenous_transform

np.set_printoptions(suppress = True)

# TODO read
# OPENCV official unit test is essentially doing what I'm doing here: https://github.com/opencv/opencv/blob/b5a9a6793b3622b01fe6c7b025f85074fec99491/modules/calib3d/test/test_calibration_hand_eye.cpp#L145
# yeah when they run eye to hand they run:
# calibrateHandEye(R_base2gripper, t_base2gripper, R_target2cam, t_target2cam, R_cam2base_est, t_cam2base_est, methods[idx]);
# otherwise they do 
# calibrateHandEye(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, R_cam2gripper_est, t_cam2gripper_est, methods[idx]);
# I could just reimplement it all? but then im learning less

# calibrateRobotWorldHandEye whats the difference with TODO : https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga41b1a8dd70eae371eba707d101729c36

# https://forum.opencv.org/t/eye-to-hand-calibration/5690/2
# TODO be very careful going from euler to rotation matrix! even numerical problems. 
# TODO nice functions in there TODO USE?
# TODO more functions in there1
def matrix_from_rtvec(rvec, tvec):
    (R, jac) = cv2.Rodrigues(rvec) # ignore the jacobian
    M = np.eye(4)
    M[0:3, 0:3] = R
    M[0:3, 3] = tvec.squeeze() # 1-D vector, row vector, column vector, whatever
    return M

def rtvec_from_matrix(M):
    (rvec, jac) = cv2.Rodrigues(M[0:3, 0:3]) # ignore the jacobian
    tvec = M[0:3, 3]
    assert M[3] == [0,0,0,1], M # sanity check
    return (rvec, tvec)


size = 0.1
origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0.0, 0.0, 0.0])

# TODO should be transforming all from origin. ok done below. 

gripper_euler_angles = [
    [0.1, 0, 0],
    [0.2, 0, 0],
    [0.3, 0, 0],
]
# if I only changed rolls then we get nice expected effects
gripper_translations = [
    [0.3, 0.0, 0.0],
    [0.4, 0.0, 0.0],
    [0.5, 0.0, 0.0],
]

# # trying more manually specified random  poses
# gripper_euler_angles = [
#     [0.1, 0, 0.5],
#     [0.2, 0.1, 0.6],
#     [0.3, 0.2, 0.9],
#     [0.3, 0.5, 0.6],
#     [0.4, 0.2, 0.7],
#     [0.6, 0.7, 0.9],
# ]
# # if I only changed rolls then we get nice expected effects
# gripper_translations = [
#     [0.3, 0.0, 0.0],
#     [0.4, 0.5, 0.6],
#     [0.5, 0.7, 0.2],
#     [0.5, 0.4, 0.2],
#     [0.7, 0.1, 0.0],
#     [0.5, 0.3, 0.4],
# ]

# random poses
# num_poses = 10
# gripper_euler_angles = []
# gripper_translations = []
# size_of_random_vecs = (3,)
# for i in range(num_poses):
#     gripper_euler_angles.append(np.random.uniform(size=size_of_random_vecs).tolist())
#     gripper_translations.append(np.random.uniform(size=size_of_random_vecs).tolist())

# Calculate all gripper frames and transformations
coordinate_frames_o3d = []
all_transformations_to_gripper = []
all_inverse_transformations_to_gripper = []
for eul_angles, translate in zip(gripper_euler_angles, gripper_translations):
    transformation = np.identity(4)
    # transformation[:3, :3] = np.identity(3)  # no rotation
    r = Rotation.from_euler("xyz", eul_angles, degrees=False)  # roll pitch yaw = xyz  
    new_rotation_matrix = r.as_matrix()
    transformation[:3, :3] = new_rotation_matrix
    transformation[0, 3] = translate[0]
    transformation[1, 3] = translate[1]
    transformation[2, 3] = translate[2]
    transformation_inverse = get_inverse_homogenous_transform(transformation)
    all_inverse_transformations_to_gripper.append(transformation_inverse)

    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0.0, 0.0, 0.0])
    coordinate_frame.transform(transformation)
    all_transformations_to_gripper.append(transformation)

    coordinate_frames_o3d.append(coordinate_frame)


####### camera transform and frame

# rvec = np.array([1.0, 1.0, 1.0])
# z, x, y, angle = euler_yzx_to_axis_angle(np.pi / 2, 0, 0)  # pointing left in yaw, 45 deg. red.
# z, x, y, angle = euler_yzx_to_axis_angle(0, np.pi / 2, 0)  # pointing up in pitch 45 degrees. 
# z, x, y, angle = euler_yzx_to_axis_angle(0, 0, np.pi / 2)  # rolling right 45 degrees.

# I want camera to the right, pointing to the left 45 degrees left in yaw and 45 degrees down
z, x, y, angle = euler_yzx_to_axis_angle(np.pi / 2, -np.pi / 2, 0)  # TODO shouldn't pi / 2 be 90 degrees and not 45??!?!!? yep


 
rvec = np.array([x, y, z])













# TODO wait i don't even need field of view since no camera in these experiments!!!!! It's just math and optimisation. 
# TODO So i could chose to not rotate camera for now just to verify everything! 
# rvec = np.array([0.0, 0.0, 0.0])
camera_position_rel_to_origin = np.array([0.0, 0.3, 0.4])
tvec = camera_position_rel_to_origin   # TODO why does 0.3 bring it down but 
cam2_arm, arm2_cam, R_tc, R_ct, pos_camera = create_homogenous_transformations(tvec, rvec)
camera_inverse_transformation = np.array(cam2_arm)
camera_transformation = np.array(arm2_cam)  # makes more intuitive sense than cam2arm?, 0.3 tvec brings bigger coordinate frame forwards.?
# camera_transformation brings points from origin to camera frame but expressed and relative to origin. arm2cam kinda means origin points in arm frame to weird rotated camera frame
# np.dot(camera_transformation, np.array([0.0, 0.0, 0.0, 1.0]))
# array([0. , 0.3, 0.4, 1. ])
# np.dot(camera_inverse_transformation, np.array([0.0, 0.0, 0.0, 1.0]))
# array([ 0.32475319,  0.04756196, -0.37719123,  1.        ])   # unintuitive due to rotation but makes sense
# np.dot(camera_transformation, camera_inverse_transformation) is identity.

camera_coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size * 1.3, origin=[0.0, 0.0, 0.0])
camera_coordinate_frame.transform(camera_transformation)


# Create new transformation from camera to origin_frame, coord_frame_1, coord_frame_2, coord_frame_3 and prove it is fine by plotting new coord frames
# TODO how? well i could just dot product. np.dot(camera_inverse_transformation, transformations[0]). 
# TODO how to visualise though? I think it's fine

all_transformation_target2cam = []  # transforms a point expressed in the target frame to the camera frame ( cTt)
all_transformation_cam2target = []  # therefore transforms a point in the camera frame to the target frame
for transformation in all_transformations_to_gripper:
    # Composing from camera to origin (cam2arm) and then normal transformation. 
    transformation_from_camera_to_coord_frame = np.dot(camera_inverse_transformation, transformation)
    # np.dot(transformation_from_camera_to_coord_frame_1, np.array([0.0, 0.0, 0.0, 1.0]))  # unituitive numbers due to rotation
    # the above takes a point in camera frame to target frame
    all_transformation_cam2target.append(transformation_from_camera_to_coord_frame)

    inverse_transformation_from_camera_to_coord_frame = get_inverse_homogenous_transform(transformation_from_camera_to_coord_frame)
    print('inverse tranforming origin (from coord frame to camera):', np.dot(inverse_transformation_from_camera_to_coord_frame, np.array([0.0, 0.0, 0.0, 1.0])))
    # array([-0.3,  0.3,  0.4,  1. ])  # correct! for first frame. Yes! 
    # the above takes a point in target frame (origin 0, 0, 0) to camera frame by going back -0.3, right 0.3 and up in z 0.4

    # all_transformation_target2cam.append(inverse_transformation_from_camera_to_coord_frame)
    
    
    
    
    # TODO WTF?
    
    
    
    
    all_transformation_target2cam.append(transformation_from_camera_to_coord_frame)

# extract R_target2cam and t_target2cam
R_target2cam = []
t_target2cam = []
for transformation in all_transformation_target2cam:
    # TODO below is a source of error, double check
    R_target2cam.append(transformation[:3, :3])
    t_target2cam.append(transformation[:3, 3])


R_gripper2base = []
t_gripper2base = []
# TODO needs to be inverted? only for eye in hand
for transformation in all_transformations_to_gripper:
    # TODO below is a source of error, double check
    R_gripper2base.append(transformation[:3, :3])
    t_gripper2base.append(transformation[:3, 3])

R_base2gripper = []
t_base2gripper = []
for transformation in all_inverse_transformations_to_gripper:  # inversely correctly bring [0,0,0] -> [-0.3, 0, 0] to go from gripper to base
    # TODO below is a source of error, double check
    R_base2gripper.append(transformation[:3, :3])
    t_base2gripper.append(transformation[:3, 3])



# TODO now sure how to plot and prove my point since everything is origin relative?
# Here all around im assuming target and gripper are same thing
# origin_frame_transformed_from_camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size * 1.3, origin=[0.0, 0.0, 0.0])
# camera_inverse_transformation


# TODO study function deeply: simulateDataEyeToHand
# R_gripper2base	Rotation part extracted from the homogeneous matrix that transforms a point expressed in the gripper frame to the robot base frame ( bTg). This is a vector (vector<Mat>) that contains the rotation, (3x3) rotation matrices or (3x1) rotation vectors, for all the transformations from gripper frame to robot base frame.
# t_gripper2base	Translation part extracted from the homogeneous matrix that transforms a point expressed in the gripper frame to the robot base frame ( bTg). This is a vector (vector<Mat>) that contains the (3x1) translation vectors for all the transformations from gripper frame to robot base frame.
# R_target2cam	Rotation part extracted from the homogeneous matrix that transforms a point expressed in the target frame to the camera frame ( cTt). This is a vector (vector<Mat>) that contains the rotation, (3x3) rotation matrices or (3x1) rotation vectors, for all the transformations from calibration target frame to camera frame.
# t_target2cam	# TODO typo in docs here Rotation part extracted from the homogeneous matrix that transforms a point expressed in the target frame to the camera frame ( cTt). This is a vector (vector<Mat>) that contains the (3x1) translation vectors for all the transformations from calibration target frame to camera frame.

# output should be  ->	R_cam2gripper, t_cam2gripper  for eye in hand
# but for eye to hand: R_cam2base_est, t_cam2base_est from unit tests in opencv

# yeah when they run eye to hand they run:
# calibrateHandEye(R_base2gripper, t_base2gripper, R_target2cam, t_target2cam, R_cam2base_est, t_cam2base_est, methods[idx]);
# otherwise they do for eye in hand
# calibrateHandEye(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, R_cam2gripper_est, t_cam2gripper_est, methods[idx]);

method = cv2.CALIB_HAND_EYE_TSAI  # default
# method = cv2.CALIB_HAND_EYE_DANIILIDIS

# seems to always be close to 0. So at least the below has better numbers
# R, T = cv2.calibrateHandEye(R_gripper2base, t_gripper2base,  # what should be done for eye-in-hand
#                             R_target2cam, t_target2cam, method=method)
# print('\nR and T for eye-in-hand')
# print(R)
# print(T)
R, T = cv2.calibrateHandEye(R_base2gripper, t_base2gripper,
                            R_target2cam, t_target2cam, method=method)
print('\nR and T for eye-to-hand')
print(R)
print(T)

# We know ground truth
print('Ground truth (camera_transform)')
print(camera_transformation)
print('Ground truth (inverse camera_transform)')
print(camera_inverse_transformation)


arm_position_coord_frames = coordinate_frames_o3d
list_of_geometry_elements = [origin_frame, camera_coordinate_frame] + arm_position_coord_frames
# list_of_geometry_elements = [origin_frame_transformed_from_camera_frame, camera_coordinate_frame] + arm_position_coord_frames
o3d.visualization.draw_geometries(list_of_geometry_elements)





sys.exit()

#--- 180 deg rotation matrix around the x axis
# R_flip       = np.zeros((3, 3), dtype=np.float32)
# R_flip[0, 0] =  1.0
# R_flip[1, 1] = -1.0
# R_flip[2, 2] = -1.0

# # TODO I need to understand conversion between all rotation types better
# # TODO why are both below the same?
# # -- Get the attitude in terms of euler 321 (Needs to be flipped first)
# roll_marker, pitch_marker, yaw_marker = rotationMatrixToEulerAngles(R_flip * R_tc)
# print('roll pitch yaw marker: ', roll_marker, pitch_marker, yaw_marker)
# # -- Get the attitude of the camera respect to the frame
# roll_camera, pitch_camera, yaw_camera = rotationMatrixToEulerAngles(R_flip * R_ct)  # todo no flip needed?
# print('roll pitch yaw camera: ', roll_camera, pitch_camera, yaw_camera)




################# hand eye


gripper_t = np.array([[0.3, 0.0, 0.0], [0.4, 0.0, 0.0], [0.5, 0.0, 0.0], [0.6, 0.0, 0.0]]) 
  
# hand_rotations = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 
#                             [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])  # non parallel motions make output be identity. ahhh
# hand_rotations = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.3], 
#                             [0.0, 0.0, 0.1], [0.0, 0.2, 0.0]]) 

# only rotation in roll since that won't ruin my simple translation. But they're still parallel? well only the x axis?
hand_rotations = []
rotations_in_roll = [0.1, -0.1, 0.2, 0]
for i in range(gripper_t.shape[0]):
    z, x, y, angle = euler_yzx_to_axis_angle(0, 0, rotations_in_roll[i])
    hand_rotations.append(np.array([x, y, z]))
hand_rotations = np.array(hand_rotations)

# TODO study spatial algebra. Find transformation between any two coordinate frames
# TODO nah, I create the transformation, so just create 3-6+ non-stupid transformations (translation is always easy), rotation matrices I'm finding above
# TODO well continuing. Easy to find all transformation from origin/arm base to gripper. a bit harder to find from camera to gripper. Can I just add both? probably yes! 

# # rotation matrix between the target and camera 
# R_target2cam = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [ 
#                         0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]) 
  
# # translation vector between the target and camera 
# t_target2cam = np.array([0.0, 0.0, 0.0, 0.0]) 

# camera_position_rel_to_origin = np.array([0.0, 0.3, 0.4])
t_target2cam = []
R_target2cam = []
print('camera_position_rel_to_origin: ', camera_position_rel_to_origin)
for idx in range(gripper_t.shape[0]):
    # print(idx, gripper_t[idx])
    diff_in_translation = camera_position_rel_to_origin - gripper_t[idx]
    # diff_in_translation = gripper_t[idx] - camera_position_rel_to_origin
    print(idx, gripper_t[idx])
    print(diff_in_translation)
    t_target2cam.append(diff_in_translation)
    R_target2cam.append(R_tc)

    # 0 [0.3 0.  0. ]
    # [-0.3  0.3  0.4]  # seems reasonable. We need to subtract 0.3 in x, move 0.3 to the right and 0.4 up in z
    # 1 [0.4 0.  0. ]
    # [-0.4  0.3  0.4]
    # 2 [0.5 0.  0. ]
    # [-0.5  0.3  0.4]


# https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#gaebfc1c9f7434196a374c382abf43439b
# InputArrayOfArrays 	R_gripper2base,
# InputArrayOfArrays 	t_gripper2base,
# InputArrayOfArrays 	R_target2cam,
# InputArrayOfArrays 	t_target2cam,

# R_gripper2base	Rotation part extracted from the homogeneous matrix that transforms a point expressed in the gripper frame to the robot base frame ( bTg). This is a vector (vector<Mat>) that contains the rotation, (3x3) rotation matrices or (3x1) rotation vectors, for all the transformations from gripper frame to robot base frame.
# t_gripper2base	Translation part extracted from the homogeneous matrix that transforms a point expressed in the gripper frame to the robot base frame ( bTg). This is a vector (vector<Mat>) that contains the (3x1) translation vectors for all the transformations from gripper frame to robot base frame.
# R_target2cam	Rotation part extracted from the homogeneous matrix that transforms a point expressed in the target frame to the camera frame ( cTt). This is a vector (vector<Mat>) that contains the rotation, (3x3) rotation matrices or (3x1) rotation vectors, for all the transformations from calibration target frame to camera frame.
# t_target2cam	Rotation part extracted from the homogeneous matrix that transforms a point expressed in the target frame to the camera frame ( cTt). This is a vector (vector<Mat>) that contains the (3x1) translation vectors for all the transformations from calibration target frame to camera frame.

# A minimum of 2 motions with non parallel rotation axes are necessary to determine the hand-eye transformation. 
# So at least 3 different poses are required, but it is strongly recommended to use many more poses.

# https://www.geeksforgeeks.org/calibratehandeye-python-opencv/

# transformation matrix 
# T, R = cv2.calibrateHandEye(gripper_t, eye_coords,  # TODO but docs say order of inputs is: Well first two could be both rotation and translation in 3vecs???
#                             R_target2cam, t_target2cam) 
# R, T = cv2.calibrateHandEye(hand_rotations, gripper_t, 
#                             hand_rotations, gripper_t)  # creates identity, which is good
R, T = cv2.calibrateHandEye(hand_rotations, gripper_t,   # probably done unless inverse TODO prove it. if gripper is 0.3 forward do I need to subtract 0.3 to get to robot base frame? but gripper frame is 0.3 forward
                            R_target2cam, t_target2cam)
print('\nR and T')
print(R)
print(T)

estimated_cam2_arm_transform = np.identity(4)
estimated_cam2_arm_transform[:3, :3] = R
estimated_cam2_arm_transform[0, 3] = T[0]
estimated_cam2_arm_transform[1, 3] = T[1]
estimated_cam2_arm_transform[2, 3] = T[2]

estimated_camera_coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size * 1.5, origin=[0.0, 0.0, 0.0])
estimated_camera_coordinate_frame.transform(estimated_cam2_arm_transform)


# arm_position_coord_frames = [coordinate_frame_1, coordinate_frame_2, coordinate_frame_3]
arm_position_coord_frames = coordinate_frames_o3d
list_of_geometry_elements = [origin_frame, camera_coordinate_frame, estimated_camera_coordinate_frame] + arm_position_coord_frames
o3d.visualization.draw_geometries(list_of_geometry_elements)


# TODO the big goal here would be to:
# 1. Using the 3 transformations from origin to the 3 coord frames of "fake arm" and then # TODO calculate simple. Identity rotation mat and translation 0.3, 0.4, 0.5
# 2. 3 transformations from camera to each of these coord frames. TODO how do I find these transformations. Translation is easy. Rotation is same as R_tc or R_ct? Can always prove I have things correct
# 3. Use opencv handeye function and then plot output transformation? 

'''
Can i do all of this in simulation without even using realsense? 
E.g. Put all objects and ground truth into simulation, visualise, then compute. 
Use Drake or Open3D? Well putting aruco into these is hard. But I don’t need to? I have all the transforms. I don’t even need to visualise depth cameras.
'''