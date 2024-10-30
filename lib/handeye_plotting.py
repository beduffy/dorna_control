import open3d as o3d

from lib.dorna_kinematics import i_k, f_k
from lib.open3d_plot_dorna import plot_open3d_Dorna
from lib.vision import get_full_pcd_from_rgbd
from lib.vision import get_camera_coordinate, create_homogenous_transformations, convert_pixel_to_arm_coordinate, convert_cam_pcd_to_arm_pcd
from lib.vision_config import pinhole_camera_intrinsic, camera_matrix, dist_coeffs


def plot_all_handeye_data(handeye_data_dict, eye_in_hand=False):
    all_gripper2base_transforms = handeye_data_dict['all_gripper2base_transforms']
    all_target2cam_transforms = handeye_data_dict['all_target2cam_transforms']
    saved_cam2arm = handeye_data_dict['saved_cam2arm']
    all_joint_angles = handeye_data_dict['all_joint_angles']
    color_images = handeye_data_dict['color_images']
    depth_images = handeye_data_dict['depth_images']

    saved_cam2arm = handeye_data_dict['saved_cam2arm']  # assuming handeye_calibrate_opencv has been called

    # use color and depth images to create point cloud from first color + depth pair
    camera_color_img = color_images[0]
    camera_depth_img = depth_images[0]
    cam_pcd_first_image_pair = get_full_pcd_from_rgbd(camera_color_img, camera_depth_img, pinhole_camera_intrinsic, visualise=False)
    # full_arm_pcd, full_pcd_numpy = convert_cam_pcd_to_arm_pcd(cam_pcd_first_image_pair, saved_cam2arm, in_milimetres=False)

    plot_one_arm_gripper_camera_frame_eye_in_hand(all_gripper2base_transforms, all_joint_angles, gripper2cam=saved_cam2arm)
    
    plot_arm_gripper_frames(all_gripper2base_transforms, all_joint_angles, plot_camera_on_gripper_if_eye_in_hand=eye_in_hand, gripper2cam=saved_cam2arm)

    plot_aruco_frames_in_camera_frame(all_target2cam_transforms, cam_pcd_first_image_pair)

    plot_blah(handeye_data_dict, cam_pcd_first_image_pair, saved_cam2arm)


def plot_blah(handeye_data_dict, cam_pcd_first_image_pair, saved_cam2arm):
    # TODO name function to what I want it to do. TODO eye in hand vs eye to hand
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
        # print('full_toolhead_fk (in metres): ', full_toolhead_fk)

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



def plot_one_arm_gripper_camera_frame_eye_in_hand(all_gripper2base_transforms, all_joint_angles, gripper2cam):
    origin_arm_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0.0, 0.0, 0.0])

    sphere_size = 0.01
    geometry_to_plot = []
    geometry_to_plot.append(origin_arm_frame)
    # TODO wrong, base2gripper, but inverse?!??!?!
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

        # TODO wrong
        combined_transform_from_arm_to_gripper_to_camera = gripper_transform @ gripper2cam
        # TODO verify again and add sphere
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
        # print('full_toolhead_fk (in metres): ', full_toolhead_fk)

        arm_plot_geometry = plot_open3d_Dorna(xyz_positions_of_all_joints, 
                          extra_geometry_elements=[coordinate_frame_shoulder_height_arm_frame],
                          do_plot=False)

        geometry_to_plot.extend(arm_plot_geometry)

    print('Visualising line mesh arms + gripper frames + arm origin frame in arm frame')
    o3d.visualization.draw_geometries(geometry_to_plot)

