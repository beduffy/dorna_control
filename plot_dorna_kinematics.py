import math

import cv2
import numpy as np
import pyrealsense2 as rs
try:
    import open3d as o3d
    from skimage.measure import find_contours
except Exception as e:
    print(e)
    print('Tried to import open3d or skimage but not installed')

from line_mesh import LineMesh
from lib.vision import get_full_pcd_from_rgbd
from lib.vision import get_camera_coordinate, create_homogenous_transformations, \
    convert_pixel_to_arm_coordinate, convert_cam_pcd_to_arm_pcd
# from lib.vision_config import class_names, class_name_to_id, pinhole_camera_intrinsic
from lib.vision_config import pinhole_camera_intrinsic

def _inch_to_mm(x):
    if x == None:
        return None
    return x * 25.4


def create_sphere_at_pos(pos, radius=4, color=[1.0, 0.0, 0.0]):
    centroid_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    centroid_sphere.compute_vertex_normals()
    centroid_sphere.paint_uniform_color(color)
    centroid_sphere.translate(pos)

    return centroid_sphere

def plot_open3d_Dorna(joint_and_link_positions, extra_geometry_elements=None):
    # todo how to make more accurate 3d model? URDF/meshes?

    print("Drawing Open3D model of arm with lines")

    # starting_x = joint_and_link_positions["starting_x"]
    # starting_y = joint_and_link_positions["starting_y"]
    # starting_z = joint_and_link_positions["starting_z"]
    x1 = joint_and_link_positions["shoulder"][0]
    y1 = joint_and_link_positions["shoulder"][1]
    z1 = joint_and_link_positions["shoulder"][2]
    x2 = joint_and_link_positions["elbow"][0]
    y2 = joint_and_link_positions["elbow"][1]
    z2 = joint_and_link_positions["elbow"][2]
    x3 = joint_and_link_positions["wrist"][0]
    y3 = joint_and_link_positions["wrist"][1]
    z3 = joint_and_link_positions["wrist"][2]
    x4 = joint_and_link_positions["toolhead"][0]
    y4 = joint_and_link_positions["toolhead"][1]
    z4 = joint_and_link_positions["toolhead"][2]

    # s_x = joint_and_link_positions["s_x"]
    # e_x = joint_and_link_positions["e_x"]
    # w_x = joint_and_link_positions["w_x"]
    # wrist_pitch_starting_side_x = joint_and_link_positions['wrist_pitch_starting_side_x']
    # wrist_pitch_starting_x = joint_and_link_positions['wrist_pitch_starting_x']
    # wrist_pitch_starting_y = joint_and_link_positions['wrist_pitch_starting_y']
    # wrist_pitch_starting_z = joint_and_link_positions['wrist_pitch_starting_z']

    # gripper model
    # gripper_back_centre_x = joint_and_link_positions.get('gripper_back_centre_x')
    # if gripper_back_centre_x:
    #     gripper_back_centre_y = joint_and_link_positions.get('gripper_back_centre_y')
    #     gripper_back_centre_z = joint_and_link_positions.get('gripper_back_centre_z')
    #     gripper_left_jaw_start_x = joint_and_link_positions.get('gripper_left_jaw_start_x')
    #     gripper_left_jaw_start_y = joint_and_link_positions.get('gripper_left_jaw_start_y')
    #     gripper_left_jaw_start_z = joint_and_link_positions.get('gripper_left_jaw_start_z')
    #     gripper_left_jaw_end_x = joint_and_link_positions.get('gripper_left_jaw_end_x')
    #     gripper_left_jaw_end_y = joint_and_link_positions.get('gripper_left_jaw_end_y')
    #     gripper_left_jaw_end_z = joint_and_link_positions.get('gripper_left_jaw_end_z')
    #     gripper_right_jaw_start_x = joint_and_link_positions.get('gripper_right_jaw_start_x')
    #     gripper_right_jaw_start_y = joint_and_link_positions.get('gripper_right_jaw_start_y')
    #     gripper_right_jaw_start_z = joint_and_link_positions.get('gripper_right_jaw_start_z')
    #     gripper_right_jaw_end_x = joint_and_link_positions.get('gripper_right_jaw_end_x')
    #     gripper_right_jaw_end_y = joint_and_link_positions.get('gripper_right_jaw_end_y')
    #     gripper_right_jaw_end_z = joint_and_link_positions.get('gripper_right_jaw_end_z')
    #     gripper_tip_x = joint_and_link_positions.get('gripper_tip_x')
    #     gripper_tip_y = joint_and_link_positions.get('gripper_tip_y')
    #     gripper_tip_z = joint_and_link_positions.get('gripper_tip_z')

    # optional IK target visualisation
    # x = joint_and_link_positions.get("x")
    # y = joint_and_link_positions.get("y")
    # z = joint_and_link_positions.get("z")
    # side_view_x_target = joint_and_link_positions.get("side_view_x_target")

    # start creating points, lines, colors and spheres to visualise model
    # points = [[starting_x, starting_y, starting_z],
    points = [[x1, y1, z1],
              [x2, y2, z2],
            #   [wrist_pitch_starting_x, wrist_pitch_starting_y, wrist_pitch_starting_z],
              [x3, y3, z3],
              [x4, y4, z4],
            #   [gripper_back_centre_x, gripper_back_centre_y, gripper_back_centre_z],
            #   [gripper_left_jaw_start_x, gripper_left_jaw_start_y, gripper_left_jaw_start_z],
            #   [gripper_right_jaw_start_x, gripper_right_jaw_start_y, gripper_right_jaw_start_z],
            #   [gripper_left_jaw_end_x, gripper_left_jaw_end_y, gripper_left_jaw_end_z],
            #   [gripper_right_jaw_end_x, gripper_right_jaw_end_y, gripper_right_jaw_end_z],
            #   [gripper_tip_x, gripper_tip_y, gripper_tip_z],
              ]

    # lines = [[0, 1], [1, 2], [2, 3], [3, 4], [5, 6], [5, 7], [6, 8], [7, 9], [4, 10]]
    lines = [[0, 1], [1, 2], [2, 3]]
    # lines = [[i, i + 1] for i, x in enumerate(points[:-1])]
    # colors = [[1, 0, 0] for i in range(len(lines))]
    colors = [[0, 0, 0], [1, 0, 0], [1, 1, 0]]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    # Create Line Mesh
    line_mesh1 = LineMesh(points, lines, colors, radius=1.5)
    line_mesh1_geoms = line_mesh1.cylinder_segments

    all_spheres = []
    for p in points:
        mesh_sphere = create_sphere_at_pos(np.array(p), color=[0.1, 0.1, 0.7], radius=4.5)
        all_spheres.append(mesh_sphere)

    list_of_geometry_elements = [line_set, *line_mesh1_geoms] + all_spheres
    if extra_geometry_elements:
        list_of_geometry_elements.extend(extra_geometry_elements)
    o3d.visualization.draw_geometries(list_of_geometry_elements)  # todo eventually nonblocking version

"""
inverse kinematics: xyz to joint
"""
def i_k(xyz, toolhead_x_length=193):
    try:
        x = xyz[0]
        y = xyz[1]
        z = xyz[2]		
        
        _delta_e = 0.001
        _bx = _inch_to_mm(3.759)  # x-distance from base to shoulder. 95.47859999999999mm Most likely.
        _bz = _inch_to_mm(8.111)  # height from ground to shoulder joint. 206.01940000000002mm confirmed
        l_shoulder_to_elbow = _inch_to_mm(8.)  # distance from shoulder to elbow. 203.2mm confirmed
        l_elbow_to_wrist = _inch_to_mm(6.)  # TODO: maybe distance from base to shoulder. NO. From elbow to wrist

        alpha = xyz[3]
        beta = xyz[4]

        alpha = math.radians(alpha)
        beta = math.radians(beta)

        # first we find the base rotation
        teta_0 = math.atan2(y, x)

        # next we assume base is not rotated and everything lives in x-z plane
        x = math.sqrt(x ** 2 + y ** 2)

        # next we update x and z based on base dimensions and hand orientation
        x -= (_bx + toolhead_x_length * math.cos(alpha))
        z -= (_bz + toolhead_x_length * math.sin(alpha))

        # at this point x and z are the summation of two vectors one from lower arm and one from upper arm of lengths l1 and l2
        # let L be the length of the overall vector
        # we can calculate the angle between l1 , l2 and L
        L = math.sqrt(x ** 2 + z ** 2)
        L = np.round(L, 13) # ???
        # not valid
        if L > (l_shoulder_to_elbow + l_elbow_to_wrist) or l_shoulder_to_elbow > (l_elbow_to_wrist + L) or l_elbow_to_wrist > (l_shoulder_to_elbow + L):  # in this case there is no solution
            print('Returning None because this shit sucks')
            # import pdb;pdb.set_trace()
            return None

        # init status
        status = 0
        if L > (l_shoulder_to_elbow + l_elbow_to_wrist) - _delta_e or l_shoulder_to_elbow > (l_elbow_to_wrist + L) - _delta_e: # in this case there is no solution
            status = 1

        teta_l1_L = math.acos((l_shoulder_to_elbow ** 2 + L ** 2 - l_elbow_to_wrist ** 2) / (2 * l_shoulder_to_elbow * L))  # l1 angle to L
        teta_L_x = math.atan2(z, x)  # L angle to x axis
        teta_1 = teta_l1_L + teta_L_x
        # note that the other solution would be to set teta_1 = teta_L_x - teta_l1_L. But for the dynamics of the robot the first solution works better.
        teta_l1_l2 = math.acos((l_shoulder_to_elbow ** 2 + l_elbow_to_wrist ** 2 - L ** 2) / (2 * l_shoulder_to_elbow * l_elbow_to_wrist))  # l1 angle to l2
        teta_2 = teta_l1_l2 - math.pi
        teta_3 = alpha - teta_1 - teta_2
        teta_4 = beta
        
        teta_0 = math.degrees(teta_0)
        teta_1 = math.degrees(teta_1)
        teta_2 = math.degrees(teta_2)
        teta_3 = math.degrees(teta_3)
        teta_4 = math.degrees(teta_4)

        return [teta_0, teta_1, teta_2, teta_3, teta_4] + xyz[5:]

    except Exception as e:
        print('Inverse kinematics failure!')
        print(e)
        return None

"""
forward kinematics: joint to xyz
"""
def f_k(joint_angles_deg, toolhead_x_length=193):
    try:
        # TODO change all variable names
        # TODO understand IK too
        # TODO would be awesome to do matplotlib pointcloud as well

        _bx = _inch_to_mm(3.759)  # x-distance from base to shoulder. 95.47859999999999mm Most likely.
        _bz = _inch_to_mm(8.111)  # height from ground to shoulder joint. 206.01940000000002mm confirmed
        l_shoulder_to_elbow = _inch_to_mm(8.)  # distance from shoulder to elbow. 203.2mm confirmed
        l_elbow_to_wrist = _inch_to_mm(6.)  # From elbow to wrist. 152.4mm

        # so in with calibrated joint angles 0, 0, 0, 0, 0, we should get 95.47859 + 203.2 + 152.4 + 193 = 644.07859

        # print('_bx, _bz, l_shoulder_to_elbow, l_elbow_to_wrist')
        # print(_bx, _bz, l_shoulder_to_elbow, l_elbow_to_wrist)

        # joint to radian
        theta = [math.radians(j) for j in joint_angles_deg]

        # first we find x, y, z assuming base rotation is zero (theta_0 = 0). Then we rotate everything
        # then we rotate the robot around z axis for theta_0

        # top-down view
        # tmp = _bx + _l1 * math.cos(theta[1]) + _l2 * math.cos(theta[1] + theta[2]) + toolhead_x_length * math.cos(theta[1] + theta[2] + theta[3])

        # TODO we're missing distance from elbow to wrist?!?!?
        shoulder_side_x = _bx
        # shoulder_side_x = _bx + l_shoulder_to_elbow * math.cos(theta[1])
        elbow_side_x = shoulder_side_x + l_shoulder_to_elbow * math.cos(theta[1])
        # elbow_side_x = shoulder_side_x + l_elbow_to_wrist * math.cos(theta[1] + theta[2])
        wrist_side_x = elbow_side_x + l_elbow_to_wrist * math.cos(theta[1] + theta[2])
        toolhead_side_x = wrist_side_x + toolhead_x_length * math.cos(theta[1] + theta[2] + theta[3])
        # print('All side x: {} {} {}'.format(shoulder_side_x, elbow_side_x, toolhead_side_x))
        
        # rotate around z-axis
        shoulder_x = shoulder_side_x * math.cos(theta[0])
        shoulder_y = shoulder_side_x * math.sin(theta[0])
        elbow_x = elbow_side_x * math.cos(theta[0])
        elbow_y = elbow_side_x * math.sin(theta[0])
        wrist_x = wrist_side_x * math.cos(theta[0])
        wrist_y = wrist_side_x * math.sin(theta[0])
        toolhead_x = toolhead_side_x * math.cos(theta[0])
        toolhead_y = toolhead_side_x * math.sin(theta[0])
        # print('shoulder_x: {} shoulder_y: {} elbow_x: {} elbow_y: {} toolhead_x: {} toolhead_y: {}'.format(shoulder_x, shoulder_y, elbow_x, elbow_y, toolhead_x, toolhead_y))

        # side-view
        # z = _bz + _l1 * math.sin(theta[1]) + _l2 * math.sin(theta[1] + theta[2]) + toolhead_x_length * math.sin(theta[1] + theta[2] + theta[3])
        # shoulder_z = _bz + l_shoulder_to_elbow * math.sin(theta[1])
        shoulder_z = _bz
        elbow_z = shoulder_z + l_shoulder_to_elbow * math.sin(theta[1])
        # elbow_z = shoulder_z + l_elbow_to_wrist * math.sin(theta[1] + theta[2])
        wrist_z = elbow_z + l_elbow_to_wrist * math.sin(theta[1] + theta[2])
        toolhead_z = wrist_z + toolhead_x_length * math.sin(theta[1] + theta[2] + theta[3])

        alpha = theta[1] + theta[2] + theta[3]
        beta = theta[4]  # just wrist roll right?
        alpha = math.degrees(alpha)
        beta = math.degrees(beta)

        xyz_toolhead_pos = [toolhead_x, toolhead_y, toolhead_z]
        full_toolhead_fk = xyz_toolhead_pos + [alpha, beta] + joint_angles_deg[5:]
        xyz_positions_of_all_joints = {'shoulder': [shoulder_x, shoulder_y, shoulder_z], 
                                       'elbow': [elbow_x, elbow_y, elbow_z], 
                                       'wrist': [wrist_x, wrist_y, wrist_z], 
                                       'toolhead': xyz_toolhead_pos}

        # TODO do I need to do this? What about it makes it in inches? ahh bx and all?
        # if _config["unit"]["length"] == "mm":
        #     _rtn = [_inch_to_mm(c) for c in _rtn]

        return full_toolhead_fk, xyz_positions_of_all_joints

    except Exception as e:
        print(e)
        return None

def check_ground_collision(xyz_positions_of_all_joints, ground_z_height=5):
    for joint in ['shoulder', 'elbow', 'wrist', 'toolhead']:
        joint_z = xyz_positions_of_all_joints[joint][2]
        if joint_z < ground_z_height:
            print('Ground collision for {} joint at z: {}'.format(joint, joint_z))
            return True

    return False

if __name__ == '__main__':
    # base, shoulder, elbow, wrist pitch, wrist roll
    full_toolhead_fk, xyz_positions_of_all_joints = f_k([0.0, 0.0, 0.0, 0.0, 0.0])
    # full_toolhead_fk, xyz_positions_of_all_joints = f_k([0.0, 0.0, 0.0, 30.0, 0.0])
    # full_toolhead_fk, xyz_positions_of_all_joints = f_k([0.0, 0.0, 0.0, 45, 0.0])
    # full_toolhead_fk, xyz_positions_of_all_joints = f_k([0.0, 0.0, 0.0, 60, 0.0])
    # full_toolhead_fk, xyz_positions_of_all_joints = f_k([45, 0.0, 0.0, 0.0, 0.0])
    # full_toolhead_fk, xyz_positions_of_all_joints = f_k([45, 45, 45, 0.0, 0.0])
    # full_toolhead_fk, xyz_positions_of_all_joints = f_k([45, 45, 45, 60, 70])
    print('full_toolhead_fk')
    print(full_toolhead_fk)
    print('xyz_positions_of_all_joints')
    print(xyz_positions_of_all_joints)

    # TODO tool z offset (kinda z) add to FK and IK and then even completely subvert dornas lower level commands and always come from joint angles
    import pdb;pdb.set_trace()

    # joint_angles = i_k(full_toolhead_fk)
    joint_angles = i_k([483.07853333332895, 0.0, 206.01941089350998, -9.999999988651159e-06, 0.0])
    full_toolhead_fk, xyz_positions_of_all_joints = f_k(joint_angles)

    # origin_frame_size = 0.01
    # origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    #     size=origin_frame_size, origin=[0.0, 0.0, 0.0])
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=25.0, origin=[0.0, 0.0, 0.0])

    # origin_frame_size = 0.01
    # origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    #     size=origin_frame_size, origin=[0.0, 0.0, 0.0])
    coordinate_frame_shoulder_height = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=25.0, origin=[0.0, 0.0, 206.01940000000002])

    # plot_open3d_Dorna(xyz_positions_of_all_joints, extra_geometry_elements=[coordinate_frame, coordinate_frame_shoulder_height])

    # user_input = input('Want to see pointcoud overlayed?\n')
    user_input = 'Y'
    if user_input.lower() == 'y':
        try:
            cam2arm = np.loadtxt('data/latest_aruco_cam2arm.txt', delimiter=' ')

            # realsense stuff 
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
            profile = pipeline.start(config)
            spatial = rs.spatial_filter()
            spatial.set_option(rs.option.holes_fill, 3)
            depth_sensor = profile.get_device().first_depth_sensor()
            # Getting the depth sensor's depth scale (see rs-align example for explanation)
            depth_scale = depth_sensor.get_depth_scale()
            print('DEPTH SCALE: ', depth_scale)
            align = rs.align(rs.stream.color)

            print('Running 20 frames to wait for auto-exposure')
            for i in range(20):
                frames = pipeline.wait_for_frames()
                aligned_frames = align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()

            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            # hole filling
            depth_frame = spatial.process(depth_frame)
            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
            color_intrin = color_frame.profile.as_video_stream_profile().intrinsics

            camera_depth_img = np.asanyarray(depth_frame.get_data())
            camera_color_img = np.asanyarray(color_frame.get_data())
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(camera_depth_img, alpha=0.03),
                cv2.COLORMAP_JET)

            camera_color_img = cv2.cvtColor(camera_color_img, cv2.COLOR_RGB2BGR)
            images = np.hstack((camera_color_img, depth_colormap))
            camera_color_img = cv2.cvtColor(camera_color_img, cv2.COLOR_BGR2RGB)  # for open3D
            cv2.imshow("image", images)
            k = cv2.waitKey(5)

            # dont need dorna import here but do need realsense
            cam_pcd = get_full_pcd_from_rgbd(camera_color_img, camera_depth_img,
                                            pinhole_camera_intrinsic, visualise=False)

            # only plot open3D arm when arm is in position. Otherwise if non-blocking
            # TODO remove q1 and z offset now that aruco and solvePnP finds correct transform
            # TODO never understood how this relates, if origin looks good but cam2arm is bad?
            # transforming rgbd pointcloud using bad cam2arm means what? . What is the thing changing again?

            full_arm_pcd, full_pcd_numpy = convert_cam_pcd_to_arm_pcd(cam_pcd, cam2arm, 0.0)

            plot_open3d_Dorna(xyz_positions_of_all_joints, extra_geometry_elements=[full_arm_pcd, coordinate_frame, coordinate_frame_shoulder_height])
        except Exception as e:
            print('Exception in pointcloud comparison')
            print(e)