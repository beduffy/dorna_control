import open3d as o3d
import numpy as np

# from line_mesh import LineMesh
from lib.line_mesh import LineMesh


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

    # TODO separate gripper into a separate function and class. 
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

    coordinate_frame_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    coordinate_frame_mesh.scale(100, center=(0, 0, 0))
    list_of_geometry_elements = [line_set, *line_mesh1_geoms] + all_spheres + [coordinate_frame_mesh]
    if extra_geometry_elements:
        list_of_geometry_elements.extend(extra_geometry_elements)
    print(list_of_geometry_elements)
    o3d.visualization.draw_geometries(list_of_geometry_elements)  # todo eventually nonblocking version

if  __name__ == '__main__':
    from dorna_kinematics import f_k
    full_toolhead_fk, xyz_positions_of_all_joints = f_k([0, 0, 30, 45, 0])
    print(xyz_positions_of_all_joints)
    
    plot_open3d_Dorna(xyz_positions_of_all_joints)