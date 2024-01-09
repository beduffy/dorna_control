import math

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib import patches, lines
from matplotlib.patches import Polygon
import numpy as np
import open3d as o3d
from skimage.measure import find_contours

from line_mesh import LineMesh
# from lib.utilities.line_mesh import LineMesh
# from kinematics import dist  # circular import
# from lib.utilities.extra import random_colors
from extra import random_colors


def create_sphere_at_pos(pos, radius=4, color=[1.0, 0.0, 0.0]):
    centroid_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    centroid_sphere.compute_vertex_normals()
    centroid_sphere.paint_uniform_color(color)
    centroid_sphere.translate(pos)

    return centroid_sphere


def plot_open3d_ROT3U(joint_and_link_positions, extra_geometry_elements=None):
    # todo how to make more accurate 3d model? URDF/meshes?

    print("Drawing Open3D model of arm with lines")

    starting_x = joint_and_link_positions["starting_x"]
    starting_y = joint_and_link_positions["starting_y"]
    starting_z = joint_and_link_positions["starting_z"]
    x1 = joint_and_link_positions["x1"]
    y1 = joint_and_link_positions["y1"]
    z1 = joint_and_link_positions["z1"]
    x2 = joint_and_link_positions["x2"]
    y2 = joint_and_link_positions["y2"]
    z2 = joint_and_link_positions["z2"]
    x3 = joint_and_link_positions["x3"]
    y3 = joint_and_link_positions["y3"]

    z3 = joint_and_link_positions["z3"]
    s_x = joint_and_link_positions["s_x"]
    e_x = joint_and_link_positions["e_x"]
    w_x = joint_and_link_positions["w_x"]
    wrist_pitch_starting_side_x = joint_and_link_positions['wrist_pitch_starting_side_x']
    wrist_pitch_starting_x = joint_and_link_positions['wrist_pitch_starting_x']
    wrist_pitch_starting_y = joint_and_link_positions['wrist_pitch_starting_y']
    wrist_pitch_starting_z = joint_and_link_positions['wrist_pitch_starting_z']

    # gripper model
    gripper_back_centre_x = joint_and_link_positions.get('gripper_back_centre_x')
    if gripper_back_centre_x:
        gripper_back_centre_y = joint_and_link_positions.get('gripper_back_centre_y')
        gripper_back_centre_z = joint_and_link_positions.get('gripper_back_centre_z')
        gripper_left_jaw_start_x = joint_and_link_positions.get('gripper_left_jaw_start_x')
        gripper_left_jaw_start_y = joint_and_link_positions.get('gripper_left_jaw_start_y')
        gripper_left_jaw_start_z = joint_and_link_positions.get('gripper_left_jaw_start_z')
        gripper_left_jaw_end_x = joint_and_link_positions.get('gripper_left_jaw_end_x')
        gripper_left_jaw_end_y = joint_and_link_positions.get('gripper_left_jaw_end_y')
        gripper_left_jaw_end_z = joint_and_link_positions.get('gripper_left_jaw_end_z')
        gripper_right_jaw_start_x = joint_and_link_positions.get('gripper_right_jaw_start_x')
        gripper_right_jaw_start_y = joint_and_link_positions.get('gripper_right_jaw_start_y')
        gripper_right_jaw_start_z = joint_and_link_positions.get('gripper_right_jaw_start_z')
        gripper_right_jaw_end_x = joint_and_link_positions.get('gripper_right_jaw_end_x')
        gripper_right_jaw_end_y = joint_and_link_positions.get('gripper_right_jaw_end_y')
        gripper_right_jaw_end_z = joint_and_link_positions.get('gripper_right_jaw_end_z')
        gripper_tip_x = joint_and_link_positions.get('gripper_tip_x')
        gripper_tip_y = joint_and_link_positions.get('gripper_tip_y')
        gripper_tip_z = joint_and_link_positions.get('gripper_tip_z')

    # optional IK target visualisation
    x = joint_and_link_positions.get("x")
    y = joint_and_link_positions.get("y")
    z = joint_and_link_positions.get("z")
    side_view_x_target = joint_and_link_positions.get("side_view_x_target")

    # start creating points, lines, colors and spheres to visualise model
    points = [[starting_x, starting_y, starting_z],
              [x1, y1, z1],
              [x2, y2, z2],
              [wrist_pitch_starting_x, wrist_pitch_starting_y, wrist_pitch_starting_z],
              [x3, y3, z3],
              [gripper_back_centre_x, gripper_back_centre_y, gripper_back_centre_z],
              [gripper_left_jaw_start_x, gripper_left_jaw_start_y, gripper_left_jaw_start_z],
              [gripper_right_jaw_start_x, gripper_right_jaw_start_y, gripper_right_jaw_start_z],
              [gripper_left_jaw_end_x, gripper_left_jaw_end_y, gripper_left_jaw_end_z],
              [gripper_right_jaw_end_x, gripper_right_jaw_end_y, gripper_right_jaw_end_z],
              [gripper_tip_x, gripper_tip_y, gripper_tip_z],
              ]

    lines = [[0, 1], [1, 2], [2, 3], [3, 4], [5, 6], [5, 7], [6, 8], [7, 9], [4, 10]]
    # lines = [[i, i + 1] for i, x in enumerate(points[:-1])]
    # colors = [[1, 0, 0] for i in range(len(lines))]
    colors = [[0, 0, 0], [1, 0, 0], [0, 1, 1], [0, 1, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0],
              [0, 1, 0], [1, 1, 0]]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    # Create Line Mesh
    line_mesh1 = LineMesh(points, lines, colors, radius=1.5)
    line_mesh1_geoms = line_mesh1.cylinder_segments

    all_spheres = []
    for p in points:
        mesh_sphere = create_sphere_at_pos(np.array(p), color=[0.1, 0.1, 0.7], radius=3.0)
        all_spheres.append(mesh_sphere)

    list_of_geometry_elements = [line_set, *line_mesh1_geoms] + all_spheres
    if extra_geometry_elements:
        list_of_geometry_elements.extend(extra_geometry_elements)
    o3d.visualization.draw_geometries(list_of_geometry_elements)  # todo eventually nonblocking version


def mypause(interval):
    backend = plt.rcParams['backend']
    if backend in matplotlib.rcsetup.interactive_bk:
        figManager = matplotlib._pylab_helpers.Gcf.get_active()
        if figManager is not None:
            canvas = figManager.canvas
            if canvas.figure.stale:
                canvas.draw()
            canvas.start_event_loop(interval)
            return


def plot_arm_ROT3U(
    joint_and_link_positions, use_wrist_pitch_offset=True, animated=False, fig=None, ax1=None, ax2=None, clear=True, boundaries=None, plot_inside=True
):
    # print('x1: {:.3f}, y1: {:.3f}, z1: {:.3f}, x2: {:.3f}, y2: {:.3f}, z2: {:.3f}, s_x: {:.3f}, e_x: {:.3f}'.format(x1, y1, z1, x2, y2, z2, s_x, e_x))
    # print keys and values in dict form with formatting of not too many digits
    # print(["{}: {:.3f}".format(k, v) for k, v in joint_and_link_positions.items()])

    base_width = 90
    base_height = 70
    target_circle_radius = 5
    # radius = 0.035
    radius = 1  # todo manually create bigger depending on min max boundary?

    starting_x = joint_and_link_positions["starting_x"]
    starting_y = joint_and_link_positions["starting_y"]
    starting_z = joint_and_link_positions["starting_z"]
    x1 = joint_and_link_positions["x1"]
    y1 = joint_and_link_positions["y1"]
    z1 = joint_and_link_positions["z1"]
    x2 = joint_and_link_positions["x2"]
    y2 = joint_and_link_positions["y2"]
    z2 = joint_and_link_positions["z2"]
    x3 = joint_and_link_positions["x3"]
    y3 = joint_and_link_positions["y3"]

    z3 = joint_and_link_positions["z3"]
    s_x = joint_and_link_positions["s_x"]
    e_x = joint_and_link_positions["e_x"]
    w_x = joint_and_link_positions["w_x"]
    wrist_pitch_starting_side_x = joint_and_link_positions['wrist_pitch_starting_side_x']
    wrist_pitch_starting_x = joint_and_link_positions['wrist_pitch_starting_x']
    wrist_pitch_starting_y = joint_and_link_positions['wrist_pitch_starting_y']
    wrist_pitch_starting_z = joint_and_link_positions['wrist_pitch_starting_z']

    # gripper model
    gripper_back_centre_x = joint_and_link_positions.get('gripper_back_centre_x')
    if gripper_back_centre_x:
        gripper_back_centre_y = joint_and_link_positions.get('gripper_back_centre_y')
        gripper_left_jaw_start_x = joint_and_link_positions.get('gripper_left_jaw_start_x')
        gripper_left_jaw_start_y = joint_and_link_positions.get('gripper_left_jaw_start_y')
        gripper_left_jaw_end_x = joint_and_link_positions.get('gripper_left_jaw_end_x')
        gripper_left_jaw_end_y = joint_and_link_positions.get('gripper_left_jaw_end_y')
        gripper_right_jaw_start_x = joint_and_link_positions.get('gripper_right_jaw_start_x')
        gripper_right_jaw_start_y = joint_and_link_positions.get('gripper_right_jaw_start_y')
        gripper_right_jaw_end_x = joint_and_link_positions.get('gripper_right_jaw_end_x')
        gripper_right_jaw_end_y = joint_and_link_positions.get('gripper_right_jaw_end_y')
        gripper_tip_x = joint_and_link_positions.get('gripper_tip_x')
        gripper_tip_y = joint_and_link_positions.get('gripper_tip_y')
        gripper_tip_z = joint_and_link_positions.get('gripper_tip_z')

    # optional IK target visualisation
    x = joint_and_link_positions.get("x")
    y = joint_and_link_positions.get("y")
    z = joint_and_link_positions.get("z")
    side_view_x_target = joint_and_link_positions.get("side_view_x_target")

    # todo replace interactive click app loose code with this function

    if fig and ax1 and ax2:
        if clear:
            ax1.cla()
            ax2.cla()
    else:
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
    # todo in else or outside?
    # set top-down limits
    # 	all_vals = [x1, y1, z1, x2, y2, z2, s_x, e_x]  # todo if I use another arm eventually?
    # all_vals_top_down = [x1, y1, x2, y2, x3, y3, s_x, e_x, w_x]
    # extra_boundary = 5
    # min_boundary_top_down = min(all_vals_top_down) - extra_boundary
    # max_boundary_top_down = max(all_vals_top_down) + extra_boundary

    # # todo just doesn't work well really. Only do if the line is vertical?
    # # ax1.set_xlim([min_boundary_top_down, max_boundary_top_down])
    # # ax1.set_ylim([min_boundary_top_down, max_boundary_top_down])
    # if boundaries:
    #     ax1.set_xlim(boundaries[0], boundaries[1])  # todo not needed to do every time?
    #     ax1.set_ylim(boundaries[0], boundaries[1])
    #
    # # all_vals_side_view = [z1, s_x, e_x, z2, starting_z, forearm_starting_z, forearm_starting_x]
    # # all_vals_side_view = [z1, s_x, e_x, z2, starting_z]
    # # min_boundary_side_view = min(all_vals_side_view) - extra_boundary
    # # max_boundary_side_view = max(all_vals_side_view) + extra_boundary
    # # ax2.set_xlim([min_boundary_side_view, max_boundary_side_view])
    # # ax2.set_ylim([min_boundary_side_view, max_boundary_side_view])
    # if boundaries:
    #     ax2.set_xlim(boundaries[0], boundaries[1])
    #     ax2.set_ylim(boundaries[0], boundaries[1])

    # ------------------
    # plotting top-down view
    # ------------------
    ax1.plot([starting_x, x1], [starting_y, y1], c="k")
    ax1.plot([x1, x2], [y1, y2], c="r")
    if use_wrist_pitch_offset:
        ax1.plot([x2, wrist_pitch_starting_x], [y2, wrist_pitch_starting_y], '--', c="c")
    ax1.plot([wrist_pitch_starting_x, x3], [wrist_pitch_starting_y, y3], c="c")

    # plot top-down gripper
    if gripper_back_centre_x:
        # todo line to gripper and remove other lines
        ax1.plot([gripper_back_centre_x, gripper_left_jaw_start_x], [gripper_back_centre_y, gripper_left_jaw_start_y], c='g')
        ax1.plot([gripper_left_jaw_start_x, gripper_left_jaw_end_x], [gripper_left_jaw_start_y, gripper_left_jaw_end_y], c='g')

        ax1.plot([gripper_back_centre_x, gripper_right_jaw_start_x], [gripper_back_centre_y, gripper_right_jaw_start_y], c='g')
        ax1.plot([gripper_right_jaw_start_x, gripper_right_jaw_end_x], [gripper_right_jaw_start_y, gripper_right_jaw_end_y], c='g')

        ax1.plot([x3, gripper_tip_x], [y3, gripper_tip_y], c='y')

    # todo is recreating these each time very slow?
    circle_origin = plt.Circle((starting_x, starting_y), radius, color="k")
    circle_shoulder_top = plt.Circle((x1, y1), radius, color="g")
    circle_elbow_top = plt.Circle((x2, y2), radius, color="b")
    circle_wrist_pitch_top = plt.Circle((x3, y3), radius, color="c")
    if x is not None and y is not None:
        circle_target = plt.Circle((x, y), target_circle_radius, color="y")
        ax1.add_artist(circle_target)
    ax1.add_artist(circle_origin)
    ax1.add_artist(circle_shoulder_top)
    ax1.add_artist(circle_elbow_top)
    ax1.add_artist(circle_wrist_pitch_top)

    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("Top view")

    if boundaries:
        ax1.set_xlim(boundaries[0], boundaries[1])  # todo not needed to do every time?
        ax1.set_ylim(boundaries[0], boundaries[1])

    # ------------------
    # plotting side view
    # ------------------
    ax2.plot([starting_x, s_x], [starting_z, z1], c="k")
    ax2.plot([s_x, e_x], [z1, z2], c="r")
    if use_wrist_pitch_offset:
        ax2.plot([e_x, wrist_pitch_starting_side_x], [z2, wrist_pitch_starting_z], '--', c="c")
    ax2.plot([wrist_pitch_starting_side_x, w_x], [wrist_pitch_starting_z, z3], c="c")
    # plot base rectangle
    ax2.plot([-base_width / 2, base_width / 2], [0, 0], c="k")  # bottom
    ax2.plot(
        [-base_width / 2, base_width / 2], [base_height, base_height], c="k"
    )  # top
    ax2.plot([base_width / 2, base_width / 2], [0, base_height], c="k")  # right
    ax2.plot([-base_width / 2, -base_width / 2], [0, base_height], c="k")  # left

    circle_origin = plt.Circle((starting_x, starting_z), radius, color="k")
    circle_shoulder_top_side_view = plt.Circle((s_x, z1), radius, color="g")
    circle_elbow_top_side_view = plt.Circle((e_x, z2), radius, color="b")
    circle_wrist_pitch_top_side_view = plt.Circle((w_x, z3), radius, color="c")

    if x is not None and y is not None:
        # origin_x_y_z = np.array((0., 0., 0.))
        # # origin_x_y_z = np.array((starting_x, starting_y, 0.0))
        # # disregarding height, get distance on xy-plane
        # side_view_target = np.array((x, y, 0.0))
        # # top down distance to target from origin
        # shoulder_elbow_plane_x = dist(origin_x_y_z, side_view_target)  # todo?
        # circle_target = plt.Circle((shoulder_elbow_plane_x, z), radius * 2, color="y")
        # todo maybe it's better to calculate the above rather than hope every interface use of this function sends it in?

        circle_target = plt.Circle((side_view_x_target, z), target_circle_radius, color="y")
        ax2.add_artist(circle_target)
    ax2.add_artist(circle_origin)
    ax2.add_artist(circle_shoulder_top_side_view)
    ax2.add_artist(circle_elbow_top_side_view)
    ax2.add_artist(circle_wrist_pitch_top_side_view)

    ax2.set_title("Side view")
    ax2.set_xlabel("shoulder-elbow plane x")
    ax2.set_ylabel("z")

    if boundaries:
        ax2.set_xlim(boundaries[0], boundaries[1])
        ax2.set_ylim(boundaries[0], boundaries[1])

    if plot_inside:
        plt.gca().set_aspect("equal", adjustable="box")
        if animated:
            plt.pause(0.05)  # todo test
            # fig.canvas.start_event_loop(0.001)  # todo now there isn't a plot at all
            # mypause(0.05)
        else:
            plt.show()

    return fig, ax1, ax2


def plot_arm(link_poses, target_x=None, target_y=None, animated=False):
    # general function to plot general revolute arms
    # todo make work on other plot

    x, y = link_poses[0]

    for i in range(1, len(link_poses)):
        plt.plot([x, link_poses[i][0]], [y, link_poses[i][1]])

        x, y = link_poses[i]

    if target_x and target_y:
        circle_target = plt.Circle((target_x, target_y), 5, color="y")

        plt.gcf().gca().add_artist(circle_target)  # todo ax

    plt.gca().set_aspect('equal', adjustable='box')
    if animated:
        try:
            plt.pause(0.05)
        except Exception as e:
            print(e)
    else:
        plt.show()

    # plt.gcf().gca()  # todo how to close?


# todo move FK functions together and inverse kinematic functions side by side???

# def plot_arm_3R(x1, y1, z1, x2, y2, z2, s_x, e_x, x=None, y=None, z=None, forearm_starting_x=None, forearm_starting_z=None):
def plot_arm_3R(joint_and_link_positions):
    # print('x1: {:.3f}, y1: {:.3f}, z1: {:.3f}, x2: {:.3f}, y2: {:.3f}, z2: {:.3f}, s_x: {:.3f}, e_x: {:.3f}'.format(x1, y1, z1, x2, y2, z2, s_x, e_x))
    # print keys and values in dict form with formatting of not too many digits
    print(["{}: {:.3f}".format(k, v) for k, v in joint_and_link_positions.items()])

    starting_x = joint_and_link_positions["starting_x"]
    starting_y = joint_and_link_positions["starting_y"]
    starting_z = joint_and_link_positions["starting_z"]
    starting_shoulder_s_x = joint_and_link_positions["starting_shoulder_s_x"]
    x1 = joint_and_link_positions["x1"]
    y1 = joint_and_link_positions["y1"]
    z1 = joint_and_link_positions["z1"]
    x2 = joint_and_link_positions["x2"]
    y2 = joint_and_link_positions["y2"]
    z2 = joint_and_link_positions["z2"]
    s_x = joint_and_link_positions["s_x"]
    e_x = joint_and_link_positions["e_x"]
    forearm_starting_x = joint_and_link_positions["forearm_starting_x"]
    forearm_starting_z = joint_and_link_positions["forearm_starting_z"]

    # optional IK target visualisation
    x = joint_and_link_positions.get("x")
    y = joint_and_link_positions.get("y")
    z = joint_and_link_positions.get("z")

    # plotting top-down view
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
    ax1.plot([starting_x, x1], [starting_y, y1], c="k")
    ax1.plot([x1, x2], [y1, y2], c="r")

    # set top-down limits
    # 	all_vals = [x1, y1, z1, x2, y2, z2, s_x, e_x]  # todo if I use another arm eventually?
    all_vals_top_down = [x1, y1, x2, y2, s_x, e_x]
    extra_boundary = 25
    min_boundary_top_down = min(all_vals_top_down) - extra_boundary
    max_boundary_top_down = max(all_vals_top_down) + extra_boundary
    ax1.set_xlim([min_boundary_top_down, max_boundary_top_down])
    ax1.set_ylim([min_boundary_top_down, max_boundary_top_down])

    # todo forearm stuff in top view. double check it works

    # radius = 0.035
    radius = 1  # todo manually create bigger depending on min max boundary?
    circle_origin = plt.Circle((starting_x, starting_y), radius, color="k")
    circle_shoulder_top = plt.Circle((x1, y1), radius, color="g")
    circle_elbow_top = plt.Circle((x2, y2), radius, color="b")
    if x and y:
        circle_target = plt.Circle((x, y), radius * 2, color="y")
        ax1.add_artist(circle_target)
    ax1.add_artist(circle_origin)
    ax1.add_artist(circle_shoulder_top)
    ax1.add_artist(circle_elbow_top)

    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("Top view")

    # plotting side view
    ax2.plot([starting_shoulder_s_x, s_x], [starting_z, z1], c="k")
    if forearm_starting_x:
        ax2.plot([forearm_starting_x, e_x], [forearm_starting_z, z2], c="r")
    else:
        ax2.plot([s_x, e_x], [z1, z2], c="r")
    circle_origin = plt.Circle((starting_shoulder_s_x, starting_z), radius, color="k")
    circle_shoulder_top_side_view = plt.Circle((s_x, z1), radius, color="g")
    circle_elbow_top_side_view = plt.Circle((e_x, z2), radius, color="b")

    if x and y:
        # origin_x_y_z = np.array((0., 0., 0.))
        # not taking z into account, because even if empire state building will be same top down view
        origin_x_y_z = np.array((starting_x, starting_y, 0.0))
        # disregarding height, get distance on xy-plane
        side_view_target = np.array((x, y, 0.0))
        # top down distance to target from origin
        shoulder_elbow_plane_x = dist(origin_x_y_z, side_view_target)
        circle_target = plt.Circle((shoulder_elbow_plane_x, z), radius * 2, color="y")
        ax2.add_artist(circle_target)
    ax2.add_artist(circle_origin)
    ax2.add_artist(circle_shoulder_top_side_view)
    ax2.add_artist(circle_elbow_top_side_view)
    all_vals_side_view = [
        z1,
        s_x,
        e_x,
        z2,
        starting_z,
        forearm_starting_z,
        forearm_starting_x,
    ]
    min_boundary_side_view = min(all_vals_side_view) - extra_boundary
    max_boundary_side_view = max(all_vals_side_view) + extra_boundary
    # ax2.set_xlim([min_boundary_side_view, max_boundary_side_view])
    # ax2.set_ylim([min_boundary_side_view, max_boundary_side_view])
    ax2.set_title("Side view")
    ax2.set_xlabel("shoulder-elbow plane x")
    ax2.set_ylabel("z")

    # fig.canvas.mpl_connect("button_press_event", click)
    # if target_x:
    #     plt.plot(target_x, target_y, 'g*')

    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()

    # todo plot 3D as well?

def display_instances(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        # assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]
        assert boxes.shape[0] == masks.shape[0] == class_ids.shape[0]

    boxes = boxes.cpu().detach().numpy()
    masks = masks.cpu().detach().numpy()
    class_ids = class_ids.cpu().detach().numpy()

    # masks = np.moveaxis(masks, [2, 3], [1, 0]).squeeze()
    masks = masks.squeeze()
    masks = np.moveaxis(masks, 0, 2)
    masks[masks > 0.7] = 1

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        if scores[i] > 0.6:
            color = colors[i]

            # Bounding box
            if not np.any(boxes[i]):
                # Skip this instance. Has no bbox. Likely lost in image cropping.
                continue
            y1, x1, y2, x2 = boxes[i]
            if show_bbox:
                # p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                p = patches.Rectangle((y1, x1), x2 - x1, y2 - y1, linewidth=2,
                                    alpha=0.7, linestyle="dashed",
                                    edgecolor=color, facecolor='none')
                ax.add_patch(p)


            # Label
            if not captions:
                class_id = class_ids[i]
                score = scores[i] if scores is not None else None
                # if class_id < 80:
                label = class_names[class_id]
                # print(label)
                # else:
                #     label = 'over 80!!'
                caption = "{} {:.3f}".format(label, score) if score else label
                print(caption)
            else:
                print('Not captions')
                caption = captions[i]
            # ax.text(x1, y1 + 8, caption,
            ax.text(y1 + 8, x1, caption,
                    color='w', size=11, backgroundcolor="none")

            # Mask
            mask = masks[:, :, i]
            # mask = masks[i, :, :]
            if show_mask:
                masked_image = apply_mask(masked_image, mask, color)

            # Mask Polygon
            # Pad to ensure proper polygons for masks that touch image edges.
            # todo understand the below and how find, pad and draw contours!!!!
            padded_mask = np.zeros(
                (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))
    if auto_show:
        plt.show()
        # plt.pause(0.05)  # todo? Not important but need to keep same axes to do


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image
