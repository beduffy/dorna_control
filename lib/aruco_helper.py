from cv2 import aruco
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from lib.vision import create_homogenous_transformations
from lib.vision_config import camera_matrix, dist_coeffs
from lib.vision import isclose, dist, isRotationMatrix, rotationMatrixToEulerAngles


def create_aruco_params():
    marker_length = 0.0275
    marker_length = 0.058  # bigger new January 2023 marker

    # marker_separation = 0.06  # TODO OMFG it is supposed to be 6 milimetres not 6 cm
    marker_separation = 0.006
    marker_separation = 0.0065  # bigger marker
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()
    parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    # parameters.adaptiveThreshWinSizeMin = 3
    # parameters.adaptiveThreshWinSizeStep = 4  # todo test more and see if it makes worse/better
    board = cv2.aruco.GridBoard_create(3, 4, marker_length, marker_separation, aruco_dict)

    return board, parameters, aruco_dict, marker_length


def create_charuco_params():
    parameters = aruco.DetectorParameters_create()
    parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)  # TODO is this the wrong thing?
    charuco_square_length = 0.028
    # charuco_marker_length = 0.019
    charuco_marker_length = 0.0145
    # charuco_marker_length = 0.026  # TODO should reprint like this....
    # TODO maybe i can tune these more
    # TODO find out how to do long distance pose estimation!!!

    # TODO aruco con More susceptible to rotational ambiguity at medium to long ranges
    # TODO read. 
    # TODO why is charuco not working well? Bad camera calibration or parameters
    # https://stackoverflow.com/questions/52222327/improve-accuracy-of-pose-with-bigger-aruco-markers
    # https://stackoverflow.com/questions/51709522/unstable-values-in-aruco-pose-estimation
    # TODO try apriltag_ros

    # board = cv2.aruco.CharucoBoard_create(7, 7, .025, .0125, aruco_dict)
    # board = cv2.aruco.CharucoBoard_create(7, 7, .025, .0125, dictionary)
    # img = board.draw((200 * 3, 200 * 3))
    board = cv2.aruco.CharucoBoard_create(7, 7, charuco_square_length, charuco_marker_length, aruco_dict)  # TODO test new values

    return board, parameters, aruco_dict, charuco_marker_length


# TODO get over the fear of creating classes
def aruco_detect_draw_get_transforms(gray_data, camera_color_img, aruco_dict, parameters, marker_length, camera_matrix, dist_coeffs):
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray_data, aruco_dict,
                                                                parameters=parameters)
    frame_markers = aruco.drawDetectedMarkers(camera_color_img, corners, ids)  # TODO separate elsewhere? This function does too much?
    all_rvec, all_tvec, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)

    return corners, ids, all_rvec, all_tvec


def show_matplotlib_all_aruco(aruco_dict):
    # TODO FileNotFoundError: [Errno 2] No such file or directory: 'data/markers.pdf' because I ran in scripts. Stop using absolute paths
    fig = plt.figure()
    nx = 4
    ny = 3
    for i in range(1, nx * ny + 1):
        ax = fig.add_subplot(ny, nx, i)
        img = aruco.drawMarker(aruco_dict, i, 700)
        plt.imshow(img, cmap=mpl.cm.gray, interpolation="nearest")
        ax.axis("off")

    plt.savefig("data/markers.pdf")
    plt.show()


def show_matplotlib_all_charuco(board):
    imboard = board.draw((2000, 2000))
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.imshow(imboard, cmap = mpl.cm.gray, interpolation = "nearest")
    ax.axis("off")
    plt.savefig("data/charuco_board.pdf")
    plt.show()
    # https://docs.opencv.org/3.4/d0/d3c/classcv_1_1aruco_1_1CharucoBoard.html#a5f1b7c91bd8bf8271931d6f06292f1bc


def calculate_pnp_12_markers(corners, ids, all_rvec, all_tvec, marker_length=0.058, marker_separation=0.0065):
    half_marker_len = marker_length / 2

    ids_list = [l[0] for l in ids.tolist()]
    ids_list_with_index_key_tuple = [(idx, id_) for idx, id_ in enumerate(ids_list)]
    ids_list_sorted = sorted(ids_list_with_index_key_tuple, key=lambda x: x[1])
    image_points = []
    ids_list = [x[0] for x in ids.tolist()]
    for idx_of_marker, id_of_marker in ids_list_sorted:
        image_points.append(corners[idx_of_marker].squeeze())

    num_ids_to_draw = 12
    id_1_rvec, id_1_tvec = None, None
    for list_idx, corner_id in enumerate(ids_list[0:num_ids_to_draw]):
        if list_idx == 0:
            rvec_aruco, tvec_aruco = all_rvec[list_idx, 0, :], all_tvec[list_idx, 0, :]
        if corner_id == 1:
            id_1_rvec, id_1_tvec = all_rvec[list_idx, 0, :], all_tvec[list_idx, 0, :]

    tvec, rvec = id_1_tvec, id_1_rvec  # so I can build object points from the ground up... 
    # TODO how to use depth here?

    # Calculate object points of all 12 markers and project back to image
    if id_1_tvec is not None and id_1_rvec is not None:
        spacing = marker_length + marker_separation
        all_obj_points_found_from_id_1 = []
        id_count = 1
        for y_ in range(3):
            for x_ in range(4):
                if id_count in ids_list:
                    top_left = np.array([-half_marker_len + x_ * spacing, half_marker_len - y_ * spacing, 0.])
                    top_right = np.array([half_marker_len + x_ * spacing, half_marker_len - y_ * spacing, 0.])
                    bottom_right = np.array([half_marker_len + x_ * spacing, -half_marker_len - y_ * spacing, 0.])
                    bottom_left = np.array([-half_marker_len + x_ * spacing, -half_marker_len - y_ * spacing, 0.])
                    # below seems about right but then ruins the projections and I could just do a 2nd transformation to ground plane instead?
                    # top_left = np.array([-half_marker_len + x_ * spacing, half_marker_len - y_ * spacing, shoulder_height / 1000.0])
                    # top_right = np.array([half_marker_len + x_ * spacing, half_marker_len - y_ * spacing, shoulder_height / 1000.0])
                    # bottom_right = np.array([half_marker_len + x_ * spacing, -half_marker_len - y_ * spacing, shoulder_height / 1000.0])
                    # bottom_left = np.array([-half_marker_len + x_ * spacing, -half_marker_len - y_ * spacing, shoulder_height / 1000.0])

                    corners_3d_points = np.array([top_left, top_right, bottom_right, bottom_left])
                    all_obj_points_found_from_id_1.append(corners_3d_points)
                    imagePointsCorners, jacobian = cv2.projectPoints(corners_3d_points, rvec, tvec, camera_matrix, dist_coeffs)
                    # for idx, (x, y) in enumerate(imagePointsCorners.squeeze().tolist()):
                    #     cv2.circle(camera_color_img, (int(x), int(y)), 4, colors[idx], -1)

                id_count += 1

    # after finding all object and image points, run PnP to get best homogenous transform
    input_obj_points_concat = np.concatenate(all_obj_points_found_from_id_1)
    input_img_points_concat = np.concatenate(image_points)
    outval, rvec_pnp_opt, tvec_pnp_opt = cv2.solvePnP(input_obj_points_concat, input_img_points_concat, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE)
    # TODO better understand insides of that function and have good descriptions of cam2arm vs arm2cam.
    cam2arm_opt, arm2cam_opt, _, _, _ = create_homogenous_transformations(tvec_pnp_opt, rvec_pnp_opt)

    return cam2arm_opt, arm2cam_opt, tvec_pnp_opt, rvec_pnp_opt, input_obj_points_concat, input_img_points_concat


def find_aruco_markers(color_img, aruco_dict, parameters, marker_length, id_on_shoulder_motor, opencv_aruco_image_text, camera_color_img_debug):
    # global ids, corners, all_rvec, all_tvec
    # TODO clean all code and better var names but for now I'm taking it out of here to avoid globals and begin this process
    # TODO less output and less params
    # TODO debug image? just changed, testing, how to avoid color and debug image params

    color_img = color_img.copy()
    bgr_color_data = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
    gray_data = cv2.cvtColor(bgr_color_data, cv2.COLOR_RGB2GRAY)

    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray_data, aruco_dict,
                                                                parameters=parameters)
    frame_markers = aruco.drawDetectedMarkers(camera_color_img_debug, corners, ids)
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
            # TODO better name for rvec, tvec here e.g. shoulder rvec and then remove all mention of shoulder since that was stupid anyway
            rvec, tvec = all_rvec[shoulder_motor_marker_id, 0, :], all_tvec[shoulder_motor_marker_id, 0, :]  # get first marker
            found_correct_marker = True
            # aruco.drawAxis(color_img, camera_matrix, dist_coeffs, rvec, tvec, marker_length)
            # TODO AttributeError: module 'cv2.aruco' has no attribute 'drawAxis'. Axes tell a lot
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

    return color_img, tvec, rvec, ids, corners, all_rvec, all_tvec