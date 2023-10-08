from cv2 import aruco
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl


def create_aruco_params():
    # marker_length = 0.0265
    # marker_length = 0.028
    marker_length = 0.0275
    # marker_length = 0.0935  # big marker
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()
    parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    # parameters.adaptiveThreshWinSizeMin = 3
    # parameters.adaptiveThreshWinSizeStep = 4  # todo test more and see if it makes worse/better
    board = cv2.aruco.GridBoard_create(3, 4, marker_length, 0.06, aruco_dict)  # marker_separation 0.06

    return board, parameters, aruco_dict, marker_length

def create_charuco_params():
    parameters = aruco.DetectorParameters_create()
    parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    # charuco_marker_length = 0.019
    charuco_marker_length = 0.0145
    charuco_square_length = 0.028
    # TODO maybe i can tune these more
    # TODO find out how to do long distance pose estimation!!!

    # TODO aruco con More susceptible to rotational ambiguity at medium to long ranges
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
    # TODO if there are two markers in image, it will not be more accurate right? or?
    all_rvec, all_tvec, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)

    return corners, ids, all_rvec, all_tvec


def show_matplotlib_all_aruco(aruco_dict):
    # TODO put into aruco helpers
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