from cv2 import aruco
import cv2


def create_aruco_params():
    marker_length = 0.0275
    # marker_length = 0.0935  # big marker
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()
    parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    # parameters.adaptiveThreshWinSizeMin = 3
    # parameters.adaptiveThreshWinSizeStep = 4  # todo test more and see if it makes worse/better
    board = cv2.aruco.GridBoard_create(3, 4, marker_length, 0.06, aruco_dict)  # marker_separation 0.06

    return board, parameters, aruco_dict, marker_length
