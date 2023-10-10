mport time

import pyrealsense2 as rs
import cv2
import cv2.aruco as aruco
import numpy as np

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
board = cv2.aruco.CharucoBoard_create(7, 7, .025, .0125, dictionary)
img = board.draw((200 * 3, 200 * 3))

# Dump the calibration board to a file
# cv2.imwrite('charuco.png', img)

# aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
# parameters = aruco.DetectorParameters_create()
# parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
marker_length = 0.0265

# old calibration
camera_matrix = np.array([[612.14801862, 0., 340.03640321],
                          [0., 611.29345062, 230.06928807],
                          [0., 0., 1.]])
dist_coeffs = np.array(
    [1.80764862e-02, 1.09549436e+00, -3.38044260e-03, 4.04543459e-03, -4.26585263e+00])

# Start capturing images for calibration
# cap = cv2.VideoCapture(3)

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)

allCorners = []
allIds = []
decimator = 0
align = rs.align(rs.stream.color)
for i in range(1500):

    # ret, frame = cap.read()

    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    camera_depth_img = np.asanyarray(depth_frame.get_data())
    frame = np.asanyarray(color_frame.get_data())
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(camera_depth_img, alpha=0.03),
                                       cv2.COLORMAP_JET)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    res = cv2.aruco.detectMarkers(gray, dictionary)

    if len(res[0]) > 0:
        res2 = cv2.aruco.interpolateCornersCharuco(res[0], res[1], gray, board)
        if res2[1] is not None and res2[2] is not None and len(res2[1]) > 3 and decimator % 3 == 0:
            allCorners.append(res2[1])
            allIds.append(res2[2])

            # todo press key to allow a frame to be used? And wait for 10.
            # todo can bad images get a better rmse calibration than good images? Right? Luck has a part to play. RMSE isn't final.

        cv2.aruco.drawDetectedMarkers(gray, res[0], res[1])

    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    decimator += 1

imsize = gray.shape

# Calibration fails for lots of reasons. Release the video if we do
try:
    # cal = cv2.aruco.calibrateCameraCharuco(allCorners, allIds, board, imsize, None, None)
    ret = cv2.aruco.calibrateCameraCharucoExtended(allCorners, allIds, board, imsize, None, None)
    retval, cameraMatrix, distCoeffs, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors = ret
    np.savetxt('data/realsense_charuco_intrinsic_calibration/camera_matrix_error_{}.txt'.format(retval), cameraMatrix, delimiter=' ')
    np.savetxt('data/realsense_charuco_intrinsic_calibration/distCoeffs_{}.txt'.format(retval), distCoeffs, delimiter=' ')
finally:
    # Stop streaming
    pipeline.stop()

# cap.release()
# cv2.destroyAllWindows()