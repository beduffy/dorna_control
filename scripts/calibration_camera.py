# https://github.com/niconielsen32/ComputerVision/blob/master/cameraCalibration.py
import numpy as np
import cv2 as cv2
import glob

from lib.vision import isclose, dist, isRotationMatrix, rotationMatrixToEulerAngles, create_homogenous_transformations
from lib.vision_config import camera_matrix, dist_coeffs
from lib.realsense_helper import setup_start_realsense, realsense_get_frames, run_10_frames_to_wait_for_auto_exposure, use_aruco_corners_and_realsense_for_3D_point
from lib.aruco_helper import create_aruco_params, aruco_detect_draw_get_transforms, show_matplotlib_all_aruco
from lib.aruco_image_text import OpenCvArucoImageText

# TODO also this: https://github.com/niconielsen32/ComputerVision/blob/master/poseEstimation.py
# more info https://learnopencv.com/camera-calibration-using-opencv/

################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################

if __name__  == '__main__':
    chessboardSize = (7, 9)
    chessboardSize = (3, 3)
    depth_intrin, color_intrin, depth_scale, pipeline, align, spatial = setup_start_realsense()

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1,2)

    size_of_chessboard_squares_mm = 20
    objp = objp * size_of_chessboard_squares_mm


    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.


    # images = glob.glob('*.png')
    run_10_frames_to_wait_for_auto_exposure(pipeline, align)

    frame_count = 0
    print('Starting loop')
    while True:
    # for image in images:

        color_frame, depth_frame = realsense_get_frames(pipeline, align, spatial)
            
        camera_depth_img = np.asanyarray(depth_frame.get_data())
        camera_color_img = np.asanyarray(color_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(camera_depth_img, alpha=0.03),
                                            cv2.COLORMAP_JET)  # todo why does it look so bad, add more contrast?
        
        frameSize = camera_color_img.shape
        bgr_color_data = cv2.cvtColor(camera_color_img, cv2.COLOR_RGB2BGR)
        gray_data = cv2.cvtColor(bgr_color_data, cv2.COLOR_RGB2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray_data, chessboardSize, None)

        # If found, add object points, image points (after refining them)
        if ret:

            objpoints.append(objp)
            # TODO should be doing the below for aruco?
            corners2 = cv2.cornerSubPix(gray_data, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(camera_color_img, chessboardSize, corners2, ret)
        else:
            print('No checkboard found')
        cv2.imshow('camera_color_img', camera_color_img)
        
        k = cv2.waitKey(1)
        if k == ord('q'):
            cv2.destroyAllWindows()
            pipeline.stop()
            break
        if k == ord('p'):
            import pdb;pdb.set_trace()



    cv2.destroyAllWindows()




    ############## CALIBRATION #######################################################

    # TODO how bad is the below live?
    ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, frameSize, None, None)


    ############## UNDISTORTION #####################################################

    img = cv2.imread('cali5.png')
    h,  w = img.shape[:2]
    newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))



    # Undistort
    dst = cv2.undistort(img, cameraMatrix, dist, None, newCameraMatrix)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.imwrite('caliResult1.png', dst)



    # Undistort with Remapping
    mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w,h), 5)
    dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.imwrite('caliResult2.png', dst)




    # Reprojection Error
    mean_error = 0

    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error

    print( "total error: {}".format(mean_error/len(objpoints)) )