import math

import cv2
import numpy as np


class OpenCvArucoImageText():
    def __init__(self) -> None:
        # Opencv text params
        self.start_y = 30
        self.jump_amt = 30
        self.text_size = 1
        #--- 180 deg rotation matrix around the x axis
        self.R_flip       = np.zeros((3, 3), dtype=np.float32)
        self.R_flip[0, 0] =  1.0
        self.R_flip[1, 1] = -1.0
        self.R_flip[2, 2] = -1.0
        #-- Font for the text in the image
        self.font = cv2.FONT_HERSHEY_PLAIN
        
    def put_marker_text(self, img, tvec, roll_marker, pitch_marker, yaw_marker):
        # TODO understand all of the below intuitively. 

        # THIS is marker relative to camera z forward, so negative x is to the left, 0 x is center, and right is positive
        # -- Print the tag position in camera frame
        str_position = "MARKER Position x={:.5f}  y={:.5f}  z={:.5f}".format(tvec[0], tvec[1], tvec[2])
        cv2.putText(img, str_position, (0, self.start_y), self.font, self.text_size, (0, 255, 0), 2, cv2.LINE_AA)

        # -- Print the marker's attitude respect to camera frame
        str_attitude = "MARKER Attitude r={:.5f}  p={:.5f}  y={:.5f}".format(
            math.degrees(roll_marker), math.degrees(pitch_marker),
            math.degrees(yaw_marker))
        cv2.putText(img, str_attitude, (0, self.start_y + self.jump_amt * 1), self.font, self.text_size, (0, 255, 0), 2, cv2.LINE_AA)

    def put_camera_text(self, img, pos_camera, roll_camera, pitch_camera, yaw_camera):
        # This is camera pose relative to marker
        
        str_position = "CAMERA Position x={:.5f}  y={:.5f}  z={:.5f}".format(
            pos_camera[0].item(), pos_camera[1].item(), pos_camera[2].item())
        cv2.putText(img, str_position, (0, self.start_y + self.jump_amt * 2), self.font, self.text_size, (0, 255, 0), 2, cv2.LINE_AA)

        str_attitude = "CAMERA Attitude r={:.5f}  p={:.5f}  y={:.5f}".format(
            math.degrees(roll_camera), math.degrees(pitch_camera),
            math.degrees(yaw_camera))
        cv2.putText(img, str_attitude, (0, self.start_y + self.jump_amt * 3), self.font, self.text_size, (0, 255, 0), 2, cv2.LINE_AA)


    def put_avg_marker_text(self, img, avg_6dof_pose):
        str_marker_pose_avg = 'M pos: ({:.5f} {:.5f} {:.5f}), angles: ({:.5f} {:.5f} {:.5f})'.format(
            avg_6dof_pose[0], avg_6dof_pose[1], avg_6dof_pose[2], avg_6dof_pose[3], avg_6dof_pose[4], avg_6dof_pose[5]
        )

        cv2.putText(img, str_marker_pose_avg, (0, self.start_y + self.jump_amt * 4), self.font, self.text_size, (0, 255, 0), 2, cv2.LINE_AA)
