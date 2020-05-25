#!/bin/bash
echo $PWD
echo $1
source activate py27
source /home/beduffy/all_projects/arm_control_ros/devel/setup.bash
python /home/beduffy/all_projects/arm_control_ros/src/arm_control/control/scripts/rosservice_call_servo_gripper.py -m $2