#!/bin/bash
echo $PWD
echo $2
source ~/archiconda3/etc/profile.d/conda.sh
#source /home/beduffy/all_projects/arm_control
#source activate py27

conda deactivate
source /home/ben/all_projects/arm_control_ros/devel/setup.bash

echo $(which python)
#python /home/beduffy/all_projects/arm_control_ros/src/arm_control/control/scripts/rosservice_call_servo_gripper.py -m $2
python /home/ben/all_projects/arm_control_ros/src/arm_control/control/scripts/rosservice_call_servo_gripper.py -m $2
