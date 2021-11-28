#!/usr/bin/env bash

SESSION="setup_arm"
SESSIONEXISTS=$(tmux list-sessions | grep $SESSION)
MODE="full"

# # TODO do I even need to create a new variable? Did I not test this that it would not work?
# export B_ROBOT_ETHERNET_IP=$B_ROBOT_ETHERNET_IP
# export robot_hostname=$HOSTNAME

# cd /home/general/bearcover_robot
# # source ./devel/setup.bash

# export special_robot="rb1"
# echo "special robot: $special_robot"
# echo "B_ROBOT_ETHERNET_IP: $B_ROBOT_ETHERNET_IP"
# echo "robot_hostname: $robot_hostname"
# echo "PATROL_FILENAME: $PATROL_FILENAME"
# echo "POLYGON_FILENAME: $POLYGON_FILENAME"
# echo "MAP_FILENAME: $MAP_FILENAME"


# TODO start loading map from .rb_config.sh env vars and if empty start new map and don't run whereami

# TODO how to fix my big tmux problem
# from operator import itemgetter as _itemgetter, eq as _eq
# ImportError: dynamic module does not define module export function (PyInit_operator)



if [ "$SESSIONEXISTS" = "" ]
then
    # Start New Session with our name
    tmux new-session -d -s $SESSION -c "/home/general/all_projects/dorna_control"

    ###############################
    tmux rename-window -t 0 'dorna_control'
    ###############################

    # tmux send-keys 'roscore' C-m

    tmux split-window -h
    tmux send-keys 'source activate py36' C-m
    tmux send-keys 'python keyboard_control.py' C-m
    

    tmux split-window -h
    tmux send-keys 'source activate py36' C-m
    tmux send-keys 'python hand_eye_calibration_dorna.py' C-m
    
    # tmux split-window -h
    # tmux send-keys 'sleep 10' C-m
    
    # tmux split-window -h
    # tmux send-keys 'sleep 10' C-m

    tmux select-layout even-vertical
fi

# Attach Session, on the Main window
tmux attach-session -t $SESSION:0