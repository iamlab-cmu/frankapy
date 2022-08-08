#!/bin/bash

control_pc_uname=${1}
control_pc_ip_address=${2}
workstation_ip_address=${3}
control_pc_franka_interface_path=${4}
robot_number=${5}
control_pc_use_password=${6}
control_pc_password=${7}

if [ "$control_pc_ip_address" = "localhost" ]; then
    cd $HOME
    cd $control_pc_franka_interface_path
    cd ros2_ws
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/$control_pc_uname/$control_pc_franka_interface_path/build/franka-interface-common:/home/$control_pc_uname/$control_pc_franka_interface_path/build/franka-interface/proto
    source install/setup.bash
    ros2 launch franka_ros_interface franka_ros_interface.launch.py robot_num:=$robot_number
    bash
else
if [ "$control_pc_use_password" = "0" ]; then
ssh -tt $control_pc_uname@$control_pc_ip_address << EOSSH
cd $control_pc_franka_interface_path
cd ros2_ws
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/$control_pc_uname/$control_pc_franka_interface_path/build/franka-interface-common:/home/$control_pc_uname/$control_pc_franka_interface_path/build/franka-interface/proto
source install/setup.bash
ros2 launch franka_ros_interface franka_ros_interface.launch.py robot_num:=$robot_number
bash
EOSSH
else
sshpass -p "$control_pc_password" ssh -tt -o StrictHostKeyChecking=no $control_pc_uname@$control_pc_ip_address << EOSSH
cd $control_pc_franka_interface_path
cd ros2_ws
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/$control_pc_uname/$control_pc_franka_interface_path/build/franka-interface-common:/home/$control_pc_uname/$control_pc_franka_interface_path/build/franka-interface/proto
source install/setup.bash
ros2 launch franka_ros_interface franka_ros_interface.launch.py robot_num:=$robot_number
bash
EOSSH
fi
fi
