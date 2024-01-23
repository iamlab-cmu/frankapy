#!/bin/bash

control_pc_uname=${1}
control_pc_ip_address=${2}
workstation_ip_address=${3}
control_pc_franka_interface_path=${4}
robot_number=${5}
robot_ip=${6}
control_pc_use_password=${7}
control_pc_password=${8}

rosmaster_path="bash_scripts/set_rosmaster.sh"
catkin_ws_setup_path="catkin_ws/devel/setup.bash"

if [ "$control_pc_ip_address" = "localhost" ]; then
    cd $HOME
    cd $control_pc_franka_interface_path
    source $catkin_ws_setup_path
    roslaunch franka_ros_interface franka_gripper.launch robot_num:=$robot_number robot_ip:=$robot_ip
    bash
else
if [ "$control_pc_use_password" = "0" ]; then
ssh -tt $control_pc_uname@$control_pc_ip_address << EOSSH
cd $control_pc_franka_interface_path
source $rosmaster_path $control_pc_ip_address $workstation_ip_address
source $catkin_ws_setup_path
roslaunch franka_ros_interface franka_gripper.launch robot_num:=$robot_number robot_ip:=$robot_ip
bash
EOSSH
else
sshpass -p "$control_pc_password" ssh -tt -o StrictHostKeyChecking=no $control_pc_uname@$control_pc_ip_address << EOSSH
cd $control_pc_franka_interface_path
source $rosmaster_path $control_pc_ip_address $workstation_ip_address
source $catkin_ws_setup_path
roslaunch franka_ros_interface franka_gripper.launch robot_num:=$robot_number robot_ip:=$robot_ip
bash
EOSSH
fi
fi