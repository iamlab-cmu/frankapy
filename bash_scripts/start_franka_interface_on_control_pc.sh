#!/bin/bash

robot_ip=${1}
with_gripper=${2}
log_on_franka_interface=${3}
stop_on_error=${4}
control_pc_uname=${5}
control_pc_ip_address=${6}
control_pc_franka_interface_path=${7}
control_pc_use_password=${8}
control_pc_password=${9}
franka_interface_stop_on_error=${10}


if [ "$control_pc_ip_address" = "localhost" ]; then
    cd $HOME
    cd $control_pc_franka_interface_path
    cd build
    ./franka_interface --robot_ip $robot_ip --with_gripper $with_gripper --log $log_on_franka_interface --stop_on_error $stop_on_error
    bash
else
if [ "$control_pc_use_password" = "0" ]; then
ssh -tt $control_pc_uname@$control_pc_ip_address << EOSSH
cd $control_pc_franka_interface_path
cd build
./franka_interface --robot_ip $robot_ip --with_gripper $with_gripper --log $log_on_franka_interface --stop_on_error $stop_on_error
bash
EOSSH
else
sshpass -p "$control_pc_password" ssh -tt -o StrictHostKeyChecking=no $control_pc_uname@$control_pc_ip_address << EOSSH
cd $control_pc_franka_interface_path
cd build
echo $stop_on_error
./franka_interface --robot_ip $robot_ip --with_gripper $with_gripper --log $log_on_franka_interface --stop_on_error $stop_on_error
bash
EOSSH
fi
fi