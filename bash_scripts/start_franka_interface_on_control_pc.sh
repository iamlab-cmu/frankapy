#!/bin/bash

with_gripper=$1
log_on_franka_interface=$2
stop_on_error=$3
control_pc_uname=$4
control_pc_ip_address=$5
control_pc_franka_interface_path=$6
control_pc_use_password=$7
control_pc_password=$8
franka_interface_stop_on_error=$9

if [ "$control_pc_ip_address" = "localhost" ]; then
    cd $HOME
    cd $control_pc_franka_interface_path
    cd build
    ./franka_interface --with_gripper $with_gripper --log $log_on_franka_interface --stop_on_error $stop_on_error
    bash
else
if [ "$control_pc_use_password" = "0" ]; then
ssh -tt $control_pc_uname@$control_pc_ip_address << EOSSH
cd $control_pc_franka_interface_path
cd build
./franka_interface --with_gripper $with_gripper --log $log_on_franka_interface --stop_on_error $stop_on_error
bash
EOSSH
else
sshpass -p "$control_pc_password" ssh -tt -o StrictHostKeyChecking=no $control_pc_uname@$control_pc_ip_address << EOSSH
cd $control_pc_franka_interface_path
cd build
echo $stop_on_error
./franka_interface --with_gripper $with_gripper --log $log_on_franka_interface --stop_on_error $stop_on_error
bash
EOSSH
fi
fi