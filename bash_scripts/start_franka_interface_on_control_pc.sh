#!/bin/bash

with_gripper=$1
control_pc_uname=$2
control_pc_ip_address=$3
control_pc_franka_interface_path=$4
control_pc_use_password=$5
control_pc_password=$6
franka_interface_stop_on_error=$7

echo $with_gripper
echo $control_pc_uname
echo $control_pc_ip_address
echo $control_pc_franka_interface_path
echo $control_pc_use_password
echo $control_pc_password
echo "FrankaInterface will stop on error: "$franka_interface_stop_on_error

stop_on_error=0
if [ "$franka_interface_stop_on_error" = "1" ]; then
    stop_on_error=1
fi

if [ "$control_pc_use_password" = "0" ]; then
ssh -tt $control_pc_uname@$control_pc_ip_address << EOSSH
cd $control_pc_franka_interface_path
cd build
./franka_interface --stop_on_error $stop_on_error
bash
EOSSH
else
sshpass -p "$control_pc_password" ssh -tt -o StrictHostKeyChecking=no $control_pc_uname@$control_pc_ip_address << EOSSH
cd $control_pc_franka_interface_path
cd build
echo $stop_on_error
./franka_interface --stop_on_error $stop_on_error
bash
EOSSH
fi
