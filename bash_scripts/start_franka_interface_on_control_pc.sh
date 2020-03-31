#!/bin/bash

control_pc_uname=$1
control_pc_ip_address=$2
control_pc_franka_interface_path=$3
control_pc_use_password=$4
control_pc_password=$5
franka_interface_stop_on_error=$6


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
