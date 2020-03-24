#!/bin/bash

control_pc_uname=$1
control_pc_ip_address=$2
control_pc_robolib_path=$3
control_pc_use_password=$4
control_pc_password=$5
robolib_stop_on_error=$6


echo $control_pc_uname
echo $control_pc_ip_address
echo $control_pc_robolib_path
echo $control_pc_use_password
echo $control_pc_password
echo "Robolib will stop on error: "$robolib_stop_on_error

stop_on_error=0
if [ "$robolib_stop_on_error" = "1" ]; then
    stop_on_error=1
fi

if [ "$control_pc_use_password" = "0" ]; then
ssh -tt $control_pc_uname@$control_pc_ip_address << EOSSH
cd $control_pc_robolib_path
cd build
./main_iam_robolib --stop_on_error $stop_on_error
bash
EOSSH
else
sshpass -p "$control_pc_password" ssh -tt -o StrictHostKeyChecking=no $control_pc_uname@$control_pc_ip_address << EOSSH
cd $control_pc_robolib_path
cd build
echo $stop_on_error
./main_iam_robolib --stop_on_error $stop_on_error
bash
EOSSH
fi
