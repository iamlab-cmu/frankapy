#!/bin/bash

control_pc_ip_address=$1
workstation_ip_address=$2

export ROS_MASTER_URI="http://$workstation_ip_address:11311"
export ROS_HOSTNAME="${control_pc_ip_address}"
export ROS_IP="${control_pc_ip_address}"
