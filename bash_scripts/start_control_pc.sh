#!/bin/bash

die () {
    echo >&2 "$@"
    exit 1
}

usage="$(basename "$0") [-h] [-i xxx.xxx.xxx.xxx] -- Start control PC from workstation

where:
    -h show this help text
    -i IP address for the control PC.
    -u Username on control PC. (default iam-lab)
    -p Control PC password
    -d Path to franka_interface on control PC (default ~/Documents/franka-interface)
    -r Robot number (default 1)
    -s Start franka-interface on control PC (0 / 1 (default))
    -g Robot has gripper (0 / 1 (default))
    -l Log at 1kHz on franka interface (0 (default) / 1)
    -e Stop franka interface when an error has occurred (0 (default) / 1)
    
    ./start_control_pc.sh -i iam-space
    ./start_control_pc.sh -i iam-space -u iam-lab -p 12345678 -d ~/Documents/franka-interface -r 1 -s 0
    "

control_pc_uname="iam-lab"
control_pc_use_password=0
control_pc_password=""
control_pc_franka_interface_path="~/Documents/franka-interface"
start_franka_interface=1
robot_number=1
with_gripper=1
log_on_franka_interface=0
stop_on_error=0

while getopts ':h:i:u:p:d:r:s:g:l:e' option; do
  case "${option}" in
    h) echo "$usage"
       exit
       ;;
    i) control_pc_ip_address=$OPTARG
       ;;
    u) control_pc_uname=$OPTARG
       ;;
    p) control_pc_password=$OPTARG
       control_pc_use_password=1
       ;;
    d) control_pc_franka_interface_path=$OPTARG
       ;;
    r) robot_number=$OPTARG
       ;;
    s) start_franka_interface=$OPTARG
       ;;
    g) with_gripper=$OPTARG
       ;;
    l) log_on_franka_interface=$OPTARG
       ;;
    e) stop_on_error=$OPTARG
       ;;
    :) printf "missing argument for -%s\n" "$OPTARG" >&2
       echo "$usage" >&2
       exit 1
       ;;
   \?) printf "illegal option: -%s\n" "$OPTARG" >&2
       echo "$usage" >&2
       exit 1
       ;;
  esac
done
shift $((OPTIND - 1))

workstation_ip_address="`hostname`"

# Notify the IP addresses being used.
echo "Control PC IP uname/address: "$control_pc_uname"@"$control_pc_ip_address
echo "Workstation IP address: "$workstation_ip_address
if [ "$control_pc_use_password" -eq 0 ]; then
  echo "Will not use password to ssh into control pc."
else
  echo "Will use input password to ssh into control pc."
fi

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Start rosmaster in a new gnome-terminal if not already running
if ! pgrep -x "roscore" > /dev/null
then
    start_rosmaster_path="$DIR/start_rosmaster.sh"
    echo "Will start ROS master in new terminal."$start_rosmaster_path
    gnome-terminal -e "bash $start_rosmaster_path"
    sleep 3
    echo "Did start ROS master in new terminal."
else
    echo "Roscore is already running"
fi

if [ "$start_franka_interface" -eq 1 ]; then
# ssh to the control pc and start franka_interface in a new gnome-terminal
start_franka_interface_on_control_pc_path="$DIR/start_franka_interface_on_control_pc.sh"
echo "Will ssh to control PC and start iam_roblib."
gnome-terminal -e "bash $start_franka_interface_on_control_pc_path $with_gripper $log_on_franka_interface $stop_on_error $control_pc_uname $control_pc_ip_address $control_pc_franka_interface_path $control_pc_use_password $control_pc_password "
echo "Done"
sleep 3
else
echo "Will not start franka_interface on the control pc."
fi

# ssh to the control pc and start ROS action server in a new gnome-terminal
start_franka_ros_interface_on_control_pc_path="$DIR/start_franka_ros_interface_on_control_pc.sh"
echo "Will ssh to control PC and start ROS action server."
gnome-terminal -e "bash $start_franka_ros_interface_on_control_pc_path $control_pc_uname $control_pc_ip_address $workstation_ip_address $control_pc_franka_interface_path $robot_number $control_pc_use_password $control_pc_password"
sleep 3
echo "Done"
