cmake_minimum_required(VERSION 3.0.2)
project(manipulation)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
)

## Declare a catkin package
catkin_package()

## Specify locations of header files
include_directories(
  ${catkin_INCLUDE_DIRS}
)

## Add executables
catkin_install_python(PROGRAMS
  src/demo_moveit.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

catkin_install_python(PROGRAMS
  src/robot_joint_converter.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Mark other files for installation (e.g. launch and config files)
install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)