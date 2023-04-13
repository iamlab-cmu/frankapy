# Author: Vibhakar Mohta vib2810@gmail.com
# Node reads current robot joint state
# - Reads manipulator joint state using ros service call through frankapy 
# - Reads gripper joint state by subscribing to /franka_gripper_1/joint_states
# Node publishes the robot joints to /real_robot_joints to be used by robot_state_publisher and eventually moveit

#!/usr/bin/env python3
import sys
import rospy
import sensor_msgs.msg
sys.path.append("/home/ros_ws/src/git_packages/frankapy")
from frankapy import FrankaArm

def gripper_callback(msg):
    global gripper_width
    gripper_width = msg.position[0]

if __name__ == "__main__":
    global gripper_width
    gripper_width = 0
    rospy.init_node('move_group_python_interface_tutorial',anonymous=True)
    fa = FrankaArm(init_node = False)
    pub = rospy.Publisher("/real_robot_joints", sensor_msgs.msg.JointState, queue_size=10)
    gripper_sub = rospy.Subscriber("/franka_gripper_1/joint_states", sensor_msgs.msg.JointState, gripper_callback)
    rate = rospy.Rate(10)
    msg = sensor_msgs.msg.JointState()
    msg.name = ["panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7", "panda_finger_joint1", "panda_finger_joint2"]
    
    while not rospy.is_shutdown():
        state = fa.get_joints().tolist()
        msg.header.stamp = rospy.Time.now()
        state.append(gripper_width); state.append(gripper_width)
        msg.position = state
        pub.publish(msg)
        rate.sleep()