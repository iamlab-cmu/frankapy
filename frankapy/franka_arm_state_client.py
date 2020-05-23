import sys, logging
from time import sleep

import numpy as np
import rospy
from franka_interface_msgs.srv import GetCurrentRobotStateCmd

from .utils import franka_pose_to_rigid_transform

class FrankaArmStateClient:

    def __init__(self, new_ros_node=True, robot_state_server_name='/get_current_robot_state_server_node/get_current_robot_state_server', offline=False):
        if new_ros_node:
            rospy.init_node('FrankaArmStateClient', anonymous=True)

        self._offline = offline
        if not self._offline:
            rospy.wait_for_service(robot_state_server_name)
            self._get_current_robot_state = rospy.ServiceProxy(robot_state_server_name, GetCurrentRobotStateCmd)

    def get_data(self):
        '''Get all fields of current robot data in a dict.

        Returns:
            dict of robot state
        '''
        if self._offline:
            logging.warn('In offline mode - FrankaArmStateClient will return 0 values.')
            return {
                'pose': franka_pose_to_rigid_transform(np.eye(4)),
                'joint_torques': np.zeros(7),
                'joint_torques_derivative': np.zeros(7),
                'joints': np.zeros(7),
                'joints_desired': np.zeros(7),
                'joint_velocities': np.zeros(7),
                'gripper_width': 0,
                'gripper_is_grasped': False,
                'ee_force_torque': np.zeros(6)
            }

        ros_data = self._get_current_robot_state().robot_state

        data = {
            'pose': franka_pose_to_rigid_transform(ros_data.O_T_EE),
            'joint_torques': np.array(ros_data.tau_J),
            'joint_torques_derivative': np.array(ros_data.dtau_J),
            'joints': np.array(ros_data.q),
            'joints_desired': np.array(ros_data.q_d),
            'joint_velocities': np.array(ros_data.dq),
            'gripper_width': ros_data.gripper_width,
            'gripper_is_grasped': ros_data.gripper_is_grasped,
            'ee_force_torque': np.array(ros_data.O_F_ext_hat_K)
        }

        return data

    def get_pose(self):
        '''Get the current pose.

        Returns:
            RigidTransform
        '''
        return self.get_data()['pose']

    def get_joints(self):
        '''Get the current joint configuration.

        Returns:
            ndarray of shape (7,)
        '''
        return self.get_data()['joints']

    def get_joint_torques(self):
        '''Get the current joint torques.

        Returns:
            ndarray of shape (7,)
        '''
        return self.get_data()['joint_torques']

    def get_joint_velocities(self):
        '''Get the current joint velocities.

        Returns:
            ndarray of shape (7,)
        '''
        return self.get_data()['joint_velocities']

    def get_gripper_width(self):
        '''Get most recent gripper width. Note this value will *not* be updated
        during a control command.

        Returns:
            float of gripper width in meters
        '''
        return self.get_data()['gripper_width']

    def get_gripper_is_grasped(self):
        '''Returns whether or not the gripper is grasping something. Note this
        value will *not* be updated during a control command.

        Returns:
            True if gripper is grasping something. False otherwise
        '''
        return self.get_data()['gripper_is_grasped']

    def get_ee_force_torque(self):
        '''Get the current ee force torque in base frame.

        Returns:
            ndarray of shape (6,)
        '''

        return self.get_data()['ee_force_torque']