
import numpy as np
import math
import rospy
import argparse
import pickle
from autolab_core import RigidTransform, Point
from frankapy import FrankaArm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pose_dmp_weights_file_path', '-f', type=str)
    args = parser.parse_args()

    print('Starting robot')
    fa = FrankaArm(with_gripper=False);

    # pose_dmp_file = open(args.pose_dmp_weights_file_path,"rb")
    # pose_dmp_info = pickle.load(pose_dmp_file)

    # position_dmp_file = open('/home/sony/data/dmp_test_July_12/data_1/robot_state_data_1_quat_position_quaternion_weights_position.pkl',"rb")
    # position_dmp_file = open('/home/sony/data/dmp_test_July_12/data_2/robot_state_data_0_quat_position_quaternion_weights_position.pkl', 'rb')
    position_dmp_file = open('/home/sony/data/dmp_test_July_12/data_2/robot_state_data_0_quat_correct_position_quaternion_weights_position.pkl', 'rb')
    position_dmp_info = pickle.load(position_dmp_file)

    # quat_dmp_file = open('/home/sony/data/dmp_test_July_12/data_1/robot_state_data_1_quat_position_quaternion_weights_quat.pkl',"rb")
    # quat_dmp_file = open('/home/sony/data/dmp_test_July_12/data_2/robot_state_data_0_quat_position_quaternion_weights_quat.pkl', 'rb')
    quat_dmp_file = open('/home/sony/data/dmp_test_July_12/data_2/robot_state_data_0_quat_correct_position_quaternion_weights_quat.pkl', 'rb')
    quat_dmp_info = pickle.load(quat_dmp_file)

    # fa.execute_pose_dmp(pose_dmp_info, duration=8)
    goal_quat = (-0.0014211, 0.99939, -0.01076, 0.03316)
    fa.execute_quaternion_pose_dmp(position_dmp_info, quat_dmp_info, 8.0, goal_quat)
