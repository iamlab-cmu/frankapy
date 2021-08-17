import numpy as np
import math
import rospy
import argparse
import pickle
from autolab_core import RigidTransform, Point
from frankapy import FrankaArm

def execute_quaternion_pose_dmp(fa, position_dmp_weights_path, quat_dmp_weights_path, 
                                goal_quat=(0.03, 1.0, -0.03, 0.01),
                                duration=10.0):
    position_dmp_file = open(position_dmp_weights_path, 'rb')
    position_dmp_info = pickle.load(position_dmp_file)

    quat_dmp_file = open(quat_dmp_weights_path, 'rb')
    quat_dmp_info = pickle.load(quat_dmp_file)

    # Should be less than duration so that the canonical system is set to 0 appropriately
    quat_goal_time = duration - 3.0
    fa.execute_quaternion_pose_dmp(position_dmp_info, quat_dmp_info, duration, goal_quat, quat_goal_time)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--position_dmp_weights_path', '-p', type=str)
    parser.add_argument('--quat_dmp_weights_path', '-q', type=str)
    args = parser.parse_args()

    print('Starting robot')
    fa = FrankaArm();

    execute_quaternion_pose_dmp(fa, 
                                args.position_dmp_weights_path, 
                                args.quat_dmp_weights_path, 
                                goal_quat=(0.03, 1.0, -0.03, 0.01),
                                duration=20.0):
