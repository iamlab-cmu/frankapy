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
    fa = FrankaArm();

    with open(args.pose_dmp_weights_file_path, 'rb') as pkl_f:
        pose_dmp_info = pickle.load(pkl_f)
    fa.execute_pose_dmp(pose_dmp_info, duration=8)