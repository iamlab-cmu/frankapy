
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

    pose_dmp_file = open(args.pose_dmp_weights_file_path,"rb")
    pose_dmp_info = pickle.load(pose_dmp_file)

    fa.execute_pose_dmp(pose_dmp_info, duration=4.2)