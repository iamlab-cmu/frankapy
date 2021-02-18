import pickle as pkl
from frankapy import FrankaArm
import numpy as np
import argparse
import logging
from autolab_core import RigidTransform

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--file', '-f', default='./data/franka_pose.pkl')
    args = parser.parse_args()

    recorded_pose = pkl.load(open(args.file,'rb'))

    print('Starting robot')
    fa = FrankaArm()

    fa.reset_joints()

    # fa.goto_pose(recorded_pose, 10, use_impedance=True, force_thresholds=[10,10,10,10,10,10])
    fa.goto_pose(recorded_pose, 10, use_impedance=False, force_thresholds=[10,10,10,10,10,10])
