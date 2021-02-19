import os
import pickle as pkl
import argparse
import time
import logging
from frankapy import FrankaArm
import numpy as np
from autolab_core import RigidTransform

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--open_gripper', '-o', action='store_true')
    parser.add_argument('--file', '-f', default='./data/franka_delta_traj.pkl')
    args = parser.parse_args()

    recorded_delta_poses = pkl.load(open(args.file, 'rb'))

    logger.info('Starting robot')
    fa = FrankaArm()
    fa.reset_joints()
    fa.open_gripper()
    time.sleep(5)
    fa.close_gripper()

    for i in range(len(recorded_delta_poses)):
        print(i)
        logger.info(f"Executing {i}th delta waypoint")
        logger.info(f"  {recorded_delta_poses[i]}")
        # fa.goto_pose_delta(recorded_delta_poses[i], 10, use_impedance=True, force_thresholds=[10,10,10,10,10,10])
        fa.goto_pose_delta(recorded_delta_poses[i], 10, use_impedance=False, force_thresholds=[10,10,10,10,10,10])
    logger.info("Resetting")
    fa.reset_joints()
