import os
import pickle as pkl
import argparse
import logging
from frankapy import FrankaArm
import numpy as np
from autolab_core import RigidTransform

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--open_gripper', '-o', action='store_true')
    parser.add_argument('--dir', '-d', default='./data/')
    parser.add_argument('--nwaypoints', '-n', type=int, default='4')
    args = parser.parse_args()

    recorded_poses = []
    for i in range(args.nwaypoints):
        recorded_poses.append(pkl.load(open(os.path.join(args.dir,
                                                         f'franka_pose{i}.pkl'),
                                            'rb')))

    logger.info('Starting robot')
    fa = FrankaArm()

    fa.reset_joints()

    for i in range(args.nwaypoints):
        logger.info(f"Executing {i}th waypoint")
        logger.info(f"  {recorded_poses[i]}")
        fa.goto_pose(recorded_poses[i], 10, use_impedance=True, force_thresholds=[10,10,10,10,10,10])
    logger.info("Resetting")
    fa.reset_joints()
