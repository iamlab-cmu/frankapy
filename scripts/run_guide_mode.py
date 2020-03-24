import argparse
from frankapy import FrankaArm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--time', '-t', type=float, default=100)
    parser.add_argument('--open_gripper', '-o', action='store_true')
    args = parser.parse_args()

    print('Starting robot')
    fa = FrankaArm()
    if args.open_gripper:
        fa.open_gripper()
    print('Applying 0 force torque control for {}s'.format(args.time))
    fa.apply_effector_forces_torques(args.time, 0, 0, 0)