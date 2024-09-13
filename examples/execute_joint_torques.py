import argparse
from frankapy import FrankaArm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--time', '-t', type=float, default=10)
    args = parser.parse_args()

    print('Starting robot')
    fa = FrankaArm()
    fa.reset_joints()
    print('Applying 0 joint torque control for {}s'.format(args.time))
    fa.execute_joint_torques([0,0,0,0,0,3,0], selection=[0,0,0,0,0,1,0], remove_gravity=[0,0,0,0,0,1,0], duration=args.time)