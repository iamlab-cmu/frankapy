import argparse
import time
from frankapy import FrankaArm
import pickle as pkl

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--time', '-t', type=float, default=10)
    parser.add_argument('--open_gripper', '-o', action='store_true')
    parser.add_argument('--file', '-f', default='franka_traj.pkl')
    args = parser.parse_args()

    print('Starting robot')
    fa = FrankaArm()
    if args.open_gripper:
        fa.open_gripper()
    print('Applying 0 force torque control for {}s'.format(args.time))
    end_effector_position = []
    fa.run_guide_mode(args.time, block=False)

    for i in range(1000):
        end_effector_position.append(fa.get_pose())
        time.sleep(0.01)

    pkl.dump(end_effector_position, open(args.file, 'wb'))