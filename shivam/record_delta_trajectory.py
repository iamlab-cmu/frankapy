import argparse
import time
from frankapy import FrankaArm
import pickle as pkl

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--time', '-t', type=float, default=10)
    parser.add_argument('--file', '-f', default='./data/franka_delta_traj.pkl')
    parser.add_argument('--nwaypoints', '-n', type=int, default='3')
    args = parser.parse_args()

    print('Starting robot')
    fa = FrankaArm()
    fa.reset_joints()
    fa.open_gripper()
    time.sleep(5)
    fa.close_gripper()

    print('Applying 0 force torque control for {}s'.format(args.time))
    end_effector_poses = []
    for i in range(args.nwaypoints + 1):
        fa.run_guide_mode(args.time, block=False)
        end_effector_pose = fa.get_pose()
        end_effector_poses.append(end_effector_pose)
        fa.wait_for_skill()

    end_eff_delta_poses = []
    for i in range(1, args.nwaypoints + 1):
        delta_pose = end_effector_poses[i-1].inverse() * end_effector_poses[i]
        end_eff_delta_poses.append(delta_pose)

    pkl.dump(end_eff_delta_poses, open(args.file, 'wb'))
