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
    fa.run_guide_mode_with_selective_pose_compliance(args.time, translational_stiffnesses=[600.0, 600.0, 0.0], 
                                                     rotational_stiffnesses=[50.0,50.0,50.0])