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
    fa.selective_guidance_mode(args.time, use_impedance=True, use_ee_frame=True, 
                               cartesian_impedances=[600.0, 0.0, 0.0, 0.0, 0.0, 0.0])