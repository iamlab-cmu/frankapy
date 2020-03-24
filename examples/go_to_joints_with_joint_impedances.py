import argparse
import sys
from frankapy import FrankaArm
from frankapy import FrankaConstants as FC

def wait_for_enter():
    if sys.version_info[0] < 3:
        raw_input('Press Enter to continue:')
    else:
        input('Press Enter to continue:')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--time', '-t', type=float, default=10)
    parser.add_argument('--open_gripper', '-o', action='store_true')
    args = parser.parse_args()

    print('Starting robot')
    fa = FrankaArm()
    if args.open_gripper:
        fa.open_gripper()

    print('Be very careful!! Make sure the robot can safely move to HOME JOINTS Position.')
    wait_for_enter()

    fa.reset_joints()
    print('Using default joint impedances to move back and forth.')
    wait_for_enter()
    fa.goto_joints(FC.READY_JOINTS, joint_impedances=FC.DEFAULT_JOINT_IMPEDANCES)
    fa.goto_joints(FC.HOME_JOINTS)
    print('Now using different joint impedances to move back and forth.')
    wait_for_enter()
    fa.goto_joints(FC.READY_JOINTS, joint_impedances=[1500, 1500, 1500, 1250, 1250, 1000, 1000])
    fa.goto_joints(FC.HOME_JOINTS)
    print('Remember to reset the joint_impedances to defaults.')
    fa.goto_joints(FC.HOME_JOINTS, joint_impedances=FC.DEFAULT_JOINT_IMPEDANCES)
    
