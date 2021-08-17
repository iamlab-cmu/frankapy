import argparse
import time
from frankapy import FrankaArm
import pickle as pkl
import numpy as np

from frankapy.utils import convert_rigid_transform_to_array


def create_formated_skill_dict(joints, end_effector_positions, time_since_skill_started):
    skill_dict = dict(skill_description='GuideMode', skill_state_dict=dict())
    skill_dict['skill_state_dict']['q'] = np.array(joints)
    skill_dict['skill_state_dict']['O_T_EE'] = np.array(end_effector_positions)
    skill_dict['skill_state_dict']['time_since_skill_started'] = np.array(time_since_skill_started)

    # The key (0 here) usually represents the absolute time when the skill was started but
    formatted_dict = {0: skill_dict}
    return formatted_dict


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
    joints = []

    time_since_skill_started = []
    fa.run_guide_mode(args.time, block=False)
    start_time = time.time()
    last_time = None

    while last_time is None or (last_time - start_time) < args.time:
        pose_array = convert_rigid_transform_to_array(fa.get_pose())
        end_effector_position.append(pose_array)
        joints.append(fa.get_joints())
        time_since_skill_started.append(time.time() - start_time)
        # add sleep to record at 100Hz
        time.sleep(0.0099)
        last_time = time.time()

    skill_dict = create_formated_skill_dict(joints, end_effector_position, time_since_skill_started)
    with open(args.file, 'wb') as pkl_f:
        pkl.dump(skill_dict, pkl_f)
        print("Did save skill dict: {}".format(args.file))
