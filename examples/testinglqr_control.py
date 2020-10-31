import numpy as np
from autolab_core import RigidTransform

from frankapy import FrankaArm
import ipdb

if __name__ == "__main__":
    fa = FrankaArm()
    
    # reset franka to its home joints
    fa.reset_joints()
    starting_position = RigidTransform.load('pickup_starting_postion.tf')
    fa.goto_pose(starting_position, duration=5, use_impedance=False, use_lqr=False)

    T_ee_world = fa.get_pose()
    ipdb.set_trace()
    T_ee_world.translation += [0., -0.4, 0.0]
    print('Desired Translation: {} |  Desired Rotation: {}'.format(T_ee_world.translation, T_ee_world.quaternion))
    ipdb.set_trace()
    fa.goto_pose(T_ee_world, duration=5, use_impedance=True, use_lqr=True, cartesian_impedances=[20, 20, 20, 30, 30, 30])
    T_ee_world = fa.get_pose()
    print('Translation: {} | Rotation: {}'.format(T_ee_world.translation, T_ee_world.quaternion))

    # T_ee_world.translation += [0,-0.2,0]
    # print('Desired Translation: {} |  Desired Rotation: {}'.format(T_ee_world.translation, T_ee_world.quaternion))
    # fa.goto_pose(T_ee_world, duration=5, use_impedance=True, use_lqr=True)
    # print('Translation: {} | Rotation: {}'.format(T_ee_world.translation, T_ee_world.quaternion))

    # reset franka back to home
    # fa.reset_joints()