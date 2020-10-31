import numpy as np
from autolab_core import RigidTransform

from frankapy import FrankaArm
import time

if __name__ == "__main__":
    fa = FrankaArm()
    
    # # reset franka to its home joints
    fa.reset_joints()
    
    # print('Rotating last joint')
    fa.goto_gripper(0.060)

    # print('Rotation in end-effector frame')
    T_ee_world_init = fa.get_pose()
    T_ee_rot = RigidTransform(
        rotation=RigidTransform.x_axis_rotation(np.deg2rad(-90)),
        from_frame='franka_tool', to_frame='franka_tool'
    )

    T_ee_world_target = T_ee_world_init * T_ee_rot
    fa.goto_pose(T_ee_world_target, duration=5, use_impedance=True, cartesian_impedances=[10, 10, 10, 15, 15, 15])

    T_ee_world = fa.get_pose()
    T_ee_rot = RigidTransform(
        rotation=RigidTransform.z_axis_rotation(np.deg2rad(-90)),
        from_frame='franka_tool', to_frame='franka_tool'
    )
    T_ee_world_target = T_ee_world * T_ee_rot
    fa.goto_pose(T_ee_world_target, duration=5, use_impedance=True, cartesian_impedances=[10, 10, 10, 15, 15, 15])

    # fa.goto_pose(T_ee_world_init)
    T_ee_world = fa.get_pose()
    print('Translation: {} | Rotation: {}'.format(T_ee_world.translation, T_ee_world.quaternion))
    T_ee_world.translation += [0.15,0.2,-0.22]
    fa.goto_pose(T_ee_world, duration=5, use_impedance=True, cartesian_impedances=[10, 10, 10, 20, 20, 20])
    print(' Target Translation: {} | Target Rotation: {}'.format(T_ee_world.translation, T_ee_world.quaternion))
    T_ee_world = fa.get_pose()
    print('Translation: {} | Rotation: {}'.format(T_ee_world.translation, T_ee_world.quaternion))

    T_ee_world.translation -= [0,0.4,0]
    print('Target Translation: {} | Target Rotation: {}'.format(T_ee_world.translation, T_ee_world.quaternion))
    fa.goto_pose(T_ee_world, duration=5, use_impedance=True, cartesian_impedances=[10, 10, 10, 20, 20, 20])
    T_ee_world = fa.get_pose()
    print('Translation: {} | Rotation: {}'.format(T_ee_world.translation, T_ee_world.quaternion))

    fa.goto_gripper(0.057)
    # # time.sleep(2)
    T_ee_world = fa.get_pose()
    T_ee_world.translation -= [0.15,0,-0.22]
    print('Target Translation: {} | Target Rotation: {}'.format(T_ee_world.translation, T_ee_world.quaternion))
    fa.goto_pose(T_ee_world, duration=5, use_impedance=True, cartesian_impedances=[10, 10, 10, 20, 20, 20])
    T_ee_world = fa.get_pose()
    print('Translation: {} | Rotation: {}'.format(T_ee_world.translation, T_ee_world.quaternion))

    T_ee_world.translation -= [0,-0.2,0]
    print('Target Translation: {} | Target Rotation: {}'.format(T_ee_world.translation, T_ee_world.quaternion))
    fa.goto_pose(T_ee_world, duration=5, use_impedance=True, cartesian_impedances=[10, 10, 10, 20, 20, 20])
    T_ee_world = fa.get_pose()
    print('Translation: {} | Rotation: {}'.format(T_ee_world.translation, T_ee_world.quaternion))


    # # joints[6] -= np.deg2rad(90)
    # # joints[4] -= np.deg2rad(-90)
    # # fa.goto_joints(joints)

    # fa.goto_pose(T_ee_world_init)
    # fa.open_gripper()
    # # reset franka back to home
    # fa.reset_joints()