import numpy as np
from autolab_core import RigidTransform

from frankapy import FrankaArm


if __name__ == "__main__":
    fa = FrankaArm()
    
    # reset franka to its home joints
    fa.reset_joints()

    # read functions
    T_ee_world = fa.get_pose()
    print('Translation: {} | Rotation: {}'.format(T_ee_world.translation, T_ee_world.quaternion))

    joints = fa.get_joints()
    print('Joints: {}'.format(joints))

    gripper_width = fa.get_gripper_width()
    print('Gripper width: {}'.format(gripper_width))

    # gripper controls
    print('Closing gripper')
    fa.close_gripper()

    print('Opening gripper to a specified position')
    fa.goto_gripper(0.02)

    print('Opening gripper all the way')
    fa.open_gripper()

    # joint controls
    print('Rotating last joint')
    joints = fa.get_joints()
    joints[6] += np.deg2rad(45)
    fa.goto_joints(joints)
    joints[6] -= np.deg2rad(45)
    fa.goto_joints(joints)

    # end-effector pose control
    print('Translation')
    T_ee_world = fa.get_pose()
    T_ee_world.translation += [0.1, 0, 0.1]
    fa.goto_pose(T_ee_world)
    T_ee_world.translation -= [0.1, 0, 0.1]
    fa.goto_pose(T_ee_world)

    print('Rotation in end-effector frame')
    T_ee_rot = RigidTransform(
        rotation=RigidTransform.x_axis_rotation(np.deg2rad(45)),
        from_frame='franka_tool', to_frame='franka_tool'
    )
    T_ee_world_target = T_ee_world * T_ee_rot
    fa.goto_pose(T_ee_world_target)
    fa.goto_pose(T_ee_world)

    # reset franka back to home
    fa.reset_joints()