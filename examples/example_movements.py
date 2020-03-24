from frankapy import FrankaArm
import numpy as np
from autolab_core import RigidTransform

if __name__ == '__main__':

    print('Starting robot')
    fa = FrankaArm()

    xtranslation_3cm = RigidTransform(rotation=np.array([
            [1,  0,  0],
            [0, 1,  0],
            [0, 0, 1]
        ]), translation=np.array([0.03, 0,  0]),
    from_frame='franka_tool', to_frame='world')

    random_position = RigidTransform(rotation=np.array([
            [0.9323473,  -0.35858258,  0.04612846],
            [-0.35996283, -0.93259467,  0.02597504],
            [0.03370496, -0.04082229, -0.99859775]
        ]), translation=np.array([0.39247965, -0.21613652,  0.3882055]),
    from_frame='franka_tool', to_frame='world')

    print(fa.get_pose().translation)

    print(fa.get_joints())

    desired_joints_1 = [-0.52733715,  0.25603565,  0.47721503, -1.26705864,  0.00600359,  1.60788199, 0.63019184]

    desired_joints_2 = [-0.16017485,  1.12476619,  0.26004398, -0.67246923,  0.04899213,  2.08439578, 0.81627789]

    fa.reset_joints()
    
    print('Opening Grippers')
    fa.open_gripper()

    #fa.reset_pose()

    # fa.goto_pose_with_cartesian_control(random_position, cartesian_impedances=[3000, 3000, 100, 300, 300, 300])

    fa.goto_joints_with_joint_control(desired_joints_1, joint_impedances=[100, 100, 100, 50, 50, 100, 100])

    fa.goto_joints_with_joint_control(desired_joints_2, joint_impedances=[100, 100, 100, 50, 50, 100, 100])

    #fa.apply_effector_forces_torques(10.0, 0, 0, 0)