import numpy as np
from autolab_core import RigidTransform

from frankapy import FrankaArm


if __name__ == "__main__":
    
    fa = FrankaArm()
    
    # reset franka to its home joints
    fa.reset_joints()

    # gripper controls
    print('Open gripper')
    fa.open_gripper()

    #load npy files
    pre_grasp_joints = np.load('pre_grasp_joint_list1.npy')
    grasp_joints = np.load('grasp_joint_list1.npy')

    print(f"pre_grasp_joints size {len(pre_grasp_joints)}")

 
    # recovery_joint = pre_grasp_joints[0]
    recovery_joint= [-0.003657742910773346, -0.45478127972319926, 0.25160863332147987, -2.462772389160959, 1.8563324756357404, 1.7923702760736147, -1.3207360251822977]

    for i in range (len(pre_grasp_joints)):
        joint_traj = pre_grasp_joints[i]
        print(f" ====== {i}th iter: approach pregrasp joints ======")
        fa.goto_joints(joint_traj)

        grasp_traj = grasp_joints[i]
        print(f" grasp joints ")
        fa.goto_joints(grasp_traj)

        fa.close_gripper()
        #push interactions
        fa.open_gripper()

        print(f" go back to pregrasp joints ")
        fa.goto_joints(joint_traj)

        fa.goto_joints(recovery_joint)
        
    