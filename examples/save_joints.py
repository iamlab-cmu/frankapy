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
    fa.goto_gripper(0.02)
   
    num_nodes = 9
    pre_grasp_joint_list = []
    grasp_joint_list = []

    for i in range (num_nodes):
        print(f'---------- guide mode start (pre-grasp) for {i+1}th node---------- ')
        guide_duration =  10
        fa.run_guide_mode(guide_duration) #hitting joint limit will end early

        joints = fa.get_joints()
        pre_grasp_joint_list.append(joints)
        print('Joints: {}'.format(joints))
        print(f"length of pre-grasp list {len(pre_grasp_joint_list)}")

        print(' ---------- guide mode start (grasp) ---------- ' )
        guide_duration =  10
        fa.run_guide_mode(guide_duration) #hitting joint limit will end early

        joints = fa.get_joints()
        grasp_joint_list.append(joints)
        print('Joints: {}'.format(joints))
        print(f"length of grasp list {len(grasp_joint_list)}")


    # reset franka back to home
    fa.reset_joints()

    print(f"================ done recording. Save ============= ")
    print(f"pre_grasp_joint_list len {len(pre_grasp_joint_list)}, grasp_joint_list{len(grasp_joint_list)} ")
    np.save('pre_grasp_joint_list',pre_grasp_joint_list)
    np.save('grasp_joint_list',grasp_joint_list)
