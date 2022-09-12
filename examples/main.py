import numpy as np
from autolab_core import RigidTransform

from frankapy import FrankaArm

import tree_interaction

if __name__ == "__main__":
    
    
    # ========= param unique for each new tree ===============
    
    # top or last branch nodes that should be take be cautious of applied force 
    branch_array = np.array([4,5,6, 8,9])
    #load npy files
    pre_grasp_joints = np.load('pre_grasp_joint_list5.npy')
    grasp_joints = np.load('grasp_joint_list5.npy')

    desired_num_pushes = 1 #2
    node_perturbation_count =  40 #40

    # recovery_joint 
    recovery_joint= [-0.003657742910773346, -0.45478127972319926, 0.25160863332147987, -2.462772389160959, 1.8563324756357404, 1.7923702760736147, -1.3207360251822977]

    

    # ============================ 

    

    num_nodes = len(pre_grasp_joints)
    print(f"pre_grasp_joints size {num_nodes}")
    
    branch_array = branch_array - 1 #to account for removal of 0th node for corresponding joints
    branch_list = list(branch_array)

    data_collect_motion = tree_interaction.tree_interaction(desired_num_pushes, num_nodes, pre_grasp_joints, grasp_joints, recovery_joint, branch_list)
    
    


    X,Y,F = data_collect_motion.collect_data(node_perturbation_count)
    print(f"====== done with data collection ======== ")
    np.save('final_X',X)
    np.save('final_Y',Y)
    np.save('final_F',F)


   
    