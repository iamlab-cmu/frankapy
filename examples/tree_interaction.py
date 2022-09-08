from xml.etree.ElementTree import PI
import numpy as np
import random
from autolab_core import RigidTransform

from frankapy import FrankaArm

import rospy
from geometry_msgs.msg import PoseStamped



class tree_interaction:
    def __init__(self, desired_num_pushes, num_nodes, pregrasp_list, grasp_list, recovery_joint, branch_list):
        self.desired_num_pushes = desired_num_pushes
        self.num_nodes = num_nodes
        self.pre_grasp_list = pregrasp_list
        self.grasp_list = grasp_list
        self.recovery_joint = recovery_joint
        self.branch_list = branch_list

        self.force_min = 10
        self.force_max = 30
        self.safe_force_max = 18


        sub1 = rospy.Subscriber("/vrpn_client_node/rb0/pose", PoseStamped, self.get_pose_rb0)
        sub1 = rospy.Subscriber("/vrpn_client_node/rb1/pose", PoseStamped, self.get_pose_rb1)
        sub2 = rospy.Subscriber("/vrpn_client_node/rb2/pose", PoseStamped, self.get_pose_rb2)
        sub3 = rospy.Subscriber("/vrpn_client_node/rb3/pose", PoseStamped, self.get_pose_rb3)
        sub4 = rospy.Subscriber("/vrpn_client_node/rb4/pose", PoseStamped, self.get_pose_rb4)
        sub5 = rospy.Subscriber("/vrpn_client_node/rb5/pose", PoseStamped, self.get_pose_rb5)
        sub6 = rospy.Subscriber("/vrpn_client_node/rb6/pose", PoseStamped, self.get_pose_rb6)
        sub7 = rospy.Subscriber("/vrpn_client_node/rb7/pose", PoseStamped, self.get_pose_rb7)
        sub8 = rospy.Subscriber("/vrpn_client_node/rb8/pose", PoseStamped, self.get_pose_rb8)
        sub9 = rospy.Subscriber("/vrpn_client_node/rb9/pose", PoseStamped, self.get_pose_rb9)


        #init franka
        self.fa = FrankaArm()  
            
        # reset franka to its home joints
        self.fa.reset_joints()

        # gripper controls
        # print('Open gripper')
        self.fa.open_gripper()

        #marker data
        self.rb1 = None

        #data to be saved
        self.vertex_init_pos_list = []
        self.vertex_final_pos_list = []
        self.force_applied_list = []

    def get_pose_rb0(self,pose_in):
        self.rb0 = pose_in.pose

    def get_pose_rb1(self,pose_in):
        self.rb1 = pose_in.pose
    
    def get_pose_rb2(self,pose_in):
        self.rb2 = pose_in.pose

    def get_pose_rb3(self,pose_in):
        self.rb3 = pose_in.pose

    def get_pose_rb4(self,pose_in):
        self.rb4 = pose_in.pose

    def get_pose_rb5(self,pose_in):
        self.rb5 = pose_in.pose

    def get_pose_rb6(self,pose_in):
        self.rb6 = pose_in.pose

    def get_pose_rb7(self,pose_in):
        self.rb7 = pose_in.pose

    def get_pose_rb8(self,pose_in):
        self.rb8 = pose_in.pose

    def get_pose_rb9(self,pose_in):
        self.rb9 = pose_in.pose

    def get_rigid_body_pose(self):
        return [self.rb0, self.rb1, self.rb2, self.rb3, self.rb4, self.rb5, self.rb6, self.rb7, self.rb8, self.rb9]

    def append_X(self,node_perturbation_count):

        X_list = []
        for i in range(node_perturbation_count):
            X_list.append(self.get_rigid_body_pose())
        return X_list


    def create_perturbation_list(self, branch, node_perturbation_count, force_min, force_max, safe_force_max):
        applied_perturbation_list = []

        if branch in self.branch_list:
            print(f"this is branch. lower force range")
            force_max = safe_force_max


        for i in range(node_perturbation_count):
            force_magnitude = random.uniform(force_min, force_max)
            theta = random.uniform(0, 2*3.14159)
            force_vector = [
                force_magnitude*np.cos(theta), force_magnitude*np.sin(theta), 0]
            applied_perturbation_list.append(force_vector)

        return applied_perturbation_list

    def random_force_interaction(self, applied_force_list,grasp_traj):
        force_duration = 3
        max_trans_meter = 0.2

        #push interactions
        for i in range(len(applied_force_list)):
            applied_force = applied_force_list[i]
            print(
                f"--- {i+1}/{len(applied_force_list)} iter - applying force {applied_force} N---")
            self.fa.apply_effector_forces_along_axis(
                force_duration, 1.0, max_trans_meter, applied_force, 0)
            #save Y data HERE
            Y = self.get_rigid_body_pose()
            self.vertex_final_pos_list.append(Y)
            print(f"returning to original node")
            self.fa.goto_joints(grasp_traj)


    def collect_data(self, node_perturbation_count):

        for iteration in range (self.desired_num_pushes):

            print(f" ------------- iteration: {iteration} -------------")
            for i in range (self.num_nodes):

                #get random F force
                perturb_list = self.create_perturbation_list(i, node_perturbation_count, self.force_min, self.force_max, self.safe_force_max)
                print(f"perturb_list {len(perturb_list)} and data {perturb_list}")

                #append X
                X_list_resized = self.append_X(node_perturbation_count)
                self.vertex_init_pos_list.append(X_list_resized)
                #append F
                self.force_applied_list.append(perturb_list)

                print(f"size of X: {len(self.vertex_init_pos_list)}, and F: {len(self.force_applied_list)}")

                #1. approach grasp point
                joint_traj = self.pre_grasp_list[i]
                print(f"approach pregrasp joints")
                self.fa.goto_joints(joint_traj)
                #2. grasp point
                grasp_traj = self.grasp_list[i]
                print(f" grasp joints ")
                self.fa.goto_joints(grasp_traj)
                #3. grasp branch
                self.fa.close_gripper()
                #4. push/pull branch 
                self.random_force_interaction(perturb_list, grasp_traj)
                #5. let go branch
                self.fa.open_gripper()
                #6. back out to approach point
                print(f"go back to pregrasp joints ")
                self.fa.goto_joints(joint_traj)
                #7. move back for clearance
                print('moving back')
                move_back = RigidTransform(translation=np.array([0,0,-0.05]), from_frame='franka_tool', to_frame='franka_tool')
                self.fa.goto_pose_delta(delta_tool_pose=move_back, duration=3)
                #8. go to safe pos to start again
                self.fa.goto_joints(self.recovery_joint)

                print(f"size of X: {len(self.vertex_init_pos_list)}, and F: {len(self.force_applied_list)}")
                print(f"size of Y: {len(self.vertex_final_pos_list)}")

            #temporarily save data
            np.save('temp_X', self.vertex_init_pos_list)
            np.save('temp_F', self.force_applied_list)
            np.save('temp_Y', self.vertex_final_pos_list)
            
        return self.vertex_init_pos_list, self.vertex_final_pos_list, self.force_applied_list
        