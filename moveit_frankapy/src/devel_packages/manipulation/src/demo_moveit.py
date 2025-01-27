# Author: Vibhakar Mohta vib2810@gmail.com
#!/usr/bin/env python3
import sys
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import sensor_msgs.msg
import numpy as np
from moveit_msgs.msg import PlanningScene, CollisionObject
from shape_msgs.msg import SolidPrimitive, Mesh
import scipy.spatial.transform as spt

sys.path.append("/home/ros_ws/src/git_packages/frankapy")
from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import JointPositionSensorMessage, ShouldTerminateSensorMessage
from franka_interface_msgs.msg import SensorDataGroup

class moveit_planner():
    def __init__(self) -> None: #None means no return value
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('move_group_python_interface_tutorial',anonymous=True)

        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        group_name = "panda_arm"
        self.group = moveit_commander.MoveGroupCommander(group_name)
        self.group.set_end_effector_link("panda_hand")
        self.obs_pub = rospy.Publisher('/planning_scene', PlanningScene, queue_size=10)

        # used to visualize the planned path
        self.display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',moveit_msgs.msg.DisplayTrajectory,queue_size=20)

        planning_frame = self.group.get_planning_frame()
        eef_link = self.group.get_end_effector_link()

        print("---------Moveit Planner Class Initialized---------")
        print("Planning frame: ", planning_frame)
        print("End effector: ", eef_link)
        print("Robot Groups:", self.robot.get_group_names())

        # frankapy 
        self.pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1000)
        self.fa = FrankaArm(init_node = False)
    
    # Utility Functions
    def print_robot_state(self):
        print("Joint Values:\n", self.group.get_current_joint_values())
        print("Pose Values (panda_hand):\n", self.group.get_current_pose())
        print("Pose Values (panda_end_effector):\nNOTE: Quaternion is (w,x,y,z)\n", self.fa.get_pose())
     
    def reset_joints(self):
        self.fa.reset_joints()

    def goto_joint(self, joint_goal):
        self.fa.goto_joints(joint_goal, duration=5, dynamic=True, buffer_time=10)
    
    def get_plan_given_pose(self, pose_goal: geometry_msgs.msg.Pose):
        """
        Plans a trajectory given a tool pose goal
        Returns joint_values 
        joint_values: numpy array of shape (N x 7)
        """
        output = self.group.plan(pose_goal)
        plan = output[1]
        joint_values = []
        for i in range(len(plan.joint_trajectory.points)):
            joint_values.append(plan.joint_trajectory.points[i].positions)
        joint_values = np.array(joint_values)
        return joint_values
    
    def get_plan_given_joint(self, joint_goal_list):
        """
        Plans a trajectory given a joint goal
        Returns joint_values and moveit plan
        joint_values: numpy array of shape (N x 7)
        plan: moveit_msgs.msg.RobotTrajectory object
        """
        joint_goal = sensor_msgs.msg.JointState()
        joint_goal.name = ["panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"]
        joint_goal.position = joint_goal_list

        output = self.group.plan(joint_goal)
        plan = output[1]
        joint_values = []
        for i in range(len(plan.joint_trajectory.points)):
            joint_values.append(plan.joint_trajectory.points[i].positions)
        joint_values = np.array(joint_values)
        return joint_values, plan

    def execute_plan(self, joints_traj):
        """
        joints_traj shape: (N x 7)
        """
        # interpolate the trajectory
        num_interp_slow = 50 # number of points to interpolate for the start and end of the trajectory
        num_interp = 20 # number of points to interpolate for the middle part of the trajectory
        interpolated_traj = []
        t_linear = np.linspace(1/num_interp, 1, num_interp)
        t_slow = np.linspace(1/num_interp_slow, 1, num_interp_slow)
        t_ramp_up = t_slow**2
        t_ramp_down = 1 - (1-t_slow)**2

        interpolated_traj.append(joints_traj[0,:])
        for t_i in range(len(t_ramp_up)):
            dt = t_ramp_up[t_i]
            interp_traj_i = joints_traj[1,:]*dt + joints_traj[0,:]*(1-dt)
            interpolated_traj.append(interp_traj_i)
            
        for i in range(2, joints_traj.shape[0]-1):
            for t_i in range(len(t_linear)):
                dt = t_linear[t_i]
                interp_traj_i = joints_traj[i,:]*dt + joints_traj[i-1,:]*(1-dt)
                interpolated_traj.append(interp_traj_i)

        for t_i in range(len(t_ramp_down)):
            dt = t_ramp_down[t_i]
            interp_traj_i = joints_traj[-1,:]*dt + joints_traj[-2,:]*(1-dt)
            interpolated_traj.append(interp_traj_i)

        interpolated_traj = np.array(interpolated_traj)

        print('Executing joints trajectory of shape: ', interpolated_traj.shape)

        rate = rospy.Rate(50)
        # To ensure skill doesn't end before completing trajectory, make the buffer time much longer than needed
        self.fa.goto_joints(interpolated_traj[1], duration=5, dynamic=True, buffer_time=20)
        init_time = rospy.Time.now().to_time()
        for i in range(2, interpolated_traj.shape[0]):
            traj_gen_proto_msg = JointPositionSensorMessage(
                id=i, timestamp=rospy.Time.now().to_time() - init_time, 
                joints=interpolated_traj[i]
            )
            ros_msg = make_sensor_group_msg(
                trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                    traj_gen_proto_msg, SensorDataMessageType.JOINT_POSITION)
            )
            self.pub.publish(ros_msg)
            rate.sleep()

        # Stop the skill
        # Alternatively can call fa.stop_skill()
        term_proto_msg = ShouldTerminateSensorMessage(timestamp=rospy.Time.now().to_time() - init_time, should_terminate=True)
        ros_msg = make_sensor_group_msg(
            termination_handler_sensor_msg=sensor_proto2ros_msg(
                term_proto_msg, SensorDataMessageType.SHOULD_TERMINATE)
            )
        self.pub.publish(ros_msg)
    
    # Test Functions
    def unit_test_joint(self, execute = False, guided = False):
        """
        Unit test for joint trajectory planning
        Resets to home and plans to the joint goal
        Displays the planned path to a fixed joint goal on rviz

        Parameters
        ----------
        execute: bool
            If True, executes the planned trajectory
        guided: bool
            If True, runs guide mode where the user can move the robot to a desired joint goal
            Else, uses a fixed joint goal
        """
        if guided:
            print("Running Guide Mode, Move Robot to Desired Pose")
            self.fa.run_guide_mode(10, block=True)
            self.fa.stop_skill()
            joint_goal_list = self.fa.get_joints()
            print("Joint Goal: ", joint_goal_list)
        else:
            # A random joint goal
            joint_goal_list = [6.14813255e-02 ,4.11382927e-01, 6.80023936e-02,-2.09547337e+00,-2.06094866e-03,2.56799173e+00,  9.20088362e-01]
        print("Resetting Joints")
        self.fa.reset_joints()
        plan_joint_vals, plan_joint = self.get_plan_given_joint(joint_goal_list)
        print("Planned Path Shape: ", plan_joint_vals.shape)
        if execute: 
            print("Executing Plan")
            self.execute_plan(plan_joint_vals)
    
    def unit_test_pose(self, execute = False, guided = False):
        """
        Unit test for pose trajectory planning
        Resets to home and plans to the pose goal
        Displays the planned path to a fixed pose goal on rviz

        Parameters
        ----------
        execute: bool
            If True, executes the planned trajectory
        guided: bool
            If True, runs guide mode where the user can move the robot to a desired pose goal
            Else, uses a fixed pose goal
        """
        print("Unit Test for Tool Pose Trajectory Planning")
        if guided:
            print("Running Guide Mode, Move Robot to Desired Pose")
            self.fa.run_guide_mode(10, block=True)
            self.fa.stop_skill()
            pose_goal_fa = self.fa.get_pose()
            pose_goal = geometry_msgs.msg.Pose()
            pose_goal.position.x = pose_goal_fa.translation[0]
            pose_goal.position.y = pose_goal_fa.translation[1]
            pose_goal.position.z = pose_goal_fa.translation[2]
            pose_goal.orientation.w = pose_goal_fa.quaternion[0]
            pose_goal.orientation.x = pose_goal_fa.quaternion[1]
            pose_goal.orientation.y = pose_goal_fa.quaternion[2]
            pose_goal.orientation.z = pose_goal_fa.quaternion[3]
        else:
            pose_goal = geometry_msgs.msg.Pose()
            # a random test pose
            pose_goal.position.x = 0.5843781940153249
            pose_goal.position.y = 0.05791107711908864
            pose_goal.position.z = 0.23098061041636195
            pose_goal.orientation.x = -0.9186984147774666
            pose_goal.orientation.y = 0.3942492534293267
            pose_goal.orientation.z = -0.012441904611284204 
            pose_goal.orientation.w = 0.020126567105018894
        
        # Convert to moveit pose
        pose_goal = self.get_moveit_pose_given_frankapy_pose(pose_goal)
        print("Pose Goal: ", pose_goal)
        print("Resetting Joints")
        self.fa.reset_joints()
        plan_pose = self.get_plan_given_pose(pose_goal)
        print("Planned Path Shape: ", plan_pose.shape)
        if execute:
            print("Executing Plan")
            self.execute_plan(plan_pose)

    def get_moveit_pose_given_frankapy_pose(self, pose):
        """
        Converts a frankapy pose (in panda_end_effector frame) to a moveit pose (in panda_hand frame) 
        by adding a 10 cm offset to z direction
        """
        transform_mat =  np.array([[1,0,0,0],
                                   [0,1,0,0],
                                   [0,0,1,-0.1034],
                                   [0,0,0,1]])
        pose_mat = self.pose_to_transformation_matrix(pose)
        transformed = pose_mat @ transform_mat
        pose_goal = self.transformation_matrix_to_pose(transformed)
        return pose_goal
    
    def pose_to_transformation_matrix(self, pose):
        """
        Converts geometry_msgs/Pose to a 4x4 transformation matrix
        """
        T = np.eye(4)
        T[0,3] = pose.position.x
        T[1,3] = pose.position.y
        T[2,3] = pose.position.z
        r = spt.Rotation.from_quat([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        T[0:3, 0:3] = r.as_matrix()
        return T

    def transformation_matrix_to_pose(self, trans_mat):   
        """
        Converts a 4x4 transformation matrix to geometry_msgs/Pose
        """
        out_pose = geometry_msgs.msg.Pose()
        out_pose.position.x = trans_mat[0,3]
        out_pose.position.y = trans_mat[1,3]
        out_pose.position.z = trans_mat[2,3]

        #convert rotation matrix to quaternion
        r = spt.Rotation.from_matrix(trans_mat[0:3, 0:3])
        quat = r.as_quat() 
        out_pose.orientation.x = quat[0]
        out_pose.orientation.y = quat[1]
        out_pose.orientation.z = quat[2]
        out_pose.orientation.w = quat[3] 
        return out_pose

    def add_box(self, name, pose: geometry_msgs.msg.PoseStamped(), size):
        """
        Adds a collision box to the planning scene

        Parameters
        ----------
        name : str
            Name of the box
        pose : geometry_msgs.msg.PoseStamped
            Pose of the box (Centroid and Orientation)
        size : list
            Size of the box in x, y, z  
        """
        self.scene.add_box(name, pose, size)

    def remove_box(self, name):
        self.scene.remove_world_object(name)

if __name__ == "__main__":
    franka_moveit = moveit_planner()

    # Print Current Robot State (Joint Values and End Effector Pose)
    franka_moveit.print_robot_state()

    # Test Planning
    # To execute the plan, set execute = True
    # To plan to a joint goal using run_guide_mode, set guided = True

    # Test Joint Planning
    franka_moveit.unit_test_joint(execute=False, guided=False) 

    # Test Tool Position Planning
    # franka_moveit.unittest_pose(execute=True, guided=True)

    # Adding and removing obstacle boxes to planning scene
    # box_pose = geometry_msgs.msg.PoseStamped()
    # box_pose.header.frame_id = "panda_link0"
    # box_pose.pose.position.x = 0.5767154
    # box_pose.pose.position.y = -0.17477638
    # box_pose.pose.position.z = 0.00872429
    # box_pose.pose.orientation.x = 0.98198148
    # box_pose.pose.orientation.y = 0.18717703
    # box_pose.pose.orientation.z = 0.02544852
    # box_pose.pose.orientation.w = -0.00495174
    # franka_moveit.add_box("box", box_pose, [0.02, 0.065, 0.015])

    # Remove added obstacle box
    # franka_moveit.remove_box("box")

