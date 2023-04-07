#!/usr/bin/env python3
import sys
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import sensor_msgs.msg
import numpy as np
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from moveit_msgs.msg import PlanningScene, CollisionObject
from shape_msgs.msg import SolidPrimitive, Mesh

sys.path.append("/home/ros_ws/src/git_packages/frankapy")
from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import JointPositionSensorMessage, ShouldTerminateSensorMessage
from franka_interface_msgs.msg import SensorDataGroup
from frankapy.utils import min_jerk

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
        print("Pose Values:\n", self.group.get_current_pose())
        pose = self.group.get_current_pose().pose
        print("Pose Euler Angles:\n", euler_from_quaternion([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]))

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
        num_interp = 20 # number of points to interpolate between each trajectory edge
        interpolated_traj = np.zeros(((joints_traj.shape[0]-1)*num_interp + 1, joints_traj.shape[1]))
        t_interp = np.linspace(1/num_interp, 1, num_interp) # expand each trajectory edge to num_interp points
        interpolated_traj[0,:] = joints_traj[0,:]
        for i in range(1, joints_traj.shape[0]):
            for t_i in range(len(t_interp)):
                dt = t_interp[t_i]
                interp_traj_i = joints_traj[i,:]*dt + joints_traj[i-1,:]*(1-dt)
                interpolated_traj[(i-1)*num_interp + t_i+1,:] = interp_traj_i
                
        # for franka robot
        print('Executing joints trajectory of shape: ', interpolated_traj.shape)
        rate = rospy.Rate(50)
        # To ensure skill doesn't end before completing trajectory, make the buffer time much longer than needed
        self.fa.goto_joints(interpolated_traj[1], duration=5, dynamic=True, buffer_time=10)
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
            
            print('Publishing: ID {}'.format(traj_gen_proto_msg.id))
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
    def unittest_joint(self, execute = False, guided = False):
        """
        Unit test for joint trajectory planning
        If guided = True:
            Robot put in run_guide_mode for 10 seconds
            Record the final joint state
        Else:
            Use a fixed joint goal
        Resets to home and plans to the recorded joint state
        Displays the planned path to a fixed joint goal on rviz
        Optionally Executes the planned path
        """
        print("Unit Test for Joint Trajectory Planning")
        if guided:
            print("Running Guide Mode")
            self.fa.run_guide_mode(10, block=True)
            self.fa.stop_skill()
            joint_goal_list = self.get_moveit_pose_given_frankapy_pose(self.fa.get_pose())
            print("Joint Goal: ", joint_goal_list)
        else:
            joint_goal_list = [6.14813255e-02 ,4.11382927e-01, 6.80023936e-02,-2.09547337e+00,-2.06094866e-03,2.56799173e+00,  9.20088362e-01]
        print("Resetting Joints")
        self.fa.reset_joints()
        plan_joint_vals, plan_joint = self.get_plan_given_joint(joint_goal_list)
        print("Planned Path Shape: ", plan_joint_vals.shape)
        if execute: 
            print("Executing Plan")
            self.execute_plan(plan_joint_vals)
    
    def unittest_pose(self, execute = False, guided = False):
        """
        Unit test for tool pose trajectory planning
        If guided = True:
            Robot put in run_guide_mode for 10 seconds
            Record the final tool pose
        Else:
            Use a fixed tool pose goal
        Resets to home and plans to the recorded tool pose
        Displays the planned path to a fixed tool pose goal on rviz
        Optionally Executes the planned path    
        """
        print("Unit Test for Tool Pose Trajectory Planning")
        if guided:
            print("Running Guide Mode")
            self.fa.run_guide_mode(10, block=True)
            self.fa.stop_skill()
            pose_goal = self.fa.get_pose()
        else:
            pose_goal = geometry_msgs.msg.Pose()
            # These states are different from the fa.get_pose() values
            # fa.get_pose() -> (-180, 0, -180)               x: 3.07138173e-01       -2.41627056e-04             4.86902806e-01
            # group.get_current_pose() -> (180, 0, 45)       x: 0.30689056659294117, -2.3679704553216237e-16, z: 0.5902820523028393
            # fa.get_pose() + 0.1z + 225 deg yaw = group.get_current_pose()
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
        Converts a frankapy pose to a moveit pose
        Adds 180 degree offset to yaw, and 10 cm offset to z
        """
        pose_goal = geometry_msgs.msg.Pose()
        pose_goal.position.x = pose[0]
        pose_goal.position.y = pose[1]
        pose_goal.position.z = pose[2] + 0.1
        # convert pose to roll, pitch, yaw
        rpy = euler_from_quaternion(pose[3:])
        # add 180 degree offset to yaw
        rpy[2] = rpy[2] + np.pi
        # convert back to quaternion
        quat = quaternion_from_euler(rpy[0], rpy[1], rpy[2])
        pose_goal.orientation.x = quat[0]
        pose_goal.orientation.y = quat[1]
        pose_goal.orientation.z = quat[2]
        pose_goal.orientation.w = quat[3]        
        return pose_goal
    
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
    franka_moveit.print_robot_state()
    
    # Adding and removing box to planning scene
    box_pose = geometry_msgs.msg.PoseStamped()
    box_pose.header.frame_id = "panda_link0"
    box_pose.pose.position.x = 0.2
    box_pose.pose.position.y = 0.0
    box_pose.pose.position.z = 0.3
    box_pose.pose.orientation.x = 0.0
    box_pose.pose.orientation.y = 0.0
    box_pose.pose.orientation.z = 0.0
    box_pose.pose.orientation.w = 1.0
    franka_moveit.add_box("box", box_pose, [0.1, 0.1, 0.6])
    # franka_moveit.remove_box("box")

    # Test Planning
    # To execute the plan, set execute = True
    # To plan to a joint goal using run_guide_mode, set guided = False
    
    # Test Joint Planning
    franka_moveit.unittest_joint(execute = False, guided=False) 

    # Test Tool Position Planning
    # franka_moveit.unittest_pose(execute = False, guided=False)