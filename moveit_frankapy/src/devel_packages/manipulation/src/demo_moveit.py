#!/usr/bin/env python3
import sys
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import sensor_msgs.msg
import numpy as np

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

        num_interp = 50
        interpolated_traj = np.zeros(((joints_traj.shape[0]-1)*num_interp + 1, joints_traj.shape[1]))
        t_interp = np.linspace(0.05, 1, num_interp) # expand each trajectory edge to num_interp points
        interpolated_traj[0,:] = joints_traj[0,:]
        for i in range(1, joints_traj.shape[0]):
            for t_i in range(len(t_interp)):
                dt = t_interp[t_i]
                interp_traj_i = joints_traj[i,:]*dt + joints_traj[i-1,:]*(1-dt)
                interpolated_traj[(i-1)*num_interp + t_i+1,:] = interp_traj_i
                
        print('interpolated_traj shape: ', interpolated_traj)

        print('min_jerk_traj shape: ', interpolated_traj.shape)

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
    
    def run_guide_mode(self, time):
        self.fa.run_guide_mode(time)
    
    def unittest_joint(self, execute = False):
        """
        Unit test for joint trajectory planning
        Resets joints to home position
        Displays the planned path to a fixed joint goal on rviz
        Optionally Executes the planned path
        """
        print("Unit Test for Joint Trajectory Planning")
        print("Resetting Joints")
        self.fa.reset_joints()
        joint_goal_list = [6.14813255e-02 ,4.11382927e-01, 6.80023936e-02,-2.09547337e+00,-2.06094866e-03,2.56799173e+00,  9.20088362e-01]
        plan_joint_vals, plan_joint = self.get_plan_given_joint(joint_goal_list)
        print("Planned Path Shape: ", plan_joint_vals.shape)
        if execute:
            print("Executing Plan")
            self.execute_plan(plan_joint_vals)
    
    def unittest_pose(self, execute = False):
        """
        Unit test for pose trajectory planning
        Resets joints to home position
        Displays the planned path to a fixed tool pose goal on rviz
        Optionally Executes the planned path
        """
        print("Unit Test for Tool Pose Trajectory Planning")
        print("Resetting Joints")
        self.fa.reset_joints()
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
        plan_pose = self.get_plan_given_pose(pose_goal)
        print("Planned Path Shape: ", plan_pose.shape)
        if execute:
            print("Executing Plan")
            self.execute_plan(plan_pose)
    
    def print_robot_state(self):
        print("Joint Values:\n", self.group.get_current_joint_values())
        print("Pose Values:\n", self.group.get_current_pose())


if __name__ == "__main__":
    franka_moveit = moveit_planner()
    # franka_moveit.print_robot_state()
    
    # Test Joint Planning
    franka_moveit.unittest_joint(execute = True) #to execute the plan, set execute = True

    # Test Tool Position Planning
    # franka_moveit.unittest_pose(execute = False) #to execute the plan, set execute = True
