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
import tf2_ros
import scipy.spatial.transform as spt

sys.path.append("/home/ros_ws/src/git_packages/frankapy")
from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import JointPositionSensorMessage, ShouldTerminateSensorMessage
from franka_interface_msgs.msg import SensorDataGroup
from frankapy.utils import min_jerk
import tf

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
        if guided:
            print("Running Guide Mode, Move Robot to Desired Pose")
            self.fa.run_guide_mode(10, block=True)
            self.fa.stop_skill()
            joint_goal_list = self.fa.get_joints()
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
            print("Running Guide Mode, Move Robot to Desired Pose")
            self.fa.run_guide_mode(10, block=True)
            self.fa.stop_skill()
            pose_goal_fa = self.fa.get_pose()
            pose_goal = geometry_msgs.msg.Pose()
            # print(dir(pose_goal_fa))
            pose_goal.position.x = pose_goal_fa.translation[0]
            pose_goal.position.y = pose_goal_fa.translation[1]
            pose_goal.position.z = pose_goal_fa.translation[2]
            pose_goal.orientation.x = pose_goal_fa.quaternion[0]
            pose_goal.orientation.y = pose_goal_fa.quaternion[1]
            pose_goal.orientation.z = pose_goal_fa.quaternion[2]
            pose_goal.orientation.w = pose_goal_fa.quaternion[3]
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
        transform_mat =  np.array([[ 1.00000000e+00,7.21391350e-06,-3.74069273e-06,-9.14757563e-07],
                                [-7.21392068e-06,1.00000000e+00,-1.91977900e-06,-2.85452459e-06],
                                [ 3.74067888e-06,1.91980599e-06,1.00000000e+00,-1.03400211e-01],
                                [ 0.00000000e+00,0.00000000e+00,0.00000000e+00,1.00000000e+00]])
        pose_goal = geometry_msgs.msg.Pose()
        pose_goal.position.x = pose.position.x
        pose_goal.position.y = pose.position.y
        pose_goal.position.z = pose.position.z + 0.10339913226933785
        # convert pose to roll, pitch, yaw
        roll, pitch, yaw = euler_from_quaternion([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        # add 180 degree offset to yaw
        # yaw = yaw + np.pi
        # roll = -roll
        # pitch = -pitch
        # convert back to quaternion
        quat = quaternion_from_euler(roll, pitch, yaw)
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

def print_diff(pose1, pose2):
    roll1, pitch1, yaw1 = euler_from_quaternion([pose1.orientation.x, pose1.orientation.y, pose1.orientation.z, pose1.orientation.w])
    roll2, pitch2, yaw2 = euler_from_quaternion([pose2.orientation.x, pose2.orientation.y, pose2.orientation.z, pose2.orientation.w])
    # print("Moveit Pose: \n", moveit_pose.position, "\nRoll: ", roll_moveit, "Pitch: ", pitch_moveit, "Yaw: ", yaw_moveit)
    # print("Fa Pose: \n", fa_pose_raw.position, "\nRoll: ", roll, "Pitch: ", pitch, "Yaw: ", yaw)
    diff_x = pose1.position.x - pose2.position.x
    diff_y = pose1.position.y - pose2.position.y
    diff_z = pose1.position.z - pose2.position.z
    diff_roll = roll1 - roll2
    diff_pitch = pitch1 - pitch2
    diff_yaw = yaw1 - yaw2
    print("Diff X: ", diff_x, "Diff Y: ", diff_y, "Diff Z: ", diff_z)
    print("Diff Roll: ", diff_roll, "Diff Pitch: ", diff_pitch, "Diff Yaw: ", diff_yaw)

def pose_to_transformation_matrix(pose):
    T = np.eye(4)
    T[0,3] = pose.position.x
    T[1,3] = pose.position.y
    T[2,3] = pose.position.z
    r = spt.Rotation.from_quat([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
    T[0:3, 0:3] = r.as_matrix()
    return T

def transformation_matrix_to_pose(trans_mat):   
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

def get_transform_pose1_pose2(pose1, pose2):
    """
    Returns transformation matrix from pose1 to pose2
    """
    # calculate transformation matrix for pose1
    trans_mat1 = pose_to_transformation_matrix(pose1)

    # calculate transformation matrix for pose2
    trans_mat2 = pose_to_transformation_matrix(pose2)

    diff_mat = np.linalg.inv(trans_mat1) @ trans_mat2
    return diff_mat

def get_transformed_pose(pose, transform_mat):
    pose_mat = pose_to_transformation_matrix(pose)
    transformed_pose_mat = pose_mat @ transform_mat
    return transformation_matrix_to_pose(transformed_pose_mat)

if __name__ == "__main__":
    franka_moveit = moveit_planner()
    # franka_moveit.print_robot_state()
    
    # Adding and removing box to planning scene
    # box_pose = geometry_msgs.msg.PoseStamped()
    # box_pose.header.frame_id = "panda_link0"
    # box_pose.pose.position.x = 0.2
    # box_pose.pose.position.y = 0.0
    # box_pose.pose.position.z = 0.3
    # box_pose.pose.orientation.x = 0.0
    # box_pose.pose.orientation.y = 0.0
    # box_pose.pose.orientation.z = 0.0
    # box_pose.pose.orientation.w = 1.0
    # # franka_moveit.add_box("box", box_pose, [0.1, 0.1, 0.6])
    # franka_moveit.remove_box("box")

    # Test Planning
    # To execute the plan, set execute = True
    # To plan to a joint goal using run_guide_mode, set guided = False
    
    # Test Joint Planning
    # franka_moveit.unittest_joint(execute = True, guided = False) 

    # Test Tool Position Planning
    # franka_moveit.unittest_pose(execute = True, guided=True)
    pose_goal_fa = franka_moveit.fa.get_pose()
    fa_pose_raw = geometry_msgs.msg.Pose()
    fa_pose_raw.position.x = pose_goal_fa.translation[0]
    fa_pose_raw.position.y = pose_goal_fa.translation[1]
    fa_pose_raw.position.z = pose_goal_fa.translation[2]
    fa_pose_raw.orientation.w = pose_goal_fa.quaternion[0]
    fa_pose_raw.orientation.x = pose_goal_fa.quaternion[1]
    fa_pose_raw.orientation.y = pose_goal_fa.quaternion[2]
    fa_pose_raw.orientation.z = pose_goal_fa.quaternion[3]
    
    moveit_pose = franka_moveit.group.get_current_pose().pose
    # print("Initial Error:")
    # print_diff(moveit_pose, fa_pose_raw)
    print("Moveit Pose: ", moveit_pose)
    print("Fa Pose Raw: ", fa_pose_raw)
    print()

    moveit_pose_mat = pose_to_transformation_matrix(moveit_pose)
    fa_pose_raw_mat = pose_to_transformation_matrix(fa_pose_raw)
    print("Moveit pose mat:", moveit_pose_mat)
    print("Fa pose raw mat:", fa_pose_raw_mat)

    # print("Moveit mat: \n", moveit_pose_mat)
    # print("Fa mat: \n", fa_pose_raw_mat)
    # get transformation matrix from moveit to fa
    transform_mat = get_transform_pose1_pose2(fa_pose_raw, moveit_pose) # fa_pose_raw^-1 x moveit_pose
    print("Transformation Matrix: \n", transform_mat)
    transform_pose = transformation_matrix_to_pose(transform_mat)
    r = spt.Rotation.from_quat([transform_pose.orientation.x, transform_pose.orientation.y, transform_pose.orientation.z, transform_pose.orientation.w])
    roll, pitch, yaw = r.as_euler('xyz', degrees=True)
    print("Transform Pose: \n", transform_pose.position, "\nRoll: ", roll, "Pitch: ", pitch, "Yaw: ", yaw)

    print()
    # get transformed pose
    fa_pose = get_transformed_pose(fa_pose_raw, transform_mat) # fa_pose_raw x transform_mat
    print("Fa pose: \n", fa_pose)
    print("Final Errors:")
    print_diff(moveit_pose, fa_pose)



    # print("Moveit Pose: \n", moveit_pose)
    # moveit_pose_mat = pose_to_transformation_matrix(moveit_pose)
    # print("Moveit mat: \n", moveit_pose_mat)
    # test_moveit = transformation_matrix_to_pose(moveit_pose_mat)
    # print("Test Moveit: \n", test_moveit)
    # print(moveit_pose)

    # get transform between panda_hand and panda_end_effector using tf wait for transform
    # listener = tf.TransformListener()
    # listener.waitForTransform('panda_link0', 'panda_end_effector', rospy.Time(), rospy.Duration(4.0))
    # transform = listener.lookupTransform('panda_hand', 'panda_end_effector', rospy.Time(0))
    # print("Transform: \n", transform)