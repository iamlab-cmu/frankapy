import sys, signal, logging
from time import time, sleep
import numpy as np
from autolab_core import RigidTransform
import quaternion
from itertools import product

import roslib
roslib.load_manifest('franka_interface_msgs')
import rospy
import actionlib
from sensor_msgs.msg import JointState
from franka_interface_msgs.msg import ExecuteSkillAction, RobolibStatus
from franka_interface_msgs.srv import GetCurrentRobolibStatusCmd

from .skill_list import *
from .exceptions import *
from .franka_arm_state_client import FrankaArmStateClient
from .franka_constants import FrankaConstants as FC
from .iam_robolib_common_definitions import *
from .ros_utils import BoxesPublisher


class FrankaArm:

    def __init__(
            self,
            rosnode_name='franka_arm_client', ros_log_level=rospy.INFO,
            robot_num=1,
            offline=False):

        self._execute_skill_action_server_name = \
                '/execute_skill_action_server_node_{}/execute_skill'.format(robot_num)
        self._robot_state_server_name = \
                '/get_current_robot_state_server_node_{}/get_current_robot_state_server'.format(robot_num)
        self._robolib_status_server_name = \
                '/get_current_robolib_status_server_node_{}/get_current_robolib_status_server'.format(robot_num)

        self._connected = False
        self._in_skill = False
        self._offline = offline

        # init ROS
        rospy.init_node(rosnode_name,
                        disable_signals=True,
                        log_level=ros_log_level)
        self._collision_boxes_pub = BoxesPublisher('franka_collision_boxes_{}'.format(robot_num))
        self._joint_state_pub = rospy.Publisher('franka_virtual_joints_{}'.format(robot_num), JointState, queue_size=10)
        
        self._state_client = FrankaArmStateClient(
                new_ros_node=False,
                robot_state_server_name=self._robot_state_server_name,
                offline=self._offline)

        if not self._offline:
            # set signal handler to handle ctrl+c and kill sigs
            signal.signal(signal.SIGINT, self._sigint_handler_gen())
            
            rospy.wait_for_service(self._robolib_status_server_name)
            self._get_current_robolib_status = rospy.ServiceProxy(
                    self._robolib_status_server_name, GetCurrentRobolibStatusCmd)

            self._client = actionlib.SimpleActionClient(
                    self._execute_skill_action_server_name, ExecuteSkillAction)
            self._client.wait_for_server()
            self.wait_for_robolib()

            # done init ROS
            self._connected = True

        # set default identity tool delta pose
        self._tool_delta_pose = RigidTransform(from_frame='franka_tool', to_frame='franka_tool_base')

        # Precompute things and preallocate np memory for collision checking
        self._collision_boxes_data = np.zeros((len(FC.COLLISION_BOX_SHAPES), 10))
        self._collision_boxes_data[:, -3:] = FC.COLLISION_BOX_SHAPES

        self._collision_box_hdiags = []
        self._collision_box_vertices_offset = []
        self._vertex_offset_signs = np.array(list(product([1, -1],[1,-1], [1,-1])))
        for sizes in FC.COLLISION_BOX_SHAPES:
            hsizes = sizes/2
            self._collision_box_vertices_offset.append(self._vertex_offset_signs * hsizes)
            self._collision_box_hdiags.append(np.linalg.norm(sizes/2))
        self._collision_box_vertices_offset = np.array(self._collision_box_vertices_offset)
        self._collision_box_hdiags = np.array(self._collision_box_hdiags)

        self._collision_proj_axes = np.zeros((3, 15))
        self._box_vertices_offset = np.ones([8, 3])
        self._box_transform = np.eye(4)

    def wait_for_robolib(self, timeout=None):
        '''Blocks execution until robolib gives ready signal.
        '''
        timeout = FC.DEFAULT_ROBOLIB_TIMEOUT if timeout is None else timeout
        t_start = time()
        while time() - t_start < timeout:
            robolib_status = self._get_current_robolib_status().robolib_status
            if robolib_status.is_ready:
                return
            sleep(1e-2)
        raise FrankaArmCommException('Robolib status not ready for {}s'.format(
            FC.DEFAULT_ROBOLIB_TIMEOUT))

    def wait_for_skill(self):
        while not self.is_skill_done():
            continue

    def is_skill_done(self, ignore_errors=True):  
        if not self._in_skill:  
            return True 

        robolib_status = self._get_current_robolib_status().robolib_status  

        e = None  
        if rospy.is_shutdown(): 
            e = RuntimeError('rospy is down!')  
        elif robolib_status.error_description:  
            e = FrankaArmException(robolib_status.error_description)  
        elif not robolib_status.is_ready: 
            e = FrankaArmRobolibNotReadyException() 

        if e is not None: 
            if ignore_errors: 
                self.wait_for_robolib() 
            else: 
                raise e 

        done = self._client.wait_for_result(rospy.Duration.from_sec(  
            FC.ACTION_WAIT_LOOP_TIME))

        if done:  
            self._in_skill = False  

        return done

    def stop_skill(self): 
        if self._connected and self._in_skill:
            self._client.cancel_goal()
        self._in_skill = False 

    def _sigint_handler_gen(self):
        def sigint_handler(sig, frame):
            if self._connected and self._in_skill:
                self._client.cancel_goal()
            sys.exit(0)

        return sigint_handler

    def _send_goal(self, goal, cb, block=True, ignore_errors=True):
        '''
        Raises:
            FrankaArmCommException if a timeout is reached
            FrankaArmException if robolib gives an error
            FrankaArmRobolibNotReadyException if robolib is not ready
        '''
        if self._offline:
            logging.warn('In offline mode, FrankaArm will not execute real robot commands.')
            return

        if not self.is_skill_done():  
            raise ValueError('Cannot send another command when the previous skill is active!')

        self._in_skill = True
        self._client.send_goal(goal, feedback_cb=cb)

        if not block:  
            return None

        self.wait_for_skill()
        return self._client.get_result()

    '''
    Controls
    '''

    def goto_pose(self,
                  tool_pose,
                  duration=3,
                  use_impedance=True,
                  stop_on_contact_forces=None,
                  stop_on_contact_torques=None,
                  cartesian_impedances=None,
                  joint_impedances=None,
                  block=True,
                  ignore_errors=True,
                  ignore_virtual_walls=False,
                  skill_desc=''):
        '''Commands Arm to the given pose via min jerk interpolation

        Args:
            tool_pose (RigidTransform) : End-effector pose in tool frame
            duration (float) : How much time this robot motion should take
            use_impedance (boolean) : Function uses our impedance controller 
                by default. If False, uses the Franka cartesian controller.
            stop_on_contact_forces (list): List of 6 floats corresponding to
                force limits on translation (xyz) and rotation about (xyz) axes.
                Default is None. If None then will not stop on contact.
            stop_on_contact_torques (list): List of 7 floats corresponding to
                torque limits on each joint. Default is None. If None then will
                not stop on contact.
            cartesian impedances (list): List of 6 floats corresponding to
                impedances on translation (xyz) and rotation about (xyz) axes.
                Default is None. If None then will use default impedances.
            joint impedances (list): List of 7 floats corresponding to
                impedances on each joint. This is used when use_impedance is 
                False. Default is None. If None then will use default impedances.
            block (boolean) : Function blocks by default. If False, the function becomes
                asynchronous and can be preempted.
            ignore_errors (boolean) : Function ignores errors by default. 
                If False, errors and some exceptions can be thrown.
            ignore_virtual_walls (boolean): Function checks for collisions with 
                virtual walls by default. If False, the robot no longer checks,
                which may be dangerous.
            skill_desc (string) : Skill description to use for logging on
                control-pc.
        '''

        if use_impedance:
            skill = Skill(SkillType.ImpedanceControlSkill, 
                          TrajectoryGeneratorType.MinJerkPoseTrajectoryGenerator,
                          feedback_controller_type=FeedbackControllerType.CartesianImpedanceFeedbackController,
                          termination_handler_type=TerminationHandlerType.FinalPoseTerminationHandler, 
                          skill_desc=skill_desc)
        else:
            skill = Skill(SkillType.CartesianPoseSkill, 
                          TrajectoryGeneratorType.MinJerkPoseTrajectoryGenerator,
                          feedback_controller_type=FeedbackControllerType.SetInternalImpedanceFeedbackController,
                          termination_handler_type=TerminationHandlerType.FinalPoseTerminationHandler, 
                          skill_desc=skill_desc)
        
        if tool_pose.from_frame != 'franka_tool' or tool_pose.to_frame != 'world':
            raise ValueError('pose has invalid frame names! Make sure pose has \
                              from_frame=franka_tool and to_frame=world')

        tool_base_pose = tool_pose * self._tool_delta_pose.inverse()

        if not ignore_virtual_walls and np.any([
            tool_base_pose.translation <= FC.WORKSPACE_WALLS[:, :3].min(axis=0),
            tool_base_pose.translation >= FC.WORKSPACE_WALLS[:, :3].max(axis=0)]):
            raise ValueError('Target pose is outside of workspace virtual walls!')

        skill.add_initial_sensor_values(FC.EMPTY_SENSOR_VALUES)

        if use_impedance:
            if cartesian_impedances is not None:
                skill.add_cartesian_impedances(cartesian_impedances)
            else:
                skill.add_cartesian_impedances(FC.DEFAULT_TRANSLATIONAL_STIFFNESSES + FC.DEFAULT_ROTATIONAL_STIFFNESSES)
        else:
            if joint_impedances is not None:
                skill.add_internal_impedances([], joint_impedances)
            elif cartesian_impedances is not None:
                skill.add_internal_impedances(cartesian_impedances, [])
            else:
                skill.add_internal_impedances([], FC.DEFAULT_JOINT_IMPEDANCES)

        if stop_on_contact_forces is not None or stop_on_contact_torques is not None:
            if stop_on_contact_forces is None:
                stop_on_contact_forces = []
            if stop_on_contact_torques is None:
                stop_on_contact_torques = []
            skill.add_contact_termination_params(FC.DEFAULT_TERM_BUFFER_TIME,
                                                 stop_on_contact_forces,
                                                 stop_on_contact_torques)
        else:
            skill.add_pose_threshold_params(FC.DEFAULT_TERM_BUFFER_TIME, FC.DEFAULT_POSE_THRESHOLDS)

        skill.add_goal_pose(duration, tool_base_pose)
        goal = skill.create_goal()

        self._send_goal(goal,
                        cb=lambda x: skill.feedback_callback(x),
                        block=block,
                        ignore_errors=ignore_errors)

    def goto_pose_delta(self,
                        delta_tool_pose,
                        duration=3,
                        use_impedance=True,
                        stop_on_contact_forces=None,
                        stop_on_contact_torques=None,
                        cartesian_impedances=None,
                        joint_impedances=None,
                        block=True,
                        ignore_errors=True,
                        ignore_virtual_walls=False,
                        skill_desc=''):
        '''Commands Arm to the given delta pose via min jerk interpolation

        Args:
            delta_tool_pose (RigidTransform) : Delta pose in tool frame
            duration (float) : How much time this robot motion should take
            use_impedance (boolean) : Function uses our impedance controller 
                by default. If False, uses the Franka cartesian controller.
            stop_on_contact_forces (list): List of 6 floats corresponding to
                force limits on translation (xyz) and rotation about (xyz) axes.
                Default is None. If None then will not stop on contact.
            stop_on_contact_torques (list): List of 7 floats corresponding to
                torque limits on each joint. Default is None. If None then will
                not stop on contact.
            cartesian impedances (list): List of 6 floats corresponding to
                impedances on translation (xyz) and rotation about (xyz) axes.
                Default is None. If None then will use default impedances.
            joint impedances (list): List of 7 floats corresponding to
                impedances on each joint. This is used when use_impedance is 
                False. Default is None. If None then will use default impedances.
            block (boolean) : Function blocks by default. If False, the function becomes
                asynchronous and can be preempted.
            ignore_errors (boolean) : Function ignores errors by default. 
                If False, errors and some exceptions can be thrown.
            ignore_virtual_walls (boolean): Function checks for collisions with 
                virtual walls by default. If False, the robot no longer checks,
                which may be dangerous.
            skill_desc (string) : Skill description to use for logging on
                control-pc.
        '''
        if use_impedance:
            skill = Skill(SkillType.ImpedanceControlSkill, 
                          TrajectoryGeneratorType.RelativeMinJerkPoseTrajectoryGenerator,
                          feedback_controller_type=FeedbackControllerType.CartesianImpedanceFeedbackController,
                          termination_handler_type=TerminationHandlerType.FinalPoseTerminationHandler, 
                          skill_desc=skill_desc)
        else:
            skill = Skill(SkillType.CartesianPoseSkill, 
                          TrajectoryGeneratorType.RelativeMinJerkPoseTrajectoryGenerator,
                          feedback_controller_type=FeedbackControllerType.SetInternalImpedanceFeedbackController,
                          termination_handler_type=TerminationHandlerType.FinalPoseTerminationHandler, 
                          skill_desc=skill_desc)

        if delta_tool_pose.from_frame != 'franka_tool' \
                or delta_tool_pose.to_frame != 'franka_tool':
            raise ValueError('delta_pose has invalid frame names! ' \
                             'Make sure delta_pose has from_frame=franka_tool ' \
                             'and to_frame=franka_tool')

        delta_tool_base_pose = self._tool_delta_pose \
                * delta_tool_pose * self._tool_delta_pose.inverse()

        if not ignore_virtual_walls and not self._offline:
            tool_base_pose = self.get_pose() * delta_tool_base_pose
            if np.any([
                tool_base_pose.translation <= FC.WORKSPACE_WALLS[:, :3].min(axis=0),
                tool_base_pose.translation >= FC.WORKSPACE_WALLS[:, :3].max(axis=0)]):
                raise ValueError('Target pose is outside of workspace virtual walls!')

        skill.add_initial_sensor_values(FC.EMPTY_SENSOR_VALUES)

        if use_impedance:
            if cartesian_impedances is not None:
                skill.add_cartesian_impedances(cartesian_impedances)
            else:
                skill.add_cartesian_impedances(FC.DEFAULT_TRANSLATIONAL_STIFFNESSES + FC.DEFAULT_ROTATIONAL_STIFFNESSES)
        else:
            if joint_impedances is not None:
                skill.add_internal_impedances([], joint_impedances)
            elif cartesian_impedances is not None:
                skill.add_internal_impedances(cartesian_impedances, [])
            else:
                skill.add_internal_impedances([], FC.DEFAULT_JOINT_IMPEDANCES)
        
        if stop_on_contact_forces is not None or stop_on_contact_torques is not None:
            if stop_on_contact_forces is None:
                stop_on_contact_forces = []
            if stop_on_contact_torques is None:
                stop_on_contact_torques = []
            skill.add_contact_termination_params(FC.DEFAULT_TERM_BUFFER_TIME,
                                                 stop_on_contact_forces,
                                                 stop_on_contact_torques)
        else:
            skill.add_pose_threshold_params(FC.DEFAULT_TERM_BUFFER_TIME, FC.DEFAULT_POSE_THRESHOLDS)

        skill.add_goal_pose(duration, delta_tool_base_pose)

        goal = skill.create_goal()

        self._send_goal(goal,
                        cb=lambda x: skill.feedback_callback(x),
                        block=block,
                        ignore_errors=ignore_errors)

    def goto_joints(self,
                    joints,
                    duration=5,
                    use_impedance=False,
                    stop_on_contact_forces=None,
                    stop_on_contact_torques=None,
                    cartesian_impedances=None,
                    joint_impedances=None,
                    k_gains=None,
                    d_gains=None,
                    block=True,
                    ignore_errors=True,
                    ignore_virtual_walls=False,
                    skill_desc=''):
        '''Commands Arm to the given joint configuration

        Args:
            joints (list): A list of 7 numbers that correspond to joint angles
                           in radians
            duration (float): How much time this robot motion should take
            use_impedance (boolean) : Function uses the Franka joint impedance  
                controller by default. If True, uses our joint impedance controller.
            stop_on_contact_forces (list): List of 6 floats corresponding to
                force limits on translation (xyz) and rotation about (xyz) axes.
                Default is None. If None then will not stop on contact.
            stop_on_contact_torques (list): List of 7 floats corresponding to
                torque limits on each joint. Default is None. If None then will
                not stop on contact.
            cartesian impedances (list): List of 6 floats corresponding to
                impedances on translation (xyz) and rotation about (xyz) axes.
                Default is None. If None then will use default impedances.
            joint impedances (list): List of 7 floats corresponding to
                impedances on each joint. This is used when use_impedance is 
                False. Default is None. If None then will use default impedances.
            k_gains (list): List of 7 floats corresponding to the k_gains on each joint
                for our impedance controller. This is used when use_impedance is 
                True. Default is None. If None then will use default k_gains.
            d_gains (list): List of 7 floats corresponding to the d_gains on each joint
                for our impedance controller. This is used when use_impedance is 
                True. Default is None. If None then will use default d_gains.
            block (boolean) : Function blocks by default. If False, the function becomes
                asynchronous and can be preempted.
            ignore_errors (boolean) : Function ignores errors by default. 
                If False, errors and some exceptions can be thrown.
            ignore_virtual_walls (boolean): Function checks for collisions with 
                virtual walls by default. If False, the robot no longer checks,
                which may be dangerous.
            skill_desc (string) : Skill description to use for logging on
                control-pc.

        Raises:
            ValueError: If is_joints_reachable(joints) returns False
        '''
        if use_impedance:
            skill = Skill(SkillType.ImpedanceControlSkill, 
                          TrajectoryGeneratorType.MinJerkJointTrajectoryGenerator,
                          feedback_controller_type=FeedbackControllerType.JointImpedanceFeedbackController,
                          termination_handler_type=TerminationHandlerType.FinalJointTerminationHandler, 
                          skill_desc=skill_desc)
        else:
            skill = Skill(SkillType.JointPositionSkill, 
                          TrajectoryGeneratorType.MinJerkJointTrajectoryGenerator,
                          feedback_controller_type=FeedbackControllerType.SetInternalImpedanceFeedbackController,
                          termination_handler_type=TerminationHandlerType.FinalJointTerminationHandler, 
                          skill_desc=skill_desc)


        if not self.is_joints_reachable(joints):
            raise ValueError('Joints not reachable!')
        if not ignore_virtual_walls and self.is_joints_in_collision_with_boxes(joints):
            raise ValueError('Target joints in collision with virtual walls!')

        skill.add_initial_sensor_values(FC.EMPTY_SENSOR_VALUES)

        if use_impedance:
            if k_gains is not None and d_gains is not None:
                skill.add_joint_gains(k_gains, d_gains)
            else:
                skill.add_joint_gains(FC.DEFAULT_K_GAINS, FC.DEFAULT_D_GAINS)
        else:
            if joint_impedances is not None:
                skill.add_internal_impedances([], joint_impedances)
            elif cartesian_impedances is not None:
                skill.add_internal_impedances(cartesian_impedances, [])
            else:
                skill.add_internal_impedances([], FC.DEFAULT_JOINT_IMPEDANCES)

        if stop_on_contact_forces is not None or stop_on_contact_torques is not None:
            if stop_on_contact_forces is None:
                stop_on_contact_forces = []
            if stop_on_contact_torques is None:
                stop_on_contact_torques = []
            skill.add_contact_termination_params(FC.DEFAULT_TERM_BUFFER_TIME,
                                                 stop_on_contact_forces,
                                                 stop_on_contact_torques)
        else:
            skill.add_joint_threshold_params(FC.DEFAULT_TERM_BUFFER_TIME, FC.DEFAULT_JOINT_THRESHOLDS)

        skill.add_goal_joints(duration, joints)
        goal = skill.create_goal()

        self._send_goal(goal,
                        cb=lambda x: skill.feedback_callback(x),
                        block=block,
                        ignore_errors=ignore_errors)

    def apply_joint_torques(self, torques, duration, ignore_errors=True, skill_desc='',):
        '''Commands Arm to apply given joint torques for duration seconds

        Args:
            torques (list): A list of 7 numbers that correspond to torques in Nm.
            duration (float): A float in the unit of seconds
            skill_desc (string) : Skill description to use for logging on
                control-pc.
        '''
        pass

    def execute_joint_dmp(self, 
                          joint_dmp_info, 
                          duration, 
                          initial_sensor_values=None,
                          use_impedance=False, 
                          stop_on_contact_forces=None,
                          stop_on_contact_torques=None,
                          cartesian_impedances=None,
                          joint_impedances=None, 
                          k_gains=None, 
                          d_gains=None, 
                          block=True, 
                          ignore_errors=True,
                          skill_desc=''):
        '''Commands Arm to execute a given joint dmp

        Args:
            joint_dmp_info (dict): Contains all the parameters of a joint DMP
                (tau, alpha, beta, num_basis, num_sensors, mu, h, and weights)
            duration (float): A float in the unit of seconds
            initial_sensor_values (list): List of initial sensor values.
                If None it will default to ones.
            use_impedance (boolean) : Function uses the Franka joint impedance  
                controller by default. If True, uses our joint impedance controller.
            stop_on_contact_forces (list): List of 6 floats corresponding to
                force limits on translation (xyz) and rotation about (xyz) axes.
                Default is None. If None then will not stop on contact.
            stop_on_contact_torques (list): List of 7 floats corresponding to
                torque limits on each joint. Default is None. If None then will
                not stop on contact.
            cartesian impedances (list): List of 6 floats corresponding to
                impedances on translation (xyz) and rotation about (xyz) axes.
                Default is None. If None then will use default impedances.
            joint impedances (list): List of 7 floats corresponding to
                impedances on each joint. This is used when use_impedance is 
                False. Default is None. If None then will use default impedances.
            k_gains (list): List of 7 floats corresponding to the k_gains on each joint
                for our impedance controller. This is used when use_impedance is 
                True. Default is None. If None then will use default k_gains.
            d_gains (list): List of 7 floats corresponding to the d_gains on each joint
                for our impedance controller. This is used when use_impedance is 
                True. Default is None. If None then will use default d_gains.
            block (boolean) : Function blocks by default. If False, the function 
                becomes asynchronous and can be preempted.
            ignore_errors (boolean) : Function ignores errors by default. 
                If False, errors and some exceptions can be thrown.
            skill_desc (string) : Skill description to use for logging on
                control-pc.
        '''

        if use_impedance:
            skill = Skill(SkillType.ImpedanceControlSkill, 
                          TrajectoryGeneratorType.JointDmpTrajectoryGenerator,
                          feedback_controller_type=FeedbackControllerType.JointImpedanceFeedbackController,
                          termination_handler_type=TerminationHandlerType.FinalJointTerminationHandler, 
                          skill_desc=skill_desc)
        else:
            skill = Skill(SkillType.JointPositionSkill, 
                          TrajectoryGeneratorType.JointDmpTrajectoryGenerator,
                          feedback_controller_type=FeedbackControllerType.SetInternalImpedanceFeedbackController,
                          termination_handler_type=TerminationHandlerType.FinalJointTerminationHandler, 
                          skill_desc=skill_desc)

        if initial_sensor_values is None:
            initial_sensor_values = np.ones(joint_dmp_info['num_sensors']).tolist()

        skill.add_initial_sensor_values(initial_sensor_values)  # sensor values
        
        skill.add_joint_dmp_params(duration, joint_dmp_info, initial_sensor_values)
        
        if use_impedance:
            if k_gains is not None and d_gains is not None:
                skill.add_joint_gains(k_gains, d_gains)
            else:
                skill.add_joint_gains(FC.DEFAULT_K_GAINS, FC.DEFAULT_D_GAINS)
        else:
            if joint_impedances is not None:
                skill.add_internal_impedances([], joint_impedances)
            elif cartesian_impedances is not None:
                skill.add_internal_impedances(cartesian_impedances, [])
            else:
                skill.add_internal_impedances([], FC.DEFAULT_JOINT_IMPEDANCES)

        if stop_on_contact_forces is not None or stop_on_contact_torques is not None:
            if stop_on_contact_forces is None:
                stop_on_contact_forces = []
            if stop_on_contact_torques is None:
                stop_on_contact_torques = []
            skill.add_contact_termination_params(FC.DEFAULT_TERM_BUFFER_TIME,
                                                 stop_on_contact_forces,
                                                 stop_on_contact_torques)
        else:
            skill.add_time_termination_params(FC.DEFAULT_TERM_BUFFER_TIME)

        goal = skill.create_goal()

        self._send_goal(goal,
                        cb=lambda x: skill.feedback_callback(x),
                        block=block,
                        ignore_errors=ignore_errors)

    def execute_pose_dmp(self, 
                         pose_dmp_info, 
                         duration, 
                         use_goal_formulation=False,
                         initial_sensor_values=None,
                         orientation_only = False,
                         position_only = False,
                         use_impedance=True, 
                         stop_on_contact_forces=None,
                         stop_on_contact_torques=None,
                         cartesian_impedances=None,
                         joint_impedances=None, 
                         block=True, 
                         ignore_errors=True,
                         skill_desc=''):
        '''Commands Arm to execute a given pose dmp

        Args:
            pose_dmp_info (dict): Contains all the parameters of a pose DMP
                (tau, alpha, beta, num_basis, num_sensors, mu, h, and weights)
            duration (float): A float in the unit of seconds
            use_goal_formulation (boolean) : Flag that represents whether to use
                the explicit goal pose dmp formulation.
            initial_sensor_values (list): List of initial sensor values.
                If None it will default to ones.
            orientation_only (boolean) : Flag that represents if the dmp weights
                are to generate a dmp only for orientation.
            position_only (boolean) : Flag that represents if the dmp weights
                are to generate a dmp only for position.
            use_impedance (boolean) : Function uses our impedance controller 
                by default. If False, uses the Franka cartesian controller.
            stop_on_contact_forces (list): List of 6 floats corresponding to
                force limits on translation (xyz) and rotation about (xyz) axes.
                Default is None. If None then will not stop on contact.
            stop_on_contact_torques (list): List of 7 floats corresponding to
                torque limits on each joint. Default is None. If None then will
                not stop on contact.
            cartesian impedances (list): List of 6 floats corresponding to
                impedances on translation (xyz) and rotation about (xyz) axes.
                Default is None. If None then will use default impedances.
            joint impedances (list): List of 7 floats corresponding to
                impedances on each joint. This is used when use_impedance is 
                False. Default is None. If None then will use default impedances.
            block (boolean) : Function blocks by default. If False, the function becomes
                asynchronous and can be preempted.
            ignore_errors (boolean) : Function ignores errors by default. 
                If False, errors and some exceptions can be thrown.
            skill_desc (string) : Skill description to use for logging on
                control-pc.
        '''

        if use_impedance:
            if use_goal_formulation:
                skill = Skill(SkillType.ImpedanceControlSkill, 
                              TrajectoryGeneratorType.GoalPoseDmpTrajectoryGenerator,
                              feedback_controller_type=FeedbackControllerType.CartesianImpedanceFeedbackController,
                              termination_handler_type=TerminationHandlerType.TimeTerminationHandler, 
                              skill_desc=skill_desc)
            else:
                skill = Skill(SkillType.ImpedanceControlSkill, 
                              TrajectoryGeneratorType.PoseDmpTrajectoryGenerator,
                              feedback_controller_type=FeedbackControllerType.CartesianImpedanceFeedbackController,
                              termination_handler_type=TerminationHandlerType.TimeTerminationHandler, 
                              skill_desc=skill_desc)
        else:
            if use_goal_formulation:
                skill = Skill(SkillType.CartesianPoseSkill, 
                              TrajectoryGeneratorType.GoalPoseDmpTrajectoryGenerator,
                              feedback_controller_type=FeedbackControllerType.SetInternalImpedanceFeedbackController,
                              termination_handler_type=TerminationHandlerType.TimeTerminationHandler, 
                              skill_desc=skill_desc)
            else:
                skill = Skill(SkillType.CartesianPoseSkill, 
                              TrajectoryGeneratorType.PoseDmpTrajectoryGenerator,
                              feedback_controller_type=FeedbackControllerType.SetInternalImpedanceFeedbackController,
                              termination_handler_type=TerminationHandlerType.TimeTerminationHandler, 
                              skill_desc=skill_desc)

        if initial_sensor_values is None:
            initial_sensor_values = np.ones(joint_dmp_info['num_sensors']).tolist()

        skill.add_initial_sensor_values(initial_sensor_values)  # sensor values

        skill.add_pose_dmp_params(orientation_only, position_only, duration, pose_dmp_info, initial_sensor_values):

        if stop_on_contact_forces is not None or stop_on_contact_torques is not None:
            if stop_on_contact_forces is None:
                stop_on_contact_forces = []
            if stop_on_contact_torques is None:
                stop_on_contact_torques = []
            skill.add_contact_termination_params(FC.DEFAULT_TERM_BUFFER_TIME,
                                                 stop_on_contact_forces,
                                                 stop_on_contact_torques)
        else:
            skill.add_time_termination_params(FC.DEFAULT_TERM_BUFFER_TIME)

        goal = skill.create_goal()

        self._send_goal(goal,
                        cb=lambda x: skill.feedback_callback(x),
                        block=block,
                        ignore_errors=ignore_errors)

    def apply_effector_forces_torques(self,
                                      run_duration,
                                      acc_duration,
                                      max_translation,
                                      max_rotation,
                                      forces=None,
                                      torques=None,
                                      block=True,
                                      ignore_errors=True,
                                      skill_desc=''):
        '''Applies the given end-effector forces and torques in N and Nm

        Args:
            run_duration (float): A float in the unit of seconds
            acc_duration (float): A float in the unit of seconds. How long to
                acc/de-acc to achieve desired force.
            forces (list): Optional (defaults to None).
                A list of 3 numbers that correspond to end-effector forces in
                    3 directions
            torques (list): Optional (defaults to None).
                A list of 3 numbers that correspond to end-effector torques in
                    3 axes

        Raises:
            ValueError if acc_duration > 0.5*run_duration, or if forces are
                too large
        '''
        if acc_duration > 0.5 * run_duration:
            raise ValueError(
                    'acc_duration must be smaller than half of run_duration!')

        forces = [0, 0, 0] if forces is None else np.array(forces).tolist()
        torques = [0, 0, 0] if torques is None else np.array(torques).tolist()

        if np.linalg.norm(forces) * run_duration > FC.MAX_LIN_MOMENTUM:
            raise ValueError('Linear momentum magnitude exceeds safety '
                    'threshold of {}'.format(FC.MAX_LIN_MOMENTUM))
        if np.linalg.norm(torques) * run_duration > FC.MAX_ANG_MOMENTUM:
            raise ValueError('Angular momentum magnitude exceeds safety '
                    'threshold of {}'.format(FC.MAX_ANG_MOMENTUM))

        skill = ForceTorqueSkill(skill_desc=skill_desc)
        skill.add_initial_sensor_values(FC.EMPTY_SENSOR_VALUES)
        skill.add_termination_params([0.1])

        skill.add_trajectory_params(
                [run_duration, acc_duration, max_translation, max_rotation]
                + forces + torques)
        goal = skill.create_goal()

        self._send_goal(goal,
                        cb=lambda x: skill.feedback_callback(x),
                        block=block,
                        ignore_errors=ignore_errors)

    def apply_effector_forces_along_axis(self,
                                         run_duration,
                                         acc_duration,
                                         max_translation,
                                         forces,
                                         block=True,
                                         ignore_errors=True,
                                         skill_desc=''):
        '''Applies the given end-effector forces and torques in N and Nm

        Args:
            run_duration (float): A float in the unit of seconds
            acc_duration (float): A float in the unit of seconds.
                How long to acc/de-acc to achieve desired force.
            max_translation (float): Max translation before the robot
                deaccelerates.
            forces (list): Optional (defaults to None).
                A list of 3 numbers that correspond to end-effector forces in
                    3 directions
        Raises:
            ValueError if acc_duration > 0.5*run_duration, or if forces are
                too large
        '''
        if acc_duration > 0.5 * run_duration:
            raise ValueError(
                    'acc_duration must be smaller than half of run_duration!')
        if np.linalg.norm(forces) * run_duration > FC.MAX_LIN_MOMENTUM_CONSTRAINED:
            raise ValueError('Linear momentum magnitude exceeds safety ' \
                    'threshold of {}'.format(FC.MAX_LIN_MOMENTUM_CONSTRAINED))

        forces = np.array(forces)
        force_axis = forces / np.linalg.norm(forces)

        skill = ForceAlongAxisSkill(skill_desc=skill_desc)
        skill.add_initial_sensor_values(FC.EMPTY_SENSOR_VALUES)
        skill.add_termination_params([0.1])
        skill.add_feedback_controller_params(
                FC.DEFAULT_FORCE_AXIS_CONTROLLER_PARAMS + force_axis.tolist())

        init_params = [run_duration, acc_duration, max_translation, 0]
        skill.add_trajectory_params(init_params + forces.tolist() + [0, 0, 0])
        goal = skill.create_goal()

        self._send_goal(goal,
                        cb=lambda x: skill.feedback_callback(x),
                        block=block,
                        ignore_errors=ignore_errors)

    def goto_gripper(self, 
                     width, 
                     grasp=False, 
                     speed=0.04, 
                     force=0.0, 
                     block=True, 
                     ignore_errors=True, 
                     skill_desc=''):
        '''Commands gripper to goto a certain width, applying up to the given
            (default is max) force if needed

        Args:
            width (float): A float in the unit of meters
            grasp (boolean): Flag that signals whether to grasp
            speed (float): Gripper operation speed in meters per sec
            force (float): Max gripper force to apply in N. Default to None,
                which gives acceptable force
            block (boolean) : Function blocks by default. If False, the function 
                becomes asynchronous and can be preempted.
            ignore_errors (boolean) : Function ignores errors by default. 
                If False, errors and some exceptions can be thrown.
            skill_desc (string) : Skill description to use for logging on
                control-pc.

        Raises:
            ValueError: If width is less than 0 or greater than TODO(jacky)
                the maximum gripper opening
        '''
        skill = Skill(SkillType.GripperSkill, 
                      TrajectoryGeneratorType.GripperTrajectoryGenerator, 
                      skill_desc=skill_desc)

        skill.add_initial_sensor_values(FC.EMPTY_SENSOR_VALUES)

        skill.add_gripper_params(grasp, width, speed, force)

        goal = skill.create_goal()

        self._send_goal(goal,
                        cb=lambda x: skill.feedback_callback(x),
                        block=block,
                        ignore_errors=ignore_errors)
        # this is so the gripper state can be updated, which happens with a
        # small lag
        sleep(FC.GRIPPER_CMD_SLEEP_TIME)

    def stay_in_position(self, duration=3, translational_stiffness=600,
                         rotational_stiffness=50, k_gains=None, d_gains=None,
                         cartesian_impedances=None, joint_impedances=None, 
                         block=True, ignore_errors=True, skill_desc='', 
                         skill_type=SkillType.ImpedanceControlSkill,
                         feedback_controller_type=FeedbackControllerType.CartesianImpedanceFeedbackController):
        '''Commands the Arm to stay in its current position with provided
        translation and rotation stiffnesses

        Args:
            duration (float) : How much time the robot should stay in place in
                seconds.
            translational_stiffness (float): Translational stiffness factor used
                in the torque controller.
                Default is 600. A value of 0 will allow free translational
                movement.
            rotational_stiffness (float): Rotational stiffness factor used in
                the torque controller.
                Default is 50. A value of 0 will allow free rotational movement.
        '''
        skill = StayInInitialPoseSkill(skill_desc, skill_type, feedback_controller_type)

        skill.add_initial_sensor_values(FC.EMPTY_SENSOR_VALUES)

        if skill_type == SkillType.ImpedanceControlSkill:
            if feedback_controller_type == FeedbackControllerType.CartesianImpedanceFeedbackController:
                if cartesian_impedances is not None:
                    skill.add_cartesian_impedances(cartesian_impedances)
                else:
                    skill.add_feedback_controller_params([translational_stiffness] + [rotational_stiffness]) 
            elif feedback_controller_type == FeedbackControllerType.JointImpedanceFeedbackController:
                if k_gains is not None and d_gains is not None:
                    skill.add_joint_gains(k_gains, d_gains)
                else:
                    skill.add_feedback_controller_params([])
            else:
                skill.add_feedback_controller_params([translational_stiffness] + [rotational_stiffness])
        elif skill_type == SkillType.CartesianPoseSkill:
            if cartesian_impedances is not None:
                skill.add_cartesian_impedances(cartesian_impedances)
            else:
                skill.add_feedback_controller_params([])
        elif skill_type == SkillType.JointPositionSkill:
            if joint_impedances is not None:
                skill.add_joint_impedances(joint_impedances)
            else:
                skill.add_feedback_controller_params([])
        else:
            skill.add_feedback_controller_params([translational_stiffness] + [rotational_stiffness]) 
        
        skill.add_run_time(duration)
        goal = skill.create_goal()

        self._send_goal(goal,
                        cb=lambda x: skill.feedback_callback(x),
                        block=block,
                        ignore_errors=ignore_errors)

    def run_guide_mode_with_selective_joint_compliance(
            self,
            duration=3, joint_impedances=None, k_gains=FC.DEFAULT_K_GAINS,
            d_gains=FC.DEFAULT_D_GAINS, block=True,
            ignore_errors=True, skill_desc='', skill_type=SkillType.ImpedanceControlSkill):
        '''Run guide mode with selective joint compliance given k and d gains
            for each joint

        Args:
            duration (float) : How much time the robot should be in selective
                               joint guide mode in seconds.
            k_gains (list): list of 7 k gains, one for each joint
                            Default is 600., 600., 600., 600., 250., 150., 50..
            d_gains (list): list of 7 d gains, one for each joint
                            Default is 50.0, 50.0, 50.0, 50.0, 30.0, 25.0, 15.0.
        '''
        skill = StayInInitialPoseSkill(skill_desc, skill_type, FeedbackControllerType.JointImpedanceFeedbackController)

        skill.add_initial_sensor_values(FC.EMPTY_SENSOR_VALUES)

        if skill_type == SkillType.ImpedanceControlSkill:
            if k_gains is not None and d_gains is not None:
                skill.add_joint_gains(k_gains, d_gains)
        elif skill_type == SkillType.JointPositionSkill:
            if joint_impedances is not None:
                skill.add_joint_impedances(joint_impedances)
            else:
                skill.add_feedback_controller_params([])
        
        skill.add_run_time(duration)
        goal = skill.create_goal()

        self._send_goal(goal,
                        cb=lambda x: skill.feedback_callback(x),
                        block=block,
                        ignore_errors=ignore_errors)

    def run_guide_mode_with_selective_pose_compliance(
            self, duration=3,
            translational_stiffnesses=FC.DEFAULT_TRANSLATIONAL_STIFFNESSES,
            rotational_stiffnesses=FC.DEFAULT_ROTATIONAL_STIFFNESSES,
            cartesian_impedances=None, block=True,
            ignore_errors=True, skill_desc='', skill_type=SkillType.ImpedanceControlSkill):
        '''Run guide mode with selective pose compliance given translational
        and rotational stiffnesses

        Args:
            duration (float) : How much time the robot should be in selective
                pose guide mode in seconds.
            translational_stiffnesses (list): list of 3 translational stiffnesses,
                one for each axis (x,y,z) Default is 600.0, 600.0, 600.0
            rotational_stiffnesses (list): list of 3 rotational stiffnesses,
                one for axis (roll, pitch, yaw) Default is 50.0, 50.0, 50.0
        '''
        skill = StayInInitialPoseSkill(skill_desc, skill_type, FeedbackControllerType.CartesianImpedanceFeedbackController)

        skill.add_initial_sensor_values(FC.EMPTY_SENSOR_VALUES)

        if skill_type == SkillType.ImpedanceControlSkill:
            if cartesian_impedances is not None:
                skill.add_cartesian_impedances(cartesian_impedances)
            else:
                skill.add_feedback_controller_params([translational_stiffnesses] + [rotational_stiffnesses]) 
        elif skill_type == SkillType.CartesianPoseSkill:
            if cartesian_impedances is not None:
                skill.add_cartesian_impedances(cartesian_impedances)
            else:
                skill.add_feedback_controller_params([])

        skill.add_feedback_controller_params(
                translational_stiffnesses + rotational_stiffnesses)

        skill.add_run_time(duration)
        goal = skill.create_goal()

        self._send_goal(
                goal,
                cb=lambda x: skill.feedback_callback(x),
                block=block,
                ignore_errors=ignore_errors)

    def run_dynamic_joint_position_interpolation(self,
         joints,
         duration=5,
         stop_on_contact_forces=None,
         joint_impedances=None,
         k_gains=None,
         d_gains=None,
         buffer_time=FC.DEFAULT_TERM_BUFFER_TIME,
         ignore_errors=True,
         skill_desc='',
         skill_type=SkillType.ImpedanceControlSkill):
        '''Commands Arm to the given joint configuration

        Args:
            joints (list): A list of 7 numbers that correspond to joint angles
                           in radians
            duration (float): How much time this robot motion should take
            joint_impedances (list): A list of 7 numbers that represent the desired
                                     joint impedances for the internal robot joint
                                     controller

        Raises:
            ValueError: If is_joints_reachable(joints) returns False
        '''
        if isinstance(joints, np.ndarray):
            joints = joints.tolist()

        if not self.is_joints_reachable(joints):
            raise ValueError('Joints not reachable!')

        skill = GoToJointsDynamicsInterpolationSkill(skill_desc, skill_type)

        skill.add_initial_sensor_values(FC.EMPTY_SENSOR_VALUES)

        if joint_impedances is not None:
            skill.add_joint_impedances(joint_impedances)
        elif k_gains is not None and d_gains is not None:
            skill.add_joint_gains(k_gains, d_gains)
        else:
            if skill_type == SkillType.ImpedanceControlSkill:
                skill.add_joint_gains(FC.DEFAULT_K_GAINS, FC.DEFAULT_D_GAINS)
            else:
                skill.add_feedback_controller_params([])

        if stop_on_contact_forces is not None:
            skill.add_contact_termination_params(buffer_time,
                                                 stop_on_contact_forces,
                                                 stop_on_contact_forces)
        else:
            skill.add_termination_params([buffer_time])

        skill.add_goal_joints(duration, joints)
        goal = skill.create_goal()

        self._send_goal(goal,
                        cb=lambda x: skill.feedback_callback(x),
                        block=False,
                        ignore_errors=ignore_errors)

    def run_dynamic_pose_interpolation(self,
                   tool_pose,
                   duration=3,
                   stop_on_contact_forces=None,
                   cartesian_impedances=None,
                   buffer_time=FC.DEFAULT_TERM_BUFFER_TIME,
                   ignore_errors=True,
                   ignore_virtual_walls=False,
                   skill_desc='',
                   skill_type=None):
        '''Commands Arm to the given pose via linear interpolation

        Args:
            tool_pose (RigidTransform) : End-effector pose in tool frame
            duration (float) : How much time this robot motion should take
            stop_on_contact_forces (list): List of 6 floats corresponding to
                force limits on translation (xyz) and rotation about (xyz) axes.
                Default is None. If None then will not stop on contact.
        '''
        if tool_pose.from_frame != 'franka_tool' or tool_pose.to_frame != 'world':
            raise ValueError('pose has invalid frame names! Make sure pose has \
                              from_frame=franka_tool and to_frame=world')

        tool_base_pose = tool_pose * self._tool_delta_pose.inverse()

        if not ignore_virtual_walls and np.any([
            tool_base_pose.translation <= FC.WORKSPACE_WALLS[:, :3].min(axis=0),
            tool_base_pose.translation >= FC.WORKSPACE_WALLS[:, :3].max(axis=0)]):
            raise ValueError('Target pose is outside of workspace virtual walls!')

        skill = GoToPoseDynamicsInterpolationSkill(skill_desc, skill_type)

        skill.add_initial_sensor_values(FC.EMPTY_SENSOR_VALUES)

        if cartesian_impedances is not None:
            skill.add_cartesian_impedances(cartesian_impedances)
        else:
            skill.add_feedback_controller_params(FC.DEFAULT_TORQUE_CONTROLLER_PARAMS)

        if stop_on_contact_forces is not None:
            skill.add_contact_termination_params(buffer_time,
                                                 stop_on_contact_forces,
                                                 stop_on_contact_forces)
        else:
            skill.add_termination_params([buffer_time])

        skill.add_goal_pose_with_matrix(duration,
                                        tool_base_pose.matrix.T.flatten().tolist())
        goal = skill.create_goal()

        self._send_goal(goal,
                        cb=lambda x: skill.feedback_callback(x),
                        block=False,
                        ignore_errors=ignore_errors)


    def open_gripper(self, block=True):
        '''Opens gripper to maximum width
        '''
        self.goto_gripper(FC.GRIPPER_WIDTH_MAX, block=block)

    def close_gripper(self, grasp=True, block=True):
        '''Closes the gripper as much as possible
        '''
        self.goto_gripper(FC.GRIPPER_WIDTH_MIN, grasp=grasp,
                          force=FC.GRIPPER_MAX_FORCE if grasp else None,
                          block=block)

    def run_guide_mode(self, duration=10, block=True):
        self.apply_effector_forces_torques(duration, 0, 0, 0, block=block)

    '''
    Reads
    '''

    def get_robot_state(self):
        '''
        Returns:
            dict of full robot state data
        '''
        return self._state_client.get_data()

    def get_pose(self):
        '''
        Returns:
            pose (RigidTransform) of the current end-effector
        '''
        tool_base_pose = self._state_client.get_pose()

        tool_pose = tool_base_pose * self._tool_delta_pose

        return tool_pose

    def get_joints(self):
        '''
        Returns:
            ndarray of shape (7,) of joint angles in radians
        '''
        return self._state_client.get_joints()

    def get_joint_torques(self):
        '''
        Returns:
            ndarray of shape (7,) of joint torques in Nm
        '''
        return self._state_client.get_joint_torques()

    def get_joint_velocities(self):
        '''
        Returns:
            ndarray of shape (7,) of joint velocities in rads/s
        '''
        return self._state_client.get_joint_velocities()

    def get_gripper_width(self):
        '''
        Returns:
            float of gripper width in meters
        '''
        return self._state_client.get_gripper_width()

    def get_gripper_is_grasped(self):
        '''
        Returns:
            True if gripper is grasping something. False otherwise
        '''
        return self._state_client.get_gripper_is_grasped()

    def get_speed(self, speed):
        '''
        Returns:
            float of current target speed parameter
        '''
        pass

    def get_tool_base_pose(self):
        '''
        Returns:
            RigidTransform of current tool base pose
        '''
        return self._tool_delta_pose.copy()

    '''
    Sets
    '''

    def set_tool_delta_pose(self, tool_delta_pose):
        '''Sets the tool pose relative to the end-effector pose

        Args:
            tool_delta_pose (RigidTransform)
        '''
        if tool_delta_pose.from_frame != 'franka_tool' \
                or tool_delta_pose.to_frame != 'franka_tool_base':
            raise ValueError('tool_delta_pose has invalid frame names! ' \
                             'Make sure it has from_frame=franka_tool, and ' \
                             'to_frame=franka_tool_base')

        self._tool_delta_pose = tool_delta_pose.copy()

    def set_speed(self, speed):
        '''Sets current target speed parameter

        Args:
            speed (float)
        '''
        pass


    '''
    Forward Kinematics, Jacobian, other offline methods
    '''

    _dh_alpha_rot = np.array([
                        [1, 0, 0, 0],
                        [0, -1, -1, 0],
                        [0, -1, -1, 0],
                        [0, 0, 0, 1]
                        ], dtype=np.float32)
    _dh_a_trans = np.array([
                        [1, 0, 0, -1],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]
                        ], dtype=np.float32)
    _dh_d_trans = np.array([
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, -1],
                        [0, 0, 0, 1]
                        ], dtype=np.float32)
    _dh_theta_rot = np.array([
                        [-1, -1, 0, 0],
                        [-1, -1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]
                        ], dtype=np.float32)

    def get_links_transforms(self, joints, use_rigid_transforms=False):
        ''' Computes the forward kinematics of all links and the end-effector

        Args:
            joints (list): A list of 7 numbers that correspond to to the joint angles in radians
            use_rigid_transforms (bool): Optional: Defaults to False.
                                        If True, converts result to RigidTransform objects. This is slower.

        Returns:
            transforms (list): A list of 9 RigidTransforms or ndarrays in panda_link0 to panda_link7, 
                                the franka_tool_base, and franka_tool frames.
            
        '''
        transforms_matrices = np.repeat(np.expand_dims(np.eye(4), 0), len(FC.DH_PARAMS) + 3, axis=0)
        prev_transform = np.eye(4)

        for i in range(len(FC.DH_PARAMS)):
            a, d, alpha, theta = FC.DH_PARAMS[i]

            if i < FC.N_REV_JOINTS:
                theta = theta + joints[i]

            ca, sa = np.cos(alpha), np.sin(alpha)
            ct, st = np.cos(theta), np.sin(theta)
            self._dh_alpha_rot[1, 1] = ca
            self._dh_alpha_rot[1, 2] = -sa
            self._dh_alpha_rot[2, 1] = sa
            self._dh_alpha_rot[2, 2] = ca

            self._dh_a_trans[0, 3] = a
            self._dh_d_trans[2, 3] = d

            self._dh_theta_rot[0, 0] = ct
            self._dh_theta_rot[0, 1] = -st
            self._dh_theta_rot[1, 0] = st
            self._dh_theta_rot[1, 1] = ct

            delta_transform_matrix = self._dh_alpha_rot @ self._dh_a_trans @ self._dh_d_trans @ self._dh_theta_rot

            transforms_matrices[i + 1] = prev_transform @ delta_transform_matrix
            prev_transform = transforms_matrices[i + 1]

        transforms_matrices[10] = transforms_matrices[9]
        transforms_matrices[11] = transforms_matrices[10] @ self._tool_delta_pose.matrix

        rigid_transforms = []
        if use_rigid_transforms:
            for i in range(8):
                rigid_transforms.append(
                    RigidTransform(rotation=transforms_matrices[i, :3, :3], translation=transforms_matrices[i, :3, 3],
                        from_frame='panda_link{}'.format(i + 1), to_frame='world'
                    ))

            rigid_transforms.append(
                RigidTransform(rotation=transforms_matrices[8, :3, :3], translation=transforms_matrices[8, :3, 3],
                    from_frame='panda_hand', to_frame='world'
                ))

            transform_tool_base = rigid_transforms[-1].as_frames(from_frame='franka_tool_base')
            transform_tool = transform_tool_base * self._tool_delta_pose

            rigid_transforms.append(transform_tool_base)
            rigid_transforms.append(transform_tool)

            return rigid_transforms
        else:
            return transforms_matrices

    def get_jacobian(self, joints):
        ''' Computes the analytical jacobian
        
        Args:
            joints (list): A list of 7 numbers that correspond to to the joint angles in radians

        Returns:
            jacobian (ndarray): a 6 by 7 jacobian matrix
        '''
        transforms = self.get_links_transforms(joints, use_rigid_transforms=False)

        joints_pos = transforms[1:FC.N_REV_JOINTS + 1, :3, 3]
        ee_pos = transforms[-1, :3, 3]
        axes = transforms[1:FC.N_REV_JOINTS + 1, :3, 2]

        J = np.r_[np.cross(axes, ee_pos - joints_pos).T, axes.T]

        return J

    def get_collision_boxes_poses(self, joints=None, use_rigid_transforms=False):
        ''' Computes the transforms of all collision boxes in world frame.

        Args:
            joints (list): Optional: Defaults to None
                            A list of 7 numbers that correspond to to the joint angles in radians
                            If None, will use current real robot joints.
            use_rigid_transforms (bool): Optional: Defaults to False.
                                        If True, converts result to RigidTransform objects. This is slower.

        Returns:
            transforms (list): A list of RigidTransforms or ndarrays for all collision boxes in world frame.
            
        '''
        if joints is None:
            joints = self.get_joints()

        fk = self.get_links_transforms(joints, use_rigid_transforms=False)

        box_poses = []
        for box_n, link in enumerate(FC.COLLISION_BOX_LINKS):
            link_transform = fk[link]
            box_pose_world = link_transform @ FC.COLLISION_BOX_POSES[box_n]
            box_poses.append(box_pose_world)

        if use_rigid_transforms:
            box_transforms = [RigidTransform(
                        rotation=box_pose[:3, :3],
                        translation=box_pose[:3, 3], 
                        from_frame='box{}'.format(box_n), to_frame='world'
                    ) for box_n, box_pose in enumerate(box_poses)]
            return box_transforms
        else:
            return box_poses

    def publish_joints(self, joints=None):
        ''' Publish the Franka joints to ROS

        Args:
            joints (list): Optional: Defaults to None
                            A list of 7 numbers that correspond to to the joint angles in radians
                            If None, will use current real robot joints.
        '''
        if joints is None:
            joints = self.get_joints()
                        
        joint_state = JointState()
        joint_state.name = FC.JOINT_NAMES
        joint_state.header.stamp = rospy.Time.now()

        if len(joints) == 7:
            joints = np.concatenate([joints, [0, 0]])
        joint_state.position = joints

        self._joint_state_pub.publish(joint_state)

    def publish_collision_boxes(self, joints=None):
        ''' Publish the Franka collsion boxes to ROS

        Args:
            joints (list): Optional: Defaults to None
                            A list of 7 numbers that correspond to to the joint angles in radians
                            If None, will use current real robot joints.
        '''
        if joints is None:
            joints = self.get_joints()

        box_poses_world = self.get_collision_boxes_poses(joints)

        for i, pose in enumerate(box_poses_world):
            self._collision_boxes_data[i, :3] = pose[:3, 3]
            q = quaternion.from_rotation_matrix(pose[:3, :3])

            for j, k in enumerate('wxyz'):
                self._collision_boxes_data[i, 3 + j] = getattr(q, k)

        self._collision_boxes_pub.publish_boxes(self._collision_boxes_data)

    def check_box_collision(self, box, joints=None):
        ''' Checks if the given joint configurations is in collision with a box

        Args:
            joints (list): Optional: Defaults to None
                            A list of 7 numbers that correspond to to the joint angles in radians
                            If None, will use current real robot joints.

            box (list): The position of the center of the box [x, y, z, r, p, y] and the length, width, and height [l, w, h]

        Returns:
            in_collision (bool)
        '''
        box_pos, box_rpy, box_hsizes = box[:3], box[3:6], box[6:]/2
        box_q = quaternion.from_euler_angles(box_rpy)
        box_axes = quaternion.as_rotation_matrix(box_q)

        self._box_vertices_offset[:,:] = self._vertex_offset_signs * box_hsizes
        box_vertices = (box_axes @ self._box_vertices_offset.T + np.expand_dims(box_pos, 1)).T

        box_hdiag = np.linalg.norm(box_hsizes)
        min_col_dists = box_hdiag + self._collision_box_hdiags

        franka_box_poses = self.get_collision_boxes_poses(joints)
        for i, franka_box_pose in enumerate(franka_box_poses):
            fbox_pos = franka_box_pose[:3, 3]
            fbox_axes = franka_box_pose[:3, :3]

            # coarse collision check
            if np.linalg.norm(fbox_pos - box_pos) > min_col_dists[i]:
                continue

            fbox_vertex_offsets = self._collision_box_vertices_offset[i]
            fbox_vertices = fbox_vertex_offsets @ fbox_axes.T + fbox_pos

            # construct axes
            cross_product_pairs = np.array(list(product(box_axes.T, fbox_axes.T)))
            cross_axes = np.cross(cross_product_pairs[:,0], cross_product_pairs[:,1]).T
            self._collision_proj_axes[:, :3] = box_axes
            self._collision_proj_axes[:, 3:6] = fbox_axes
            self._collision_proj_axes[:, 6:] = cross_axes

            # projection
            box_projs = box_vertices @ self._collision_proj_axes
            fbox_projs = fbox_vertices @ self._collision_proj_axes
            min_box_projs, max_box_projs = box_projs.min(axis=0), box_projs.max(axis=0)
            min_fbox_projs, max_fbox_projs = fbox_projs.min(axis=0), fbox_projs.max(axis=0)

            # check if no separating planes exist
            if np.all([min_box_projs <= max_fbox_projs, max_box_projs >= min_fbox_projs]):
                return True
        
        return False

    def is_joints_in_collision_with_boxes(self, joints=None, boxes=None):
        if boxes is None:
            boxes = FC.WORKSPACE_WALLS

        for box in boxes:
            if self.check_box_collision(box, joints=joints):
                return True

        return False

    '''
    Misc
    '''
    def reset_joints(self, duration=5, skill_desc='', block=True, ignore_errors=True):
        '''Commands Arm to goto hardcoded home joint configuration
        '''
        self.goto_joints(FC.HOME_JOINTS, duration=duration, skill_desc=skill_desc, block=block, ignore_errors=ignore_errors)

    def reset_pose(self, duration=5, skill_desc='', block=True, ignore_errors=True):
        '''Commands Arm to goto hardcoded home pose
        '''
        self.goto_pose(FC.HOME_POSE, duration=duration, skill_desc=skill_desc, block=block, ignore_errors=ignore_errors)

    def is_joints_reachable(self, joints):
        '''
        Returns:
            True if all joints within joint limits
        '''
        for i, val in enumerate(joints):
            if val <= FC.JOINT_LIMITS_MIN[i] or val >= FC.JOINT_LIMITS_MAX[i]:
                return False

        return True
