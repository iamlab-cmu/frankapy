"""
franka_arm.py
====================================
The core module of frankapy
"""

import sys, signal, logging
from time import time, sleep
import numpy as np
from autolab_core import RigidTransform
import quaternion
from itertools import product

import roslib
roslib.load_manifest('franka_interface_msgs')
import rospy
from actionlib import SimpleActionClient
from sensor_msgs.msg import JointState
from franka_interface_msgs.msg import ExecuteSkillAction
from franka_interface_msgs.srv import GetCurrentFrankaInterfaceStatusCmd
from franka_gripper.msg import *

from .skill_list import *
from .exceptions import *
from .franka_arm_state_client import FrankaArmStateClient
from .franka_constants import FrankaConstants as FC
from .franka_interface_common_definitions import *
from .ros_utils import BoxesPublisher


class FrankaArm:

    def __init__(
            self,
            rosnode_name='franka_arm_client', 
            ros_log_level=rospy.INFO,
            robot_num=1,
            with_gripper=True,
            old_gripper=False,
            offline=False,
            init_node=True):

        """
        Initialize a FrankaArm.

        Parameters
        ----------
        rosnode_name : :obj:`str`
            A name for the FrankaArm ROS Node.

        ros_log_level : :obj:`rospy.log_level` of int
            Passed to rospy.init_node.

        robot_num : :obj:`int`
            Number assigned to the current robot.

        with_gripper : :obj:`bool`
            Flag for whether the Franka gripper is attached.

        old_gripper : :obj:`bool`
            Flag for whether to run the franka_ros gripper or the old_gripper 
            implementation.

        offline : :obj:`bool`
            Flag for whether to run the robot in the real world.

        """

        self._execute_skill_action_server_name = \
                '/execute_skill_action_server_node_{}/execute_skill'.format(robot_num)
        self._robot_state_server_name = \
                '/get_current_robot_state_server_node_{}/get_current_robot_state_server'.format(robot_num)
        self._franka_interface_status_server_name = \
                '/get_current_franka_interface_status_server_node_{}/get_current_franka_interface_status_server'.format(robot_num)
        self._gripper_homing_action_server_name = \
                '/franka_gripper_{}/homing'.format(robot_num)
        self._gripper_move_action_server_name = \
                '/franka_gripper_{}/move'.format(robot_num)
        self._gripper_stop_action_server_name = \
                '/franka_gripper_{}/stop'.format(robot_num)
        self._gripper_grasp_action_server_name = \
                '/franka_gripper_{}/grasp'.format(robot_num)
        self._gripper_joint_states_name = \
                '/franka_gripper_{}/joint_states'.format(robot_num)

        self._connected = False
        self._in_skill = False
        self._offline = offline
        self._with_gripper = with_gripper
        self._old_gripper = old_gripper
        self._last_gripper_command = None

        # init ROS
        if init_node:
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
            
            rospy.wait_for_service(self._franka_interface_status_server_name)
            self._get_current_franka_interface_status = rospy.ServiceProxy(
                    self._franka_interface_status_server_name, GetCurrentFrankaInterfaceStatusCmd)

            self._client = SimpleActionClient(
                    self._execute_skill_action_server_name, ExecuteSkillAction)
            self._client.wait_for_server()
            self.wait_for_franka_interface()

            if self._with_gripper and not self._old_gripper:
                self._gripper_homing_client = SimpleActionClient(
                    self._gripper_homing_action_server_name, HomingAction)
                self._gripper_homing_client.wait_for_server()
                self._gripper_move_client = SimpleActionClient(
                    self._gripper_move_action_server_name, MoveAction)
                self._gripper_move_client.wait_for_server()
                self._gripper_grasp_client = SimpleActionClient(
                    self._gripper_grasp_action_server_name, GraspAction)
                self._gripper_grasp_client.wait_for_server()
                self._gripper_stop_client = SimpleActionClient(
                    self._gripper_stop_action_server_name, StopAction)
                self._gripper_stop_client.wait_for_server()

            # done init ROS
            self._connected = True

        # set default identity tool delta pose
        self._tool_delta_pose = RigidTransform(from_frame='franka_tool', 
                                               to_frame='franka_tool_base')

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

    def wait_for_franka_interface(self, timeout=None):
        """
        Blocks execution until franka-interface gives ready signal.

        Parameters
        ----------
        timeout : :obj:`float`
            Timeout in seconds to wait for franka-interface.
        """
        timeout = FC.DEFAULT_FRANKA_INTERFACE_TIMEOUT if timeout is None else timeout
        t_start = time()
        while time() - t_start < timeout:
            franka_interface_status = self._get_current_franka_interface_status().franka_interface_status
            if franka_interface_status.is_ready:
                return
            sleep(1e-2)
        raise FrankaArmCommException('FrankaInterface status not ready for {}s'.format(
            FC.DEFAULT_FRANKA_INTERFACE_TIMEOUT))

    def wait_for_skill(self):
        """
        Blocks execution until skill is done.
        """
        while not self.is_skill_done():
            continue

    def wait_for_gripper(self):
        """
        Blocks execution until gripper is done.
        """
        if self._last_gripper_command == "Grasp":
            done = self._gripper_grasp_client.wait_for_result()
        elif self._last_gripper_command == "Homing":
            done = self._gripper_homing_client.wait_for_result()
        elif self._last_gripper_command == "Stop":
            done = self._gripper_stop_client.wait_for_result()
        elif self._last_gripper_command == "Move":
            done = self._gripper_move_client.wait_for_result()
        sleep(2)


    def is_skill_done(self, ignore_errors=True): 
        """
        Checks whether skill is done.

        Parameters
        ----------
        ignore_errors : :obj:`bool`
            Flag of whether to ignore errors.

        Returns
        -------
        :obj:`bool`
            Flag of whether the skill is done.
        """ 
        if not self._in_skill:  
            return True 

        franka_interface_status = self._get_current_franka_interface_status().franka_interface_status  

        e = None  
        if rospy.is_shutdown(): 
            e = RuntimeError('rospy is down!')  
        elif franka_interface_status.error_description:  
            e = FrankaArmException(franka_interface_status.error_description)  
        elif not franka_interface_status.is_ready: 
            e = FrankaArmFrankaInterfaceNotReadyException() 

        if e is not None: 
            if ignore_errors: 
                self.wait_for_franka_interface() 
            else: 
                raise e 

        done = self._client.wait_for_result(rospy.Duration.from_sec(  
            FC.ACTION_WAIT_LOOP_TIME))

        if done:  
            self._in_skill = False  

        return done

    def stop_skill(self): 
        """
        Stops the current skill.
        """
        if self._connected and self._in_skill:
            self._client.cancel_goal()
        self.wait_for_skill()

    def _sigint_handler_gen(self):
        def sigint_handler(sig, frame):
            if self._connected and self._in_skill:
                self._client.cancel_goal()
            sys.exit(0)

        return sigint_handler

    def _send_goal(self, goal, cb, block=True, ignore_errors=True):
        """
        Sends the goal to the Franka Interface ROS Action Server

        Parameters
        ----------

        goal : :obj:`ExecuteSkillGoal`
            Skill with parameters that are sent to franka-interface.

        cb : :obj:`function`
            Function that takes in ExecuteSkillFeedback during skill execution.

        block : :obj:`bool`
            Flag that determines whether to block until skill is finished.

        ignore_errors : :obj:`bool`
            Flag of whether to ignore errors.

        Returns
        -------
        :obj:`ExecuteSkillResult`
            Skill Results that are sent from franka-interface.

        Raises
        ------
            FrankaArmCommException if a timeout is reached
            FrankaArmException if franka-interface gives an error
            FrankaArmFrankaInterfaceNotReadyException if franka-interface is not ready
        """
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

    """
    Controls
    """

    def goto_pose(self,
                  tool_pose,
                  duration=3,
                  use_impedance=True,
                  dynamic=False,
                  buffer_time=FC.DEFAULT_TERM_BUFFER_TIME,
                  force_thresholds=None,
                  torque_thresholds=None,
                  cartesian_impedances=None,
                  joint_impedances=None,
                  block=True,
                  ignore_errors=True,
                  ignore_virtual_walls=False,
                  skill_desc='GoToPose'):
        """
        Commands Arm to the given pose via min jerk interpolation

        Parameters
        ----------
            tool_pose : :obj:`autolab_core.RigidTransform`
                End-effector pose in tool frame
            duration : :obj:`float`
                How much time this robot motion should take.
            use_impedance : :obj:`bool`
                Function uses our impedance controller by default. 
                If False, uses the Franka cartesian controller.
            dynamic : :obj:`bool`
                Flag that states whether the skill is dynamic.  
                If True, it will use our joint impedance controller 
                and sensor values.
            buffer_time : :obj:`float`
                How much extra time the termination handler will wait
                before stopping the skill after duration has passed.
            force_thresholds : :obj:`list`
                List of 6 floats corresponding to force limits on 
                translation (xyz) and rotation about (xyz) axes.
                Default is None. If None then will not stop on contact.
            torque_thresholds : :obj:`list`
                List of 7 floats corresponding to torque limits on each joint. 
                Default is None. If None then will not stop on contact.
            cartesian_impedances : :obj:`list`
                List of 6 floats corresponding to impedances on translation 
                (xyz) and rotation about (xyz) axes. Default is None. 
                If None then will use default impedances.
            joint_impedances : :obj:`list` 
                List of 7 floats corresponding to impedances on each joint. 
                This is used when use_impedance is False. Default is None. 
                If None then will use default impedances.
            block : :obj:`bool` 
                Function blocks by default. If False, the function becomes
                asynchronous and can be preempted.
            ignore_errors : :obj:`bool` 
                Function ignores errors by default. If False, errors and 
                some exceptions can be thrown.
            ignore_virtual_walls : :obj:`bool` 
                Function checks for collisions with virtual walls by default. 
                If False, the robot no longer checks, which may be dangerous.
            skill_desc : :obj:`str` 
                Skill description to use for logging on the Control PC.

        Raises
        ------
            ValueError: If tool_pose does not have from_frame=franka_tool and 
                        to_frame=world or tool_pose is outside the virtual walls.
        """
        if dynamic:
            skill = Skill(SkillType.ImpedanceControlSkill, 
                          TrajectoryGeneratorType.PassThroughPoseTrajectoryGenerator,
                          feedback_controller_type=FeedbackControllerType.CartesianImpedanceFeedbackController,
                          termination_handler_type=TerminationHandlerType.TimeTerminationHandler, 
                          skill_desc=skill_desc)
            use_impedance=True
            block = False
        else:
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

        skill.set_cartesian_impedances(use_impedance, cartesian_impedances, joint_impedances)

        if not skill.check_for_contact_params(buffer_time, force_thresholds, torque_thresholds):
            if dynamic:
                skill.add_time_termination_params(buffer_time)
            else:
                skill.add_pose_threshold_params(buffer_time, FC.DEFAULT_POSE_THRESHOLDS)

        skill.add_goal_pose(duration, tool_base_pose)
        goal = skill.create_goal()

        self._send_goal(goal,
                        cb=lambda x: skill.feedback_callback(x),
                        block=block,
                        ignore_errors=ignore_errors)

        if dynamic:
            sleep(FC.DYNAMIC_SKILL_WAIT_TIME)

    def goto_pose_delta(self,
                        delta_tool_pose,
                        duration=3,
                        use_impedance=True,
                        buffer_time=FC.DEFAULT_TERM_BUFFER_TIME,
                        force_thresholds=None,
                        torque_thresholds=None,
                        cartesian_impedances=None,
                        joint_impedances=None,
                        block=True,
                        ignore_errors=True,
                        ignore_virtual_walls=False,
                        skill_desc='GoToPoseDelta'):
        """
        Commands Arm to the given delta pose via min jerk interpolation

        Parameters
        ----------
            delta_tool_pose : :obj:`autolab_core.RigidTransform` 
                Delta pose in tool frame
            duration : :obj:`float` 
                How much time this robot motion should take.
            use_impedance : :obj:`bool` 
                Function uses our impedance controller by default. 
                If False, uses the Franka cartesian controller.
            buffer_time : :obj:`float` 
                How much extra time the termination handler will wait
                before stopping the skill after duration has passed.
            force_thresholds : :obj:`list` 
                List of 6 floats corresponding to force limits on translation 
                (xyz) and rotation about (xyz) axes.
                Default is None. If None then will not stop on contact.
            torque_thresholds : :obj:`list` 
                List of 7 floats corresponding to torque limits on each joint. 
                Default is None. If None then will not stop on contact.
            cartesian_impedances : :obj:`list` 
                List of 6 floats corresponding to impedances on translation 
                (xyz) and rotation about (xyz) axes. Default is None. 
                If None then will use default impedances.
            joint_impedances : :obj:`list` 
                List of 7 floats corresponding to impedances on each joint. 
                This is used when use_impedance is False. Default is None. 
                If None then will use default impedances.
            block : :obj:`bool` 
                Function blocks by default. If False, the function becomes
                asynchronous and can be preempted.
            ignore_errors : :obj:`bool` 
                Function ignores errors by default. If False, errors and 
                some exceptions can be thrown.
            ignore_virtual_walls : :obj:`bool` 
                Function checks for collisions with virtual walls by default. 
                If False, the robot no longer checks, which may be dangerous.
            skill_desc : :obj:`str` 
                Skill description to use for logging on the Control PC.

        Raises:
            ValueError: If delta_pose does not have from_frame=franka_tool and 
                        to_frame=franka_tool or from_frame=world and to_frame=world.
        """
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

        skill.add_initial_sensor_values(FC.EMPTY_SENSOR_VALUES)

        skill.set_cartesian_impedances(use_impedance, cartesian_impedances, joint_impedances)

        if not skill.check_for_contact_params(buffer_time, force_thresholds, torque_thresholds):
            skill.add_pose_threshold_params(buffer_time, FC.DEFAULT_POSE_THRESHOLDS)

        if delta_tool_pose.from_frame == 'world' \
                and delta_tool_pose.to_frame == 'world':

            skill.add_goal_pose(duration, delta_tool_pose)

        elif delta_tool_pose.from_frame == 'franka_tool' \
                and delta_tool_pose.to_frame == 'franka_tool':
            starting_pose = self.get_pose()

            starting_tool_base_pose = starting_pose * self._tool_delta_pose.inverse()

            final_goal_pose = starting_pose * delta_tool_pose

            final_tool_base_pose = final_goal_pose * self._tool_delta_pose.inverse()

            delta_tool_base_pose = starting_tool_base_pose.inverse() * final_tool_base_pose

            starting_rotation = RigidTransform(
                rotation=starting_pose.rotation,
                from_frame='franka_tool', to_frame='franka_tool'
            )
            predicted_translation = RigidTransform(
                translation=delta_tool_base_pose.translation,
                from_frame='franka_tool', to_frame='franka_tool'
            )
            actual_translation = starting_rotation * predicted_translation
            delta_tool_base_pose.translation = actual_translation.translation

            if not ignore_virtual_walls and not self._offline:
                if np.any([
                    final_tool_base_pose.translation <= FC.WORKSPACE_WALLS[:, :3].min(axis=0),
                    final_tool_base_pose.translation >= FC.WORKSPACE_WALLS[:, :3].max(axis=0)]):
                    raise ValueError('Target pose is outside of workspace virtual walls!')

            skill.add_goal_pose(duration, delta_tool_base_pose)

        else:
            raise ValueError('delta_pose has invalid frame names! ' \
                             'Make sure delta_pose has from_frame=franka_tool ' \
                             'and to_frame=franka_tool or from_frame=world and \
                             to_frame=world')

        goal = skill.create_goal()

        self._send_goal(goal,
                        cb=lambda x: skill.feedback_callback(x),
                        block=block,
                        ignore_errors=ignore_errors)
            

    def goto_joints(self,
                    joints,
                    duration=5,
                    use_impedance=False,
                    dynamic=False,
                    buffer_time=FC.DEFAULT_TERM_BUFFER_TIME,
                    force_thresholds=None,
                    torque_thresholds=None,
                    cartesian_impedances=None,
                    joint_impedances=None,
                    k_gains=None,
                    d_gains=None,
                    block=True,
                    ignore_errors=True,
                    ignore_virtual_walls=False,
                    skill_desc='GoToJoints'):
        """
        Commands Arm to the given joint configuration

        Parameters
        ----------
            joints : :obj:`list` 
                A list of 7 numbers that correspond to joint angles in radians.
            duration : :obj:`float` 
                How much time this robot motion should take.
            use_impedance : :obj:`bool` 
                Function uses the Franka joint impedance controller by default. 
                If True, uses our joint impedance controller.
            dynamic : :obj:`bool` 
                Flag that states whether the skill is dynamic. If True, it 
                will use our joint impedance controller and sensor values.
            buffer_time : :obj:`float` 
                How much extra time the termination handler will wait
                before stopping the skill after duration has passed.
            force_thresholds : :obj:`list` 
                List of 6 floats corresponding to force limits on translation 
                (xyz) and rotation about (xyz) axes. Default is None. 
                If None then will not stop on contact.
            torque_thresholds : :obj:`list` 
                List of 7 floats corresponding to torque limits on each joint. 
                Default is None. If None then will not stop on contact.
            cartesian_impedances : :obj:`list` 
                List of 6 floats corresponding to impedances on translation 
                (xyz) and rotation about (xyz) axes. Default is None. If None 
                then will use default impedances.
            joint_impedances : :obj:`list` 
                List of 7 floats corresponding to impedances on each joint. 
                This is used when use_impedance is False. Default is None. 
                If None then will use default impedances.
            k_gains : :obj:`list` 
                List of 7 floats corresponding to the k_gains on each joint for 
                our impedance controller. This is used when use_impedance is 
                True. Default is None. If None then will use default k_gains.
            d_gains : :obj:`list` 
                List of 7 floats corresponding to the d_gains on each joint for 
                our impedance controller. This is used when use_impedance is 
                True. Default is None. If None then will use default d_gains.
            block : :obj:`bool` 
                Function blocks by default. If False, the function becomes
                asynchronous and can be preempted.
            ignore_errors : :obj:`bool` 
                Function ignores errors by default. If False, errors and some 
                exceptions can be thrown.
            ignore_virtual_walls : :obj:`bool` 
                Function checks for collisions with virtual walls by default. 
                If False, the robot no longer checks, which may be dangerous.
            skill_desc : :obj:`str` 
                Skill description to use for logging on the Control PC.

        Raises:
            ValueError: If is_joints_reachable(joints) returns False
        """

        joints = np.array(joints).tolist() 

        if dynamic:
            skill = Skill(SkillType.ImpedanceControlSkill, 
                          TrajectoryGeneratorType.PassThroughJointTrajectoryGenerator,
                          feedback_controller_type=FeedbackControllerType.JointImpedanceFeedbackController,
                          termination_handler_type=TerminationHandlerType.TimeTerminationHandler, 
                          skill_desc=skill_desc)
            use_impedance=True
            block = False
        else:
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

        skill.set_joint_impedances(use_impedance, cartesian_impedances, joint_impedances, k_gains, d_gains)

        if not skill.check_for_contact_params(buffer_time, force_thresholds, torque_thresholds):
            if dynamic:
                skill.add_time_termination_params(buffer_time)
            else:
                skill.add_joint_threshold_params(buffer_time, FC.DEFAULT_JOINT_THRESHOLDS)

        skill.add_goal_joints(duration, joints)
        goal = skill.create_goal()

        self._send_goal(goal,
                        cb=lambda x: skill.feedback_callback(x),
                        block=block,
                        ignore_errors=ignore_errors)

        if dynamic:
            sleep(FC.DYNAMIC_SKILL_WAIT_TIME)

    def execute_joint_dmp(self, 
                          joint_dmp_info, 
                          duration, 
                          initial_sensor_values=None,
                          use_impedance=False, 
                          buffer_time=FC.DEFAULT_TERM_BUFFER_TIME,
                          force_thresholds=None,
                          torque_thresholds=None,
                          cartesian_impedances=None,
                          joint_impedances=None, 
                          k_gains=None, 
                          d_gains=None, 
                          block=True, 
                          ignore_errors=True,
                          skill_desc='JointDmp'):
        """
        Commands Arm to execute a given joint dmp

        Parameters
        ----------
            joint_dmp_info : :obj:`dict` 
                Contains all the parameters of a joint DMP
                (tau, alpha, beta, num_basis, num_sensors, mu, h, and weights)
            duration : :obj:`float` 
                How much time this robot motion should take.
            initial_sensor_values : :obj:`list` 
                List of initial sensor values. If None it will default to ones.
            use_impedance : :obj:`bool` 
                Function uses the Franka joint impedance controller by default. 
                If True, uses our joint impedance controller.
            buffer_time : :obj:`float` 
                How much extra time the termination handler will wait
                before stopping the skill after duration has passed.
            force_thresholds : :obj:`list` 
                List of 6 floats corresponding to force limits on translation 
                (xyz) and rotation about (xyz) axes. Default is None. 
                If None then will not stop on contact.
            torque_thresholds : :obj:`list` 
                List of 7 floats corresponding to torque limits on each joint. 
                Default is None. If None then will not stop on contact.
            cartesian_impedances : :obj:`list` 
                List of 6 floats corresponding to impedances on translation 
                (xyz) and rotation about (xyz) axes. Default is None. 
                If None then will use default impedances.
            joint_impedances : :obj:`list` 
                List of 7 floats corresponding to impedances on each joint. 
                This is used when use_impedance is False. Default is None. 
                If None then will use default impedances.
            k_gains : :obj:`list` 
                List of 7 floats corresponding to the k_gains on each joint
                for our impedance controller. This is used when use_impedance is 
                True. Default is None. If None then will use default k_gains.
            d_gains : :obj:`list` 
                List of 7 floats corresponding to the d_gains on each joint
                for our impedance controller. This is used when use_impedance is 
                True. Default is None. If None then will use default d_gains.
            block : :obj:`bool`
                Function blocks by default. If False, the function 
                becomes asynchronous and can be preempted.
            ignore_errors : :obj:`bool`
                Function ignores errors by default. 
                If False, errors and some exceptions can be thrown.
            skill_desc : :obj:`str` 
                Skill description to use for logging on control-pc.
        """

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
        
        skill.set_joint_impedances(use_impedance, cartesian_impedances, joint_impedances, k_gains, d_gains)

        if not skill.check_for_contact_params(buffer_time, force_thresholds, torque_thresholds):
            skill.add_time_termination_params(buffer_time)

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
                         ee_frame = False,
                         use_impedance=True, 
                         buffer_time=FC.DEFAULT_TERM_BUFFER_TIME,
                         force_thresholds=None,
                         torque_thresholds=None,
                         cartesian_impedances=None,
                         joint_impedances=None, 
                         block=True, 
                         ignore_errors=True,
                         skill_desc='PoseDmp'):
        """
        Commands Arm to execute a given pose dmp

        Parameters
        ----------
            pose_dmp_info : :obj:`dict`
                Contains all the parameters of a pose DMP
                (tau, alpha, beta, num_basis, num_sensors, mu, h, and weights)
            duration : :obj:`float`
                How much time this robot motion should take.
            use_goal_formulation : :obj:`bool`
                Flag that represents whether to use the explicit goal pose 
                dmp formulation.
            initial_sensor_values : :obj:`list` 
                List of initial sensor values. If None it will default to ones.
            orientation_only : :obj:`bool`
                Flag that represents if the dmp weights are to generate a dmp 
                only for orientation.
            position_only : :obj:`bool`
                Flag that represents if the dmp weights are to generate a dmp 
                only for position.
            ee_frame : :obj:`bool`
                Flag that indicates whether the dmp is in terms of the 
                end-effector frame.
            use_impedance : :obj:`bool`
                Function uses our impedance controller by default. If False, 
                uses the Franka cartesian controller.
            buffer_time : :obj:`float`
                How much extra time the termination handler will wait
                before stopping the skill after duration has passed.
            force_thresholds : :obj:`list` 
                List of 6 floats corresponding to force limits on translation 
                (xyz) and rotation about (xyz) axes. Default is None. 
                If None then will not stop on contact.
            torque_thresholds : :obj:`list` 
                List of 7 floats corresponding to torque limits on each joint. 
                Default is None. If None then will not stop on contact.
            cartesian_impedances : :obj:`list` 
                List of 6 floats corresponding to impedances on translation 
                (xyz) and rotation about (xyz) axes. Default is None. 
                If None then will use default impedances.
            joint_impedances : :obj:`list` 
                List of 7 floats corresponding to impedances on each joint. 
                This is used when use_impedance is False. Default is None. 
                If None then will use default impedances.
            block : :obj:`bool`
                Function blocks by default. If False, the function becomes
                asynchronous and can be preempted.
            ignore_errors : :obj:`bool`
                Function ignores errors by default. If False, errors and 
                some exceptions can be thrown.
            skill_desc : :obj:`str` 
                Skill description to use for logging on control-pc.
        """

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
            if orientation_only or position_only:
                initial_sensor_values = np.ones(3*pose_dmp_info['num_sensors']).tolist()
            else:
                initial_sensor_values = np.ones(6*pose_dmp_info['num_sensors']).tolist()

        skill.add_initial_sensor_values(initial_sensor_values)  # sensor values

        skill.add_pose_dmp_params(orientation_only, position_only, ee_frame, duration, pose_dmp_info, initial_sensor_values)

        skill.set_cartesian_impedances(use_impedance, cartesian_impedances, joint_impedances)

        if not skill.check_for_contact_params(buffer_time, force_thresholds, torque_thresholds):
            skill.add_time_termination_params(buffer_time)

        goal = skill.create_goal()

        self._send_goal(goal,
                        cb=lambda x: skill.feedback_callback(x),
                        block=block,
                        ignore_errors=ignore_errors)

    def execute_quaternion_pose_dmp(self, 
                                    position_dmp_info,
                                    quat_dmp_info,
                                    duration,
                                    goal_quat,
                                    goal_quat_time, 
                                    use_goal_formulation=False,
                                    initial_sensor_values=None,
                                    ee_frame=False,
                                    use_impedance=True, 
                                    buffer_time=FC.DEFAULT_TERM_BUFFER_TIME,
                                    force_thresholds=None,
                                    torque_thresholds=None,
                                    cartesian_impedances=None,
                                    joint_impedances=None, 
                                    block=True, 
                                    ignore_errors=True,
                                    skill_desc='QuaternionPoseDmp'):
        '''Commands Arm to execute a given quaternion pose dmp

        Args:
            position_dmp_info : :obj:`dict`
                Contains all the parameters of a position DMP
                (tau, alpha, beta, num_basis, num_sensors, mu, h, and weights)
            quat_dmp_info : :obj:`dict`
                Contains all the parameters of a quaternion DMP
                (tau, alpha, beta, num_basis, num_sensors, mu, h, and weights)
            duration : :obj:`float`
                How much time this robot motion should take.
            goal_quat : :obj:`list` 
                List of 4 floats that represents the quaternion goal to reach. 
                This assumes an explicit goal formulation for the quaternion DMPs.
            goal_quat_time : :obj:`float`
                Time to reach goal for quaternion DMPs.
            use_goal_formulation : :obj:`bool`
                Flag that represents whether to use the explicit goal pose dmp formulation.
            initial_sensor_values : :obj:`list` 
                List of initial sensor values. If None it will default to ones.
            ee_frame : :obj:`bool`
                Flag that indicates whether the dmp is in terms of the 
                end-effector frame.
            use_impedance : :obj:`bool`
                Function uses our impedance controller by default. If False, 
                uses the Franka cartesian controller.
            buffer_time : :obj:`float`
                How much extra time the termination handler will wait
                before stopping the skill after duration has passed.
            force_thresholds : :obj:`list` 
                List of 6 floats corresponding to force limits on translation 
                (xyz) and rotation about (xyz) axes. Default is None. 
                If None then will not stop on contact.
            torque_thresholds : :obj:`list` 
                List of 7 floats corresponding to torque limits on each joint. 
                Default is None. If None then will not stop on contact.
            cartesian_impedances : :obj:`list` 
                List of 6 floats corresponding to impedances on translation 
                (xyz) and rotation about (xyz) axes. Default is None. 
                If None then will use default impedances.
            joint_impedances : :obj:`list` 
                List of 7 floats corresponding to impedances on each joint. 
                This is used when use_impedance is False. Default is None. 
                If None then will use default impedances.
            block : :obj:`bool`
                Function blocks by default. If False, the function becomes
                asynchronous and can be preempted.
            ignore_errors : :obj:`bool`
                Function ignores errors by default. If False, errors and some
                exceptions can be thrown.
            skill_desc : :obj:`str` 
                Skill description to use for logging on control-pc.
        '''

        if use_impedance:
            skill = Skill(SkillType.ImpedanceControlSkill, 
                            TrajectoryGeneratorType.QuaternionPoseDmpTrajectoryGenerator,
                            feedback_controller_type=FeedbackControllerType.CartesianImpedanceFeedbackController,
                            termination_handler_type=TerminationHandlerType.TimeTerminationHandler, 
                            skill_desc=skill_desc)
        else:
            skill = Skill(SkillType.CartesianPoseSkill, 
                            TrajectoryGeneratorType.QuaternionPoseDmpTrajectoryGenerator,
                            feedback_controller_type=FeedbackControllerType.SetInternalImpedanceFeedbackController,
                            termination_handler_type=TerminationHandlerType.TimeTerminationHandler, 
                            skill_desc=skill_desc)

        if initial_sensor_values is None:
            initial_sensor_values = np.ones(3*position_dmp_info['num_sensors']).tolist()

        skill.add_initial_sensor_values(initial_sensor_values)  # sensor values

        skill.add_quaternion_pose_dmp_params(ee_frame, duration, position_dmp_info, quat_dmp_info, initial_sensor_values, goal_quat, goal_quat_time)
        skill.set_cartesian_impedances(use_impedance, cartesian_impedances, joint_impedances)

        if not skill.check_for_contact_params(buffer_time, force_thresholds, torque_thresholds):
            skill.add_time_termination_params(buffer_time)

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
                                      buffer_time=FC.DEFAULT_TERM_BUFFER_TIME,
                                      force_thresholds=None,
                                      torque_thresholds=None,
                                      block=True,
                                      ignore_errors=True,
                                      skill_desc='ForceTorque'):
        """
        Applies the given end-effector forces and torques in N and Nm

        Parameters
        ----------
            run_duration : :obj:`float`
                How much time this robot motion should take.
            acc_duration : :obj:`float`
                How long to acc/de-acc to achieve desired force.
            max_translation : :obj:`float`
                A float in the unit of meters. Max translation before the 
                robot deaccelerates.
            max_rotation : :obj:`float`
                A float in the unit of rad. Max rotation before the robot
                deaccelerates.
            forces : :obj:`list` 
                Optional (defaults to None). A list of 3 numbers that correspond 
                to end-effector forces in (xyz) directions.
            torques : :obj:`list` 
                Optional (defaults to None). A list of 3 numbers that correspond 
                to end-effector torques in 3 rotational axes.
            buffer_time : :obj:`float`
                How much extra time the termination handler will wait
                before stopping the skill after duration has passed.
            force_thresholds : :obj:`list` 
                List of 6 floats corresponding to force limits on translation 
                (xyz) and rotation about (xyz) axes. Default is None. 
                If None then will not stop on contact.
            torque_thresholds : :obj:`list` 
                List of 7 floats corresponding to torque limits on each joint. 
                Default is None. If None then will not stop on contact.
            block : :obj:`bool`
                Function blocks by default. If False, the function becomes
                asynchronous and can be preempted.
            ignore_errors : :obj:`bool`
                Function ignores errors by default. If False, errors and some 
                exceptions can be thrown.
            skill_desc : :obj:`str` 
                Skill description to use for logging on control-pc.

        Raises:
            ValueError if acc_duration > 0.5*run_duration, or if forces are
            too large
        """
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

        skill = Skill(SkillType.ForceTorqueSkill, 
                      TrajectoryGeneratorType.ImpulseTrajectoryGenerator,
                      feedback_controller_type=FeedbackControllerType.PassThroughFeedbackController,
                      termination_handler_type=TerminationHandlerType.TimeTerminationHandler, 
                      skill_desc=skill_desc)

        skill.add_initial_sensor_values(FC.EMPTY_SENSOR_VALUES)
        
        skill.add_impulse_params(run_duration, acc_duration, max_translation, max_rotation, forces, torques)

        if not skill.check_for_contact_params(buffer_time, force_thresholds, torque_thresholds):
            skill.add_time_termination_params(buffer_time)

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
                                         buffer_time=FC.DEFAULT_TERM_BUFFER_TIME,
                                         force_thresholds=None,
                                         torque_thresholds=None,
                                         block=True,
                                         ignore_errors=True,
                                         skill_desc='ForcesAlongAxis'):
        """
        Applies the given end-effector forces and torques in N and Nm

        Parameters
        ----------
            run_duration : :obj:`float`
                How much time this robot motion should take.
            acc_duration : :obj:`float`
                How long to acc/de-acc to achieve desired force.
            max_translation : :obj:`float`
                A float in the unit of meters. Max translation before the 
                robot deaccelerates.
            forces : :obj:`list` 
                Optional (defaults to None). A list of 3 numbers that correspond 
                to the magnitude of the end-effector forces and the axis.
            buffer_time : :obj:`float`
                How much extra time the termination handler will wait
                before stopping the skill after duration has passed.
            force_thresholds : :obj:`list` 
                List of 6 floats corresponding to force limits on translation 
                (xyz) and rotation about (xyz) axes. Default is None. 
                If None then will not stop on contact.
            torque_thresholds : :obj:`list` 
                List of 7 floats corresponding to torque limits on each joint. 
                Default is None. If None then will not stop on contact.
            block : :obj:`bool`
                Function blocks by default. If False, the function becomes
                asynchronous and can be preempted.
            ignore_errors : :obj:`bool`
                Function ignores errors by default. If False, errors and some 
                exceptions can be thrown.
            skill_desc : :obj:`str` 
                Skill description to use for logging on control-pc.
                    
        Raises:
            ValueError if acc_duration > 0.5*run_duration, or if forces are
            too large
        """
        if acc_duration > 0.5 * run_duration:
            raise ValueError(
                    'acc_duration must be smaller than half of run_duration!')
        if np.linalg.norm(forces) * run_duration > FC.MAX_LIN_MOMENTUM_CONSTRAINED:
            raise ValueError('Linear momentum magnitude exceeds safety ' \
                    'threshold of {}'.format(FC.MAX_LIN_MOMENTUM_CONSTRAINED))

        forces = np.array(forces)
        force_axis = forces / np.linalg.norm(forces)

        skill = Skill(SkillType.ForceTorqueSkill, 
                      TrajectoryGeneratorType.ImpulseTrajectoryGenerator,
                      feedback_controller_type=FeedbackControllerType.ForceAxisImpedenceFeedbackController,
                      termination_handler_type=TerminationHandlerType.TimeTerminationHandler, 
                      skill_desc=skill_desc)

        skill.add_initial_sensor_values(FC.EMPTY_SENSOR_VALUES)
        
        skill.add_impulse_params(run_duration, acc_duration, max_translation, 0, forces.tolist(), [0, 0, 0])

        skill.add_force_axis_params(FC.DEFAULT_FORCE_AXIS_TRANSLATIONAL_STIFFNESS,
                                    FC.DEFAULT_FORCE_AXIS_ROTATIONAL_STIFFNESS,
                                    force_axis.tolist())

        if not skill.check_for_contact_params(buffer_time, force_thresholds, torque_thresholds):
            skill.add_time_termination_params(buffer_time)

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
                     epsilon_inner=0.08,
                     epsilon_outer=0.08, 
                     block=True, 
                     ignore_errors=True, 
                     skill_desc='GoToGripper'):
        """
        Commands gripper to goto a certain width, applying up to the given
            (default is max) force if needed

        Parameters
        ----------
            width : :obj:`float`
                Desired width of the gripper in the unit of meters.
            grasp : :obj:`bool`
                Flag that signals whether to grasp.
            speed : :obj:`float`
                Gripper operation speed in meters per sec.
            force : :obj:`float`
                Max gripper force to apply in N. Default to None,
                which gives acceptable force.
            epsilon_inner : :obj:`float`
                Max tolerated deviation when the actual grasped 
                width is smaller than the commanded grasp width.
            epsilon_outer : :obj:`float`
                Max tolerated deviation when the actual grasped 
                width is larger than the commanded grasp width.
            block : :obj:`bool`
                Function blocks by default. If False, the function 
                becomes asynchronous and can be preempted.
            ignore_errors : :obj:`bool`
                Function ignores errors by default. If False, errors 
                and some exceptions can be thrown.
            skill_desc : :obj:`str` 
                Skill description to use for logging on control-pc.

        Raises:
            ValueError: If width is less than 0 or greater than TODO(jacky)
                the maximum gripper opening
        """

        if width < FC.GRIPPER_WIDTH_MIN or width > FC.GRIPPER_WIDTH_MAX:
            raise ValueError(
                    'gripper width must be within the gripper limits of ' \
                    '{} and {}'.format(FC.GRIPPER_WIDTH_MIN,FC.GRIPPER_WIDTH_MAX))

        if self._old_gripper:
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
        else:
            if grasp:
                grasp_skill = GraspGoal()
                grasp_skill.width = width
                grasp_skill.speed = speed
                grasp_skill.force = force
                grasp_skill.epsilon.inner = epsilon_inner
                grasp_skill.epsilon.outer = epsilon_outer
                self._gripper_grasp_client.send_goal(grasp_skill)
            else:
                move_skill = MoveGoal()
                move_skill.width = width
                move_skill.speed = speed
                self._gripper_move_client.send_goal(move_skill)
            if block:
                self.wait_for_gripper()

    def selective_guidance_mode(self, 
                                duration=5,
                                use_joints=False, 
                                use_impedance=False,
                                use_ee_frame=False,
                                buffer_time=FC.DEFAULT_TERM_BUFFER_TIME,
                                force_thresholds=None,
                                torque_thresholds=None,
                                cartesian_impedances=None,
                                joint_impedances=None,
                                k_gains=None,
                                d_gains=None,
                                block=True,
                                ignore_errors=True,
                                ignore_virtual_walls=False,
                                skill_desc='SelectiveGuidance'):
        """
        Commands the Arm to stay in its current position with selective impedances
        that allow guidance in either certain joints or in cartesian pose.

        Parameters
        ----------
            duration : :obj:`float`
                How much time this guidance should take
            use_joints : :obj:`bool`
                Function uses cartesian impedance controller by default. 
                If True, it uses joint impedance.
            use_impedance : :obj:`bool`
                Function uses the Franka joint impedance controller by default. 
                If True, uses our joint impedance controller.
            use_ee_frame : :obj:`bool`
                Function uses the end-effector cartesian feedback controller
                only when use_impedance is True.
            buffer_time : :obj:`float`
                How much extra time the termination handler will wait
                before stopping the skill after duration has passed.
            force_thresholds : :obj:`list` 
                List of 6 floats corresponding to force limits on translation 
                (xyz) and rotation about (xyz) axes. Default is None. 
                If None then will not stop on contact.
            torque_thresholds : :obj:`list` 
                List of 7 floats corresponding to torque limits on each joint. 
                Default is None. If None then will not stop on contact.
            cartesian_impedances : :obj:`list` 
                List of 6 floats corresponding to impedances on translation 
                (xyz) and rotation about (xyz) axes. Default is None. 
                If None then will use default impedances.
            joint_impedances : :obj:`list` 
                List of 7 floats corresponding to impedances on each joint. 
                This is used when use_impedance is False. Default is None. 
                If None then will use default impedances.
            k_gains : :obj:`list` 
                List of 7 floats corresponding to the k_gains on each joint
                for our impedance controller. This is used when use_impedance is 
                True. Default is None. If None then will use default k_gains.
            d_gains : :obj:`list` 
                List of 7 floats corresponding to the d_gains on each joint
                for our impedance controller. This is used when use_impedance is 
                True. Default is None. If None then will use default d_gains.
            block : :obj:`bool`
                Function blocks by default. If False, the function becomes
                asynchronous and can be preempted.
            ignore_errors : :obj:`bool`
                Function ignores errors by default. If False, errors and some 
                exceptions can be thrown.
            ignore_virtual_walls : :obj:`bool`
                Function checks for collisions with virtual walls by default. 
                If False, the robot no longer checks, which may be dangerous.
            skill_desc : :obj:`str` 
                Skill description to use for logging on control-pc.
        """
        if use_joints:
            if use_impedance:
                skill = Skill(SkillType.ImpedanceControlSkill, 
                              TrajectoryGeneratorType.StayInInitialJointsTrajectoryGenerator,
                              feedback_controller_type=FeedbackControllerType.JointImpedanceFeedbackController,
                              termination_handler_type=TerminationHandlerType.FinalJointTerminationHandler, 
                              skill_desc=skill_desc)
            else:
                skill = Skill(SkillType.JointPositionSkill, 
                              TrajectoryGeneratorType.StayInInitialJointsTrajectoryGenerator,
                              feedback_controller_type=FeedbackControllerType.SetInternalImpedanceFeedbackController,
                              termination_handler_type=TerminationHandlerType.FinalJointTerminationHandler, 
                              skill_desc=skill_desc)
        else:
            if use_impedance:
                if use_ee_frame:
                    skill = Skill(SkillType.ImpedanceControlSkill, 
                                  TrajectoryGeneratorType.StayInInitialPoseTrajectoryGenerator,
                                  feedback_controller_type=FeedbackControllerType.EECartesianImpedanceFeedbackController,
                                  termination_handler_type=TerminationHandlerType.TimeTerminationHandler, 
                                  skill_desc=skill_desc)
                else:
                    skill = Skill(SkillType.ImpedanceControlSkill, 
                                  TrajectoryGeneratorType.StayInInitialPoseTrajectoryGenerator,
                                  feedback_controller_type=FeedbackControllerType.CartesianImpedanceFeedbackController,
                                  termination_handler_type=TerminationHandlerType.TimeTerminationHandler, 
                                  skill_desc=skill_desc)
            else:
                skill = Skill(SkillType.CartesianPoseSkill, 
                              TrajectoryGeneratorType.StayInInitialPoseTrajectoryGenerator,
                              feedback_controller_type=FeedbackControllerType.SetInternalImpedanceFeedbackController,
                              termination_handler_type=TerminationHandlerType.TimeTerminationHandler, 
                              skill_desc=skill_desc)

        skill.add_initial_sensor_values(FC.EMPTY_SENSOR_VALUES)
        
        skill.add_run_time(duration)

        if use_joints:
            skill.set_joint_impedances(use_impedance, cartesian_impedances, joint_impedances, k_gains, d_gains)
        else:
            skill.set_cartesian_impedances(use_impedance, cartesian_impedances, joint_impedances)

        if not skill.check_for_contact_params(buffer_time, force_thresholds, torque_thresholds):
            skill.add_time_termination_params(buffer_time)

        goal = skill.create_goal()

        self._send_goal(goal,
                        cb=lambda x: skill.feedback_callback(x),
                        block=block,
                        ignore_errors=ignore_errors)

    def open_gripper(self, block=True, skill_desc='OpenGripper'):
        """
        Opens gripper to maximum width

        Parameters
        ----------
            block : :obj:`bool`
                Function blocks by default. If False, the function becomes
                asynchronous and can be preempted.
            skill_desc : :obj:`str` 
                Skill description to use for logging on control-pc.
        """
        self.goto_gripper(FC.GRIPPER_WIDTH_MAX, block=block, skill_desc=skill_desc)

    def close_gripper(self, grasp=True, block=True, skill_desc='CloseGripper'):
        """
        Closes the gripper as much as possible

        Parameters
        ----------
            grasp : :obj:`bool`
                Flag that signals whether to grasp.
            block : :obj:`bool`
                Function blocks by default. If False, the function becomes
                asynchronous and can be preempted.
            skill_desc : :obj:`str` 
                Skill description to use for logging on control-pc.
        """
        self.goto_gripper(FC.GRIPPER_WIDTH_MIN, grasp=grasp,
                          force=FC.GRIPPER_MAX_FORCE if grasp else None,
                          block=block, skill_desc=skill_desc)

    def home_gripper(self, block=True, skill_desc='HomeGripper'):
        """
        Homes the gripper.

        Parameters
        ----------
            block : :obj:`bool`
                Function blocks by default. If False, the function becomes
                asynchronous and can be preempted.
            skill_desc : :obj:`str` 
                Skill description to use for logging on control-pc.
        """
        self._last_gripper_command = 'Homing'
        homing_skill = HomingGoal()
        self._gripper_homing_client.send_goal(homing_skill)
        if block:
            self.wait_for_gripper()

    def stop_gripper(self, block=True, skill_desc='StopGripper'):
        """
        Stops the gripper.

        Parameters
        ----------
            block : :obj:`bool`
                Function blocks by default. If False, the function becomes
                asynchronous and can be preempted.
            skill_desc : :obj:`str` 
                Skill description to use for logging on control-pc.
        """
        self._last_gripper_command = 'Stop'
        stop_skill = StopGoal()
        self._gripper_stop_client.send_goal(stop_skill)
        if block:
            self.wait_for_gripper()

    def run_guide_mode(self, duration=10, block=True, skill_desc='GuideMode'):
        """
        Runs guide mode which allows you to move the robot freely.
        
        Parameters
        ----------
            duration : :obj:`float`
                How much time this guidance should take
            block : :obj:`bool`
                Function blocks by default. If False, the function becomes
                asynchronous and can be preempted.
            skill_desc : :obj:`str` 
                Skill description to use for logging on control-pc.
        """
        self.apply_effector_forces_torques(duration, 0, 0, 0, block=block, skill_desc=skill_desc)

    def run_dynamic_force_position(self,
                  duration=3,
                  buffer_time=FC.DEFAULT_TERM_BUFFER_TIME,
                  force_thresholds=None,
                  torque_thresholds=None,
                  position_kps_cart=FC.DEFAULT_TRANSLATIONAL_STIFFNESSES + FC.DEFAULT_ROTATIONAL_STIFFNESSES,
                  force_kps_cart=FC.DEFAULT_HFPC_FORCE_GAIN,
                  position_kps_joint=FC.DEFAULT_K_GAINS,
                  force_kps_joint=FC.DEFAULT_HFPC_FORCE_GAIN,
                  S=FC.DEFAULT_HFPC_S,
                  interpolate=False,
                  use_cartesian_gains=True,
                  ignore_errors=True,
                  ignore_virtual_walls=False,
                  skill_desc=''):
        """
        Commands Arm to run dynamic hybrid force position control

        Parameters
        ----------
            duration : :obj:`float`
                How much time this robot motion should take.
            use_impedance : :obj:`bool`
                Function uses our impedance controller by default. 
                If False, uses the Franka cartesian controller.
            buffer_time : :obj:`float`
                How much extra time the termination handler will wait
                before stopping the skill after duration has passed.
            force_thresholds : :obj:`list` 
                List of 6 floats corresponding to force limits on translation 
                (xyz) and rotation about (xyz) axes. Default is None. 
                If None then will not stop on contact.
            torque_thresholds : :obj:`list` 
                List of 7 floats corresponding to torque limits on each joint. 
                Default is None. If None then will not stop on contact.
            position_kp_cart : :obj:`list` 
                List of 6 floats corresponding to proportional gain used for 
                position errors in cartesian space.
            force_kp_cart : :obj:`list` 
                List of 6 floats corresponding to proportional gain used for 
                force errors in cartesian space.
            position_kp_joint : :obj:`list` 
                List of 7 floats corresponding to proportional gain used for 
                position errors in joint space.
            force_kp_joint : :obj:`list` 
                List of 6 floats corresponding to proportional gain used for 
                force errors in joint space.
            S : :obj:`list` 
                List of 6 numbers between 0 and 1 for the HFPC selection matrix.
            interpolate : :obj:`bool`
                Whether or not to perform linear interpolation in between way points.
            use_cartesian_gains : :obj:`bool`
                Whether to use cartesian gains or joint gains.
            ignore_errors : :obj:`bool`
                Function ignores errors by default. If False, errors and some 
                exceptions can be thrown.
            ignore_virtual_walls : :obj:`bool`
                Function checks for collisions with virtual walls by default. 
                If False, the robot no longer checks, which may be dangerous.
            skill_desc : :obj:`str` 
                Skill description to use for logging on control-pc.
        """
        if interpolate:
            traj_gen = TrajectoryGeneratorType.LinearForcePositionTrajectoryGenerator
        else:
            traj_gen = TrajectoryGeneratorType.PassThroughForcePositionTrajectoryGenerator
        skill = Skill(SkillType.ForceTorqueSkill, 
                        traj_gen,
                        feedback_controller_type=FeedbackControllerType.ForcePositionFeedbackController,
                        termination_handler_type=TerminationHandlerType.TimeTerminationHandler, 
                        skill_desc=skill_desc)

        skill.add_initial_sensor_values(FC.EMPTY_SENSOR_VALUES)
        skill.add_force_position_params(position_kps_cart, force_kps_cart, position_kps_joint, force_kps_joint, S, use_cartesian_gains)
        skill.add_run_time(duration)

        if not skill.check_for_contact_params(buffer_time, force_thresholds, torque_thresholds):
            skill.add_time_termination_params(buffer_time)
            
        goal = skill.create_goal()

        self._send_goal(goal,
                        cb=lambda x: skill.feedback_callback(x),
                        block=False,
                        ignore_errors=ignore_errors)

        sleep(FC.DYNAMIC_SKILL_WAIT_TIME)

    """
    Reads
    """

    def get_robot_state(self):
        """
        Returns
        -------
            robot_state : :obj:`dict`
                Dict that contains all of the robot's current
                state information.
        """
        return self._state_client.get_data()

    def get_pose(self, include_tool_offset=True):
        """
        Returns
        -------
            pose : :obj:`autolab_core.RigidTransform`
                Current pose of the end-effector including the transform
                to the end of the tool.
        """
        tool_base_pose = self._state_client.get_pose()

        if include_tool_offset:
            tool_pose = tool_base_pose * self._tool_delta_pose
            return tool_pose
        else:
            return tool_base_pose

    def get_joints(self):
        """
        Returns the current joint positions.

        Returns
        -------
            joint_positions : :obj:`numpy.ndarray`
                7 floats that represent each joint's position in radians.
        """
        return self._state_client.get_joints()

    def get_joint_torques(self):
        """
        Returns the current joint torques.

        Returns
        -------
            joint_torques : :obj:`numpy.ndarray`
                7 floats that represent each joint's torque in Nm.
        """
        return self._state_client.get_joint_torques()

    def get_joint_velocities(self):
        """
        Returns the current joint velocities.

        Returns
        -------
            joint_velocities : :obj:`numpy.ndarray`
                7 floats that represent each joint's velocity in rads/s.
        """
        return self._state_client.get_joint_velocities()

    def get_gripper_width(self):
        """
        Returns the current gripper width.

        Returns
        -------
            gripper_width : :obj:`float`
                Robot gripper width in meters.
        """
        if self._old_gripper:
            return self._state_client.get_gripper_width()
        else:
            gripper_data = rospy.wait_for_message(self._gripper_joint_states_name, JointState)
            return gripper_data.position[0] + gripper_data.position[1]

    def get_gripper_is_grasped(self):
        """
        Returns a flag that represents if the gripper is grasping something.

        WARNING: THIS FUNCTION HAS BEEN DEPRECATED WHEN WE SWITCHED
        TO THE FRANKA_ROS GRIPPER CONTROLLER BECAUSE IT DOES NOT PUBLISH
        THE IS GRASPED FLAG. YOU WILL NEED TO DETERMINE WHEN THE ROBOT
        IS GRASPING AN ITEM YOURSELF USING THE GRIPPER WIDTH.

        Returns
        -------
            is_grasped : :obj:`bool`
                Returns True if gripper is grasping something. False otherwise.
        """
        return self._state_client.get_gripper_is_grasped()

    def get_tool_base_pose(self):
        """
        Returns the tool delta pose.

        Returns
        -------
            tool_delta_pose : :obj:`autolab_core.RigidTransform`
                RigidTransform that represents the transform between the 
                end-effector and the end of the tool.
        """
        return self._tool_delta_pose.copy()

    def get_ee_force_torque(self):
        """
        Returns the current end-effector's sensed force_torque.

        Returns
        -------
            ee_force_torque : :obj:`numpy.ndarray`
                A numpy ndarray of 6 floats that represents the current
                end-effector's sensed force_torque.
        """
        return self._state_client.get_ee_force_torque()

    def get_finger_poses(self):
        """
        Returns the current poses of the left and right fingers.

        Returns
        -------
            left_finger_pose : :obj:`autolab_core.RigidTransform`
                RigidTransform that represents the transform of the left finger.
                The left finger is +y from the gripper pose
            right_finger_pose : :obj:`autolab_core.RigidTransform`
                RigidTransform that represents the transform of the right finger.
                The right finger is -y from the gripper pose
        """
        ee_pose = self.get_pose()
        delta_gripper_width = self.get_gripper_width() / 2

        left_finger_pose = ee_pose * RigidTransform(
            translation=np.array([0, delta_gripper_width, 0]),
            from_frame='finger_left', to_frame=ee_pose.from_frame
        )
        right_finger_pose = ee_pose * RigidTransform(
            translation=np.array([0, -delta_gripper_width, 0]),
            from_frame='finger_right', to_frame=ee_pose.from_frame
        )
        return left_finger_pose, right_finger_pose

    """
    Sets
    """

    def set_tool_delta_pose(self, tool_delta_pose):
        """
        Sets the tool pose relative to the end-effector pose

        Parameters
        ----------
            tool_delta_pose : :obj:`autolab_core.RigidTransform`
                RigidTransform that represents the transform between the 
                end-effector and the end of the tool.
        """
        if tool_delta_pose.from_frame != 'franka_tool' \
                or tool_delta_pose.to_frame != 'franka_tool_base':
            raise ValueError('tool_delta_pose has invalid frame names! ' \
                             'Make sure it has from_frame=franka_tool, and ' \
                             'to_frame=franka_tool_base')

        self._tool_delta_pose = tool_delta_pose.copy()


    """
    Forward Kinematics, Jacobian, other offline methods
    """

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
        """ 
        Computes the forward kinematics of all links and the end-effector

        Parameters
        ----------
            joints : :obj:`list` 
                A list of 7 numbers that correspond to to the joint angles in radians
            use_rigid_transforms : :obj:`bool`  
                Optional: Defaults to False.
                If True, converts result to RigidTransform objects. This is slower.

        Returns
        -------
            transforms : :obj:`list` 
                A list of 9 RigidTransforms or ndarrays in panda_link0 to panda_link7, 
                the franka_tool_base, and franka_tool frames.
            
        """
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
        """ 
        Computes the geometric jacobian
        
        Parameters
        ----------
            joints : :obj:`list` 
                A list of 7 numbers that correspond to the joint angles in radians.

        Returns
        -------
            jacobian : :obj:`numpy.ndarray`
                A 6 by 7 jacobian matrix.
        """
        transforms = self.get_links_transforms(joints, use_rigid_transforms=False)

        joints_pos = transforms[1:FC.N_REV_JOINTS + 1, :3, 3]
        ee_pos = transforms[-1, :3, 3]
        axes = transforms[1:FC.N_REV_JOINTS + 1, :3, 2]

        J = np.r_[np.cross(axes, ee_pos - joints_pos).T, axes.T]

        return J

    def get_collision_boxes_poses(self, joints=None, use_rigid_transforms=False):
        """ 
        Computes the transforms of all collision boxes in world frame.

        Parameters
        ----------
            joints : :obj:`list` 
                Optional: Defaults to None
                A list of 7 numbers that correspond to to the joint angles in radians
                If None, will use current real robot joints.
            use_rigid_transforms : :obj:`bool` 
                Optional: Defaults to False.
                If True, converts result to RigidTransform objects. This is slower.

        Returns
        -------
            transforms : :obj:`list` 
                A list of RigidTransforms or ndarrays for all collision boxes in world frame.
            
        """
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
        """ 
        Publish the Franka joints to ROS

        Parameters
        ----------
            joints : :obj:`list` 
                Optional: Defaults to None
                A list of 7 numbers that correspond to to the joint angles in radians
                If None, will use current real robot joints.
        """
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
        """ 
        Publish the Franka collsion boxes to ROS

        Parameters
        ----------
            joints : :obj:`list` 
                Optional: Defaults to None
                A list of 7 numbers that correspond to to the joint angles in radians.
                If None, will use current real robot joints.
        """
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
        """ 
        Checks if the given joint configuration is in collision with a box

        Parameters
        ----------
            joints : :obj:`list` 
                Optional: Defaults to None
                A list of 7 numbers that correspond to to the joint angles in radians
                If None, will use current real robot joints.

            box : :obj:`list` 
                The position of the center of the box [x, y, z, r, p, y] and the 
                length, width, and height [l, w, h].

        Returns
        -------
            in_collision : :obj:`bool`
                Flag that indicates whether the robot would be in collision. 
        """
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
        """ 
        Checks if the given joint configuration is in collision with boxes
        or the workspace walls.

        Parameters
        ----------
            joints : :obj:`list` 
                Optional: Defaults to None
                A list of 7 numbers that correspond to the joint angles in radians
                If None, will use current real robot joints.

            boxes : :obj:`list` 
                List of boxes where each box is a list of 9 floats that contains 
                the position of the center of the box [x, y, z, r, p, y] and the 
                length, width, and height [l, w, h].

        Returns
        -------
            in_collision : :obj:`bool`
                Flag that indicates whether the robot would be in collision. 
        """
        if boxes is None:
            boxes = FC.WORKSPACE_WALLS

        for box in boxes:
            if self.check_box_collision(box, joints=joints):
                return True

        return False

    """
    Misc
    """
    def reset_joints(self, duration=5, block=True, ignore_errors=True, skill_desc=''):
        """
        Commands Arm to go to hardcoded home joint configuration

        Parameters
        ----------
            duration : :obj:`float`
                How much time this robot motion should take.
            block : :obj:`bool`
                Function blocks by default. If False, the function becomes
                asynchronous and can be preempted.
            ignore_errors : :obj:`bool`
                Function ignores errors by default. If False, errors and some 
                exceptions can be thrown.
            skill_desc : :obj:`str` 
                Skill description to use for logging on control-pc.
        """
        self.goto_joints(FC.HOME_JOINTS, duration=duration, skill_desc=skill_desc, block=block, ignore_errors=ignore_errors)

    def reset_pose(self, duration=5, block=True, ignore_errors=True, skill_desc=''):
        """
        Commands Arm to go to hardcoded home pose

        Parameters
        ----------
            duration : :obj:`float`
                How much time this robot motion should take.
            block : :obj:`bool`
                Function blocks by default. If False, the function becomes
                asynchronous and can be preempted.
            ignore_errors : :obj:`bool`
                Function ignores errors by default. If False, errors and some 
                exceptions can be thrown.
            skill_desc : :obj:`str` 
                Skill description to use for logging on control-pc.
        """
        self.goto_pose(FC.HOME_POSE, duration=duration, skill_desc=skill_desc, block=block, ignore_errors=ignore_errors)

    def is_joints_reachable(self, joints):
        """
        Checks if the given joint configuration is within joint limits.

        Parameters
        ----------
            joints : :obj:`list` 
                A list of 7 numbers that correspond to the joint angles in radians.

        Returns
        -------
            joints_reachable : :obj:`bool`
                Flag that is True if all joints are within joint limits.
        """
        for i, val in enumerate(joints):
            if val <= FC.JOINT_LIMITS_MIN[i] or val >= FC.JOINT_LIMITS_MAX[i]:
                return False

        return True

    """
    Unimplemented
    """

    def apply_joint_torques(self, torques, duration, ignore_errors=True, skill_desc='',):
        """
        NOT IMPLEMENTED YET

        Commands the arm to apply given joint torques for duration seconds.

        Parameters
        ----------
            torques : :obj:`list` 
                A list of 7 numbers that correspond to torques in Nm.
            duration : :obj:`float`
                How much time this robot motion should take.
            skill_desc : :obj:`str` 
                Skill description to use for logging on control-pc.
        """
        pass

    def set_speed(self, speed):
        """
        NOT IMPLEMENTED YET

        Sets the current target speed parameter

        Parameters
        ----------
            speed : :obj:`float`
                Current target speed parameter
        """
        pass

    def get_speed(self, speed):
        """
        NOT IMPLEMENTED YET

        Returns the current target speed parameter

        Returns
        -------
            speed : :obj:`float`
                Current target speed parameter
        """
        pass
