#!/usr/bin/env python

import roslib
roslib.load_manifest('franka_interface_msgs')
import rospy
import actionlib
import numpy as np
from autolab_core import RigidTransform
from .iam_robolib_common_definitions import *
from .franka_constants import FrankaConstants as FC
from .proto_utils import *

from franka_interface_msgs.msg import ExecuteSkillAction, ExecuteSkillGoal

class Skill:
    def __init__(self, 
                 skill_type,
                 trajectory_generator_type,
                 feedback_controller_type = FeedbackControllerType.NoopFeedbackController,
                 termination_handler_type = TerminationHandlerType.NoopTerminationHandler,
                 meta_skill_type = MetaSkillType.BaseMetaSkill,
                 meta_skill_id = 0,
                 sensor_topics = None,
                 timer_type = 1,
                 skill_desc = ''):
        self._skill_type = skill_type
        self._skill_desc = skill_desc
        self._meta_skill_type = meta_skill_type
        self._meta_skill_id = meta_skill_id
        self._sensor_topics = sensor_topics if sensor_topics not None else ['']
        self._trajectory_generator_type = trajectory_generator_type
        self._feedback_controller_type = 
        self._termination_handler_type = TerminationHandlerType.NoopTerminationHandler
        self._timer_type = timer_type

        self._sensor_value_sizes = 0
        self._initial_sensor_values = []

        # Add trajectory params
        self._trajectory_generator_param_data = []
        self._trajectory_generator_param_data_size = 0

        # Add feedback controller params
        self._feedback_controller_param_data = []
        self._feedback_controller_param_data_size = 0

        # Add termination params
        self._termination_handler_param_data = []
        self._termination_handler_param_data_size = 0

        # Add timer params
        self._timer_params = []
        self._num_timer_params = 0

    def set_meta_skill_type(self, meta_skill_type):
        self._meta_skill_type = meta_skill_type

    def set_meta_skill_id(self, meta_skill_id):
        self._meta_skill_id = meta_skill_id

    def add_initial_sensor_values(self, values):
        self._initial_sensor_values = values
        self._sensor_value_sizes = [len(values)]

    def add_trajectory_params(self, params):
        self._trajectory_generator_param_data = params
        self._trajectory_generator_param_data_size = len(params)

    def add_feedback_controller_params(self, params):
        self._feedback_controller_param_data = params
        self._feedback_controller_param_data_size = len(params)

    def add_termination_params(self, params):
        self._termination_handler_param_data = params
        self._termination_handler_param_data_size = len(params)

    def add_timer_params(self, params):
        self._timer_params = params
        self._num_timer_params = len(params)

    ## Feedback Controllers

    def add_cartesian_impedances(self, cartesian_impedances):
        assert type(cartesian_impedances) is list, \
                "Incorrect cartesian impedances type. Should be list."
        assert len(cartesian_impedances) == 6, \
                "Incorrect cartesian impedances len. Should be 6."
        assert self._skill_type == SkillType.ImpedanceControlSkill, \
                "Incorrect skill type. Should be ImpedanceControlSkill."

        cartesian_impedance_feedback_controller_msg_proto = \
            make_cartesian_impedance_feedback_controller_msg_proto(cartesian_impedances[:3], 
                                                                   cartesian_impedances[3:])

        self.add_feedback_controller_params(cartesian_impedance_feedback_controller_msg_proto.SerializeToString())

    def add_internal_impedances(self, cartesian_impedances, joint_impedances):
        assert type(cartesian_impedances) is list, \
                "Incorrect joint impedances type. Should be list."
        assert type(joint_impedances) is list, \
                "Incorrect joint impedances type. Should be list."
        assert len(cartesian_impedances) == 0 or len(cartesian_impedances) == 6, \
                "Incorrect cartesian impedances len. Should be 0 or 6."
        assert len(joint_impedances) == 0 or len(joint_impedances) == 7, \
                "Incorrect joint impedances len. Should be 0 or 7."
        assert self._skill_type == SkillType.CartesianPoseSkill or self._skill_type == SkillType.JointPositionSkill, \
                "Incorrect skill type. Should be CartesianPoseSkill or JointPositionSkill."

        internal_feedback_controller_msg_proto = \
            make_internal_feedback_controller_msg_proto(cartesian_impedances, joint_impedances)

        self.add_feedback_controller_params(internal_feedback_controller_msg_proto.SerializeToString())

    def add_joint_gains(self, k_gains, d_gains):
        assert type(k_gains) is list, "Incorrect k_gains type. Should be list."
        assert type(d_gains) is list, "Incorrect d_gains type. Should be list."
        assert len(k_gains) == 7, "Incorrect k_gains len. Should be 7."
        assert len(d_gains) == 7, "Incorrect d_gains len. Should be 7."
        assert self._skill_type == SkillType.ImpedanceControlSkill, \
                "Incorrect skill type. Should be ImpedanceControlSkill"

        joint_feedback_controller_msg_proto = \
            make_joint_feedback_controller_msg_proto(k_gains, d_gains)

        self.add_feedback_controller_params(joint_feedback_controller_msg_proto.SerializeToString())

    ## Termination Handlers

    def add_contact_termination_params(self, buffer_time,
                                       force_thresholds,
                                       torque_thresholds):
        assert type(buffer_time) is float or type(buffer_time) is int, \
                "Incorrect buffer time type. Should be int or float."
        assert buffer_time >= 0, "Incorrect buffer time. Should be non negative."
        assert type(force_thresholds) is list, \
                "Incorrect force thresholds type. Should be list."
        assert type(torque_thresholds) is list, \
                "Incorrect torque thresholds type. Should be list."
        assert len(force_thresholds) == 0 or len(force_thresholds) == 6, \
                "Incorrect force thresholds length. Should be 0 or 6."
        assert len(torque_thresholds) == 0 or len(torque_thresholds) == 7, \
                "Incorrect torque thresholds length. Should be 0 or 7."

        self._termination_handler_type = TerminationHandlerType.ContactTerminationHandler

        contact_termination_handler_msg_proto = \
            make_contact_termination_handler_msg_proto(buffer_time, force_thresholds,
                                                       torque_thresholds)

        self.add_termination_params(contact_termination_handler_msg_proto.SerializeToString())

    def add_joint_threshold_params(self, buffer_time, joint_thresholds):
        assert type(buffer_time) is float or type(buffer_time) is int, \
                "Incorrect buffer time type. Should be int or float."
        assert buffer_time >= 0, "Incorrect buffer time. Should be non negative."
        assert type(joint_thresholds) is list, \
                "Incorrect joint thresholds type. Should be list."
        assert len(joint_thresholds) == 0 or len(joint_thresholds) == 7, \
                "Incorrect joint thresholds length. Should be 0 or 7."
        assert self._termination_handler_type == TerminationHandlerType.FinalJointTerminationHandler, \
                "Incorrect termination handler type. Should be FinalJointTerminationHandler"

        joint_threshold_msg_proto = make_joint_threshold_msg_proto(buffer_time, joint_thresholds)

        self.add_termination_params(joint_threshold_msg_proto.SerializeToString())

    def add_pose_threshold_params(self, buffer_time, pose_thresholds):
        assert type(buffer_time) is float or type(buffer_time) is int, \
                "Incorrect buffer time type. Should be int or float."
        assert buffer_time >= 0, "Incorrect buffer time. Should be non negative."
        assert type(pose_thresholds) is list, \
                "Incorrect pose thresholds type. Should be list."
        assert len(pose_thresholds) == 0 or len(pose_thresholds) == 6, \
                "Incorrect pose thresholds length. Should be 0 or 6."
        assert self._termination_handler_type == TerminationHandlerType.FinalPoseTerminationHandler, \
                "Incorrect termination handler type. Should be FinalPoseTerminationHandler"

        pose_threshold_msg_proto = make_pose_threshold_msg_proto(buffer_time, pose_thresholds)

        self.add_termination_params(pose_threshold_msg_proto.SerializeToString())

    def add_time_termination_params(self, buffer_time):
        assert type(buffer_time) is float or type(buffer_time) is int, \
                "Incorrect buffer time type. Should be int or float."
        assert buffer_time >= 0, "Incorrect buffer time. Should be non negative."
        assert self._termination_handler_type == TerminationHandlerType.TimeTerminationHandler, \
                "Incorrect termination handler type. Should be TimeTerminationHandler"

        time_termination_handler_msg_proto = make_time_termination_handler_msg_proto(buffer_time)

        self.add_termination_params(time_termination_handler_msg_proto.SerializeToString())

    ## Trajectory Generators

    def add_gripper_params(self, grasp, width, speed, force):
        assert type(grasp) is bool, \
                "Incorrect grasp type. Should be bool."
        assert type(width) is float or type(width) is int, \
                "Incorrect width type. Should be int or float."
        assert type(speed) is float or type(speed) is int, \
                "Incorrect speed type. Should be int or float."
        assert type(force) is float or type(force) is int, \
                "Incorrect force type. Should be int or float."
        assert width >= 0, "Incorrect width. Should be non negative."
        assert speed >= 0, "Incorrect speed. Should be non negative."
        assert force >= 0, "Incorrect force. Should be non negative."
        assert self._skill_type == SkillType.GripperSkill, \
                "Incorrect skill type. Should be GripperSkill"
        assert self._trajectory_generator_type == TrajectoryGeneratorType.GripperTrajectoryGenerator, \
                "Incorrect trajectory generator type. Should be GripperTrajectoryGenerator"

        gripper_trajectory_generator_msg_proto = make_gripper_trajectory_generator_msg_proto(grasp, width, speed, force)

        self.add_trajectory_params(gripper_trajectory_generator_msg_proto.SerializeToString())

    def add_goal_pose(self, time, goal_pose):
        assert type(time) is float or type(time) is int, \
                "Incorrect time type. Should be int or float."
        assert time >= 0, "Incorrect time. Should be non negative."
        assert type(goal_pose) is RigidTransform, "Incorrect goal_pose type. Should be RigidTransform."

        pose_trajectory_generator_msg_proto = make_pose_trajectory_generator_msg_proto(time, goal_pose)

        self.add_trajectory_params(pose_trajectory_generator_msg_proto.SerializeToString())

    def add_goal_joints(self, time, joints):
        assert type(time) is float or type(time) is int,\
                "Incorrect time type. Should be int or float."
        assert time >= 0, "Incorrect time. Should be non negative."
        assert type(joints) is list, "Incorrect joints type. Should be list."
        assert len(joints) == 7, "Incorrect joints len. Should be 7."

        joint_trajectory_generator_msg_proto = make_joint_trajectory_generator_msg_proto(time, joints)

        self.add_trajectory_params(joint_trajectory_generator_msg_proto.SerializeToString())

    def add_joint_dmp_params(self, time, joint_dmp_info, initial_sensor_values):
        assert type(time) is float or type(time) is int,\
                "Incorrect time type. Should be int or float."
        assert time >= 0, "Incorrect time. Should be non negative."

        assert type(joint_dmp_info['tau']) is float or type(joint_dmp_info['tau']) is int,\
                "Incorrect tau type. Should be int or float."
        assert joint_dmp_info['tau'] >= 0, "Incorrect tau. Should be non negative."

        assert type(joint_dmp_info['alpha']) is float or type(joint_dmp_info['alpha']) is int,\
                "Incorrect alpha type. Should be int or float."
        assert joint_dmp_info['alpha'] >= 0, "Incorrect alpha. Should be non negative."

        assert type(joint_dmp_info['beta']) is float or type(joint_dmp_info['beta']) is int,\
                "Incorrect beta type. Should be int or float."
        assert joint_dmp_info['beta'] >= 0, "Incorrect beta. Should be non negative."

        assert type(joint_dmp_info['num_basis']) is float or type(joint_dmp_info['num_basis']) is int,\
                "Incorrect num basis type. Should be int or float."
        assert joint_dmp_info['num_basis'] >= 0, "Incorrect num basis. Should be non negative."

        assert type(joint_dmp_info['num_sensors']) is float or type(joint_dmp_info['num_sensors']) is int,\
                "Incorrect num sensors type. Should be int or float."
        assert joint_dmp_info['num_sensors'] >= 0, "Incorrect num sensors. Should be non negative."

        assert type(joint_dmp_info['mu']) is list, "Incorrect basis mean type. Should be list."
        assert len(joint_dmp_info['mu']) == joint_dmp_info['num_basis'], \
                "Incorrect basis mean len. Should be equal to num basis."

        assert type(joint_dmp_info['h']) is list, "Incorrect basis std dev type. Should be list."
        assert len(joint_dmp_info['h']) == joint_dmp_info['num_basis'], \
                "Incorrect basis std dev len. Should be equal to num basis."

        assert type(initial_sensor_values) is list, "Incorrect initial sensor values type. Should be list."
        assert len(initial_sensor_values) == joint_dmp_info['num_sensors'], \
                "Incorrect initial sensor values len. Should be equal to num sensors."

        weights = np.array(joint_dmp_info['weights']).reshape(-1).tolist()
        num_weights = 7 * int(joint_dmp_info['num_basis']) * int(joint_dmp_info['num_sensors'])

        assert len(weights) == num_weights, \
                "Incorrect weights len. Should be equal to 7 * num basis * num sensors."

        assert self._skill_type == SkillType.ImpedanceControlSkill or \
               self._skill_type == SkillType.JointPositionSkill, \
                "Incorrect skill type. Should be ImpedanceControlSkill or JointPositionSkill."
        assert self._trajectory_generator_type == TrajectoryGeneratorType.JointDmpTrajectoryGenerator, \
                "Incorrect trajectory generator type. Should be JointDmpTrajectoryGenerator"

        joint_dmp_trajectory_generator_msg_proto = make_joint_dmp_trajectory_generator_msg_proto(run_time, 
                                                   joint_dmp_info['tau'], joint_dmp_info['alpha'], joint_dmp_info['beta'], 
                                                   joint_dmp_info['num_basis'], joint_dmp_info['num_sensors'], 
                                                   joint_dmp_info['mu'], joint_dmp_info['h'], 
                                                   np.array(joint_dmp_info['weights']).reshape(-1).tolist(), 
                                                   initial_sensor_values)

        self.add_trajectory_params(joint_dmp_trajectory_generator_msg_proto.SerializeToString())

    def add_pose_dmp_params(self, orientation_only, position_only, time, pose_dmp_info, initial_sensor_values):
        assert type(time) is float or type(time) is int,\
                "Incorrect time type. Should be int or float."
        assert time >= 0, "Incorrect time. Should be non negative."

        assert type(pose_dmp_info['tau']) is float or type(pose_dmp_info['tau']) is int,\
                "Incorrect tau type. Should be int or float."
        assert pose_dmp_info['tau'] >= 0, "Incorrect tau. Should be non negative."

        assert type(pose_dmp_info['alpha']) is float or type(pose_dmp_info['alpha']) is int,\
                "Incorrect alpha type. Should be int or float."
        assert pose_dmp_info['alpha'] >= 0, "Incorrect alpha. Should be non negative."

        assert type(pose_dmp_info['beta']) is float or type(pose_dmp_info['beta']) is int,\
                "Incorrect beta type. Should be int or float."
        assert pose_dmp_info['beta'] >= 0, "Incorrect beta. Should be non negative."

        assert type(pose_dmp_info['num_basis']) is float or type(pose_dmp_info['num_basis']) is int,\
                "Incorrect num basis type. Should be int or float."
        assert pose_dmp_info['num_basis'] >= 0, "Incorrect num basis. Should be non negative."

        assert type(pose_dmp_info['num_sensors']) is float or type(pose_dmp_info['num_sensors']) is int,\
                "Incorrect num sensors type. Should be int or float."
        assert pose_dmp_info['num_sensors'] >= 0, "Incorrect num sensors. Should be non negative."

        assert type(pose_dmp_info['mu']) is list, "Incorrect basis mean type. Should be list."
        assert len(pose_dmp_info['mu']) == pose_dmp_info['num_basis'], \
                "Incorrect basis mean len. Should be equal to num basis."

        assert type(pose_dmp_info['h']) is list, "Incorrect basis std dev type. Should be list."
        assert len(pose_dmp_info['h']) == pose_dmp_info['num_basis'], \
                "Incorrect basis std dev len. Should be equal to num basis."

        assert type(initial_sensor_values) is list, "Incorrect initial sensor values type. Should be list."
        assert len(initial_sensor_values) == pose_dmp_info['num_sensors'], \
                "Incorrect initial sensor values len. Should be equal to num sensors."

        weights = np.array(pose_dmp_info['weights']).reshape(-1).tolist()

        if orientation_only or position_only:
            num_weights = 3 * int(pose_dmp_info['num_basis']) * int(pose_dmp_info['num_sensors'])
            assert len(weights) == num_weights, \
                    "Incorrect weights len. Should be equal to 3 * num basis * num sensors."
        else:
            num_weights = 6 * int(pose_dmp_info['num_basis']) * int(pose_dmp_info['num_sensors'])
            assert len(weights) == num_weights, \
                    "Incorrect weights len. Should be equal to 6 * num basis * num sensors."

        assert self._skill_type == SkillType.CartesianPoseSkill or \
               self._skill_type == SkillType.ImpedanceControlSkill, \
                "Incorrect skill type. Should be CartesianPoseSkill or ImpedanceControlSkill."
        assert self._trajectory_generator_type == TrajectoryGeneratorType.PoseDmpTrajectoryGenerator, \
                "Incorrect trajectory generator type. Should be PoseDmpTrajectoryGenerator"

        pose_dmp_trajectory_generator_msg_proto = make_pose_dmp_trajectory_generator_msg_proto(orientation_only,
                                                   position_only, run_time, 
                                                   pose_dmp_info['tau'], pose_dmp_info['alpha'], pose_dmp_info['beta'], 
                                                   pose_dmp_info['num_basis'], pose_dmp_info['num_sensors'], 
                                                   pose_dmp_info['mu'], pose_dmp_info['h'], 
                                                   np.array(pose_dmp_info['weights']).reshape(-1).tolist(), 
                                                   initial_sensor_values)

        self.add_trajectory_params(pose_dmp_trajectory_generator_msg_proto.SerializeToString())

    def add_run_time(self, time):
        assert type(time) is float or type(time) is int, \
                "Incorrect time type. Should be int or float."
        assert time >= 0, "Incorrect time. Should be non negative."

        run_time_msg_proto = make_run_time_msg_proto(time)
        self.add_trajectory_params(run_time_msg_proto.SerializeToString())

    # Add checks for these
    def create_goal(self):
        goal = ExecuteSkillGoal()
        goal.skill_type = self._skill_type
        goal.skill_description = self._skill_desc
        goal.meta_skill_type = self._meta_skill_type
        goal.meta_skill_id = self._meta_skill_id
        goal.sensor_topics = self._sensor_topics
        goal.initial_sensor_values = self._initial_sensor_values
        goal.sensor_value_sizes = self._sensor_value_sizes
        goal.trajectory_generator_type = self._trajectory_generator_type
        goal.trajectory_generator_param_data = self._trajectory_generator_param_data
        goal.trajectory_generator_param_data_size = self._trajectory_generator_param_data_size
        goal.feedback_controller_type = self._feedback_controller_type
        goal.feedback_controller_param_data = self._feedback_controller_param_data
        goal.feedback_controller_param_data_size = \
                self._feedback_controller_param_data_size
        goal.termination_handler_type = self._termination_handler_type
        goal.termination_handler_param_data = self._termination_handler_param_data
        goal.termination_handler_param_data_size = self._termination_handler_param_data_size
        goal.timer_type = self._timer_type
        goal.timer_params = self._timer_params
        goal.num_timer_params = self._num_timer_params
        return goal

    def feedback_callback(self, feedback):
        pass

    
class GoToJointsDynamicsInterpolationSkill(Skill):

    def __init__(self, skill_desc='', skill_type=SkillType.JointPositionSkill):
        if len(skill_desc) == 0:
            skill_desc = GoToJointsDynamicsInterpolationSkill.__name__

        if skill_type == SkillType.JointPositionSkill:
            super(GoToJointsDynamicsInterpolationSkill, self).__init__(
                SkillType.JointPositionSkill,
                skill_desc,
                MetaSkillType.BaseMetaSkill,
                0,
                ['/franka_robot/camera'],
                TrajectoryGeneratorType.CubicHermiteSplineJointTrajectoryGenerator,
                FeedbackControllerType.JointImpedanceFeedbackController,
                TerminationHandlerType.TimeTerminationHandler,
                1)
        elif skill_type == SkillType.ImpedanceControlSkill:
            super(GoToJointsDynamicsInterpolationSkill, self).__init__(
                SkillType.ImpedanceControlSkill,
                skill_desc,
                MetaSkillType.BaseMetaSkill,
                0,
                ['/franka_robot/camera'],
                TrajectoryGeneratorType.CubicHermiteSplineJointTrajectoryGenerator,
                FeedbackControllerType.JointImpedanceFeedbackController,
                TerminationHandlerType.TimeTerminationHandler,
                1)
        else:
            super(GoToJointsDynamicsInterpolationSkill, self).__init__(
                SkillType.ImpedanceControlSkill,
                skill_desc,
                MetaSkillType.BaseMetaSkill,
                0,
                ['/franka_robot/camera'],
                TrajectoryGeneratorType.CubicHermiteSplineJointTrajectoryGenerator,
                FeedbackControllerType.JointImpedanceFeedbackController,
                TerminationHandlerType.TimeTerminationHandler,
                1)

    
class GoToPoseDynamicsInterpolationSkill(Skill):

    def __init__(self, skill_desc='', skill_type=SkillType.ImpedanceControlSkill):
        if len(skill_desc) == 0:
            skill_desc = GoToPoseDynamicsInterpolationSkill.__name__

        if skill_type == SkillType.ImpedanceControlSkill:
            super(GoToPoseDynamicsInterpolationSkill, self).__init__(
                SkillType.ImpedanceControlSkill,
                skill_desc,
                MetaSkillType.BaseMetaSkill,
                0,
                ['/franka_robot/camera'],
                TrajectoryGeneratorType.CubicHermiteSplinePoseTrajectoryGenerator,
                FeedbackControllerType.CartesianImpedanceFeedbackController,
                TerminationHandlerType.TimeTerminationHandler,
                1)
        else:
            super(GoToPoseDynamicsInterpolationSkill, self).__init__(
                SkillType.ImpedanceControlSkill,
                skill_desc,
                MetaSkillType.BaseMetaSkill,
                0,
                ['/franka_robot/camera'],
                TrajectoryGeneratorType.CubicHermiteSplinePoseTrajectoryGenerator,
                FeedbackControllerType.CartesianImpedanceFeedbackController,
                TerminationHandlerType.TimeTerminationHandler,
                1)


# Define skill that stays in pose
class StayInInitialPoseSkill(Skill):
    def __init__(self, skill_desc='', skill_type=SkillType.ImpedanceControlSkill,
                 feedback_controller_type=FeedbackControllerType.CartesianImpedanceFeedbackController):
        if len(skill_desc) == 0:
            skill_desc = StayInInitialPoseSkill.__name__
        if skill_type == SkillType.ImpedanceControlSkill:
            if feedback_controller_type == FeedbackControllerType.CartesianImpedanceFeedbackController:
                super(StayInInitialPoseSkill, self).__init__(
                      SkillType.ImpedanceControlSkill,
                      skill_desc,
                      MetaSkillType.BaseMetaSkill,
                      0,
                      ['/franka_robot/camera'],
                      TrajectoryGeneratorType.StayInInitialPoseTrajectoryGenerator,
                      FeedbackControllerType.CartesianImpedanceFeedbackController,
                      TerminationHandlerType.TimeTerminationHandler,
                      1)
            elif feedback_controller_type == FeedbackControllerType.JointImpedanceFeedbackController:
                super(StayInInitialPoseSkill, self).__init__(
                      SkillType.ImpedanceControlSkill,
                      skill_desc,
                      MetaSkillType.BaseMetaSkill,
                      0,
                      ['/franka_robot/camera'],
                      TrajectoryGeneratorType.StayInInitialJointsTrajectoryGenerator,
                      FeedbackControllerType.JointImpedanceFeedbackController,
                      TerminationHandlerType.TimeTerminationHandler,
                      1)
            else:
                super(StayInInitialPoseSkill, self).__init__(
                      SkillType.ImpedanceControlSkill,
                      skill_desc,
                      MetaSkillType.BaseMetaSkill,
                      0,
                      ['/franka_robot/camera'],
                      TrajectoryGeneratorType.StayInInitialPoseTrajectoryGenerator,
                      FeedbackControllerType.CartesianImpedanceFeedbackController,
                      TerminationHandlerType.TimeTerminationHandler,
                      1)
        elif skill_type == SkillType.CartesianPoseSkill:
            super(StayInInitialPoseSkill, self).__init__(
                  SkillType.CartesianPoseSkill,
                  skill_desc,
                  MetaSkillType.BaseMetaSkill,
                  0,
                  ['/franka_robot/camera'],
                  TrajectoryGeneratorType.StayInInitialPoseTrajectoryGenerator,
                  FeedbackControllerType.SetInternalImpedanceFeedbackController,
                  TerminationHandlerType.TimeTerminationHandler,
                  1)
        elif skill_type == SkillType.JointPositionSkill:
            super(StayInInitialPoseSkill, self).__init__(
                  SkillType.JointPositionSkill,
                  skill_desc,
                  MetaSkillType.BaseMetaSkill,
                  0,
                  ['/franka_robot/camera'],
                  TrajectoryGeneratorType.StayInInitialJointsTrajectoryGenerator,
                  FeedbackControllerType.SetInternalImpedanceFeedbackController,
                  TerminationHandlerType.TimeTerminationHandler,
                  1)
        else:
            super(StayInInitialPoseSkill, self).__init__(
                  SkillType.ImpedanceControlSkill,
                  skill_desc,
                  MetaSkillType.BaseMetaSkill,
                  0,
                  ['/franka_robot/camera'],
                  TrajectoryGeneratorType.StayInInitialPoseTrajectoryGenerator,
                  FeedbackControllerType.CartesianImpedanceFeedbackController,
                  TerminationHandlerType.TimeTerminationHandler,
                  1)

# Define skill that uses force control
class ForceTorqueSkill(Skill):
    def __init__(self, skill_desc=''):
        if len(skill_desc) == 0:
            skill_desc = ForceTorqueSkill.__name__
        super(ForceTorqueSkill, self).__init__(
              SkillType.ForceTorqueSkill,
              skill_desc,
              MetaSkillType.BaseMetaSkill,
              0,
              ['/franka_robot/camera'],
              TrajectoryGeneratorType.ImpulseTrajectoryGenerator,
              FeedbackControllerType.PassThroughFeedbackController,
              TerminationHandlerType.TimeTerminationHandler,
              1)

# Define skill that uses force control along a specific axis
class ForceAlongAxisSkill(Skill):
    def __init__(self, skill_desc=''):
        if len(skill_desc) == 0:
            skill_desc = ForceTorqueSkill.__name__
        super(ForceAlongAxisSkill, self).__init__(
              SkillType.ForceTorqueSkill,
              skill_desc,
              MetaSkillType.BaseMetaSkill,
              0,
              ['/franka_robot/camera'],
              TrajectoryGeneratorType.ImpulseTrajectoryGenerator,
              FeedbackControllerType.ForceAxisImpedenceFeedbackController,
              TerminationHandlerType.TimeTerminationHandler,
              1)
