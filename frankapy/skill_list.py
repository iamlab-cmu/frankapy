#!/usr/bin/env python

import roslib
roslib.load_manifest('franka_action_lib_msgs')
import rospy
import actionlib
import numpy as np
from .iam_robolib_common_definitions import *
from .franka_constants import FrankaConstants as FC

from franka_action_lib_msgs.msg import ExecuteSkillAction, ExecuteSkillGoal

class BaseSkill(object):
    def __init__(self, 
                 skill_type,
                 skill_desc,
                 meta_skill_type,
                 meta_skill_id,
                 sensor_topics,
                 trajectory_generator_type,
                 feedback_controller_type,
                 termination_type,
                 timer_type):
        self._skill_type = skill_type
        self._skill_desc = skill_desc
        self._meta_skill_type = meta_skill_type
        self._meta_skill_id = meta_skill_id
        self._sensor_topics = sensor_topics
        self._trajectory_generator_type = trajectory_generator_type
        self._feedback_controller_type = feedback_controller_type
        self._termination_type = termination_type
        self._timer_type = timer_type

        self._sensor_value_sizes = 0
        self._initial_sensor_values = []

        # Add trajectory params
        self._trajectory_generator_params = []
        self._num_trajectory_generator_params = 0

        # Add feedback controller params
        self._feedback_controller_params = []
        self._num_feedback_controller_params = 0

        # Add termination params
        self._termination_params = []
        self._num_termination_params = 0

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
        assert type(params) is list, \
                "Invalid type of params provided {}".format(params)
        self._trajectory_generator_params = params
        self._num_trajectory_generator_params = len(params)

    def add_feedback_controller_params(self, params):
        self._feedback_controller_params = params
        self._num_feedback_controller_params = len(params)

    def add_termination_params(self, params):
        self._termination_params = params
        self._num_termination_params = len(params)

    def add_timer_params(self, params):
        self._timer_params = params
        self._num_timer_params = len(params)

    def add_goal_pose_with_matrix(self, time, matrix):
        assert type(time) is float or type(time) is int, \
                "Incorrect time type. Should be int or float."
        assert time >= 0, "Incorrect time. Should be non negative."
        assert type(matrix) is list, "Incorrect matrix type. Should be list."
        assert len(matrix) == 16, "Incorrect matrix len. Should be 16."
        self.add_trajectory_params([time] + matrix)

    def add_goal_pose_with_quaternion(self, time, position, quaternion):
        assert type(time) is float or type(time) is int, \
                "Incorrect time type. Should be int or float."
        assert time >= 0, "Incorrect time. Should be non negative."
        assert type(position) is list, "Incorrect position type. Should be list."
        assert type(quaternion) is list, "Incorrect quaternion type. Should be list."
        assert len(position) == 3, "Incorrect position length. Should be 3."
        assert len(quaternion) == 4, "Incorrect quaternion length. Should be 4."
        self.add_trajectory_params([time] + position + quaternion)

    def add_relative_motion_with_matrix(self, time, matrix):
        assert type(time) is float or type(time) is int, \
                "Incorrect time type. Should be int or float."
        assert time >= 0, "Incorrect time. Should be non negative."
        assert type(matrix) is list, "Incorrect matrix type. Should be list."
        assert len(matrix) == 16, "Incorrect matrix len. Should be 16."
        self.add_trajectory_params([time] + matrix)

    def add_relative_motion_with_quaternion(self, time, position, quaternion):
        assert type(time) is float or type(time) is int, \
                "Incorrect time type. Should be int or float."
        assert time >= 0, "Incorrect time. Should be non negative."
        assert type(position) is list, "Incorrect position type. Should be list."
        assert type(quaternion) is list, "Incorrect quaternion type. Should be list."
        assert len(position) == 3, "Incorrect position length. Should be 3."
        assert len(quaternion) == 4, "Incorrect quaternion length. Should be 4."
        self.add_trajectory_params([time] + position + quaternion)

    def add_goal_joints(self, time, joints):
        assert type(time) is float or type(time) is int,\
                "Incorrect time type. Should be int or float."
        assert time >= 0, "Incorrect time. Should be non negative."
        assert type(joints) is list, "Incorrect joints type. Should be list."
        assert len(joints) == 7, "Incorrect joints len. Should be 7."
        self.add_trajectory_params([time] + joints)

    def add_run_time(self, time):
        assert type(time) is float or type(time) is int, \
                "Incorrect time type. Should be int or float."
        assert time >= 0, "Incorrect time. Should be non negative."
        self.add_trajectory_params([time])

    def add_cartesian_impedances(self, cartesian_impedances):
        assert type(cartesian_impedances) is list, \
                "Incorrect cartesian impedances type. Should be list."
        assert len(cartesian_impedances) == 6, \
                "Incorrect cartesian impedances len. Should be 6."
        if(self._skill_type == SkillType.CartesianPoseSkill):
            self._feedback_controller_type = \
                    FeedbackControllerType.SetInternalImpedanceFeedbackController

        self.add_feedback_controller_params(cartesian_impedances)

    def add_joint_impedances(self, joint_impedances):
        assert type(joint_impedances) is list, \
                "Incorrect joint impedances type. Should be list."
        assert len(joint_impedances) == 7, \
                "Incorrect joint impedances len. Should be 7."
        assert self._skill_type == SkillType.JointPositionSkill, \
                "Incorrect skill type. Should be JointPositionSkill"

        self._feedback_controller_type = \
                FeedbackControllerType.SetInternalImpedanceFeedbackController

        self.add_feedback_controller_params(joint_impedances)

    def add_joint_gains(self, k_gains, d_gains):
        assert type(k_gains) is list, "Incorrect k_gains type. Should be list."
        assert type(d_gains) is list, "Incorrect d_gains type. Should be list."
        assert len(k_gains) == 7, "Incorrect k_gains len. Should be 7."
        assert len(d_gains) == 7, "Incorrect d_gains len. Should be 7."
        assert self._skill_type == SkillType.ImpedanceControlSkill, \
                "Incorrect skill type. Should be ImpedanceControlSkill"

        self.add_feedback_controller_params(k_gains + d_gains)

    def add_contact_termination_params(self, buffer_time,
                                       lower_force_thresholds_accel,
                                       lower_force_thresholds_nominal):
        assert type(buffer_time) is float or type(buffer_time) is int, \
                "Incorrect buffer time type. Should be int or float."
        assert buffer_time >= 0, "Incorrect buffer time. Should be non negative."
        assert type(lower_force_thresholds_accel) is list, \
                "Incorrect lower force thresholds accel type. Should be list."
        assert type(lower_force_thresholds_nominal) is list, \
                "Incorrect lower force thresholds nominal type. Should be list."
        assert len(lower_force_thresholds_accel) == 6, \
                "Incorrect lower force thresholds accel length. Should be 6."
        assert len(lower_force_thresholds_nominal) == 6, \
                "Incorrect lowern force thresholds nominal length. Should be 6."

        self._termination_type = TerminationHandlerType.ContactTerminationHandler

        params = [buffer_time] \
                + FC.DEFAULT_LOWER_TORQUE_THRESHOLDS_ACCEL \
                + FC.DEFAULT_UPPER_TORQUE_THRESHOLDS_ACCEL \
                + FC.DEFAULT_LOWER_TORQUE_THRESHOLDS_NOMINAL \
                + FC.DEFAULT_UPPER_TORQUE_THRESHOLDS_NOMINAL \
                + lower_force_thresholds_accel \
                + FC.DEFAULT_UPPER_FORCE_THRESHOLDS_ACCEL \
                + lower_force_thresholds_nominal \
                + FC.DEFAULT_UPPER_FORCE_THRESHOLDS_NOMINAL

        self.add_termination_params(params)

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
        goal.traj_gen_type = self._trajectory_generator_type
        goal.traj_gen_params = self._trajectory_generator_params
        goal.num_traj_gen_params = self._num_trajectory_generator_params
        goal.feedback_controller_type = self._feedback_controller_type
        goal.feedback_controller_params = self._feedback_controller_params
        goal.num_feedback_controller_params = \
                self._num_feedback_controller_params
        goal.termination_type = self._termination_type
        goal.termination_params = self._termination_params
        goal.num_termination_params = self._num_termination_params
        goal.timer_type = self._timer_type
        goal.timer_params = self._timer_params
        goal.num_timer_params = self._num_timer_params
        return goal

    def feedback_callback(self, feedback):
        pass

# Define an instance of a gripper skill
class GripperSkill(BaseSkill):
    def __init__(self, skill_desc=''):
        if len(skill_desc) == 0:
            skill_desc = GripperSkill.__name__

        super(GripperSkill, self).__init__(
              SkillType.GripperSkill,
              skill_desc,
              MetaSkillType.BaseMetaSkill,
              0,
              ['/franka_robot/camera'],
              TrajectoryGeneratorType.GripperTrajectoryGenerator,
              FeedbackControllerType.NoopFeedbackController,
              TerminationHandlerType.NoopTerminationHandler,
              1)

# Define an instance of a DMP skill using joint position control
class JointDMPSkill(BaseSkill):
    def __init__(self, skill_desc='', skill_type=SkillType.JointPositionSkill):
        if len(skill_desc) == 0:
            skill_desc = JointDMPSkill.__name__

        if skill_type == SkillType.JointPositionSkill:
            super(JointDMPSkill, self).__init__(
                  SkillType.JointPositionSkill,
                  skill_desc,
                  MetaSkillType.BaseMetaSkill,
                  0,
                  ['/franka_robot/camera'],
                  TrajectoryGeneratorType.JointDmpTrajectoryGenerator,
                  FeedbackControllerType.NoopFeedbackController,
                  TerminationHandlerType.TimeTerminationHandler,
                  1)
        elif skill_type == SkillType.ImpedanceControlSkill:
            super(JointDMPSkill, self).__init__(
                  SkillType.ImpedanceControlSkill,
                  skill_desc,
                  MetaSkillType.BaseMetaSkill,
                  0,
                  ['/franka_robot/camera'],
                  TrajectoryGeneratorType.JointDmpTrajectoryGenerator,
                  FeedbackControllerType.JointImpedanceFeedbackController,
                  TerminationHandlerType.TimeTerminationHandler,
                  1)
        else:
            super(JointDMPSkill, self).__init__(
                  SkillType.JointPositionSkill,
                  skill_desc,
                  MetaSkillType.BaseMetaSkill,
                  0,
                  ['/franka_robot/camera'],
                  TrajectoryGeneratorType.JointDmpTrajectoryGenerator,
                  FeedbackControllerType.NoopFeedbackController,
                  TerminationHandlerType.TimeTerminationHandler,
                  1)

# Define an instance of a DMP skill using cartesian control
class PoseDMPSkill(BaseSkill):
    def __init__(self, skill_desc='', skill_type=SkillType.CartesianPoseSkill):
        if len(skill_desc) == 0:
            skill_desc = PoseDMPSkill.__name__

        if skill_type == SkillType.CartesianPoseSkill:
            super(PoseDMPSkill, self).__init__(
                  SkillType.CartesianPoseSkill,
                  skill_desc,
                  MetaSkillType.BaseMetaSkill,
                  0,
                  ['/franka_robot/camera'],
                  TrajectoryGeneratorType.PoseDmpTrajectoryGenerator,
                  FeedbackControllerType.NoopFeedbackController,
                  TerminationHandlerType.TimeTerminationHandler,
                  1)
        elif skill_type == SkillType.ImpedanceControlSkill:
            super(PoseDMPSkill, self).__init__(
                  SkillType.ImpedanceControlSkill,
                  skill_desc,
                  MetaSkillType.BaseMetaSkill,
                  0,
                  ['/franka_robot/camera'],
                  TrajectoryGeneratorType.PoseDmpTrajectoryGenerator,
                  FeedbackControllerType.CartesianImpedanceFeedbackController,
                  TerminationHandlerType.TimeTerminationHandler,
                  1)
        else:
            super(PoseDMPSkill, self).__init__(
                  SkillType.CartesianPoseSkill,
                  skill_desc,
                  MetaSkillType.BaseMetaSkill,
                  0,
                  ['/franka_robot/camera'],
                  TrajectoryGeneratorType.PoseDmpTrajectoryGenerator,
                  FeedbackControllerType.NoopFeedbackController,
                  TerminationHandlerType.TimeTerminationHandler,
                  1)

# Define an instance of a DMP skill (used for Sony Project, ask Kevin Zhang)
class GoalPoseDMPSkill(BaseSkill):
    def __init__(self, skill_desc='', skill_type=SkillType.CartesianPoseSkill):
        if len(skill_desc) == 0:
            skill_desc = GoalPoseDMPSkill.__name__

        if skill_type == SkillType.CartesianPoseSkill:
            super(GoalPoseDMPSkill, self).__init__(
                  SkillType.CartesianPoseSkill,
                  skill_desc,
                  MetaSkillType.BaseMetaSkill,
                  0,
                  ['/franka_robot/camera'],
                  TrajectoryGeneratorType.GoalPoseDmpTrajectoryGenerator,
                  FeedbackControllerType.NoopFeedbackController,
                  TerminationHandlerType.TimeTerminationHandler,
                  1)
        elif skill_type == SkillType.ImpedanceControlSkill:
            super(GoalPoseDMPSkill, self).__init__(
                  SkillType.ImpedanceControlSkill,
                  skill_desc,
                  MetaSkillType.BaseMetaSkill,
                  0,
                  ['/franka_robot/camera'],
                  TrajectoryGeneratorType.GoalPoseDmpTrajectoryGenerator,
                  FeedbackControllerType.CartesianImpedanceFeedbackController,
                  TerminationHandlerType.TimeTerminationHandler,
                  1)
        else:
            super(GoalPoseDMPSkill, self).__init__(
                  SkillType.CartesianPoseSkill,
                  skill_desc,
                  MetaSkillType.BaseMetaSkill,
                  0,
                  ['/franka_robot/camera'],
                  TrajectoryGeneratorType.GoalPoseDmpTrajectoryGenerator,
                  FeedbackControllerType.NoopFeedbackController,
                  TerminationHandlerType.TimeTerminationHandler,
                  1)
    
# Define skill that uses joint position control relative to base
class GoToJointsSkill(BaseSkill):
    def __init__(self, skill_desc='', skill_type=SkillType.JointPositionSkill):
        if len(skill_desc) == 0:
            skill_desc = GoToJointsSkill.__name__

        if skill_type == SkillType.JointPositionSkill:
            super(GoToJointsSkill, self).__init__(
                  SkillType.JointPositionSkill,
                  skill_desc,
                  MetaSkillType.BaseMetaSkill,
                  0,
                  ['/franka_robot/camera'],
                  TrajectoryGeneratorType.MinJerkJointTrajectoryGenerator,
                  FeedbackControllerType.NoopFeedbackController,
                  TerminationHandlerType.FinalJointTerminationHandler,
                  1)
        elif skill_type == SkillType.ImpedanceControlSkill:
            super(GoToJointsSkill, self).__init__(
                  SkillType.ImpedanceControlSkill,
                  skill_desc,
                  MetaSkillType.BaseMetaSkill,
                  0,
                  ['/franka_robot/camera'],
                  TrajectoryGeneratorType.MinJerkJointTrajectoryGenerator,
                  FeedbackControllerType.JointImpedanceFeedbackController,
                  TerminationHandlerType.FinalJointTerminationHandler,
                  1)
        else:
            super(GoToJointsSkill, self).__init__(
                  SkillType.JointPositionSkill,
                  skill_desc,
                  MetaSkillType.BaseMetaSkill,
                  0,
                  ['/franka_robot/camera'],
                  TrajectoryGeneratorType.MinJerkJointTrajectoryGenerator,
                  FeedbackControllerType.NoopFeedbackController,
                  TerminationHandlerType.FinalJointTerminationHandler,
                  1)
    
class GoToJointsDynamicsInterpolationSkill(BaseSkill):

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

    
class GoToPoseDynamicsInterpolationSkill(BaseSkill):

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


# Define skill that uses cartesian impedance control relative to base
class GoToPoseSkill(BaseSkill):
    def __init__(self, skill_desc='', skill_type=SkillType.ImpedanceControlSkill):
        if len(skill_desc) == 0:
            skill_desc = GoToPoseSkill.__name__

        if skill_type == SkillType.ImpedanceControlSkill:
            super(GoToPoseSkill, self).__init__(
                  SkillType.ImpedanceControlSkill,
                  skill_desc,
                  MetaSkillType.BaseMetaSkill,
                  0,
                  ['/franka_robot/camera'],
                  TrajectoryGeneratorType.MinJerkPoseTrajectoryGenerator,
                  FeedbackControllerType.CartesianImpedanceFeedbackController,
                  TerminationHandlerType.FinalPoseTerminationHandler,
                  1)
        elif skill_type == SkillType.CartesianPoseSkill:
            super(GoToPoseSkill, self).__init__(
                  SkillType.CartesianPoseSkill,
                  skill_desc,
                  MetaSkillType.BaseMetaSkill,
                  0,
                  ['/franka_robot/camera'],
                  TrajectoryGeneratorType.MinJerkPoseTrajectoryGenerator,
                  FeedbackControllerType.NoopFeedbackController,
                  TerminationHandlerType.FinalPoseTerminationHandler,
                  1)
        else:
            super(GoToPoseSkill, self).__init__(
                  SkillType.ImpedanceControlSkill,
                  skill_desc,
                  MetaSkillType.BaseMetaSkill,
                  0,
                  ['/franka_robot/camera'],
                  TrajectoryGeneratorType.MinJerkPoseTrajectoryGenerator,
                  FeedbackControllerType.CartesianImpedanceFeedbackController,
                  TerminationHandlerType.FinalPoseTerminationHandler,
                  1)

# Define skill that uses cartesian impedance to go position relative to current position
class GoToPoseDeltaSkill(BaseSkill):
    def __init__(self, skill_desc='', skill_type=SkillType.ImpedanceControlSkill):
        if len(skill_desc) == 0:
            skill_desc = GoToPoseDeltaSkill.__name__

        if skill_type == SkillType.ImpedanceControlSkill:
            super(GoToPoseDeltaSkill, self).__init__(
                  SkillType.ImpedanceControlSkill,
                  skill_desc,
                  MetaSkillType.BaseMetaSkill,
                  0,
                  ['/franka_robot/camera'],
                  TrajectoryGeneratorType.RelativeMinJerkPoseTrajectoryGenerator,
                  FeedbackControllerType.CartesianImpedanceFeedbackController,
                  TerminationHandlerType.FinalPoseTerminationHandler,
                  1)
        elif skill_type == SkillType.CartesianPoseSkill:
            super(GoToPoseDeltaSkill, self).__init__(
                  SkillType.CartesianPoseSkill,
                  skill_desc,
                  MetaSkillType.BaseMetaSkill,
                  0,
                  ['/franka_robot/camera'],
                  TrajectoryGeneratorType.RelativeMinJerkPoseTrajectoryGenerator,
                  FeedbackControllerType.NoopFeedbackController,
                  TerminationHandlerType.FinalPoseTerminationHandler,
                  1)
        else:
            super(GoToPoseDeltaSkill, self).__init__(
                  SkillType.ImpedanceControlSkill,
                  skill_desc,
                  MetaSkillType.BaseMetaSkill,
                  0,
                  ['/franka_robot/camera'],
                  TrajectoryGeneratorType.RelativeMinJerkPoseTrajectoryGenerator,
                  FeedbackControllerType.CartesianImpedanceFeedbackController,
                  TerminationHandlerType.FinalPoseTerminationHandler,
                  1)

# Define skill that stays in pose
class StayInInitialPoseSkill(BaseSkill):
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
                  FeedbackControllerType.NoopFeedbackController,
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
                  FeedbackControllerType.NoopFeedbackController,
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
class ForceTorqueSkill(BaseSkill):
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
class ForceAlongAxisSkill(BaseSkill):
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
