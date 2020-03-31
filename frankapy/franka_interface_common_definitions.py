#######################################################################
#                                                                     #
#   Important: Any Changes here should also be reflected in changes   #
#   in the frankapy franka_interface_common_definitions.py file as well.   #
#                                                                     #
#   The order of the enums matter!!                                   #
#                                                                     #
####################################################################### 
from enum import Enum

# From https://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python
def enum_auto(*sequential, **named):
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)


SkillType = enum_auto(
    'CartesianPoseSkill',
    'ForceTorqueSkill',
    'GripperSkill',
    'ImpedanceControlSkill',
    'JointPositionSkill'
)


MetaSkillType = enum_auto(
    'BaseMetaSkill',
    'JointPositionContinuousSkill'
)


TrajectoryGeneratorType = enum_auto(
    'CubicHermiteSplineJointTrajectoryGenerator',
    'CubicHermiteSplinePoseTrajectoryGenerator',
    'GoalPoseDmpTrajectoryGenerator',
    'GripperTrajectoryGenerator',
    'ImpulseTrajectoryGenerator',
    'JointDmpTrajectoryGenerator',
    'LinearJointTrajectoryGenerator',
    'LinearPoseTrajectoryGenerator',
    'MinJerkJointTrajectoryGenerator',
    'MinJerkPoseTrajectoryGenerator',
    'PoseDmpTrajectoryGenerator',
    'RelativeLinearPoseTrajectoryGenerator',
    'RelativeMinJerkPoseTrajectoryGenerator',
    'SineJointTrajectoryGenerator',
    'SinePoseTrajectoryGenerator',
    'StayInInitialJointsTrajectoryGenerator',
    'StayInInitialPoseTrajectoryGenerator'
)


FeedbackControllerType = enum_auto(
    'CartesianImpedanceFeedbackController',
    'ForceAxisImpedenceFeedbackController',
    'JointImpedanceFeedbackController',
    'NoopFeedbackController',
    'PassThroughFeedbackController',
    'SetInternalImpedanceFeedbackController'
)


TerminationHandlerType = enum_auto(
    'ContactTerminationHandler',
    'FinalJointTerminationHandler',
    'FinalPoseTerminationHandler',
    'NoopTerminationHandler',
    'TimeTerminationHandler'
)


SkillStatus = enum_auto(
    'TO_START',
    'RUNNING',
    'FINISHED',
    'VIRT_COLL_ERR'
)

SensorDataMessageType = enum_auto(
    'JOINT_POSITION_VELOCITY',
    'POSE_POSITION_VELOCITY',
    'BOUNDING_BOX'
)
