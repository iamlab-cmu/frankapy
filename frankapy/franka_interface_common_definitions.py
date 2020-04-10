#######################################################################
#                                                                     #
#   Important: Any Changes here should also be reflected in changes   #
#   in the franka-interface-common definitions.h file as well.        #
#                                                                     #
#   The order of the enums matter!!                                   #
#                                                                     #
####################################################################### 


_ENUM_COUNTER = {}
def _enum_auto(key):
    if key not in _ENUM_COUNTER:
        _ENUM_COUNTER[key] = 0
    val = _ENUM_COUNTER[key]
    _ENUM_COUNTER[key] += 1
    return val


class SkillType:
    CartesianPoseSkill = _enum_auto('SkillType')
    ForceTorqueSkill = _enum_auto('SkillType')
    GripperSkill = _enum_auto('SkillType')
    ImpedanceControlSkill = _enum_auto('SkillType')
    JointPositionSkill = _enum_auto('SkillType')


class MetaSkillType:
    BaseMetaSkill = _enum_auto('MetaSkillType')
    JointPositionContinuousSkill = _enum_auto('MetaSkillType')


class TrajectoryGeneratorType:
    CubicHermiteSplineJointTrajectoryGenerator = _enum_auto('TrajectoryGeneratorType')
    CubicHermiteSplinePoseTrajectoryGenerator = _enum_auto('TrajectoryGeneratorType')
    GoalPoseDmpTrajectoryGenerator = _enum_auto('TrajectoryGeneratorType')
    GripperTrajectoryGenerator = _enum_auto('TrajectoryGeneratorType')
    ImpulseTrajectoryGenerator = _enum_auto('TrajectoryGeneratorType')
    JointDmpTrajectoryGenerator = _enum_auto('TrajectoryGeneratorType')
    LinearForcePositionTrajectoryGenerator = _enum_auto('TrajectoryGeneratorType')
    LinearJointTrajectoryGenerator = _enum_auto('TrajectoryGeneratorType')
    LinearPoseTrajectoryGenerator = _enum_auto('TrajectoryGeneratorType')
    MinJerkJointTrajectoryGenerator = _enum_auto('TrajectoryGeneratorType')
    MinJerkPoseTrajectoryGenerator = _enum_auto('TrajectoryGeneratorType')
    PassThroughForcePositionTrajectoryGenerator = _enum_auto('TrajectoryGeneratorType')
    PassThroughJointTrajectoryGenerator = _enum_auto('TrajectoryGeneratorType')
    PassThroughPoseTrajectoryGenerator = _enum_auto('TrajectoryGeneratorType')
    PoseDmpTrajectoryGenerator = _enum_auto('TrajectoryGeneratorType')
    RelativeLinearPoseTrajectoryGenerator = _enum_auto('TrajectoryGeneratorType')
    RelativeMinJerkPoseTrajectoryGenerator = _enum_auto('TrajectoryGeneratorType')
    SineJointTrajectoryGenerator = _enum_auto('TrajectoryGeneratorType')
    SinePoseTrajectoryGenerator = _enum_auto('TrajectoryGeneratorType')
    StayInInitialJointsTrajectoryGenerator = _enum_auto('TrajectoryGeneratorType')
    StayInInitialPoseTrajectoryGenerator = _enum_auto('TrajectoryGeneratorType')


class FeedbackControllerType:
    CartesianImpedanceFeedbackController = _enum_auto('FeedbackControllerType')
    EECartesianImpedanceFeedbackController = _enum_auto('FeedbackControllerType')
    ForceAxisImpedenceFeedbackController = _enum_auto('FeedbackControllerType')
    ForcePositionFeedbackController = _enum_auto('FeedbackControllerType')
    JointImpedanceFeedbackController = _enum_auto('FeedbackControllerType')
    NoopFeedbackController = _enum_auto('FeedbackControllerType')
    PassThroughFeedbackController = _enum_auto('FeedbackControllerType')
    SetInternalImpedanceFeedbackController = _enum_auto('FeedbackControllerType')


class TerminationHandlerType:
    ContactTerminationHandler = _enum_auto('TerminationHandlerType')
    FinalJointTerminationHandler = _enum_auto('TerminationHandlerType')
    FinalPoseTerminationHandler = _enum_auto('TerminationHandlerType')
    NoopTerminationHandler = _enum_auto('TerminationHandlerType')
    TimeTerminationHandler = _enum_auto('TerminationHandlerType')


class SkillStatus:
    TO_START = _enum_auto('SkillStatus')
    RUNNING = _enum_auto('SkillStatus')
    FINISHED = _enum_auto('SkillStatus')
    VIRT_COLL_ERR = _enum_auto('SkillStatus')


class SensorDataMessageType:
    BOUNDING_BOX = _enum_auto('SensorDataMessageType')
    CARTESIAN_IMPEDANCE = _enum_auto('SensorDataMessageType')
    FORCE_POSITION = _enum_auto('SensorDataMessageType')
    FORCE_POSITION_GAINS = _enum_auto('SensorDataMessageType')
    JOINT_POSITION_VELOCITY = _enum_auto('SensorDataMessageType')
    JOINT_POSITION = _enum_auto('SensorDataMessageType')
    POSE_POSITION_VELOCITY = _enum_auto('SensorDataMessageType')
    POSE_POSITION = _enum_auto('SensorDataMessageType')
    SHOULD_TERMINATE = _enum_auto('SensorDataMessageType')
