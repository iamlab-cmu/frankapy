#######################################################################
#                                                                     #
#   Important: Any Changes here should also be reflected in changes   #
#   in the frankapy iam_robolib_common_definitions.py file as well.    #
#                                                                     #
####################################################################### 

class SkillType:
    CartesianPoseSkill = 0
    ForceTorqueSkill = 1
    GripperSkill = 2
    ImpedanceControlSkill = 3
    JointPositionSkill = 4
    JointPositionDynamicInterpolationSkill = 5

class MetaSkillType:
    BaseMetaSkill = 0
    JointPositionContinuousSkill = 1

class TrajectoryGeneratorType:
    CubicHermiteSplineJointTrajectoryGenerator = 0
    GoalPoseDmpTrajectoryGenerator = 1
    GripperTrajectoryGenerator = 2
    ImpulseTrajectoryGenerator = 3
    JointDmpTrajectoryGenerator = 4
    LinearPoseTrajectoryGenerator = 5
    LinearJointTrajectoryGenerator = 6
    MinJerkJointTrajectoryGenerator = 7
    MinJerkPoseTrajectoryGenerator = 8
    PoseDmpTrajectoryGenerator = 9
    RelativeLinearPoseTrajectoryGenerator = 10
    RelativeMinJerkPoseTrajectoryGenerator = 11
    SineJointTrajectoryGenerator = 12
    SinePoseTrajectoryGenerator = 13
    StayInInitialJointsTrajectoryGenerator = 14
    StayInInitialPoseTrajectoryGenerator = 15

class FeedbackControllerType:
    CartesianImpedanceFeedbackController = 0
    ForceAxisImpedenceFeedbackController = 1
    JointImpedanceFeedbackController = 2
    NoopFeedbackController = 3
    PassThroughFeedbackController = 4
    SetInternalImpedanceFeedbackController = 5

class TerminationHandlerType:
    ContactTerminationHandler = 0
    FinalJointTerminationHandler = 1
    FinalPoseTerminationHandler = 2
    NoopTerminationHandler = 3
    TimeTerminationHandler = 4

class SkillStatus: 
    TO_START = 0
    RUNNING = 1
    FINISHED = 2 

class SensorDataMessageType:
    JOINT_POSITION_VELOCITY = 0
    BOUNDING_BOX = 1
