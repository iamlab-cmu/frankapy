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

class MetaSkillType:
    BaseMetaSkill = 0
    JointPositionContinuousSkill = 1

class TrajectoryGeneratorType:
    CubicHermiteSplinePoseTrajectoryGenerator = 0
    CubicHermiteSplineJointTrajectoryGenerator = 1
    GoalPoseDmpTrajectoryGenerator = 2
    GripperTrajectoryGenerator = 3
    ImpulseTrajectoryGenerator = 4
    JointDmpTrajectoryGenerator = 5
    LinearPoseTrajectoryGenerator = 6
    LinearJointTrajectoryGenerator = 7
    MinJerkJointTrajectoryGenerator = 8
    MinJerkPoseTrajectoryGenerator = 9
    PoseDmpTrajectoryGenerator = 10
    RelativeLinearPoseTrajectoryGenerator = 11
    RelativeMinJerkPoseTrajectoryGenerator = 12
    SineJointTrajectoryGenerator = 13
    SinePoseTrajectoryGenerator = 14
    StayInInitialJointsTrajectoryGenerator = 15
    StayInInitialPoseTrajectoryGenerator = 16

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
    POSE_POSITION_VELOCITY = 1
    BOUNDING_BOX = 2
