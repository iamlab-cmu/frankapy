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
    GoalPoseDmpTrajectoryGenerator = 0
    GripperTrajectoryGenerator = 1
    ImpulseTrajectoryGenerator = 2
    JointDmpTrajectoryGenerator = 3
    LinearPoseTrajectoryGenerator = 4
    LinearJointTrajectoryGenerator = 5
    MinJerkJointTrajectoryGenerator = 6
    MinJerkPoseTrajectoryGenerator = 7
    PoseDmpTrajectoryGenerator = 8
    RelativeLinearPoseTrajectoryGenerator = 9
    RelativeMinJerkPoseTrajectoryGenerator = 10
    SineJointTrajectoryGenerator = 11
    SinePoseTrajectoryGenerator = 12
    StayInInitialJointsTrajectoryGenerator = 13
    StayInInitialPoseTrajectoryGenerator = 14

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
