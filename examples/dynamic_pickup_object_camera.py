from ipaddress import ip_address
import numpy as np
from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import PosePositionSensorMessage, ShouldTerminateSensorMessage, CartesianImpedanceSensorMessage
from franka_interface_msgs.msg import SensorDataGroup
from frankapy.utils import min_jerk, min_jerk_weight
import rospy
import scipy.stats
import os
from shapes import Rectangle
from cv_bridge import CvBridge, CvBridgeError

import time
import numpy as np
from autolab_core import RigidTransform, YamlConfig
from perception_utils.apriltags import AprilTagDetector
from perception import Kinect2SensorFactory, KinectSensorBridged
from sensor_msgs.msg import Image
from perception.camera_intrinsics import CameraIntrinsics




class Perception():
    def __init__(self, visualize=False):
        self.bridge = CvBridge()
        self.visualize=visualize
        self.rate = 10 #Hz publish rate
        self.id = 0
        # self.init_time = rospy.Time.now().to_time()
        self.setup_perception()
        # self.pose_publisher = rospy.Publisher("/block_pose", PoseStamped, queue_size=10)
        # self.franka_sensor_buffer_pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1000)

    def setup_perception(self):
        self.cfg = YamlConfig("cfg/saumya_april_tag_cfg.yaml") #TODO replace with your yaml file
        self.T_camera_world = RigidTransform.load(self.cfg['T_k4a_franka_path'])
        self.sensor = Kinect2SensorFactory.sensor('bridged', self.cfg)  # Kinect sensor object
        self.sensor.start()
        self.april = AprilTagDetector(self.cfg['april_tag'])
        # {"_frame": "azure_kinect_overhead", "_fx": 975.7285766601562, "_fy": 975.496826171875, "_cx": 1026.7061767578125, "_cy": 776.0191650390625, "_skew": 0.0, "_height": 1536, "_width": 2048, "_K": 0}
        self.intr = CameraIntrinsics('k4a', fx=975.7285766601562,cx=1026.7061767578125, fy=975.496826171875, cy=776.0191650390625, height=1536, width=2048) #fx fy cx cy overhead
    
    def detect_ar_world_pos(self,straighten=True, shape_class = Rectangle, goal=False):
        """
        O, 1, 2, 3 left hand corner. average [0,2] then [1,3]
        @param bool straighten - whether the roll and pitch should be
                    forced to 0 as prior information that the object is flat
        @param shape_class Shape \in {Rectangle, etc} - type of shape the detector should look for
        @param goal whether the detector should look for the object or goal hole
        """
        T_tag_cameras = []
        detections = self.april.detect(self.sensor, self.intr, vis=0)# Set vis=1 for debug

        detected_ids = []
        for new_detection in detections:
            detected_ids.append(int(new_detection.from_frame.split("/")[1])) #won't work for non-int values
            T_tag_cameras.append(new_detection)
        T_tag_camera = shape_class.tforms_to_pose(detected_ids, T_tag_cameras, goal=goal) #as if there were a tag in the center #assumes 1,2,3,4
        if T_tag_camera == None:
            return None
        T_tag_camera.to_frame="azure_kinect_overhead"
        T_tag_world = self.T_camera_world * T_tag_camera

        if straighten:
            T_tag_world  = straighten_transform(T_tag_world)
        return T_tag_world

    def group_rosmsg(self, timestamp, i, stiffness, T_tag_world):

        traj_gen_proto_msg = PosePositionSensorMessage(
            id=i, timestamp=timestamp, 
            position=T_tag_world.translation, quaternion=T_tag_world.quaternion
        )

        fb_ctrlr_proto = CartesianImpedanceSensorMessage(
            id=i, timestamp=timestamp,
            translational_stiffnesses=stiffness,
            rotational_stiffnesses=FC.DEFAULT_ROTATIONAL_STIFFNESSES
        )
        ros_msg = make_sensor_group_msg(
            trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                traj_gen_proto_msg, SensorDataMessageType.POSE_POSITION),
            feedback_controller_sensor_msg=sensor_proto2ros_msg(
                fb_ctrlr_proto, SensorDataMessageType.CARTESIAN_IMPEDANCE)
            )

        return traj_gen_proto_msg, ros_msg

def straighten_transform(rt):
    """
    Straightens object assuming that the roll and pitch are 0. 
    """
    angles = rt.euler_angles
    roll_off = angles[0]
    pitch_off = angles[1]
    roll_fix = RigidTransform(rotation = RigidTransform.x_axis_rotation(np.pi-roll_off),  from_frame = rt.from_frame, to_frame=rt.from_frame)
    pitch_fix = RigidTransform(rotation = RigidTransform.y_axis_rotation(pitch_off), from_frame = rt.from_frame, to_frame=rt.from_frame)
    new_rt = rt*roll_fix*pitch_fix
    return new_rt

if __name__ == "__main__":
    fa = FrankaArm()
    percept = Perception(visualize=True)
    fa.open_gripper()

    fa.reset_joints()
    import ipdb; ipdb.set_trace()

    starting_position = RigidTransform.load('examples/pickup_starting_position.tf')
    fa.goto_pose(starting_position, duration=5, use_impedance=False)
    p0 = fa.get_pose()
    
    object = p0.copy()
    T_tag_world = None
    while T_tag_world is None:
        T_tag_world = percept.detect_ar_world_pos(shape_class=Rectangle)
    object.translation = T_tag_world.translation.copy()
    object.translation[0] += -0.03
    object.translation[1] += -0.06
    # object.translation[2] += -0.12 # works with tape
    object.translation[2] += -0.12

    goal_transform = p0.copy()
    goal_transform.translation[0] = p0.translation[0]-0.0
    goal_transform.translation[1] = p0.translation[1]-0.4
    goal_transform.translation[2] = p0.translation[2]+0.15

    T = 10
    dt = 0.02
    ts = np.arange(0, T, dt)

    x_stiffness_traj = [min_jerk(40, 70, t, T) for t in ts]
    y_stiffness_traj = [min_jerk(20, 120, t, T) for t in ts]
    z_stiffness_traj1 = [min_jerk(20, 100, t, T) for t in ts]
    z_stiffness_traj2 = [min_jerk(30, 130, t, T) for t in ts]

    rospy.loginfo('Initializing Sensor Publisher')
    pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1000)
    rate = rospy.Rate(1 / dt)

    rospy.loginfo('Publishing pose trajectory...')

    # To ensure skill doesn't end before completing trajectory, make the buffer time much longer than needed
    fa.goto_pose(p0, duration=T, dynamic=True, buffer_time=2,
        cartesian_impedances=[x_stiffness_traj[0]] + [y_stiffness_traj[0]] + [z_stiffness_traj1[0]] + FC.DEFAULT_ROTATIONAL_STIFFNESSES
    )
    prev_pose = object.copy()
    init_time = rospy.Time.now().to_time()
    for i in range(2, len(ts)):
        timestamp = rospy.Time.now().to_time() - init_time
        p_now = fa.get_pose()
        if p_now.translation[1]>object.translation[1]+0.1:
            print(f'p_now.translation= {p_now.translation[1]}')
            print(f'object.translation[1]= {object.translation[1]}')
            # T_tag_world = percept.detect_ar_world_pos(shape_class=Rectangle)
            # if T_tag_world == None:
            #     T_tag_world = prev_pose.copy()
            # else:
            #     T_tag_world.translation[2] = object.translation[2]
            #     T_tag_world.rotation = p0.rotation.copy()
            #     prev_pose = T_tag_world.copy()
            traj_gen_proto_msg, ros_msg = percept.group_rosmsg(timestamp, i, [x_stiffness_traj[i]] + [y_stiffness_traj[i]] + [z_stiffness_traj1[i]], object)

            if p_now.translation[1]<object.translation[1]+0.108:
                print("closing gripper")
                current_gripper_width = 0.06
                fa.goto_gripper(current_gripper_width, block=False)

        else:
            print("reached object")
            current_gripper_width = 0.06
            fa.goto_gripper(current_gripper_width, block=False)
            traj_gen_proto_msg, ros_msg = percept.group_rosmsg(timestamp, i, [x_stiffness_traj[i]] + [y_stiffness_traj[i]] + [z_stiffness_traj1[i]], goal_transform)

        
        # rospy.loginfo('Publishing: ID {}'.format(traj_gen_proto_msg.id))
        pub.publish(ros_msg)
        rate.sleep()

    # Stop the skill
    # Alternatively can call fa.stop_skill()
    term_proto_msg = ShouldTerminateSensorMessage(timestamp=rospy.Time.now().to_time() - init_time, should_terminate=True)
    ros_msg = make_sensor_group_msg(
        termination_handler_sensor_msg=sensor_proto2ros_msg(
            term_proto_msg, SensorDataMessageType.SHOULD_TERMINATE)
        )
    pub.publish(ros_msg)

    rospy.loginfo('Done')