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
import cv_bridge
import time
import numpy as np
from autolab_core import RigidTransform, YamlConfig
from perception_utils.apriltags import AprilTagDetector
from perception import Kinect2SensorFactory, KinectSensorBridged
from sensor_msgs.msg import Image
from perception.camera_intrinsics import CameraIntrinsics
import ipdb
import argparse
import pickle

class Perception():
    def __init__(self, visualize=False):
        self.bridge = cv_bridge.CvBridge()
        self.visualize=visualize
        self.rate = 10 #Hz publish rate
        self.id = 0
        # self.init_time = rospy.Time.now().to_time()
        self.setup_perception()
        # self.pose_publisher = rospy.Publisher("/block_pose", PoseStamped, queue_size=10)
        # self.franka_sensor_buffer_pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1000)

    def setup_perception(self):
        self.cfg = YamlConfig("saumya_ar_cfg.yaml") #TODO replace with your yaml file
        self.T_camera_world = RigidTransform.load(self.cfg['T_k4a_franka_path'])
        self.sensor = Kinect2SensorFactory.sensor('bridged', self.cfg)  # Kinect sensor object
        self.sensor.start()
        self.april = AprilTagDetector(self.cfg['april_tag'])
        # {"_frame": "kinect2_overhead", "_fx": 971.8240356445312, "_fy": 971.56640625, "_cx": 1022.5419311523438, "_cy": 787.8853759765625, "_skew": 0.0, "_height": 1536, "_width": 2048, "_K": 0}
        self.intr = CameraIntrinsics('k4a', fx=971.8240356445312,cx=1022.5419311523438, fy=971.56640625, cy=787.8853759765625, height=1536, width=2048) #fx fy cx cy overhead

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
        # ipdb.set_trace()
        detected_ids = []
        for new_detection in detections:
            detected_ids.append(int(new_detection.from_frame.split("/")[1])) #won't work for non-int values
            T_tag_cameras.append(new_detection)
        T_tag_camera = shape_class.tforms_to_pose(detected_ids, T_tag_cameras, goal=goal) #as if there were a tag in the center #assumes 1,2,3,4
        if T_tag_camera == None:
            return None
        T_tag_camera.to_frame="kinect2_overhead"
        T_tag_world = self.T_camera_world * T_tag_camera
        if straighten:
            T_tag_world  = straighten_transform(T_tag_world)
        return T_tag_world

    def rosmsg_shape_location(self, shape_type, timestamp, p0, z_stiffness_traj, i, prev_pose):

        T_tag_world = self.detect_ar_world_pos(shape_class=shape_type)
        if T_tag_world == None:
            T_tag_world = prev_pose.copy()
        # else:
            # T_tag_world.translation[2] = p0.translation[2] - 0.15
            # T_tag_world.translation += [0.0, 0.07, 0]

        traj_gen_proto_msg = PosePositionSensorMessage(
            id=i, timestamp=timestamp, 
            position=T_tag_world.translation, quaternion=p0.quaternion
        )

        fb_ctrlr_proto = CartesianImpedanceSensorMessage(
            id=i, timestamp=timestamp,
            translational_stiffnesses=[z_stiffness_traj[i]] + [z_stiffness_traj[i]] + [z_stiffness_traj[i]],
            rotational_stiffnesses=FC.DEFAULT_ROTATIONAL_STIFFNESSES
        )
        ros_msg = make_sensor_group_msg(
            trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                traj_gen_proto_msg, SensorDataMessageType.POSE_POSITION),
            feedback_controller_sensor_msg=sensor_proto2ros_msg(
                fb_ctrlr_proto, SensorDataMessageType.CARTESIAN_IMPEDANCE)
            )

        return traj_gen_proto_msg, ros_msg, T_tag_world

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--pose_dmp_weights_file_path', '-f', type=str)
    args = parser.parse_args()

    pose_dmp_file = open(args.pose_dmp_weights_file_path,"rb")
    pose_dmp_info = pickle.load(pose_dmp_file)

   
    percept = Perception(visualize=True)
    fa.reset_joints()
    fa.goto_gripper(0.0677)
    # rospy.loginfo('Generating Trajectory')
    # p0 = fa.get_pose()
    # p1 = p0.copy()
    # T_delta = RigidTransform(
    #     translation=np.array([0, 0, 0.2]),
    #     rotation=RigidTransform.z_axis_rotation(np.deg2rad(30)), 
    #                         from_frame=p1.from_frame, to_frame=p1.from_frame)
    # p1 = p1 * T_delta
    # fa.goto_pose(p1)

    starting_position = RigidTransform.load('pickup_starting_postion.tf')
    fa.goto_pose(starting_position, duration=5, use_impedance=False)
    p0 = fa.get_pose()

    T = 10
    dt = 0.001
    ts = np.arange(0, T, dt)
    # ipdb.set_trace()
    # weights = [min_jerk_weight(t, T) for t in ts]
    # pose_traj = [p1.interpolate_with(p0, w) for w in weights]

    z_stiffness_traj = [min_jerk(150, 400, t, T) for t in ts]

    rospy.loginfo('Initializing Sensor Publisher')
    pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=5000)
    rate = rospy.Rate(1 / dt)

    rospy.loginfo('Publishing pose trajectory...')

    T_tag_world = percept.detect_ar_world_pos(shape_class=Rectangle)

    # fa.goto_pose(p0, duration=T, dynamic=True, use_lqr=True, buffer_time=2,
    #     cartesian_impedances=[z_stiffness_traj[1]] + [z_stiffness_traj[1]] + [z_stiffness_traj[1]] + FC.DEFAULT_ROTATIONAL_STIFFNESSES
    # )

    fa.execute_pose_dmp(pose_dmp_info, position_only= True, duration=12, use_impedance=True, use_lqr_skill=True, cartesian_impedances=[1000, 1000, 2000, 50, 50, 50])

    # prev_pose = fa.get_pose()
    prev_pose = percept.detect_ar_world_pos(shape_class=Rectangle)
    init_time = rospy.Time.now().to_time()
    for i in range(2, int(len(ts)/4)):
        timestamp = rospy.Time.now().to_time() - init_time
        # print("z_stiffness_traj ", z_stiffness_traj[i])
        # print("time ", timestamp)
        
        traj_gen_proto_msg, ros_msg, prev_pose = percept.rosmsg_shape_location(Rectangle, timestamp, p0, z_stiffness_traj, i, prev_pose)
        
        rospy.loginfo('Publishing: ID {}'.format(traj_gen_proto_msg.id))
        pub.publish(ros_msg)
        rate.sleep()
        p1 = fa.get_pose()
        print(' Final Translation: {} | Final Rotation: {}'.format(p1.translation, p1.quaternion))
    # Stop the skill
    # Alternatively can call fa.stop_skill()
    term_proto_msg = ShouldTerminateSensorMessage(timestamp=rospy.Time.now().to_time() - init_time, should_terminate=True)
    ros_msg = make_sensor_group_msg(
        termination_handler_sensor_msg=sensor_proto2ros_msg(
            term_proto_msg, SensorDataMessageType.SHOULD_TERMINATE)
        )
    pub.publish(ros_msg)
    p1 = fa.get_pose()
    print(' Final Translation: {} | Final Rotation: {}'.format(p1.translation, p1.quaternion))
    rospy.loginfo('Done')
