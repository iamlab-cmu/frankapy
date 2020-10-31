import scipy.stats
import os
from shapes import Rectangle
import rospy
import cv_bridge
import time
import numpy as np
from autolab_core import RigidTransform, YamlConfig
from perception_utils.apriltags import AprilTagDetector
from perception import Kinect2SensorFactory, KinectSensorBridged
from sensor_msgs.msg import Image
from perception.camera_intrinsics import CameraIntrinsics

from frankapy import SensorDataMessageType
from frankapy import FrankaConstants as FC
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import PosePositionSensorMessage
from franka_interface_msgs.msg import SensorDataGroup
from geometry_msgs.msgs import PoseStamped

rospy.init_node("estimate_pose")

class Perception():
    def __init__(self, visualize=False):
        self.bridge = cv_bridge.CvBridge()
        self.visualize=visualize
        self.rate = 10 #Hz publish rate
        self.id = 0
        self.init_time = rospy.Time.now().to_time()
        self.setup_perception()
        self.pose_publisher = rospy.Publisher("/block_pose", PoseStamped, queue_size=10)
        self.franka_sensor_buffer_pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1000)

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
        detected_ids = []
        for new_detection in detections:
            detected_ids.append(int(new_detection.from_frame.split("/")[1])) #won't work for non-int values
            T_tag_cameras.append(new_detection)
        T_tag_camera = shape_class.tforms_to_pose(detected_ids, T_tag_cameras, goal=goal) #as if there were a tag in the center #assumes 1,2,3,4
        T_tag_camera.to_frame="kinect2_overhead"
        T_tag_world = self.T_camera_world * T_tag_camera
        if straighten:
            T_tag_world  = straighten_transform(T_tag_world)
        return T_tag_world

    def publish_shape_location(self, shape_type):
        rate = rospy.Rate(self.rate)
        while(True):
            T_tag_world = self.detect_ar_world_pos(shape_class=shape_type)
            pose_stamped_msg = PoseStamped()
            pose_stamped_msg.pose.position.x = T_tag_world.translation.x
            pose_stamped_msg.pose.position.y = T_tag_world.translation.y
            pose_stamped_msg.pose.position.z = T_tag_world.translation.z
            pose_stamped_msg.pose.orientation.w = T_tag_world.quaternion[0] #wxyz
            pose_stamped_msg.pose.orientation.x = T_tag_world.quaternion[1] 
            pose_stamped_msg.pose.orientation.y = T_tag_world.quaternion[2] 
            pose_stamped_msg.pose.orientation.z = T_tag_world.quaternion[3] 

            obj_pose_proto_msg = PosePositionSensorMessage(
                id=self.id, timestamp=rospy.Time.now().to_time() - self.init_time, 
                position=T_tag_world.translation,
                quaternion=T_tag_world.quaternion
            )

            ros_msg = make_sensor_group_msg(
                trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                    traj_gen_proto_msg, SensorDataMessageType.JOINT_POSITION)
            )
            self.franka_sensor_buffer_pub.publish(ros_msg)
            self.pose_publisher(pose_stamped_msg)
            self.id+=1
            rate.sleep() #Maintain speed. Get rid of to run as fast as possible

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
    robot = Perception(visualize=True)
    print("Publishing pose information...")
    robot.publish_shape_location(Rectangle)
    
