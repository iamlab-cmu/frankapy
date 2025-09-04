import numpy as np
from cv_bridge import CvBridge

from perception.camera_intrinsics import CameraIntrinsics
from autolab_core import RigidTransform, Point
from perception import RgbdSensorFactory
from frankapy.proto import PosePositionSensorMessage
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy import SensorDataMessageType


class Perception:
    def __init__(self, cfg=None):
        self.cfg = cfg
        self.bridge = CvBridge()
        self.sensor_name = list(cfg.sensors.keys())[0]
        self.sensor_config = cfg.sensors[self.sensor_name].sensor_config
        self.azure_kinect_intrinsics = CameraIntrinsics.load(cfg.sensors[self.sensor_name].intrinsics_file)
        self.azure_kinect_to_world_transform = RigidTransform.load(cfg.sensors[self.sensor_name].extrinsics_tf)

        self.setup_perception()

    def setup_perception(self):
        self.sensor_config['frame'] = self.sensor_name
        self.sensor = RgbdSensorFactory.sensor(self.sensor_config.type, self.sensor_config)
        self.sensor.start()
    
    def get_object_center_point_in_world(self, object_image_center_x, object_image_center_y, depth_image):    
    
        object_center = Point(np.array([object_image_center_x, object_image_center_y]), self.sensor_name)
        object_depth = depth_image[object_image_center_y, object_image_center_x]
        object_center_point_in_world = self.azure_kinect_to_world_transform * self.sensor.ir_intrinsics.deproject_pixel(object_depth, object_center)    

        return object_center_point_in_world 

    def get_pose_msg(self, timestamp, id, T_tag_world):
        traj_gen_proto_msg = PosePositionSensorMessage(
            id=id, timestamp=timestamp, 
            position=T_tag_world.translation, quaternion=T_tag_world.quaternion
        )
        
        ros_msg = make_sensor_group_msg(
            trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                traj_gen_proto_msg, SensorDataMessageType.POSE_POSITION)
        )
        return ros_msg, traj_gen_proto_msg