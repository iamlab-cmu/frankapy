from franka_action_lib_msgs.msg import SensorData
from frankapy.proto import sensor_msg_pb2


def make_joint_position_velocity_proto(id, timestamp, seg_run_time, joints, joint_velocities):
    joint_position_velocity_proto = sensor_msg_pb2.JointPositionVelocitySensorMessage()   
    joint_position_velocity_proto.id = id
    joint_position_velocity_proto.timestamp = timestamp
    joint_position_velocity_proto.seg_run_time = seg_run_time

    for n_joint in range(7):
        setattr(joint_position_velocity_proto, 'q{}'.format(n_joint + 1), joints[n_joint])
        setattr(joint_position_velocity_proto, 'dq{}'.format(n_joint + 1), joint_velocities[n_joint])

    return joint_position_velocity_proto


def sensor_proto2ros_msg(sensor_proto_msg, sensor_data_type, info=''):
    sensor_ros_msg = SensorData()
    sensor_ros_msg.type = sensor_data_type
    sensor_ros_msg.info = info 

    sensor_data_bytes = sensor_proto_msg.SerializeToString()
    sensor_ros_msg.size = len(sensor_data_bytes)
    sensor_ros_msg.sensorData = sensor_data_bytes

    return sensor_ros_msg