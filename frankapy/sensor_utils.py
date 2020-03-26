from franka_action_lib_msgs.msg import SensorData
from frankapy.proto import sensor_msg_pb2


def sensor_proto2ros_msg(sensor_proto_msg, sensor_data_type, info=''):
    sensor_ros_msg = SensorData()
    sensor_ros_msg.type = sensor_data_type
    sensor_ros_msg.info = info 

    sensor_data_bytes = sensor_proto_msg.SerializeToString()
    sensor_ros_msg.size = len(sensor_data_bytes)
    sensor_ros_msg.sensorData = sensor_data_bytes

    return sensor_ros_msg


def make_joint_position_velocity_proto(id, timestamp, seg_run_time, joints, joint_velocities):
    joint_position_velocity_proto = sensor_msg_pb2.JointPositionVelocitySensorMessage()   
    joint_position_velocity_proto.id = id
    joint_position_velocity_proto.timestamp = timestamp
    joint_position_velocity_proto.seg_run_time = seg_run_time

    for n_joint in range(7):
        setattr(joint_position_velocity_proto, 'q{}'.format(n_joint + 1), joints[n_joint])
        setattr(joint_position_velocity_proto, 'dq{}'.format(n_joint + 1), joint_velocities[n_joint])

    return joint_position_velocity_proto


def make_pose_position_velocity_proto(id, timestamp, seg_run_time, pose_rigid_transform, pose_velocities):
    '''
    pose_rigid_transform is a RigidTransform object
    pose_velocities is an ndarray of length 6 with velocities in xyz and euler angles xyz
    '''
    pose_position_velocity_proto = sensor_msg_pb2.PosePositionVelocitySensorMessage()   
    pose_position_velocity_proto.id = id
    pose_position_velocity_proto.timestamp = timestamp
    pose_position_velocity_proto.seg_run_time = seg_run_time

    pose_position_velocity_proto.x = pose_rigid_transform.translation[0]
    pose_position_velocity_proto.y = pose_rigid_transform.translation[1]
    pose_position_velocity_proto.z = pose_rigid_transform.translation[2]
    q_wxyz = pose_rigid_transform.quaternion
    pose_position_velocity_proto.qw = q_wxyz[0]
    pose_position_velocity_proto.qx = q_wxyz[1]
    pose_position_velocity_proto.qy = q_wxyz[2]
    pose_position_velocity_proto.qz = q_wxyz[3]

    pose_position_velocity_proto.dx = pose_velocities[0]
    pose_position_velocity_proto.dy = pose_velocities[1]
    pose_position_velocity_proto.dz = pose_velocities[2]
    pose_position_velocity_proto.drx = pose_velocities[3]
    pose_position_velocity_proto.dry = pose_velocities[4]
    pose_position_velocity_proto.drz = pose_velocities[5]

    return pose_position_velocity_proto