from franka_interface_msgs.msg import SensorData, SensorDataGroup


def sensor_proto2ros_msg(sensor_proto_msg, sensor_data_type, info=''):
    sensor_ros_msg = SensorData()
    sensor_ros_msg.type = sensor_data_type
    sensor_ros_msg.info = info 

    sensor_data_bytes = sensor_proto_msg.SerializeToString()
    sensor_ros_msg.size = len(sensor_data_bytes)
    sensor_ros_msg.sensorData = sensor_data_bytes

    return sensor_ros_msg


def make_sensor_group_msg(trajectory_generator_sensor_msg=None, feedback_controller_sensor_msg=None, termination_handler_sensor_msg=None):
    sensor_group_msg = SensorDataGroup()

    if trajectory_generator_sensor_msg is not None:
        sensor_group_msg.has_trajectory_generator_sensor_data = True
        sensor_group_msg.trajectoryGeneratorSensorData = trajectory_generator_sensor_msg
    if feedback_controller_sensor_msg is not None:
        sensor_group_msg.has_feedback_controller_sensor_data = True
        sensor_group_msg.feedbackControllerSensorData = feedback_controller_sensor_msg
    if termination_handler_sensor_msg is not None:
        sensor_group_msg.has_termination_handler_sensor_data = True
        sensor_group_msg.terminationHandlerSensorData = termination_handler_sensor_msg

    return sensor_group_msg
