from franka_interface_msgs.msg import SensorData, SensorDataGroup
from frankapy.proto import *


## Feedback Controller Protobufs

def make_cartesian_impedance_feedback_controller_msg_proto(translational_stiffnesses, rotational_stiffnesses):
    cartesian_impedance_feedback_controller_msg_proto = feedback_controller_params_msg_pb2.CartesianImpedanceFeedbackControllerMessage()
    cartesian_impedance_feedback_controller_msg_proto.translational_stiffnesses.extend(translational_stiffnesses)
    cartesian_impedance_feedback_controller_msg_proto.rotational_stiffnesses.extend(rotational_stiffnesses)

    return cartesian_impedance_feedback_controller_msg_proto


def make_force_axis_feedback_controller_msg_proto(translational_stiffness, rotational_stiffness, axis):
    force_axis_feedback_controller_msg_proto = feedback_controller_params_msg_pb2.ForceAxisFeedbackControllerMessage()
    force_axis_feedback_controller_msg_proto.translational_stiffness = translational_stiffness
    force_axis_feedback_controller_msg_proto.rotational_stiffness = rotational_stiffness
    force_axis_feedback_controller_msg_proto.axis.extend(axis)

    return force_axis_feedback_controller_msg_proto


def make_joint_feedback_controller_msg_proto(k_gains, d_gains):
    joint_feedback_controller_msg_proto = feedback_controller_params_msg_pb2.JointImpedanceFeedbackControllerMessage()
    joint_feedback_controller_msg_proto.k_gains.extend(k_gains)
    joint_feedback_controller_msg_proto.d_gains.extend(d_gains)

    return joint_feedback_controller_msg_proto


def make_internal_feedback_controller_msg_proto(cartesian_impedances, joint_impedances):
    internal_feedback_controller_msg_proto = feedback_controller_params_msg_pb2.InternalImpedanceFeedbackControllerMessage()
    internal_feedback_controller_msg_proto.cartesian_impedances.extend(cartesian_impedances)
    internal_feedback_controller_msg_proto.joint_impedances.extend(joint_impedances)

    return internal_feedback_controller_msg_proto


## Termination Handler Protobufs

def make_contact_termination_handler_msg_proto(buffer_time, force_thresholds, torque_thresholds):
    contact_termination_handler_msg_proto = termination_handler_params_msg_pb2.ContactTerminationHandlerMessage()
    contact_termination_handler_msg_proto.buffer_time = buffer_time
    contact_termination_handler_msg_proto.force_thresholds.extend(force_thresholds)
    contact_termination_handler_msg_proto.torque_thresholds.extend(torque_thresholds)

    return contact_termination_handler_msg_proto


def make_joint_threshold_msg_proto(buffer_time, joint_thresholds):
    joint_threshold_msg_proto = termination_handler_params_msg_pb2.JointThresholdMessage()
    joint_threshold_msg_proto.buffer_time = buffer_time
    joint_threshold_msg_proto.joint_thresholds.extend(joint_thresholds)

    return joint_threshold_msg_proto


def make_pose_threshold_msg_proto(buffer_time, pose_thresholds):
    pose_threshold_msg_proto = termination_handler_params_msg_pb2.PoseThresholdMessage()
    pose_threshold_msg_proto.buffer_time = buffer_time
    pose_threshold_msg_proto.position_thresholds.extend(pose_thresholds[:3])
    pose_threshold_msg_proto.orientation_thresholds.extend(pose_thresholds[3:])

    return pose_threshold_msg_proto


def make_time_termination_handler_msg_proto(buffer_time):
    time_termination_handler_msg_proto = termination_handler_params_msg_pb2.TimeTerminationHandlerMessage()
    time_termination_handler_msg_proto.buffer_time = buffer_time

    return time_termination_handler_msg_proto


## Trajector Generator Protobufs

def make_gripper_trajectory_generator_msg_proto(grasp, width, speed, force):
    gripper_trajectory_generator_msg_proto = trajectory_generator_params_msg_pb2.GripperTrajectoryGeneratorMessage()
    gripper_trajectory_generator_msg_proto.grasp = grasp
    gripper_trajectory_generator_msg_proto.width = width
    gripper_trajectory_generator_msg_proto.speed = speed
    gripper_trajectory_generator_msg_proto.force = force

    return gripper_trajectory_generator_msg_proto


def make_impulse_trajectory_generator_msg_proto(run_time, acc_time, max_trans, max_rot, forces, torques):
    impulse_trajectory_generator_msg_proto = trajectory_generator_params_msg_pb2.ImpulseTrajectoryGeneratorMessage()
    impulse_trajectory_generator_msg_proto.run_time = run_time
    impulse_trajectory_generator_msg_proto.acc_time = acc_time
    impulse_trajectory_generator_msg_proto.max_trans = max_trans
    impulse_trajectory_generator_msg_proto.max_rot = max_rot
    impulse_trajectory_generator_msg_proto.forces.extend(forces)
    impulse_trajectory_generator_msg_proto.torques.extend(torques)

    return impulse_trajectory_generator_msg_proto


def make_joint_trajectory_generator_msg_proto(run_time, joints):
    joint_trajectory_generator_msg_proto = trajectory_generator_params_msg_pb2.JointTrajectoryGeneratorMessage()
    joint_trajectory_generator_msg_proto.run_time = run_time
    joint_trajectory_generator_msg_proto.joints.extend(joints)

    return joint_trajectory_generator_msg_proto


def make_pose_trajectory_generator_msg_proto(run_time, pose_rigid_transform):
    pose_trajectory_generator_msg_proto = trajectory_generator_params_msg_pb2.PoseTrajectoryGeneratorMessage()
    pose_trajectory_generator_msg_proto.run_time = run_time
    pose_trajectory_generator_msg_proto.position.extend(pose_rigid_transform.translation)
    pose_trajectory_generator_msg_proto.quaternion.extend(pose_rigid_transform.quaternion)
    pose_trajectory_generator_msg_proto.pose.extend(pose_rigid_transform.matrix.T.flatten().tolist())

    return pose_trajectory_generator_msg_proto


def make_joint_dmp_trajectory_generator_msg_proto(run_time, tau, alpha, beta, num_basis, num_sensor_values, 
                                                  basis_mean, basis_std, weights, initial_sensor_values):
    joint_dmp_trajectory_generator_msg_proto = trajectory_generator_params_msg_pb2.JointDMPTrajectoryGeneratorMessage()
    joint_dmp_trajectory_generator_msg_proto.run_time = run_time
    joint_dmp_trajectory_generator_msg_proto.tau = tau
    joint_dmp_trajectory_generator_msg_proto.alpha = alpha
    joint_dmp_trajectory_generator_msg_proto.beta = beta
    joint_dmp_trajectory_generator_msg_proto.num_basis = num_basis
    joint_dmp_trajectory_generator_msg_proto.num_sensor_values = num_sensor_values
    joint_dmp_trajectory_generator_msg_proto.basis_mean.extend(basis_mean)
    joint_dmp_trajectory_generator_msg_proto.basis_std.extend(basis_std)
    joint_dmp_trajectory_generator_msg_proto.weights.extend(weights)
    joint_dmp_trajectory_generator_msg_proto.initial_sensor_values.extend(initial_sensor_values)

    return joint_dmp_trajectory_generator_msg_proto


def make_pose_dmp_trajectory_generator_msg_proto(orientation_only, position_only,
                                                 run_time, tau, alpha, beta, num_basis, num_sensor_values, 
                                                 basis_mean, basis_std, y0, weights, initial_sensor_values):
    pose_dmp_trajectory_generator_msg_proto = trajectory_generator_params_msg_pb2.PoseDMPTrajectoryGeneratorMessage()
    pose_dmp_trajectory_generator_msg_proto.orientation_only = orientation_only
    pose_dmp_trajectory_generator_msg_proto.position_only = position_only
    pose_dmp_trajectory_generator_msg_proto.run_time = run_time
    pose_dmp_trajectory_generator_msg_proto.tau = tau
    pose_dmp_trajectory_generator_msg_proto.alpha = alpha
    pose_dmp_trajectory_generator_msg_proto.beta = beta
    pose_dmp_trajectory_generator_msg_proto.num_basis = num_basis
    pose_dmp_trajectory_generator_msg_proto.num_sensor_values = num_sensor_values
    pose_dmp_trajectory_generator_msg_proto.basis_mean.extend(basis_mean)
    pose_dmp_trajectory_generator_msg_proto.basis_std.extend(basis_std)
    pose_dmp_trajectory_generator_msg_proto.y0.extend(y0)
    pose_dmp_trajectory_generator_msg_proto.weights.extend(weights)
    pose_dmp_trajectory_generator_msg_proto.initial_sensor_values.extend(initial_sensor_values)

    return pose_dmp_trajectory_generator_msg_proto


def make_run_time_msg_proto(run_time):
    run_time_msg_proto = trajectory_generator_params_msg_pb2.RunTimeMessage()
    run_time_msg_proto.run_time = run_time

    return run_time_msg_proto


## Sensor Msg Protobufs

def make_joint_position_velocity_proto(id, timestamp, seg_run_time, joints, joint_velocities):
    joint_position_velocity_proto = sensor_msg_pb2.JointPositionVelocitySensorMessage()   
    joint_position_velocity_proto.id = id
    joint_position_velocity_proto.timestamp = timestamp
    joint_position_velocity_proto.seg_run_time = seg_run_time
    joint_position_velocity_proto.joints.extend(joints)
    joint_position_velocity_proto.joint_vels.extend(joint_velocities)

    return joint_position_velocity_proto


def make_joint_position_proto(id, timestamp, joints):
    joint_position_velocity_proto = sensor_msg_pb2.JointPositionSensorMessage()   
    joint_position_velocity_proto.id = id
    joint_position_velocity_proto.timestamp = timestamp
    joint_position_velocity_proto.joints.extend(joints)

    return joint_position_velocity_proto


def make_pose_position_velocity_proto(id, timestamp, seg_run_time, pose_rigid_transform, pose_velocities):
    '''
    pose_rigid_transform is a RigidTransform object
    pose_velocities is an ndarray of length 6 with velocities in xyz and euler angles xyz
    '''
    pose_position_velocity_proto = sensor_msg_pb2.PosePositionVelocitySensorMessage()   
    pose_position_velocity_proto.id = id
    pose_position_velocity_proto.timestamp = rospy.Time.now().to_time()
    pose_position_velocity_proto.seg_run_time = seg_run_time

    pose_position_velocity_proto.position.extend(pose_rigid_transform.translation)
    pose_position_velocity_proto.quaternion.extend(pose_rigid_transform.quaternion)
    pose_position_velocity_proto.pose_velocities.extend(pose_velocities)

    return pose_position_velocity_proto


def make_pose_position_proto(id, timestamp, pose_rigid_transform):
    '''
    pose_rigid_transform is a RigidTransform object
    pose_velocities is an ndarray of length 6 with velocities in xyz and euler angles xyz
    '''
    pose_position_proto = sensor_msg_pb2.PosePositionSensorMessage()   
    pose_position_proto.id = id
    pose_position_proto.timestamp = timestamp

    pose_position_proto.position.extend(pose_rigid_transform.translation)
    pose_position_proto.quaternion.extend(pose_rigid_transform.quaternion)

    return pose_position_proto


def make_should_terminate_proto(timestamp, should_terminate):
    should_terminate_proto = sensor_msg_pb2.ShouldTerminateSensorMessage()
    should_terminate_proto.timestamp = timestamp

    should_terminate_proto.should_terminate = should_terminate
    return should_terminate_proto


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
