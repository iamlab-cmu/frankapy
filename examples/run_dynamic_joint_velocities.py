import numpy as np
import time

from frankapy import FrankaArm, SensorDataMessageType
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import JointVelocitySensorMessage, ShouldTerminateSensorMessage
from franka_interface_msgs.msg import SensorDataGroup

from frankapy.utils import min_jerk

if __name__ == "__main__":
    fa = FrankaArm()
    fa.reset_joints()

    fa.log_info('Generating Trajectory')
    p = fa.get_pose()
    p.translation[2] -= 0.2
    fa.goto_pose(p, duration=5, block=False)

    joint_accelerations = [1, 1, 1, 1, 1, 1, 1]

    T = 5
    dt = 0.001
    ts = np.arange(0, T, dt)
    joint_velocities = []

    for i in range(len(ts)):
        joint_velocities.append(fa.get_robot_state()['joint_velocities'])
        time.sleep(dt)

    fa.wait_for_skill()

    fa.reset_joints()

    fa.log_info('Initializing Sensor Publisher')

    fa.log_info('Publishing joints trajectory...')
    # To ensure skill doesn't end before completing trajectory, make the buffer time much longer than needed
    fa.execute_joint_velocities(joint_velocities[0], joint_accelerations, duration=T, dynamic=True, buffer_time=10)
    init_time = fa.get_time()
    for i in range(len(ts)):
        traj_gen_proto_msg = JointVelocitySensorMessage(
            id=i, timestamp=fa.get_time() - init_time, 
            joint_velocities=joint_velocities[i]
        )
        ros_msg = make_sensor_group_msg(
            trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                traj_gen_proto_msg, SensorDataMessageType.JOINT_VELOCITY)
        )
        
        fa.log_info('Publishing: ID {}'.format(traj_gen_proto_msg.id))
        fa.publish_sensor_data(ros_msg)
        time.sleep(dt)

    # Stop the skill
    # Alternatively can call fa.stop_skill()
    term_proto_msg = ShouldTerminateSensorMessage(timestamp=fa.get_time() - init_time, should_terminate=True)
    ros_msg = make_sensor_group_msg(
        termination_handler_sensor_msg=sensor_proto2ros_msg(
            term_proto_msg, SensorDataMessageType.SHOULD_TERMINATE)
        )
    fa.publish_sensor_data(ros_msg)

    fa.log_info('Done')
