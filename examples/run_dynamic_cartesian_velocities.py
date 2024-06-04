import numpy as np
import time

from frankapy import FrankaConstants as FC
from frankapy import FrankaArm, SensorDataMessageType
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import CartesianVelocitySensorMessage, ShouldTerminateSensorMessage
from franka_interface_msgs.msg import SensorDataGroup

from frankapy.utils import min_jerk
import rospy


if __name__ == "__main__":
    fa = FrankaArm()
    fa.reset_joints()

    rospy.loginfo('Generating Trajectory')
    p = fa.get_pose()
    p.translation[2] -= 0.2
    fa.goto_pose(p, duration=5, block=False)

    cartesian_accelerations = [1, 1, 1, 0.1, 0.1, 0.1]

    T = 5
    dt = 0.01
    ts = np.arange(0, T, dt)
    cartesian_velocities = []

    for i in range(len(ts)):
        cartesian_velocities.append(fa.get_robot_state()['cartesian_velocities'])
        time.sleep(dt)

    fa.wait_for_skill()

    print(cartesian_velocities)

    pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1000)
    rate = rospy.Rate(1 / dt)

    fa.reset_joints()

    rospy.loginfo('Initializing Sensor Publisher')

    rospy.loginfo('Publishing cartesian velocity trajectory...')
    # To ensure skill doesn't end before completing trajectory, make the buffer time much longer than needed
    fa.execute_cartesian_velocities(cartesian_velocities[0], cartesian_accelerations, duration=T, dynamic=True, buffer_time=10)
    init_time = rospy.Time.now().to_time()
    for i in range(len(ts)):
        traj_gen_proto_msg = CartesianVelocitySensorMessage(
            id=i, timestamp=rospy.Time.now().to_time() - init_time, 
            cartesian_velocities=cartesian_velocities[i]
        )
        ros_msg = make_sensor_group_msg(
            trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                traj_gen_proto_msg, SensorDataMessageType.CARTESIAN_VELOCITY)
        )
        
        rospy.loginfo('Publishing: ID {}'.format(traj_gen_proto_msg.id))
        pub.publish(ros_msg)
        time.sleep(dt)

    # Stop the skill
    # Alternatively can call fa.stop_skill()
    term_proto_msg = ShouldTerminateSensorMessage(timestamp=rospy.Time.now().to_time() - init_time, should_terminate=True)
    ros_msg = make_sensor_group_msg(
        termination_handler_sensor_msg=sensor_proto2ros_msg(
            term_proto_msg, SensorDataMessageType.SHOULD_TERMINATE)
        )
    pub.publish(ros_msg)

    rospy.loginfo('Done')
