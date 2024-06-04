import numpy as np
import time

from frankapy import FrankaConstants as FC
from frankapy import FrankaArm, SensorDataMessageType
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import JointVelocitySensorMessage, ShouldTerminateSensorMessage
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

    joint_accelerations = [1, 1, 1, 1, 1, 1, 1]

    T = 5
    dt = 0.01
    ts = np.arange(0, T, dt)
    joint_velocities = []

    for i in range(len(ts)):
        joint_velocities.append(fa.get_robot_state()['joint_velocities'])
        time.sleep(dt)

    fa.wait_for_skill()

    pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1000)
    rate = rospy.Rate(1 / dt)

    fa.reset_joints()

    rospy.loginfo('Initializing Sensor Publisher')

    rospy.loginfo('Publishing joints trajectory...')
    # To ensure skill doesn't end before completing trajectory, make the buffer time much longer than needed
    fa.execute_joint_velocities(joint_velocities[0], joint_accelerations, duration=T, dynamic=True, buffer_time=10)
    init_time = rospy.Time.now().to_time()
    for i in range(len(ts)):
        traj_gen_proto_msg = JointVelocitySensorMessage(
            id=i, timestamp=rospy.Time.now().to_time() - init_time, 
            joint_velocities=joint_velocities[i]
        )
        ros_msg = make_sensor_group_msg(
            trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                traj_gen_proto_msg, SensorDataMessageType.JOINT_VELOCITY)
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
