from time import sleep
import numpy as np

from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from frankapy.proto_utils import make_joint_position_velocity_proto, sensor_proto2ros_msg
from franka_msgs.msg import SensorData
from frankapy.utils import min_jerk, min_jerk_delta

import rospy


if __name__ == "__main__":
    fa = FrankaArm()
    fa.reset_joints()

    rospy.loginfo('Generating Trajectory')
    joints_0 = fa.get_joints()
    p = fa.get_pose()
    p.translation[2] -= 0.2
    fa.goto_pose(p)
    joints_1 = fa.get_joints()

    T = 5
    dt = 0.02
    ts = np.arange(0, T, dt)
    joints_traj = [min_jerk(joints_1, joints_0, t, T) for t in ts]
    joint_velocities_traj = [min_jerk_delta(joints_1, joints_0, t, T, dt) for t in ts]

    rospy.loginfo('Initializing Sensor Publisher')
    pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorData, queue_size=1000)
    rate = rospy.Rate(1 / dt)

    rospy.loginfo('Publishing joints trajectory...')
    fa.run_dynamic_joint_position_interpolation(joints_traj[1], duration=T, buffer_time=4)
    init_time = rospy.Time.now().to_time()
    for i in range(2, len(ts)):
        proto_msg = make_joint_position_velocity_proto(i, rospy.Time.now().to_time() - init_time, 
                                                      dt, joints_traj[i], joint_velocities_traj[i])
        ros_msg = sensor_proto2ros_msg(proto_msg, SensorDataMessageType.JOINT_POSITION_VELOCITY)
        
        rospy.loginfo('Publishing: ID {}'.format(proto_msg.id))
        pub.publish(ros_msg)
        rate.sleep()

    fa.wait_for_skill()
    rospy.loginfo('Done')
