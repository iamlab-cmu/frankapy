from time import sleep
import numpy as np

from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from frankapy.proto_utils import make_joint_position_proto, sensor_proto2ros_msg, make_sensor_group_msg
from franka_interface_msgs.msg import SensorDataGroup
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

    rospy.loginfo('Initializing Sensor Publisher')
    pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1000)
    rate = rospy.Rate(1 / dt)

    rospy.loginfo('Publishing joints trajectory...')
    fa.goto_joints(joints_traj[1], duration=T, dynamic=True, buffer_time=5)
    init_time = rospy.Time.now().to_time()
    for i in range(2, len(ts)):
        traj_gen_proto_msg = make_joint_position_proto(i, rospy.Time.now().to_time() - init_time, joints_traj[i])
        traj_gen_ros_msg = sensor_proto2ros_msg(traj_gen_proto_msg, SensorDataMessageType.JOINT_POSITION)
        ros_msg = make_sensor_group_msg(trajectory_generator_sensor_msg=traj_gen_ros_msg)
        
        rospy.loginfo('Publishing: ID {}'.format(traj_gen_proto_msg.id))
        pub.publish(ros_msg)
        rate.sleep()

    fa.wait_for_skill()
    rospy.loginfo('Done')
