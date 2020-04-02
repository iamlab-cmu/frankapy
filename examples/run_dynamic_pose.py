from time import sleep
import numpy as np

from autolab_core import RigidTransform
from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from frankapy.proto_utils import make_pose_position_proto, sensor_proto2ros_msg, make_sensor_group_msg, make_should_terminate_proto
from franka_interface_msgs.msg import SensorDataGroup
from frankapy.utils import min_jerk, min_jerk_delta, min_jerk_weight

import rospy


if __name__ == "__main__":
    fa = FrankaArm()
    fa.reset_joints()

    rospy.loginfo('Generating Trajectory')
    p0 = fa.get_pose()
    p1 = p0.copy()
    T_delta = RigidTransform(
        translation=np.array([0, 0, 0.2]),
        rotation=RigidTransform.z_axis_rotation(np.deg2rad(30)), 
                            from_frame=p1.from_frame, to_frame=p1.from_frame)
    p1 = p1 * T_delta
    fa.goto_pose(p1)

    T = 5
    dt = 0.02
    ts = np.arange(0, T, dt)

    weights = [min_jerk_weight(t, T) for t in ts]
    pose_traj = [p1.interpolate_with(p0, w) for w in weights]

    rospy.loginfo('Initializing Sensor Publisher')
    pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1000)
    rate = rospy.Rate(1 / dt)

    rospy.loginfo('Publishing pose trajectory...')
    # To ensure skill doesn't end before completing trajectory, make the buffer time much longer than needed
    fa.goto_pose(pose_traj[1], duration=T, dynamic=True, buffer_time=10)
    init_time = rospy.Time.now().to_time()
    for i in range(2, len(ts)):
        traj_gen_proto_msg = make_pose_position_proto(i, rospy.Time.now().to_time() - init_time, pose_traj[i])
        ros_msg = make_sensor_group_msg(
            trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                traj_gen_proto_msg, SensorDataMessageType.POSE_POSITION)
            )
        
        rospy.loginfo('Publishing: ID {}'.format(traj_gen_proto_msg.id))
        pub.publish(ros_msg)
        rate.sleep()

    # Stop the skill
    # Alternatively can call fa.stop_skill()
    term_proto_msg = make_should_terminate_proto(rospy.Time.now().to_time() - init_time, True)
    ros_msg = make_sensor_group_msg(
        termination_handler_sensor_msg=sensor_proto2ros_msg(
            term_proto_msg, SensorDataMessageType.SHOULD_TERMINATE)
        )
    pub.publish(ros_msg)  

    rospy.loginfo('Done')
