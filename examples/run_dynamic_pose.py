from time import sleep
import numpy as np

from autolab_core import RigidTransform
from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from frankapy.sensor_utils import make_pose_position_velocity_proto, sensor_proto2ros_msg
from franka_action_lib_msgs.msg import SensorData
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

    p0_array = np.concatenate([p0.translation, p0.euler_angles])
    p1_array = np.concatenate([p1.translation, p1.euler_angles])
    pose_velocities_traj = [min_jerk_delta(p1_array, p0_array, t, T, dt) for t in ts]

    rospy.loginfo('Initializing Sensor Publisher')
    pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorData, queue_size=1000)
    rate = rospy.Rate(1 / dt)

    rospy.loginfo('Publishing pose trajectory...')
    fa.run_dynamic_pose_interpolation(pose_traj[1], duration=T, buffer_time=4)
    init_time = rospy.Time.now().to_time()
    for i in range(2, len(ts)):
        proto_msg = make_pose_position_velocity_proto(i, rospy.Time.now().to_time() - init_time, 
                                                      dt, pose_traj[2], pose_velocities_traj[2])
        ros_msg = sensor_proto2ros_msg(proto_msg, SensorDataMessageType.POSE_POSITION_VELOCITY)
        
        rospy.loginfo('Publishing: ID {}'.format(proto_msg.id))
        pub.publish(ros_msg)
        rate.sleep()

    fa.wait_for_skill()
    rospy.loginfo('Done')
