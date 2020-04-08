import numpy as np

from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import ForcePositionSensorMessage, ForcePositionControllerSensorMessage
from franka_interface_msgs.msg import SensorDataGroup
from frankapy.utils import transform_to_list, min_jerk

from tqdm import trange

import rospy


if __name__ == "__main__":
    fa = FrankaArm()
    fa.reset_joints()
    fa.close_gripper()

    while True:
        input('Presse [Enter] to enter guide mode and move robot to be on top of a flat surface.')
        fa.run_guide_mode()
        while True:
            inp = input('[r]etry or [c]ontinue? ')
            if inp not in ('r', 'c'):
                print('Please give valid input!')
            else:
                break
        if inp == 'c':
            break

    rospy.loginfo('Generating Trajectory')
    # EE will follow a 2D circle while pressing down with a target force
    dt = 0.01
    T = 10
    ts = np.arange(0, T, dt)
    N = len(ts)
    dthetas = np.linspace(-np.pi / 2, 3 * np.pi / 2, N)
    r = 0.07
    circ = r * np.c_[np.sin(dthetas), np.cos(dthetas)]

    start_pose = fa.get_pose()
    start_pose.rotation = FC.HOME_POSE.rotation
    target_poses = []
    for i, t in enumerate(ts):
        pose = start_pose.copy()
        pose.translation[0] += r + circ[i, 0]
        pose.translation[1] += circ[i, 1]
        target_poses.append(pose)
    target_force = [0, 0, -10, 0, 0, 0]
    S = [1, 1, 1, 1, 1, 1]
    position_kps_cart = FC.DEFAULT_TRANSLATIONAL_STIFFNESSES + FC.DEFAULT_ROTATIONAL_STIFFNESSES
    force_kps_cart = [0.1] * 6
    position_kps_joint = FC.DEFAULT_K_GAINS
    force_kps_joint = [0.1] * 7

    rospy.loginfo('Initializing Sensor Publisher')
    pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1)
    rate = rospy.Rate(1 / dt)
    n_times = 1

    rospy.loginfo('Publishing HFPC trajectory w/ cartesian gains...')
    fa.run_dynamic_force_position(duration=T * n_times, buffer_time=3, S=S,
                                use_cartesian_gains=True,
                                position_kps_cart=position_kps_cart,
                                force_kps_cart=force_kps_cart)
    init_time = rospy.Time.now().to_time()
    for i in trange(N * n_times):
        t = i % N
        timestamp = rospy.Time.now().to_time() - init_time
        traj_gen_proto_msg = ForcePositionSensorMessage(
            id=i, timestamp=timestamp, seg_run_time=dt,
            pose=transform_to_list(target_poses[t]),
            force=target_force
        )
        fb_ctrlr_proto = ForcePositionControllerSensorMessage(
            id=i, timestamp=timestamp,
            position_kps_cart=position_kps_cart,
            force_kps_cart=force_kps_cart,
            selection=S
        )
        ros_msg = make_sensor_group_msg(
            trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                traj_gen_proto_msg, SensorDataMessageType.FORCE_POSITION),
            feedback_controller_sensor_msg=sensor_proto2ros_msg(
                fb_ctrlr_proto, SensorDataMessageType.FORCE_POSITION_GAINS)
            )
        pub.publish(ros_msg)
        rate.sleep()
    fa.stop_skill()

    rospy.loginfo('Publishing HFPC trajectory w/ joint gains...')
    fa.run_dynamic_force_position(duration=T * n_times, buffer_time=3, S=S,
                                use_cartesian_gains=False,
                                position_kps_joint=position_kps_joint,
                                force_kps_joint=force_kps_joint)
    init_time = rospy.Time.now().to_time()
    for i in trange(N * n_times):
        t = i % N
        timestamp = rospy.Time.now().to_time() - init_time
        traj_gen_proto_msg = ForcePositionSensorMessage(
            id=i, timestamp=timestamp, seg_run_time=dt,
            pose=transform_to_list(target_poses[t]),
            force=target_force
        )
        fb_ctrlr_proto = ForcePositionControllerSensorMessage(
            id=i, timestamp=timestamp,
            position_kps_joint=position_kps_joint,
            force_kps_joint=force_kps_joint,
            selection=S
        )
        ros_msg = make_sensor_group_msg(
            trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                traj_gen_proto_msg, SensorDataMessageType.FORCE_POSITION),
            feedback_controller_sensor_msg=sensor_proto2ros_msg(
                fb_ctrlr_proto, SensorDataMessageType.FORCE_POSITION_GAINS)
            )
        pub.publish(ros_msg)
        rate.sleep()
    fa.stop_skill()

    fa.reset_joints()
    fa.open_gripper()
    
    rospy.loginfo('Done')
