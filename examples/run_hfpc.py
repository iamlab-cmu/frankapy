import numpy as np

from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import ForcePositionSensorMessage
from franka_interface_msgs.msg import SensorDataGroup

from frankapy.utils import transform_to_list

import rospy


if __name__ == "__main__":
    fa = FrankaArm()
    fa.reset_joints()
    # fa.close_gripper()

    # while True:
    #     input('Presse [Enter] to enter guide mode and move robot to be on top of a flat surface.')
    #     fa.run_guide_mode()
    #     while True:
    #         inp = input('[r]etry or [c]ontinue? ')
    #         if inp not in ('r', 'c'):
    #             print('Please give valid input!')
    #         else:
    #             break
    #     if inp == 'c':
    #         break

    rospy.loginfo('Generating Trajectory')
    # EE will follow a 2D circle while pressing down with a target force

    start_pose = fa.get_pose()

    dt = 0.02
    T = 20
    dts = np.arange(0, T, dt)
    N = len(dts)
    dthetas = np.linspace(-np.pi / 2, 3 * np.pi / 2, N)
    r = 0.05
    circ = r * np.c_[np.cos(dthetas), np.sin(dthetas)]

    target_poses = []
    for i, t in enumerate(dts):
        pose = start_pose.copy()
        pose.translation[0] += r + circ[i, 0]
        pose.translation[1] += circ[i, 1]
        target_poses.append(pose)

    target_force = [0, 0, 0, 0, 0, 0]
    S = [1, 1, 1, 1, 1, 1]

    rospy.loginfo('Initializing Sensor Publisher')
    pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1000)
    rate = rospy.Rate(1 / dt)

    rospy.loginfo('Publishing HFPC trajectory...')
    # To ensure skill doesn't end before completing trajectory, make the buffer time much longer than needed
    fa.run_dynamic_force_position(duration=T, buffer_time=10, S=S, position_kp=100, force_kp=0.1)
    init_time = rospy.Time.now().to_time()
    for i in range(N):
        timestamp = rospy.Time.now().to_time() - init_time
        traj_gen_proto_msg = ForcePositionSensorMessage(
            id=i, timestamp=timestamp, 
            pose=transform_to_list(target_poses[i]),
            force=target_force
        )
        # fb_ctrlr_proto = CartesianImpedanceSensorMessage(
        #     id=i, timestamp=timestamp,
        #     translational_stiffnesses=FC.DEFAULT_TRANSLATIONAL_STIFFNESSES[:2] + [z_stiffness_traj[i]],
        #     rotational_stiffnesses=FC.DEFAULT_ROTATIONAL_STIFFNESSES
        # )
        ros_msg = make_sensor_group_msg(
            trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                traj_gen_proto_msg, SensorDataMessageType.FORCE_POSITION))
            # feedback_controller_sensor_msg=sensor_proto2ros_msg(
            #     fb_ctrlr_proto, SensorDataMessageType.CARTESIAN_IMPEDANCE)
            # )
        
        # rospy.loginfo('Publishing: ID {}'.format(i))
        pub.publish(ros_msg)
        rate.sleep()

    fa.stop_skill()

    fa.reset_joints()
    fa.open_gripper()
    
    rospy.loginfo('Done')
