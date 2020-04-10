import numpy as np

from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import ForcePositionSensorMessage, ForcePositionControllerSensorMessage
from franka_interface_msgs.msg import SensorDataGroup
from frankapy.utils import transform_to_list

from tqdm import trange

import rospy


if __name__ == "__main__":
    fa = FrankaArm()
    fa.reset_joints()
    fa.close_gripper()

    input('Please hold a flat surface right beneath the robot gripper. [ENTER] to continue.')

    dt = 0.01
    T = 20
    N = int(T / dt)

    position_kps_cart = FC.DEFAULT_TRANSLATIONAL_STIFFNESSES + FC.DEFAULT_ROTATIONAL_STIFFNESSES
    force_kps_cart = FC.DEFAULT_HFPC_FORCE_GAIN
    S = [1, 1, 1, 1, 1, 1]

    rospy.loginfo('Initializing Sensor Publisher')
    pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1)
    rate = rospy.Rate(1 / dt)

    rospy.loginfo('Running HFPC...')
    fa.run_dynamic_force_position(duration=T, buffer_time=3, S=S,
                                use_cartesian_gains=True,
                                position_kps_cart=position_kps_cart,
                                force_kps_cart=force_kps_cart)
    for i in range(N):
        timestamp = rospy.Time.now().to_time()

        force_torque = fa.get_force_torque()
        force = force_torque[:3]
        force_mag = np.linalg.norm(force)
        force_axis = force / force_mag
        in_contact = force_mag > 2

        current_pose = fa.get_pose()
        
        # if in contact, maintain z-direction contact
        if in_contact:
            print('in contact', force_mag)
            target_force = np.array([0, 0, -10, 0, 0, 0])
            target_pose = current_pose
            S = [0, 0, 0, 1, 1, 1]
            force_kis_cart = [0.01 * v for v in FC.DEFAULT_HFPC_FORCE_GAIN]
            reset_force_integral_error = False
        # if not in contact, stay in place
        else:
            print('not in contact', force_mag)
            target_pose = current_pose
            target_force = np.zeros(6)
            S = [0, 0, 0, 1, 1, 1]
            force_kis_cart = [0] * 6
            reset_force_integral_error = True

        err_frame = np.eye(3)
        traj_gen_proto_msg = ForcePositionSensorMessage(
            id=i, timestamp=timestamp, seg_run_time=dt,
            pose=transform_to_list(target_pose), force=target_force.tolist()
        )
        fb_ctrlr_proto = ForcePositionControllerSensorMessage(
            id=i, timestamp=timestamp,
            position_kps_cart=position_kps_cart,
            force_kps_cart=force_kps_cart,
            force_kis_cart=force_kis_cart,
            selection=S, error_frame=err_frame.flatten().tolist(),
            reset_force_integral_error=reset_force_integral_error
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
