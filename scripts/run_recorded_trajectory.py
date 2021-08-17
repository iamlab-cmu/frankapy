import argparse
import pickle
import numpy as np
import time

from autolab_core import RigidTransform
from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import PosePositionSensorMessage, ShouldTerminateSensorMessage
from franka_interface_msgs.msg import SensorDataGroup

from frankapy.utils import convert_array_to_rigid_transform

import rospy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--trajectory_pickle', '-t', type=str, required=True,
                        help='Path to trajectory (in pickle format) to replay.')
    parser.add_argument('--open_gripper', '-o', action='store_true')
    args = parser.parse_args()

    print('Starting robot')
    fa = FrankaArm()
    fa.reset_joints()

    rospy.loginfo('Loading Trajectory')

    with open(args.trajectory_pickle, 'rb') as pkl_f:
        skill_data = pickle.load(pkl_f)
    
    assert skill_data[0]['skill_description'] == 'GuideMode', \
        "Trajectory not collected in guide mode"
    skill_state_dict = skill_data[0]['skill_state_dict']

    T = float(skill_state_dict['time_since_skill_started'][-1])
    dt = 0.01
    ts = np.arange(0, T, dt)

    pose_traj = skill_state_dict['O_T_EE']
    # Goto the first position in the trajectory.
    print(convert_array_to_rigid_transform(pose_traj[0]).matrix)
    fa.goto_pose(convert_array_to_rigid_transform(pose_traj[0]), 
                 duration=4.0, 
                 cartesian_impedances=[600.0, 600.0, 600.0, 50.0, 50.0, 50.0])
    
    rospy.loginfo('Initializing Sensor Publisher')
    pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=10)
    rate = rospy.Rate(1 / dt)

    rospy.loginfo('Publishing pose trajectory...')
    # To ensure skill doesn't end before completing trajectory, make the buffer time much longer than needed
    fa.goto_pose(convert_array_to_rigid_transform(pose_traj[1]), 
                 duration=T, 
                 dynamic=True, 
                 buffer_time=10, 
                 cartesian_impedances=[600.0, 600.0, 600.0, 50.0, 50.0, 50.0]
    )
    init_time = rospy.Time.now().to_time()
    for i in range(2, len(ts)):
        timestamp = rospy.Time.now().to_time() - init_time
        pose_tf = convert_array_to_rigid_transform(pose_traj[i])
        traj_gen_proto_msg = PosePositionSensorMessage(
            id=i, 
            timestamp=timestamp,
            position=pose_tf.translation, 
            quaternion=pose_tf.quaternion
		)
        ros_msg = make_sensor_group_msg(
            trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                traj_gen_proto_msg, 
                SensorDataMessageType.POSE_POSITION),
            )

        # Sleep the same amount as the trajectory was recorded in
        dt = skill_state_dict['time_since_skill_started'][i] - skill_state_dict['time_since_skill_started'][i-1]
        rospy.loginfo('Publishing: ID {}, dt: {:.4f}'.format(traj_gen_proto_msg.id, dt))
        pub.publish(ros_msg)
        time.sleep(dt)
        # Finished trajectory
        if i >= pose_traj.shape[0] - 1:
            break

    # Stop the skill
    # Alternatively can call fa.stop_skill()
    term_proto_msg = ShouldTerminateSensorMessage(timestamp=rospy.Time.now().to_time() - init_time, 
                                                  should_terminate=True)
    ros_msg = make_sensor_group_msg(
        termination_handler_sensor_msg=sensor_proto2ros_msg(
            term_proto_msg, SensorDataMessageType.SHOULD_TERMINATE)
        )
    pub.publish(ros_msg)

    rospy.loginfo('Done')
