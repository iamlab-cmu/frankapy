import numpy as np

from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import ForcePositionSensorMessage, ForcePositionControllerSensorMessage
from franka_interface_msgs.msg import SensorDataGroup
#from frankapy.utils import transform_to_list, min_jerk
#from frankapy.utils import DMPPositionTrajectoryGenerator
from frankapy.utils import *

from tqdm import trange

from autolab_core import RigidTransform

import rospy
import pickle 

# def get_dmp_traj_poses_reformatted(y, starting_rotation):
#     '''
#     this converts y (xyz position traject) to list of 1x16 
#     '''
#     target_poses = []
#     last_row = np.array([0, 0, 0, 1])
#     for t in range(y.shape[0]):
#         transl = np.array([y[t,:]]).T
#         r_t = np.hstack((starting_rotation, transl))
#         TF_matrix = np.vstack((r_t,last_row)) # TF matrix
#         flattened_TF_matrix = TF_matrix.T.flatten().tolist()
#         target_poses.append(flattened_TF_matrix)    
#     return target_poses

# def downsample_dmp_traject(original_dmp_traject, og_dt, new_downsampled_dt):
#     '''
#     downsample original dmp_traject (default is dt = 0.001) (e.g. new dmp_traject dt = 0.01)
#     '''
#     downsmpled_dmp_traject = np.empty((0,original_dmp_traject.shape[1]))
#     samples_to_skip = int(new_downsampled_dt/og_dt)

#     inds_og_traject = np.arange(0,original_dmp_traject.shape[0],samples_to_skip)
#     for i in inds_og_traject:
#         downsmpled_dmp_traject = np.vstack((downsmpled_dmp_traject, original_dmp_traject[i,:]))
    
#     return downsmpled_dmp_traject
    
if __name__ == "__main__":
    fa = FrankaArm()
    # TODO: Move to desired cutting position/orientation prior to getting starting pose 
   
    # go to home joints position
    reset_joint_positions = [ 0.02846037, -0.51649966, -0.12048514, -2.86642333, -0.05060268,  2.30209197, 0.7744833 ]
    fa.goto_joints(reset_joint_positions)
    
    start_cut_pose = RigidTransform(rotation=np.array([
        [-0.06875206,  0.97926097, -0.19053231],
       [ 0.99702057,  0.06080344, -0.04726211],
       [-0.03469692, -0.193214  , -0.98054257]
       ]), translation=np.array([0.48059782, 0.03, 0.2]), 
       from_frame='franka_tool', to_frame='world')

    fa.goto_pose(start_cut_pose, duration=5, use_impedance = False)

    rospy.loginfo('Generating Trajectory')
    start_pose = fa.get_pose()
    starting_rotation = start_pose.rotation

    # define length of dmp trajectory
    traject_time = 5 # TODO: make this a CL arg    
    # load dmp traject params
    #dmp_wts_pkl_filepath = '/home/sony/Desktop/debug_dmp_wts.pkl'
    dmp_wts_pkl_filepath = '/home/sony/092420_normal_cut_dmp_weights_zeroY.pkl' # y weights zero-ed out
    dmp_traj = DMPPositionTrajectoryGenerator(traject_time)
    dmp_traj.load_saved_dmp_params_from_pkl_file(dmp_wts_pkl_filepath)
    dmp_traj.parse_dmp_params_dict()

    # define starting position - need to change this based on where robot actually starts
    y0 = start_pose.translation
    # calculate dmp position trajectory - NOTE: this assumes a 0.001 dt for calc the dmp traject
    dmp_traject, dy, _, _, _ = dmp_traj.run_dmp_with_weights(y0) # y: np array(tx3)
    #target_poses = get_dmp_traj_poses_reformatted(y, starting_rotation) # target_poses is a nx16 list of target poses at each time step

    # sampling info for sending msgs via ROS
    dt = 0.01 #0.001
    T = traject_time
    ts = np.arange(0, T, dt)
    N = len(ts)

    # downsample dmp traject 
    downsmpled_dmp_traject = downsample_dmp_traject(dmp_traject, 0.001, dt)
    target_poses = get_dmp_traj_poses_reformatted(downsmpled_dmp_traject, starting_rotation) # target_poses is a nx16 list of target poses at each time step
    
    target_force = [0, 0, 0, 0, 0, 0]  #[0, 0, -10, 0, 0, 0] 
    S = [1, 1, 1, 1, 1, 1] #[1, 1, 0, 1, 1, 1]
    position_kps_cart = FC.DEFAULT_TRANSLATIONAL_STIFFNESSES + FC.DEFAULT_ROTATIONAL_STIFFNESSES
    force_kps_cart = [0.1] * 6
    position_kps_joint = FC.DEFAULT_K_GAINS
    force_kps_joint = [0.1] * 7

    rospy.loginfo('Initializing Sensor Publisher')
    pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1)
    rate = rospy.Rate(1 / dt)
    n_times = 1

    rospy.loginfo('Publishing HFPC trajectory w/ cartesian gains...')

    current_ht = fa.get_pose().translation[2]
    dmp_num = 0 
    import pdb; pdb.set_trace()

    # start FP skill
    fa.run_dynamic_force_position(duration=T * 10, buffer_time = 3, S=S,
                                    use_cartesian_gains=True,
                                    position_kps_cart=position_kps_cart,
                                    force_kps_cart=force_kps_cart,block=False)
    print('DEBUG')

    while current_ht > 0.0446:
        print('starting dmp', dmp_num)        
        init_time = rospy.Time.now().to_time()
        for i in trange(N * n_times):
            t = i % N
            timestamp = rospy.Time.now().to_time() - init_time
            #NOTE: format of pose sent is: 1x16 Transform matrix 
            
            traj_gen_proto_msg = ForcePositionSensorMessage(
                id=i, timestamp=timestamp, seg_run_time=dt,
                pose=target_poses[t],
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
        
        current_ht = fa.get_pose().translation[2]
        print('current_ht', current_ht)
        dmp_num+=1       

        # calculate new dmp traject based on current position
        y0 = fa.get_pose().translation
        # calculate dmp position trajectory - NOTE: this assumes a 0.001 dt for calc the dmp traject
        dmp_traject, dy, _, _, _ = dmp_traj.run_dmp_with_weights(y0) # y: np array(tx3)
        # downsample dmp traject and reformat target_poses
        downsmpled_dmp_traject = downsample_dmp_traject(dmp_traject, 0.001, dt)
        target_poses = get_dmp_traj_poses_reformatted(downsmpled_dmp_traject, starting_rotation) # target_poses is a nx16 list of target poses at each time step
    
    

    fa.stop_skill()
   
    fa.goto_joints(reset_joint_positions)
    
    rospy.loginfo('Done')
