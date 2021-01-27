''' Notes: 
TODO: add task success count/tracking, time to complete cut metrics and save

- try w/ hard vs soft objects (carrot/celery/potato, vs. cucumber vs. tomato)
- try w/ pivoted cut and variable cartesian gains/stiffnesses 
'''
# TODO: refactor this script to combine piv chop and normal cut
import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
plt.close("all")
import math
import rospy
import argparse
import pickle
from autolab_core import RigidTransform, Point
from frankapy import FrankaArm
from cv_bridge import CvBridge

from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import ForcePositionSensorMessage, ForcePositionControllerSensorMessage, ShouldTerminateSensorMessage
from franka_interface_msgs.msg import SensorDataGroup
from frankapy.utils import *

from tqdm import trange
from rl_utils import reps
import time
import copy

'''note: dmp weights trained with:
python 04_train_dmp.py -i ~/robot_state_data_0.npz -d position -o ~/092420_normal_cut_dmp_weights -b 6 -s 1 -a 5 -t 0.7  -v --time 4.5
'/home/sony/092420_normal_cut_dmp_weights_zeroY.pkl' --> w/ y weights not zeroed out
'/home/sony/092420_normal_cut_dmp_weights_wY.pkl' --> with y weights 

to start script from previous data (i.e. later epoch/sample):
python examples/run_cutting_RL.py -w /home/sony/092420_normal_cut_dmp_weights_wY.pkl -n 6 -sfp True -pd '/home/sony/Documents/cutting_RL_experiments/data/celery/exp_6/' -start_epoch 1 -s 15 --use_all_dmp_dims True

 python examples/run_cutting_RL.py -w /home/sony/092420_normal_cut_dmp_weights_wY.pkl -n 6 -sfp True -pd '/home/sony/Documents/cutting_RL_experiments/data/celery/exp_6/' -start_epoch 2 -start_sample 14 -s 15 --use_all_dmp_dims True
'''
# TODO: run comparison experiments of force control + var stiffness vs. position control + var stiffness 

def plot_sampled_new_dmp_traject_and_original_dmp(epoch, sample, save_dir, new_pitch_stiffness, traject_time, \
    initial_dmp_weights_pkl_file, new_dmp_traject, y0):
    #original_dmp_wts_pkl_filepath = '/home/sony/Desktop/debug_dmp_wts.pkl'
    dmp_traj = DMPPositionTrajectoryGenerator(traject_time)
    dmp_traj.load_saved_dmp_params_from_pkl_file(initial_dmp_weights_pkl_file)
    dmp_traj.parse_dmp_params_dict()
    # calculate dmp position trajectory - NOTE: this assumes a 0.001 dt for calc the dmp traject
    original_traject, dy, _, _, _ = dmp_traj.run_dmp_with_weights(y0) # y: np array(tx3)
    
    axes = ['x', 'y','z']   
    fig, ax = plt.subplots(3,1) 
    for i in range(3):
        ax[i].plot(np.arange(0, traject_time, 0.001), original_traject[:,i])
        ax[i].plot(np.arange(0, traject_time, 0.001), new_dmp_traject[:,i])           
        if i!=0:
            ax[i].set_title('Cartesian Position - '+str(axes[i]))
        else:     
            if new_pitch_stiffness == 'NA':
                ax[i].set_title('Cartesian Position - '+str(axes[i]))
            else:
                ax[i].set_title('Cartesian Position - '+str(axes[i]) + ', ' + 'Cart_pitch_stiffness: '+str(new_pitch_stiffness))
        ax[i].set_ylabel('Position (m)')
        ax[i].legend((axes[i] + '-original traject', axes[i] + '-new sampled traject'))
    
    ax[2].set_xlabel('Time (s)')
    plt.show()
    # save figure to working dir
    fig.savefig(work_dir + '/' + 'dmp_traject_plots' + '/sampledDMP_' + 'epoch_'+str(epoch) + '_ep_'+str(sample)+'.png')

def plot_updated_policy_mean_traject(work_dir, position_dmp_weights_file_path, epoch, dmp_traject_time, control_type_z_axis, init_dmp_info_dict, \
    initial_wts, REPS_updated_mean):
    if control_type_z_axis == 'force':                
        new_weights = np.expand_dims(np.vstack((REPS_updated_mean[0:7],initial_wts[1,:,:],initial_wts[2,:,:])),axis=1)
        new_z_force = REPS_updated_mean[-2]
    elif control_type_z_axis == 'position':
        new_weights = np.expand_dims(np.vstack((initial_wts[0,:,:],initial_wts[1,:,:],REPS_updated_mean[0:7])),axis=1)
        new_z_force = REPS_updated_mean[-1] #0

    # Save new weights to dict
    data_dict = {
        'tau':           init_dmp_info_dict['tau'],
        'alpha':         init_dmp_info_dict['alpha'],
        'beta':          init_dmp_info_dict['beta'],
        'num_dims':      init_dmp_info_dict['num_dims'],
        'num_basis':     init_dmp_info_dict['num_basis'],
        'num_sensors':   init_dmp_info_dict['num_sensors'],
        'mu':            init_dmp_info_dict['mu'],
        'h':             init_dmp_info_dict['h'],
        'phi_j':         init_dmp_info_dict['phi_j'],
        'weights':       new_weights.tolist(),                
        }

    # save new sampled weights to pkl file
    weight_save_file = os.path.join(work_dir, 'meanWeightsUpdatedPol' + '.pkl')
    save_weights(weight_save_file, data_dict)

    # Calculate dmp trajectory             
    traject_time = dmp_traject_time   # define length of dmp trajectory  
    # Load dmp traject params
    dmp_traj = DMPPositionTrajectoryGenerator(traject_time)
    dmp_traj.load_saved_dmp_params_from_pkl_file(weight_save_file)
    dmp_traj.parse_dmp_params_dict()

    # Define starting position 
    start_pose = fa.get_pose()
    starting_rotation = start_pose.rotation
    y0 = start_pose.translation 
    # calculate dmp position trajectory - NOTE: this assumes a 0.001 dt for calc the dmp traject
    dmp_traject, dy, _, _, _ = dmp_traj.run_dmp_with_weights(y0) # y: np array(tx3)
    
    # check new dmp sampled wt trajectory vs original
    sample = 0
    plot_sampled_new_dmp_traject_and_original_dmp(epoch, sample, work_dir, new_z_force, traject_time, \
        position_dmp_weights_file_path, dmp_traject, y0)
    import pdb; pdb.set_trace()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--position_dmp_weights_file_path', '-w', type=str, default='/home/sony/raw_IL_trajects/Jan-2021/011321_piv_chop_potato_position_weights_zeroXY.pkl')#'/home/sony/raw_IL_trajects/100220_piv_chop_position_dmp_weights_zeroXY_2.pkl')
    parser.add_argument('--use_all_dmp_dims', type=bool, default = False)
    parser.add_argument('--control_type_z_axis', type=str, default = 'position', help='position or force')
    parser.add_argument('--dmp_traject_time', '-t', type=int, default = 6)  
    parser.add_argument('--dmp_wt_sampling_var', type=float, default = 0.01)
    parser.add_argument('--num_epochs', '-e', type=int, default = 5)  
    parser.add_argument('--num_samples', '-s', type=int, default = 25)    
    parser.add_argument('--data_savedir', '-d', type=str, default='/home/sony/Documents/cutting_RL_experiments/data/Jan-2021-LL-param-exps/pivchop/mozz/')
    parser.add_argument('--exp_num', '-n', type=int)
    parser.add_argument('--food_type', type=str, default='hard') #hard or soft
    parser.add_argument('--start_from_previous', '-sfp', type=bool, default=False)
    parser.add_argument('--previous_datadir', '-pd', type=str)
    parser.add_argument('--prev_epochs_to_calc_pol_update', '-num_prev_epochs', type=int, default = 1)
    parser.add_argument('--starting_epoch_num', '-start_epoch', type=int, default = 0)
    parser.add_argument('--starting_sample_num', '-start_sample', type=int, default = 0)
    args = parser.parse_args()

    # create folders to save data
    if not os.path.isdir(args.data_savedir + 'exp_' + str(args.exp_num)):
        createFolder(args.data_savedir + 'exp_' + str(args.exp_num))
    work_dir = args.data_savedir + 'exp_' + str(args.exp_num)
    if not os.path.isdir(work_dir + '/' + 'all_polParamRew_data'):
        createFolder(work_dir + '/' + 'all_polParamRew_data')
    if not os.path.isdir(work_dir + '/' + 'dmp_traject_plots'):
        createFolder(work_dir + '/' + 'dmp_traject_plots')    
    if not os.path.isdir(work_dir + '/' + 'dmp_wts'):
        createFolder(work_dir + '/' + 'dmp_wts')
    if not os.path.isdir(work_dir + '/' + 'forces_positions'):
        createFolder(work_dir + '/' + 'forces_positions')

    position_dmp_pkl = open(args.position_dmp_weights_file_path,"rb")
    init_dmp_info_dict = pickle.load(position_dmp_pkl)

    print('Starting robot')
    fa = FrankaArm()
    
    reset_joint_positions = [ 0.02846037, -0.51649966, -0.12048514, -2.86642333, -0.05060268,  2.30209197, 0.7744833 ]
    fa.goto_joints(reset_joint_positions)    


    # piv chop angle - 3D printed knife (20 deg) - TODO: sample this starting angle as well??
    # knife_orientation = np.array([[0.0,   0.9397,  -0.3420],
    #                               [ 1.0,   0.0,  0.0],
    #                               [ 0.0, -0.3420,  -0.9397]])

    # metal knife (26 deg tilt forward)
    knife_orientation = np.array([[0.0,   0.8988,  -0.4384],
                                  [ 1.0,   0.0,  0.0],
                                  [ 0.0, -0.4384,  -0.8988]])
    
    # go to initial cutting pose
    starting_position = RigidTransform(rotation=knife_orientation, \
        translation=np.array([0.454, 0.01, 0.145]), #z=0.05
        from_frame='franka_tool', to_frame='world')    
    fa.goto_pose(starting_position, duration=5, use_impedance=False)

    # move down to contact
    move_down_to_contact = RigidTransform(translation=np.array([0.0, 0.0, -0.1]),
    from_frame='world', to_frame='world')   
    # fa.goto_pose_delta(move_down_to_contact, duration=5, use_impedance=False, force_thresholds=[10.0, 10.0, 3.0, 10.0, 10.0, 10.0], ignore_virtual_walls=True)
    fa.goto_pose_delta(move_down_to_contact, duration=5, use_impedance=False, force_thresholds=[10.0, 10.0, 1.9, 10.0, 10.0, 10.0], ignore_virtual_walls=True)
    
    # Initialize Gaussian policy params (DMP weights) - mean and sigma
    if args.start_from_previous: # load previous data collected and start from updated policy and/or sample/epoch        
        prev_data_dir = args.previous_datadir
        if args.use_all_dmp_dims:
            policy_params_mean, policy_params_sigma = parse_policy_params_and_rews_from_file(prev_data_dir, args.prev_epochs_to_calc_pol_update, hfpc = False)
        else:
            policy_params_mean, policy_params_sigma = parse_policy_params_and_rews_from_file(prev_data_dir, args.prev_epochs_to_calc_pol_update)

        initial_mu, initial_sigma = policy_params_mean, policy_params_sigma
        mu, sigma = initial_mu, initial_sigma
        print('starting from updated policy - mean', policy_params_mean)
        initial_wts = np.array(init_dmp_info_dict['weights'])
        
        # if args.start_from_previous and args.start_epoch!=0 and args.start_sample==0:
        #     np.savez(os.path.join(work_dir, 'REPSupdatedMean_' + 'epoch_'+str(epoch) +'.npz'), \
        #         updated_mean = policy_params_mean, updated_cov = policy_params_sigma)

        if args.food_type == 'hard':
            S = [0,1,1,1,1,1]
        elif args.food_type == 'soft':
            S = [1,1,1,1,1,1]
        import pdb; pdb.set_trace()


    else: # start w/ initial DMP weights from IL
        initial_wts = np.array(init_dmp_info_dict['weights'])
        cart_pitch_stiffness_initial = 20 #50 #20 

        if args.use_all_dmp_dims: # use position control in all dims (use all dmp wt dims (x/y/z))
            initial_mu = initial_wts.flatten() 
            initial_sigma = np.diag(np.repeat(args.dmp_wt_sampling_var, initial_mu.shape[0]))

        else: # use only z wts and var impedance
            if args.control_type_z_axis == 'position': # no force control, only z axis position control + var pitch stiffness
                initial_mu = np.append(initial_wts[2,:,:], cart_pitch_stiffness_initial)  
                initial_sigma = np.diag(np.repeat(args.dmp_wt_sampling_var, initial_mu.shape[0]))
                initial_sigma[-1,-1] = 500 # change exploration variance for force parameter - TODO: increase

                if args.food_type == 'hard':
                    S = [0,1,1,1,1,1]
                elif args.food_type == 'soft':
                    S = [1,1,1,1,1,1]

            elif args.control_type_z_axis == 'force': # no position control, only z axis force control + var pitch stiffness
                f_initial = -10
                initial_mu = np.append(f_initial, cart_pitch_stiffness_initial)  
                initial_sigma = np.diag([120, 500])   
                S = [1,1,0,1,1,1]  
            
        print('initial mu', initial_mu)        
        mu, sigma = initial_mu, initial_sigma
    
    mean_params_each_epoch, cov_each_epoch = [], []
    # if we're starting from a later epoch: load previous data
    if os.path.isfile(os.path.join(work_dir, 'policy_mean_each_epoch.npy')):
        mean_params_each_epoch = np.load(os.path.join(work_dir, 'policy_mean_each_epoch.npy')).tolist()
        cov_each_epoch = np.load(os.path.join(work_dir, 'policy_cov_each_epoch.npy')).tolist()
    else:   
        mean_params_each_epoch.append(initial_mu)   
        cov_each_epoch.append(initial_sigma) 
    
    # track success metrics
    time_to_complete_cut, task_success = [], [] # task_success defined as 0 (unsuccessful cut), 1 (average cut), 2 (good cut)
    # track: cut through (0/1), cut through except for small tag (0/1), housing bumped into food/pushed out of gripper (0/1)
    task_success_more_granular = [] 
    # save reward features (for easier data post-processing)
    reward_features_all_samples = []

    if args.starting_sample_num !=0 or args.starting_epoch_num!=0:
        prev_data_time = np.load(os.path.join(work_dir, 'cut_times_all_samples.npy'))
        prev_data_task_succ = np.load(os.path.join(work_dir, 'task_success_all_samples.npy'))
        time_to_complete_cut = prev_data_time.tolist()
        task_success = prev_data_task_succ.tolist()
        
        # load previous sample granular task success and reward features
        if os.path.isfile(os.path.join(work_dir, 'task_success_more_granular_all_samples.npy')):
            prev_data_task_succ_more_granular = np.load(os.path.join(work_dir, 'task_success_more_granular_all_samples.npy'))
            task_success_more_granular = prev_data_task_succ_more_granular.tolist() 
        else:
            for i in range(len(task_success)):
                task_success_more_granular.append([np.inf, np.inf, np.inf])

        if os.path.isfile(os.path.join(work_dir, 'reward_features_all_samples.npy')):
            prev_data_reward_feats_all_samples = np.load(os.path.join(work_dir, 'reward_features_all_samples.npy'))        
            reward_features_all_samples = prev_data_reward_feats_all_samples.tolist()
        else: 
            for i in range(len(task_success)):
                reward_features_all_samples.append([np.inf]*7)
        import pdb; pdb.set_trace()

    for epoch in range(args.starting_epoch_num, args.num_epochs):
        policy_params_all_samples, rewards_all_samples =[], []
        for sample in range(args.starting_sample_num, args.num_samples):
            print('Epoch: %i Sample: %i'%(epoch,sample))
            # Sample new policy params from mu and sigma 
            new_params = np.random.multivariate_normal(mu, sigma)    
            if args.use_all_dmp_dims:
                new_weights = initial_wts #new_params.reshape(initial_wts.shape)
                new_pitch_stiffness = 'NA'
                                
            else:    
                if args.control_type_z_axis == 'position':
                    new_z_weights = new_params[0:-1]
                    new_cart_pitch_stiffness = new_params[-1]
                    # cap value
                    new_cart_pitch_stiffness = np.clip(new_cart_pitch_stiffness, 5, 600)           
                    new_params[-1] = int(new_cart_pitch_stiffness)  
                    print('clipped sampled new_cart_pitch_stiffness', new_cart_pitch_stiffness)  

                elif args.control_type_z_axis == 'force':  
                    new_z_weights = initial_wts[2,:,:] # not actually using this in this case b/c no position control in z
                    new_z_force = new_params[0]
                    new_cart_pitch_stiffness = new_params[1]
                    # cap force value
                    new_z_force = np.clip(new_z_force, -40, -3)    # TODO: up force to -50N        
                    new_params[0] = int(new_z_force)  
                    print('clipped sampled z force', new_z_force)  
                    new_cart_pitch_stiffness = np.clip(new_cart_pitch_stiffness, 5, 600)           
                    new_params[1] = int(new_cart_pitch_stiffness)  
                    print('clipped sampled new_cart_pitch_stiffness', new_cart_pitch_stiffness)  

            # save to policy params buffer
            policy_params_all_samples.append(new_params.tolist())
            
            # concat new sampled x weights w/ old y (zero's) and z weights if we're only sampling x weights
            if not args.use_all_dmp_dims: 
                new_weights = np.expand_dims(np.vstack((initial_wts[0,:,:],initial_wts[1,:,:], new_z_weights)),axis=1)
            
            # Save new weights to dict
            data_dict = {
                'tau':           init_dmp_info_dict['tau'],
                'alpha':         init_dmp_info_dict['alpha'],
                'beta':          init_dmp_info_dict['beta'],
                'num_dims':      init_dmp_info_dict['num_dims'],
                'num_basis':     init_dmp_info_dict['num_basis'],
                'num_sensors':   init_dmp_info_dict['num_sensors'],
                'mu':            init_dmp_info_dict['mu'],
                'h':             init_dmp_info_dict['h'],
                'phi_j':         init_dmp_info_dict['phi_j'],
                'weights':       new_weights.tolist(),                
                }

            # save new sampled weights to pkl file
            weight_save_file = os.path.join(work_dir + '/' + 'dmp_wts', 'weights_' + 'epoch_'+str(epoch) + '_ep_'+str(sample)+ '.pkl')
            save_weights(weight_save_file, data_dict)

            # Calculate dmp trajectory             
            traject_time = args.dmp_traject_time   # define length of dmp trajectory  
            # Load dmp traject params
            dmp_traj = DMPPositionTrajectoryGenerator(traject_time)
            dmp_traj.load_saved_dmp_params_from_pkl_file(weight_save_file)
            dmp_traj.parse_dmp_params_dict()

            # Define starting position 
            start_pose = fa.get_pose()
            starting_rotation = start_pose.rotation
            y0 = start_pose.translation 
            # calculate dmp position trajectory - NOTE: this assumes a 0.001 dt for calc the dmp traject
            dmp_traject, dy, _, _, _ = dmp_traj.run_dmp_with_weights(y0) # y: np array(tx3)
            
            # check new dmp sampled wt trajectory vs original
            plot_sampled_new_dmp_traject_and_original_dmp(epoch, sample, work_dir, new_cart_pitch_stiffness, traject_time, \
                args.position_dmp_weights_file_path, dmp_traject, y0)
            # import pdb; pdb.set_trace()

            # sampling info for sending msgs via ROS
            dt = 0.01 
            T = traject_time
            ts = np.arange(0, T, dt)
            N = len(ts)

            # downsample dmp traject 
            downsmpled_dmp_traject = downsample_dmp_traject(dmp_traject, 0.001, dt)
            target_poses = get_dmp_traj_poses_reformatted(downsmpled_dmp_traject, starting_rotation) # target_poses is a nx16 list of target poses at each time step


            if args.control_type_z_axis == 'position':
                #S = [1, 1, 1, 1, 1, 1] # position control in all axes
                target_force = [0, 0, 0, 0, 0, 0]

            elif args.control_type_z_axis == 'force': 
                #S = [1, 1, 0, 1, 1, 1] # force control in z, zero-position control in x/y
                target_force = [0, 0, new_z_force, 0, 0, 0]

            position_kps_cart = FC.DEFAULT_TRANSLATIONAL_STIFFNESSES + FC.DEFAULT_ROTATIONAL_STIFFNESSES
            # set pitch axis cartesian gain to be sampled value
            position_kps_cart[-2] = new_cart_pitch_stiffness
            force_kps_cart = [0.1] * 6   

            rospy.loginfo('Initializing Sensor Publisher')
            pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1)
            rate = rospy.Rate(1 / dt)
            n_times = 1
            rospy.loginfo('Publishing HFPC trajectory w/ cartesian gains...')
            
            current_ht = fa.get_pose().translation[2]
            dmp_num = 0 

            # sample from gaussian to get dmp weights for this execution            
            dmp_num = 0            
            peak_z_forces_all_dmps, x_mvmt_all_dmps, forw_vs_back_x_mvmt_all_dmps, diff_up_down_z_mvmt_all_dmps = [], [], [], []# sum of abs (+x/-x mvmt)  
            y_mvmt_all_dmps, peak_y_force_all_dmps, z_mvmt_all_dmps, upward_z_penalty_all_dmps = [], [], [], []
            total_cut_time_all_dmps = 0          
            while current_ht > 0.023:   
                fa.run_dynamic_force_position(duration=T *100, buffer_time = 3, 
                                        force_thresholds = [60.0, 60.0, 60.0, 30.0, 30.0, 30.0],
                                        S=S, use_cartesian_gains=True,
                                        position_kps_cart=position_kps_cart,
                                        force_kps_cart=force_kps_cart, block=False)
                print('starting dmp', dmp_num) 
                robot_positions = np.zeros((0,3))
                robot_forces = np.zeros((0,6))       
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
                    
                    robot_positions = np.vstack((robot_positions, fa.get_pose().translation.reshape(1,3)))
                    robot_forces = np.vstack((robot_forces, fa.get_ee_force_torque().reshape(1,6)))
                    
                    pub.publish(ros_msg)
                    rate.sleep()

                #fa.stop_skill()
                term_proto_msg = ShouldTerminateSensorMessage(timestamp=rospy.Time.now().to_time() - init_time, should_terminate=True)
                ros_msg = make_sensor_group_msg(
                    termination_handler_sensor_msg=sensor_proto2ros_msg(
                        term_proto_msg, SensorDataMessageType.SHOULD_TERMINATE)
                    )
                pub.publish(ros_msg)

                # calc stats from dmp
                cut_time = rospy.Time.now().to_time() - init_time
                peak_z_force = np.max(np.abs(robot_forces[:,2]))
                if (robot_positions[-1,2]-robot_positions[0,2]) > 0.02:
                    upward_z_penalty = (robot_positions[-1,2]-robot_positions[0,2])
                else:
                    upward_z_penalty = 0          

                up_z_mvmt = np.abs(robot_positions[-1,2]) - np.min(np.abs(robot_positions[:,2])) 
                down_z_mvmt = np.abs(robot_positions[0,2]) - np.min(np.abs(robot_positions[:,2]))
                total_z_mvmt = up_z_mvmt + down_z_mvmt
                diff_up_down_z_mvmt = np.abs(up_z_mvmt - down_z_mvmt)

                # # data only for use in 3-dim xyz position dmp reward
                forward_x_mvmt = (np.max(np.abs(robot_positions[:,0]) - np.abs(robot_positions[0,0])))
                backward_x_mvmt = (np.max(np.abs(robot_positions[:,0]) - np.abs(robot_positions[-1,0])))
                total_x_mvmt = forward_x_mvmt + backward_x_mvmt
                #diff_forw_back_x_mvmt = np.abs(forward_x_mvmt - backward_x_mvmt)
                forward_y_mvmt = (np.max(np.abs(robot_positions[:,1]) - np.abs(robot_positions[0,1])))
                backward_y_mvmt = (np.max(np.abs(robot_positions[:,1]) - np.abs(robot_positions[-1,1])))
                total_y_mvmt = forward_y_mvmt + backward_y_mvmt
                peak_y_force = np.max(np.abs(robot_forces[:,1]))

                # save to buffers 
                total_cut_time_all_dmps += cut_time
                peak_z_forces_all_dmps.append(peak_z_force)
                z_mvmt_all_dmps.append(total_z_mvmt)
                upward_z_penalty_all_dmps.append(upward_z_penalty)
                diff_up_down_z_mvmt_all_dmps.append(diff_up_down_z_mvmt)

                x_mvmt_all_dmps.append(total_x_mvmt)
                #forw_vs_back_x_mvmt_all_dmps.append(diff_forw_back_x_mvmt)
                y_mvmt_all_dmps.append(total_y_mvmt)
                peak_y_force_all_dmps.append(peak_y_force)
                #diff_up_down_z_mvmt_all_dmps.append(diff_up_down_z_mvmt)

                np.savez(work_dir + '/' + 'forces_positions/' + 'epoch_'+str(epoch) + '_ep_'+str(sample) + '_trial_info_'+str(dmp_num)+'.npz', robot_positions=robot_positions, \
                    robot_forces=robot_forces)
                
                completed_cut = input('cut complete? (0-n, 1-y, 2-cannot complete): ')

                while completed_cut not in ['0', '1', '2']:
                    completed_cut = input('please enter valid answer. cut complete? (0/1/2): ') 

                if completed_cut == '1': 
                    break

                elif completed_cut == '2': 
                    # if cut can't be completed, give very high penalty for time 
                    total_cut_time_all_dmps = 200
                    break

                elif completed_cut == '0':
                    current_ht = fa.get_pose().translation[2]
                    print('current_ht', current_ht)
                    dmp_num += 1  

                    # calculate new dmp traject based on current position
                    y0 = fa.get_pose().translation
                    # calculate dmp position trajectory - NOTE: this assumes a 0.001 dt for calc the dmp traject
                    dmp_traject, dy, _, _, _ = dmp_traj.run_dmp_with_weights(y0) # y: np array(tx3)
                    # downsample dmp traject and reformat target_poses
                    downsmpled_dmp_traject = downsample_dmp_traject(dmp_traject, 0.001, dt)
                    target_poses = get_dmp_traj_poses_reformatted(downsmpled_dmp_traject, fa.get_pose().rotation) # target_poses is a nx16 list of target poses at each time step
                
            
            # After finishing set of dmps for a full slice - calculate avg reward here            
            time.sleep(1.5) # pause to let skill fully stop

            # track success metrics
            time_to_complete_cut.append(total_cut_time_all_dmps) 
            if completed_cut == '2':
                success = 0
            else:
                success = input('enter task success - 1 (average cut) or 2 (good cut): ')
                while success not in ['1', '2']:
                    success = input('enter task success - 1 (average cut) or 2 (good cut): ')
            task_success.append(int(success))

            # track more granular success metrics
            detailed_success = input('enter: cut through? (0/1), cut through w/ small tag (0/1)?, housing bumped into food/pushed out of gripper? (0/1): ')
            while detailed_success not in ['000','001','010','100','110','101','011','111']:
                detailed_success = input('enter: cut through? (0/1), cut through w/ small tag (0/1)?, housing bumped into food/pushed out of gripper? (0/1): ')
            split_str = [int(char) for char in detailed_success]
            task_success_more_granular.append(split_str) 

            # calc averages/max across all cut types - NOTE: switched to max instead of avg to handle dmps that vary as they are chained
            avg_peak_y_force = np.max(peak_y_force_all_dmps)
            avg_peak_z_force = np.max(peak_z_forces_all_dmps) #np.mean(peak_forces_all_dmps)
            avg_x_mvmt = np.max(x_mvmt_all_dmps)
            avg_y_mvmt = np.max(y_mvmt_all_dmps)
            avg_z_mvmt = np.max(z_mvmt_all_dmps)
            #avg_diff_up_down_z_mvmt = np.max(diff_up_down_z_mvmt_all_dmps) #np.mean(diff_up_down_z_mvmt_all_dmps)
            avg_upward_z_penalty = np.max(upward_z_penalty_all_dmps)
            # import pdb; pdb.set_trace() 
                   
            if args.use_all_dmp_dims:  
                # reward = -0.05*avg_peak_force -0.1*avg_peak_y_force - 10*avg_x_mvmt -20*avg_y_mvmt -50*avg_upward_z_mvmt - 50*avg_diff_forw_back_x_mvmt + -0.2*total_cut_time_all_dmps
                reward = -0.05*avg_peak_z_force -0.1*avg_peak_y_force - 10*avg_x_mvmt -100*avg_y_mvmt -100*avg_upward_z_mvmt \
                    - 50*avg_diff_forw_back_x_mvmt + -0.2*total_cut_time_all_dmps

            else:
                # pivchop-specific reward:
                #reward = -0.15*avg_peak_force - 10*avg_z_mvmt - 200*avg_diff_up_down_z_mvmt - avg_upward_z_penalty -0.2*total_cut_time_all_dmps
                
                # trying out more generalized cutting reward function - remove not returning to start penalty:
                reward = -0.1*avg_peak_y_force -0.15*avg_peak_z_force - 10*avg_x_mvmt -100*avg_y_mvmt - 10*avg_z_mvmt \
                    -100*avg_upward_z_penalty -0.2*total_cut_time_all_dmps # NOTE: in normal cut, -100*avg_upward_z_penalty term = -100*avg_upward_z_mvmt (calc diff for each cut type)

            # save reward to buffer
            print('Epoch: %i Sample: %i Reward: '%(epoch,sample), reward)
            rewards_all_samples.append(reward)
            reward_features = [avg_peak_y_force, avg_peak_z_force, avg_x_mvmt, avg_y_mvmt, avg_z_mvmt, avg_upward_z_penalty, total_cut_time_all_dmps]
            reward_features_all_samples.append(reward_features)
            import pdb; pdb.set_trace()

            #import pdb; pdb.set_trace()
            # save intermediate rewards/pol params 
            if args.starting_sample_num !=0:
                prev_sample_data = np.load(os.path.join(work_dir + '/' + 'all_polParamRew_data', 'polParamsRews_' + 'epoch_'+str(epoch) + '_ep_'+str(args.starting_sample_num-1) + '.npy'))
                new_sample_data = np.concatenate((np.array(policy_params_all_samples), np.array([rewards_all_samples]).T), axis=1)
                combined_data = np.concatenate((prev_sample_data, new_sample_data), axis=0)
                np.save(os.path.join(work_dir + '/' + 'all_polParamRew_data', 'polParamsRews_' + 'epoch_'+str(epoch) + '_ep_'+str(sample) + '.npy'),
                    combined_data)

            else:
                np.save(os.path.join(work_dir + '/' + 'all_polParamRew_data', 'polParamsRews_' + 'epoch_'+str(epoch) + '_ep_'+str(sample) + '.npy'), \
                    np.concatenate((np.array(policy_params_all_samples), np.array([rewards_all_samples]).T), axis=1))
            
            # save task success metrics           
            np.save(os.path.join(work_dir, 'cut_times_all_samples.npy'), np.array(time_to_complete_cut))
            np.save(os.path.join(work_dir, 'task_success_all_samples.npy'), np.array(task_success)) 
            np.save(os.path.join(work_dir, 'task_success_more_granular_all_samples.npy'), np.array(task_success_more_granular))
            # save reward features each samples
            np.save(os.path.join(work_dir, 'reward_features_all_samples.npy'), np.array(reward_features_all_samples))
                        
            # reset to starting cut position            
            new_position = copy.deepcopy(starting_position)
            new_position.translation[1] = fa.get_pose().translation[1]
            fa.goto_pose(new_position, duration=5, use_impedance=False)

            # move over a bit (y dir)          
            y_shift = 0.004 #float(input('enter how far to shift in y dir (m): '))
            move_over_slice_thickness = RigidTransform(translation=np.array([0.0, y_shift, 0.0]),
                from_frame='world', to_frame='world') 
            # move_over_slice_thickness = RigidTransform(translation=np.array([0.0, 0.005, 0.0]),
            #     from_frame='world', to_frame='world') 
            fa.goto_pose_delta(move_over_slice_thickness, duration=3, use_impedance=False)

            # y_shift = float(input('enter how far to shift in y dir (m): '))
            # move_over_slice_thickness = RigidTransform(translation=np.array([0.0, y_shift, 0.0]),
            #     from_frame='world', to_frame='world') 
            # fa.goto_pose_delta(move_over_slice_thickness, duration=3, use_impedance=False)
            import pdb; pdb.set_trace()

            # move down to contact
            move_down_to_contact = RigidTransform(translation=np.array([0.0, 0.0, -0.1]),
            from_frame='world', to_frame='world')   
            #fa.goto_pose_delta(move_down_to_contact, duration=5, use_impedance=False, force_thresholds=[10.0, 10.0, 3.0, 10.0, 10.0, 10.0], ignore_virtual_walls=True)
            fa.goto_pose_delta(move_down_to_contact, duration=5, use_impedance=False, force_thresholds=[10.0, 10.0, 1.9, 10.0, 10.0, 10.0], ignore_virtual_walls=True)
        
        # save reward 
        import pdb; pdb.set_trace()
        np.save(os.path.join(work_dir, 'polParamsRews_' + 'epoch_'+str(epoch) +'.npy'), \
            np.concatenate((np.array(policy_params_all_samples), np.array([rewards_all_samples]).T), axis=1))

        # update policy mean and cov (REPS) 
        if args.starting_sample_num !=0:
            all_data = work_dir + '/all_polParamRew_data/' +'polParamsRews_epoch_' + str(epoch) + '_ep_'+ str(sample) + '.npy'
            policy_params_all_samples = (np.load(all_data)[:,0:-1]).tolist()
            rewards_all_samples = (np.load(all_data)[:,-1]).tolist()

        reps_agent = reps.Reps(rel_entropy_bound=1.5,min_temperature=0.001) #Create REPS object
        policy_params_mean, policy_params_sigma, reps_info = \
            reps_agent.policy_from_samples_and_rewards(policy_params_all_samples, rewards_all_samples)
        
        print('updated policy params mean')
        print(policy_params_mean)
        print('updated policy cov')
        print(policy_params_sigma)

        mu, sigma = policy_params_mean, policy_params_sigma
        mean_params_each_epoch.append(policy_params_mean)
        cov_each_epoch.append(policy_params_sigma)
        np.save(os.path.join(work_dir, 'policy_mean_each_epoch.npy'),np.array(mean_params_each_epoch))
        np.save(os.path.join(work_dir, 'policy_cov_each_epoch.npy'),np.array(cov_each_epoch))

        import pdb; pdb.set_trace()
        # TODO: roll out updated policy mean and evaluate
        # plot updated policy mean trajectory to visualize
        plot_updated_policy_mean_traject(work_dir, args.position_dmp_weights_file_path, epoch, args.dmp_traject_time, args.control_type_z_axis, init_dmp_info_dict,\
            initial_wts,policy_params_mean)
        import pdb; pdb.set_trace()

        # save new policy params mean and cov   
        np.savez(os.path.join(work_dir, 'REPSupdatedMean_' + 'epoch_'+str(epoch) +'.npz'), \
            updated_mean = policy_params_mean, updated_cov = policy_params_sigma)

        # after epoch is complete, reset start_sample to 0
        args.starting_sample_num = 0
        plt.figure()
        plt.plot(time_to_complete_cut)
        plt.title('epoch %i time to complete cut'%epoch)
        plt.xlabel('samples')
        plt.ylabel('time to complete cut')
        plt.figure()
        plt.plot(task_success)
        plt.title('epoch %i task_success'%epoch)
        plt.xlabel('samples')
        plt.ylabel('task_success')
        plt.show()


    fa.goto_joints(reset_joint_positions)

    