'''
UPDATES: 
Added iterative updates (1 sample at a time for GP reward model --> compute EPD --> update
model w/ sample w/ highest EPD, until EPD is below threshold)

updating policy after 0th epoch w/ GP reward model rewards after GP reward model is trained!

Workflow:
- initialize GP reward model
- initialize policy
- for epoch in num_epochs:
    -for sample in num_samples:
        - sample policy params
        - get/save features from sample 
        - query human for reward (save as expert reward)
        - get/save task success metrics like before
        - get reward from GP reward model based on current reward model (GP model not trained yet if epoch =0, so need to first train)
        - save expected GP model reward mean and var to buffers and training data buffer
    - if epoch == 0:
        - initialize GP reward model w/ training data from samples in 1st epoch 
    - elif epoch!=0:
        - update policy w/ REPS
        - compute EPD for each sample (using kl_div) and get samples_to_query and queried_outcomes
        - query expert for rewards if samples_to_query!=[]
        - save all queried outcomes and queried rewards in buffer to send to GPytorch model as training data everytime it get updated
        - Add samples to query to running list of queried_samples, Keep track of number of queried samples 
        - Update Reward model GP if there are any outcomes to query

UPDATES: 
-- refactored lower level parameter experiment scripts (prev separate script for each cut type) to support all 3 cut types in 1 script
-- added in option to standardize reward features before inputting to GP model

NOTES:
training_data_list = n_samples x 4 np array: [outcomes_from_sample/reward_features (7,), mean_expected_reward[0] (scalar), var_expected_reward[0] (scalar), sampled_policy_params (8,)]

polParamsRews_epoch_ep saved as: [polParams, analytical_reward, GP_reward_model_mean, expert_reward (slow, fast)]

outcomes/reward features: [avg_peak_y_force, avg_peak_z_force, avg_x_mvmt, avg_y_mvmt, avg_z_mvmt, avg_upward_z_penalty, total_cut_time_all_dmps]

ex. CL args: 
python -m pdb run_learn_LL_params_w_ARL_v2.py --exp_num 3 --food_type soft --food_name tomato --cut_type scoring --start_from_previous True --previous_datadir /home/sony/Documents/cutting_RL_experiments/data/Jan-2021-HIL-ARL-exps/scoring/tomato/exp_3/ --starting_epoch_num 3 --num_samples 20 --desired_cutting_behavior quality_cut --num_GP_training_samples 19

'''
import argparse

import torch
import gpytorch
from matplotlib import cm
import os
import sys
sys.path.append('/home/sony/Documents/frankapy/examples/cutting_RL/spring-2021-scripts/HIL_ARL_exp/') # TODO: update this is using different path

from reward_learner import RewardLearner
from policy_learner import REPSPolicyLearner
from gp_regression_model import GPRegressionModel

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_all_dmp_dims', type=bool, default = False)
    parser.add_argument('--dmp_traject_time', '-t', type=int, default = 5)  
    parser.add_argument('--dmp_wt_sampling_var', type=float, default = 0.01)
    parser.add_argument('--num_epochs', '-e', type=int, default = 5)  
    parser.add_argument('--num_samples', '-s', type=int, default = 25)    
    parser.add_argument('--data_savedir', '-d', type=str, default='/home/sony/Documents/cutting_RL_experiments/data/Jan-2021-HIL-ARL-exps/')
    parser.add_argument('--exp_num', '-n', type=int)
    parser.add_argument('--food_type', type=str, default='hard') #hard or soft
    parser.add_argument('--food_name', type=str, help = 'potato, carrot, celery, tomator, banana, mozz') #potato, carrot, celery, tomato, banana, mozz
    parser.add_argument('--cut_type', type=str, default = 'normal', help ='options: normal, pivchop, scoring') # options: 'normal', 'pivchop', 'scoring'
    parser.add_argument('--start_from_previous', '-sfp', type=bool, default=False)
    parser.add_argument('--previous_datadir', '-pd', type=str)
    parser.add_argument('--prev_epochs_to_calc_pol_update', '-num_prev_epochs', type=int, default = 1, help='num \
        previous epochs of data to use to calculate REPS policy update')
    parser.add_argument('--starting_epoch_num', '-start_epoch', type=int, default = 0)
    parser.add_argument('--starting_sample_num', '-start_sample', type=int, default = 0)
    parser.add_argument('--debug', type=bool, default=False)
    
    # GP reward model-related args
    parser.add_argument('--kappa', type=float, default = 0.1) 
    parser.add_argument('--rel_entropy_bound', type=float, default = 1.5)
    parser.add_argument('--num_EPD_epochs', type=int, default = 5)
    parser.add_argument('--GP_training_epochs_initial', type=int, default = 1)
    parser.add_argument('--GP_training_epochs_later', type=int, default = 0)
    parser.add_argument('--desired_cutting_behavior', type=str, help='options: fast, slow, quality_cut') # fast, slow, quality_cut
    parser.add_argument('--standardize_reward_feats', type=bool, default = True)
    parser.add_argument('--scale_pol_params_for_KLD', type=bool, default = True)
    parser.add_argument('--add_ridge_to_pol_cov_for_KLD', type=bool, default = False)
    parser.add_argument('--sampl_or_weight_kld_calc', type=str, default = 'weight', help = 'sampling or weight')

    parser.add_argument('--num_GP_training_samples', type=int, default = 1, help = 'number of samples to use when training initial GP reward model')
    parser.add_argument('--GPsignal_var_initial', type=int, default = 3) 
    parser.add_argument('--init_GP_lengthscale_hyperparam_from_prior', type=bool, default = True)
    args = parser.parse_args()

    kappa = args.kappa   
    initial_var = args.dmp_wt_sampling_var   
    num_episodes_initial_epoch = args.num_samples     
    num_episodes_later_epochs = args.num_samples
    rel_entropy_bound = args.rel_entropy_bound
    num_EPD_epochs = args.num_EPD_epochs
    GP_training_epochs_initial = args.GP_training_epochs_initial
    GP_training_epochs_later = args.GP_training_epochs_later    

    if args.desired_cutting_behavior == 'fast' or args.desired_cutting_behavior == 'slow':
        num_expert_rews_each_sample = 2
    elif args.desired_cutting_behavior == 'quality_cut':
        num_expert_rews_each_sample = 1
        
    # create folders to save data
    if not os.path.isdir(args.data_savedir + args.cut_type + '/' + args.food_name + '/'):
        createFolder(args.data_savedir + args.cut_type + '/' + args.food_name + '/')
    args.data_savedir = args.data_savedir + args.cut_type + '/' + args.food_name + '/'
    if not os.path.isdir(args.data_savedir + 'exp_' + str(args.exp_num)):
        createFolder(args.data_savedir + 'exp_' + str(args.exp_num))

    work_dir = args.data_savedir + 'exp_' + str(args.exp_num)
    
    if not os.path.isdir(work_dir + '/' + 'all_polParamRew_data'):
        createFolder(work_dir + '/' + 'all_polParamRew_data')
    if not os.path.isdir(work_dir + '/' + 'GP_reward_model_data'):
        createFolder(work_dir + '/' + 'GP_reward_model_data')
    if not os.path.isdir(work_dir + '/' + 'GP_reward_model_data' + '/' + 'GP_cov_mat'):
        createFolder(work_dir + '/' + 'GP_reward_model_data' + '/' + 'GP_cov_mat')
    if not os.path.isdir(work_dir + '/' + 'GP_reward_model_data' + '/' + 'policy_pi_star_tilda_data'):
        createFolder(work_dir + '/' + 'GP_reward_model_data' + '/' + 'policy_pi_star_tilda_data')
    if not os.path.isdir(work_dir + '/' + 'dmp_traject_plots'):
        createFolder(work_dir + '/' + 'dmp_traject_plots')    
    if not os.path.isdir(work_dir + '/' + 'dmp_wts'):
        createFolder(work_dir + '/' + 'dmp_wts')
    if not os.path.isdir(work_dir + '/' + 'forces_positions'):
        createFolder(work_dir + '/' + 'forces_positions')
    
    # Instantiate Policy Learner (agent)
    agent = REPSPolicyLearner(num_expert_rews_each_sample, args.cut_type, args.food_type, args.dmp_wt_sampling_var, args.start_from_previous, args.previous_datadir,\
        args.prev_epochs_to_calc_pol_update, args.starting_epoch_num, args.dmp_traject_time)

    # Instantiate reward learner - note: GPR model not instantiated yet
    reward_learner = RewardLearner(work_dir, kappa, args.cut_type, args.food_type, args.desired_cutting_behavior)
    reward_learner.scale_pol_params = args.scale_pol_params_for_KLD
    reward_learner.add_ridge_to_pol_cov = args.add_ridge_to_pol_cov_for_KLD
    reward_learner.sampl_or_weight_kld_calc = args.sampl_or_weight_kld_calc
    reward_learner.n_GP_training_samples = args.num_GP_training_samples
    beta = 0.001 # fixed gaussian noise likelihood
 
    # load dmp weights and starting knife orientation
    dmp_wts_file, knife_orientation = load_dmp_wts_and_knife_orientation(args.cut_type)
    
    position_dmp_pkl = open(dmp_wts_file,"rb")
    init_dmp_info_dict = pickle.load(position_dmp_pkl)

    print('Starting robot')
    fa = FrankaArm()
    
    if not args.debug:
        reset_joint_positions = [ 0.02846037, -0.51649966, -0.12048514, -2.86642333, -0.05060268,  2.30209197, 0.7744833 ]
        fa.goto_joints(reset_joint_positions)    

    # go to initial cutting pose
    starting_position = RigidTransform(rotation=knife_orientation, \
        translation=np.array([0.507, 0.127, 0.144]), #z=0.05
        from_frame='franka_tool', to_frame='world')    
    fa.goto_pose(starting_position, duration = 5, use_impedance=False)

    # move down to contact
    if not args.debug:
        move_down_to_contact = RigidTransform(translation=np.array([0.0, 0.0, -0.1]),
        from_frame='world', to_frame='world')   
        if args.food_name == 'banana': 
            fa.goto_pose_delta(move_down_to_contact, duration=5, use_impedance=False, force_thresholds=[10.0, 10.0, 2.0, 10.0, 10.0, 10.0], ignore_virtual_walls=True)
        else:
            fa.goto_pose_delta(move_down_to_contact, duration=5, use_impedance=False, force_thresholds=[10.0, 10.0, 3.0, 10.0, 10.0, 10.0], ignore_virtual_walls=True)    


    mean_params_each_epoch, cov_each_epoch = [], []
        
    # Buffers for GP reward model data
    total_queried_samples_each_epoch, mean_reward_model_rewards_all_epochs = [], [] #track number of queries to expert for rewards and average rewards for each epoch
    training_data_list, queried_samples_all = [], []
    GP_training_data_x_all = np.empty([0,7])
    GP_training_data_y_all, GP_training_data_y_all_slow, GP_training_data_y_all_fast =  np.empty([0]), np.empty([0]), np.empty([0]) 
    queried_outcomes, queried_expert_rewards, policy_params_all_epochs, block_poses_all_epochs = [], [], [], []
    reward_model_rewards_all_mean_buffer, expert_rewards_all_epochs = [], [] # TODO: update expert_rewards_all_epochs to load previous data 
    total_samples = 0
    
    # Track success metrics
    '''
    task_success: 0 (unsuccessful cut), 1 (average cut), 2 (good cut)
    task_success_more_granular: cut through (0/1), cut through except for small tag (0/1), housing bumped into food/pushed out of gripper (0/1)
    '''
    time_to_complete_cut, task_success, task_success_more_granular = [], [], []    
    # save reward features (for easier data post-processing)
    reward_features_all_samples = []
    # load previous data in starting from later sample/epoch
    if args.starting_sample_num !=0 or args.starting_epoch_num!=0:
        reward_features_all_samples, time_to_complete_cut, task_success, task_success_more_granular = load_prev_task_success_data(work_dir)        
        
        # load prev GP training data list
        prev_GP_training_data_list = np.load(work_dir + '/' + 'GP_reward_model_data/' + 'training_data_list.npy', allow_pickle=True)
        training_data_list = prev_GP_training_data_list.tolist()

        prev_expert_rewards_all_epochs = np.load(work_dir + '/' + 'GP_reward_model_data/' + 'expert_rewards_all_epochs.npy')     
        expert_rewards_all_epochs = prev_expert_rewards_all_epochs.tolist()

        # load prev queried_samples_all
        if args.desired_cutting_behavior == 'slow' and args.starting_epoch_num!=0:            
            prev_queried_samples_all = np.load(work_dir + '/' + 'GP_reward_model_data/' + 'queried_samples_all_slowCut_epoch_'+str(args.starting_epoch_num-1)+'.npy')            
            queried_samples_all = prev_queried_samples_all.tolist()

            prev_total_queried_samples_each_epoch = np.load(work_dir + '/' 'GP_reward_model_data/' + 'total_queried_samples_each_epoch_slowCut.npy')
            total_queried_samples_each_epoch = prev_total_queried_samples_each_epoch.tolist()    

        elif args.desired_cutting_behavior == 'fast' and args.starting_epoch_num!=0:            
            prev_queried_samples_all = np.load(work_dir + '/' + 'GP_reward_model_data/' + 'queried_samples_all_fastCut_epoch_'+str(args.starting_epoch_num-1)+'.npy')
            queried_samples_all = prev_queried_samples_all.tolist()

            prev_total_queried_samples_each_epoch = np.load(work_dir + '/' 'GP_reward_model_data/' + 'total_queried_samples_each_epoch_fastCut.npy')
            total_queried_samples_each_epoch = prev_total_queried_samples_each_epoch.tolist()    

        elif args.desired_cutting_behavior == 'quality_cut' and args.starting_epoch_num!=0:    
            prev_queried_samples_all = np.load(work_dir + '/' + 'GP_reward_model_data/' + 'queried_samples_all_qualityCut_epoch_'+str(args.starting_epoch_num-1)+'.npy')
            queried_samples_all = prev_queried_samples_all.tolist()

            prev_total_queried_samples_each_epoch = np.load(work_dir + '/' 'GP_reward_model_data/' + 'total_queried_samples_each_epoch_qualityCut.npy')
            total_queried_samples_each_epoch = prev_total_queried_samples_each_epoch.tolist()    
        
        import pdb; pdb.set_trace()
    
    # if not starting from 0-th epoch - need to load/train GP reward model based on previous data
    if args.starting_epoch_num > 0:
        print('training GP reward model from previously saved data')
        import pdb; pdb.set_trace()                
        if args.desired_cutting_behavior == 'slow':
            prev_GP_training_data = np.load(work_dir + '/' 'GP_reward_model_data/' + 'GP_reward_model_training_data_slowCut_epoch_' +str(args.starting_epoch_num-1) + '.npz')
            GP_training_data_x_all = prev_GP_training_data['GP_training_data_x_all']
            GP_training_data_y_all = prev_GP_training_data['GP_training_data_y_all']
            GP_training_data_y_all_slow = GP_training_data_y_all            
            import pdb; pdb.set_trace()   

        elif args.desired_cutting_behavior == 'fast': 
            prev_GP_training_data = np.load(work_dir + '/' 'GP_reward_model_data/' + 'GP_reward_model_training_data_fastCut_epoch_' +str(args.starting_epoch_num-1) + '.npz')
            GP_training_data_x_all = prev_GP_training_data['GP_training_data_x_all']
            GP_training_data_y_all = prev_GP_training_data['GP_training_data_y_all']
            GP_training_data_y_all_fast = GP_training_data_y_all            
            import pdb; pdb.set_trace()   
        
        else:
            prev_GP_training_data = np.load(work_dir + '/' 'GP_reward_model_data/' + 'GP_reward_model_training_data_qualityCut_epoch_' +str(args.starting_epoch_num-1) + '.npz')
            GP_training_data_x_all = prev_GP_training_data['GP_training_data_x_all']
            GP_training_data_y_all = prev_GP_training_data['GP_training_data_y_all']            
            import pdb; pdb.set_trace()   

        # train with previous data
        print('initializing and training GP reward model from previous data')
        likelihood = gpytorch.likelihoods.GaussianLikelihood()      
        # import pdb; pdb.set_trace()      
        train_x = torch.from_numpy(GP_training_data_x_all)
        train_x = train_x.float()
        train_y = torch.from_numpy(GP_training_data_y_all)
        train_y = train_y.float()
        print('train_y variance', train_y.var())
        # add white noise 
        gpr_reward_model = GPRegressionModel(train_x, train_y, likelihood) 

        # Set prior on GP model hyperparams
        gpr_reward_model.covar_module.outputscale = args.GPsignal_var_initial
        # adding to initialize hyperparams based on prior data!
        if args.init_GP_lengthscale_hyperparam_from_prior:
            gpr_reward_model.covar_module.base_kernel.lengthscale = torch.from_numpy(np.array([[5.1391, 5.0817, 5.1031, 5.1450, 5.1920, 5.1543, 2.8334]]))
        import pdb; pdb.set_trace()

        optimizer = torch.optim.Adam([                
            {'params': gpr_reward_model.covar_module.parameters()},
            {'params': gpr_reward_model.mean_module.parameters()},
            {'params': gpr_reward_model.likelihood.parameters()},
        ], lr=0.01) # lr = 0.01 originally 
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gpr_reward_model)
        
        # train reward GP model given initial train_x and train_y data (voxel_data, queried human rewards)
        gpr_reward_model = reward_learner.train_GPmodel(work_dir, GP_training_epochs_initial, optimizer, gpr_reward_model, likelihood, mll, train_x, train_y)

        print('shape GP_training_data_x_all', GP_training_data_x_all.shape)
        print('shape GP_training_data_y_all', GP_training_data_y_all.shape)
        
        # calculate GP model rewards for ALL samples in training set under current reward model
        GP_mean_rews_all_data_current_reward_model, GP_var_rews_all_data_current_reward_model = reward_learner.calc_expected_reward_for_observed_outcome_w_GPmodel \
            (gpr_reward_model, likelihood, np.array(np.array(training_data_list)[:,0].tolist()))     
        agent.GP_mean_rews_all_data_current_reward_model = GP_mean_rews_all_data_current_reward_model       
    import pdb; pdb.set_trace()
    
    # Initialize Gaussian policy params (DMP weights) - mean and sigma
    initial_wts, initial_mu, initial_sigma, S, control_type_z_axis = agent.initialize_gaussian_policy(init_dmp_info_dict, work_dir, dmp_wts_file)
    print('initial mu', initial_mu)        
    mu, sigma = initial_mu, initial_sigma   
    import pdb; pdb.set_trace()

    # if starting from a later epoch: load previous data    
    if os.path.isfile(os.path.join(work_dir, 'policy_mean_each_epoch.npy')):
        mean_params_each_epoch = np.load(os.path.join(work_dir, 'policy_mean_each_epoch.npy')).tolist()
        cov_each_epoch = np.load(os.path.join(work_dir, 'policy_cov_each_epoch.npy')).tolist()
    else:   
        mean_params_each_epoch.append(initial_mu)   
        cov_each_epoch.append(initial_sigma) 

    if args.starting_epoch_num > 0:
        agent.init_mu_0 = np.array(mean_params_each_epoch[0])
        agent.init_cov_0 = np.array(cov_each_epoch[0])
    import pdb; pdb.set_trace()

    for epoch in range(args.starting_epoch_num, args.num_epochs):
        #Initialize lists to save epoch's data                 
        expert_rewards_all_samples, reward_model_mean_rewards_all_samples, reward_model_rewards_all_cov, \
            policy_params_all_samples, outcomes_all, analytical_rewards_all_samples = [],[],[],[],[],[]        
        # if starting from nonzero sample num, need to load previous reward_model_mean_rewards_all_samples, reward_model_rewards_all_cov, outcomes_all
        # TODO: pull this out and clean up!!
        if args.starting_sample_num !=0:
            prev_data = np.load(work_dir + '/' + 'GP_reward_model_data/' + 'training_data_list.npy', allow_pickle=True)
            if epoch == 0:
                start_idx = 0
            elif epoch == 1:
                start_idx = 25
            elif epoch == 2:
                start_idx = 50
            elif epoch == 3:
                start_idx = 75
            elif epoch == 4:
                start_idx = 95
            outcomes_all = prev_data[start_idx:,0].tolist()
            # reward_model_mean_rewards_all_samples = prev_data[start_idx:,1].tolist()
            reward_model_rewards_all_cov = prev_data[start_idx:,2].tolist()            
        import pdb; pdb.set_trace()    

        for sample in range(args.starting_sample_num, args.num_samples):
            print('Epoch: %i Sample: %i'%(epoch,sample))
            total_samples += 1            

            # Sample new policy params from mu and sigma - NOTE: cap force to be [-1, -40]
            new_params, new_weights, new_z_force, new_cart_pitch_stiffness  = agent.sample_new_params_from_policy(mu, sigma, args.use_all_dmp_dims, initial_wts,  args.cut_type, S)

            # save to policy params buffer
            policy_params_all_samples.append(new_params.tolist())
            policy_params_all_epochs.append(new_params) 

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
            plot_sampled_new_dmp_traject_and_original_dmp(epoch, work_dir, sample, new_z_force, traject_time, \
                dmp_wts_file, dmp_traject, y0)
                    
            import pdb; pdb.set_trace()

            # sampling info for sending msgs via ROS
            dt = 0.01 
            T = traject_time
            ts = np.arange(0, T, dt)
            N = len(ts)

            # downsample dmp traject 
            downsmpled_dmp_traject = downsample_dmp_traject(dmp_traject, 0.001, dt)
            target_poses = get_dmp_traj_poses_reformatted(downsmpled_dmp_traject, starting_rotation) # target_poses is a nx16 list of target poses at each time step
           
            if S[2] == 0:
                target_force = [0, 0, new_z_force, 0, 0, 0] 

            elif S[2] == 1:
                target_force = [0, 0, 0, 0, 0, 0] 

            position_kps_cart = FC.DEFAULT_TRANSLATIONAL_STIFFNESSES + FC.DEFAULT_ROTATIONAL_STIFFNESSES
            # set pitch axis cartesian gain to be sampled value
            position_kps_cart[-2] = new_cart_pitch_stiffness            
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

            # sample from gaussian to get dmp weights for this execution       
            dmp_num = 0            
            peak_z_forces_all_dmps, x_mvmt_all_dmps, forw_vs_back_x_mvmt_all_dmps = [], [], []# sum of abs (+x/-x mvmt)  
            y_mvmt_all_dmps, peak_y_force_all_dmps, z_mvmt_all_dmps, upward_z_mvmt_all_dmps, upward_z_penalty_all_dmps = [], [], [], [], []
            total_cut_time_all_dmps = 0          
            while current_ht > 0.023:   
                # start FP skill
                fa.run_dynamic_force_position(duration=T *100000000000000000, buffer_time = 3, 
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

                # stop skill here w/ proto msg
                term_proto_msg = ShouldTerminateSensorMessage(timestamp=rospy.Time.now().to_time() - init_time, should_terminate=True)
                ros_msg = make_sensor_group_msg(
                    termination_handler_sensor_msg=sensor_proto2ros_msg(
                        term_proto_msg, SensorDataMessageType.SHOULD_TERMINATE)
                    )
                pub.publish(ros_msg)

                # calc stats from dmp
                cut_time = rospy.Time.now().to_time() - init_time
                peak_y_force = np.max(np.abs(robot_forces[:,1]))
                peak_z_force = np.max(np.abs(robot_forces[:,2]))

                forward_x_mvmt = (np.max(np.abs(robot_positions[:,0]) - np.abs(robot_positions[0,0])))
                backward_x_mvmt = (np.max(np.abs(robot_positions[:,0]) - np.abs(robot_positions[-1,0])))
                total_x_mvmt = forward_x_mvmt + backward_x_mvmt

                forward_y_mvmt = (np.max(np.abs(robot_positions[:,1]) - np.abs(robot_positions[0,1])))
                backward_y_mvmt = (np.max(np.abs(robot_positions[:,1]) - np.abs(robot_positions[-1,1])))
                total_y_mvmt = forward_y_mvmt + backward_y_mvmt               
                
                if args.cut_type == 'normal' or args.cut_type == 'scoring':
                    upward_z_mvmt = np.max(robot_positions[:,2]) - robot_positions[0,2]
                    up_z_mvmt = np.abs(robot_positions[-1,2]) - np.min(np.abs(robot_positions[:,2])) 
                    down_z_mvmt = np.abs(robot_positions[0,2]) - np.min(np.abs(robot_positions[:,2]))
                    total_z_mvmt = up_z_mvmt + down_z_mvmt

                elif args.cut_type == 'pivchop':
                    if (robot_positions[-1,2]-robot_positions[0,2]) > 0.02:
                        upward_z_penalty = (robot_positions[-1,2]-robot_positions[0,2])
                    else:
                        upward_z_penalty = 0          
                    up_z_mvmt = np.abs(robot_positions[-1,2]) - np.min(np.abs(robot_positions[:,2])) 
                    down_z_mvmt = np.abs(robot_positions[0,2]) - np.min(np.abs(robot_positions[:,2]))
                    total_z_mvmt = up_z_mvmt + down_z_mvmt
                    diff_up_down_z_mvmt = np.abs(up_z_mvmt - down_z_mvmt)

                # save to buffers 
                total_cut_time_all_dmps += cut_time
                peak_z_forces_all_dmps.append(peak_z_force)
                x_mvmt_all_dmps.append(total_x_mvmt)
                y_mvmt_all_dmps.append(total_y_mvmt)
                peak_y_force_all_dmps.append(peak_y_force)           
                z_mvmt_all_dmps.append(total_z_mvmt)

                if args.cut_type == 'normal' or args.cut_type =='scoring':
                    upward_z_mvmt_all_dmps.append(upward_z_mvmt)

                elif args.cut_type == 'pivchop':
                    upward_z_penalty_all_dmps.append(upward_z_penalty)

                np.savez(work_dir + '/' + 'forces_positions/' + 'epoch_'+str(epoch) + '_ep_'+str(sample) + '_trial_info_'+str(dmp_num)+'.npz', robot_positions=robot_positions, \
                    robot_forces=robot_forces)
                
                completed_cut = input('cut complete? (0-n, 1-y, 2-cannot complete): ')

                while completed_cut not in ['0', '1', '2']:
                    completed_cut = input('please enter valid answer. cut complete? (0/1/2): ') 

                if completed_cut == '1': 
                    #fa.stop_skill()
                    break

                elif completed_cut == '2': 
                    # if cut can't be completed, give very high penalty for time 
                    total_cut_time_all_dmps = 200
                    #fa.stop_skill()
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
                    target_poses = get_dmp_traj_poses_reformatted(downsmpled_dmp_traject, starting_rotation) # target_poses is a nx16 list of target poses at each time step
                
            
            # pause to let skill fully stop
            time.sleep(1.5)

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

            # Query human for continuous reward: between -2 to 2 
            '''NOTE: supports 2 different experiments: 1) desired behavior is fast or slow cut (human specified reward for each of these)
            2) desired behavior is quality cut (e.g. scoring tomato)
            '''
            if args.desired_cutting_behavior == 'fast' or args.desired_cutting_behavior == 'slow':
                expert_reward_slow = input('enter human reward, desired behavior is slow: -2 to 2: ')
                expert_reward_fast = input('enter human reward, desired behavior is fast: -2 to 2: ')
                
                while expert_reward_slow not in ['-2', '-1', '0', '1', '2']:
                    expert_reward_slow = input('enter human reward, desired behavior is slow: -2 to 2: ')

                while expert_reward_fast not in ['-2', '-1', '0', '1', '2']:
                    expert_reward_fast = input('enter human reward, desired behavior is fast: -2 to 2:')
                
                expert_reward = [int(expert_reward_slow), int(expert_reward_fast)]
                expert_rewards_all_samples.append(expert_reward)
                expert_rewards_all_epochs.append(expert_reward)

            elif args.desired_cutting_behavior == 'quality_cut':
                expert_reward = input('enter human reward, desired behavior is quality cut: -2 to 2: ')
                while expert_reward not in ['-2', '-1', '0', '1', '2']:
                    expert_reward = input('enter human reward, desired behavior is quality cut: -2 to 2: ')
                expert_reward = int(expert_reward)
                expert_rewards_all_samples.append(expert_reward)
                expert_rewards_all_epochs.append(expert_reward)

            # calc averages/max across all cut types - NOTE: switched to max instead of avg to handle dmps that vary as they are chained            
            avg_peak_y_force = np.max(peak_y_force_all_dmps)
            avg_peak_z_force = np.max(peak_z_forces_all_dmps)
            avg_x_mvmt = np.max(x_mvmt_all_dmps)
            avg_y_mvmt = np.max(y_mvmt_all_dmps)
            avg_z_mvmt = np.max(z_mvmt_all_dmps)

            if args.cut_type == 'normal' or args.cut_type == 'scoring':
                avg_upward_z_mvmt = np.max(upward_z_mvmt_all_dmps)
                avg_upward_z_penalty = avg_upward_z_mvmt
            
            elif args.cut_type == 'pivchop':
                avg_upward_z_penalty = np.max(upward_z_penalty_all_dmps)
            
            # more generalized cutting reward function:
            analytical_reward = -0.1*avg_peak_y_force -0.15*avg_peak_z_force - 10*avg_x_mvmt -100*avg_y_mvmt - 10*avg_z_mvmt \
                -100*avg_upward_z_penalty -0.2*total_cut_time_all_dmps 

            # save reward to buffer
            if args.desired_cutting_behavior == 'slow':
                print('----------------------------------')
                print('Epoch: %i Sample: %i Analytical Reward: %f Expert Reward: %f'%(epoch,sample,analytical_reward, expert_reward[0]))
            elif args.desired_cutting_behavior == 'fast':
                print('Epoch: %i Sample: %i Analytical Reward: %f Expert Reward: %f'%(epoch,sample,analytical_reward, expert_reward[1]))
            else: 
                print('Epoch: %i Sample: %i Analytical Reward: %f Expert Reward: %f'%(epoch,sample,analytical_reward, expert_reward))

            analytical_rewards_all_samples.append(analytical_reward)
            reward_features = [avg_peak_y_force, avg_peak_z_force, avg_x_mvmt, avg_y_mvmt, avg_z_mvmt, avg_upward_z_penalty, total_cut_time_all_dmps]
            reward_features_all_samples.append(reward_features)
            
            # save reward features as sample outcome 
            outcomes_from_sample = np.array(reward_features)
            # if standardizing reward features:
            if args.standardize_reward_feats:
                outcomes_from_sample = reward_learner.standardize_reward_feature(args.cut_type, np.array(reward_features))
            outcomes_all.append(outcomes_from_sample)     
            # import pdb; pdb.set_trace()

            if epoch == 0: #if epoch = 0, GP model hasn't been trained so mean = 0
                '''
                might not need to add these since we don't have reward model rewards in first epoch
                since GP model hasn't been trained yet
                '''
                reward_model_mean_rewards_all_samples.append(0)
                reward_model_rewards_all_cov.append(0)
                reward_model_rewards_all_mean_buffer.append(0)

                training_data_list.append([outcomes_from_sample, 0, 0, new_params])

            elif epoch != 0: 
                mean_expected_reward, var_expected_reward = reward_learner.calc_expected_reward_for_observed_outcome_w_GPmodel(gpr_reward_model, \
                    likelihood, outcomes_from_sample)
                print('reward model reward = ', mean_expected_reward[0])
                import pdb; pdb.set_trace()
                
                #Save expected reward mean and var to lists and add to training_data_list of all training data
                reward_model_mean_rewards_all_samples.append(mean_expected_reward[0])
                print('GP model rewards mean all eps', reward_model_mean_rewards_all_samples)
                reward_model_rewards_all_cov.append(var_expected_reward[0])
                print('GP model rewards var all eps', reward_model_rewards_all_cov)
                reward_model_rewards_all_mean_buffer.append(mean_expected_reward[0])
                training_data_list.append([outcomes_from_sample, mean_expected_reward[0], var_expected_reward[0], new_params])                               
            
            # save intermediate rewards/pol params 
            '''NOTE: saving these in format: [polParams, analytical_reward, GP_reward_model_mean, expert_reward],
            where expert_reward can be 1 or 2-element list depending on type of desired behavior
            '''
            if args.starting_sample_num !=0:
                # import pdb; pdb.set_trace()
                # policy param data
                prev_sample_data = np.load(os.path.join(work_dir + '/' + 'all_polParamRew_data', 'polParamsRews_' + 'epoch_'+str(epoch) + '_ep_'+str(args.starting_sample_num-1) + '.npy'))
                    
                if num_expert_rews_each_sample == 1:
                    # import pdb; pdb.set_trace()
                    new_analytRew_GPmodelRew_ExpertRew = np.concatenate((np.array([analytical_rewards_all_samples]).T, np.array([reward_model_mean_rewards_all_samples]).T, np.array([expert_rewards_all_samples]).T),axis=1)
                elif num_expert_rews_each_sample == 2:                  
                    new_analytRew_GPmodelRew_ExpertRew = np.concatenate((np.array([analytical_rewards_all_samples]).T, np.array([reward_model_mean_rewards_all_samples]).T, np.array(expert_rewards_all_samples)),axis=1)
                                               
                new_sample_data = np.concatenate((np.array(policy_params_all_samples), new_analytRew_GPmodelRew_ExpertRew), axis=1)
                combined_data = np.concatenate((prev_sample_data, new_sample_data), axis=0)
                np.save(os.path.join(work_dir + '/' + 'all_polParamRew_data', 'polParamsRews_' + 'epoch_'+str(epoch) + '_ep_'+str(sample) + '.npy'),
                    combined_data)

            else:
                if num_expert_rews_each_sample == 1:
                    new_analytRew_GPmodelRew_ExpertRew = np.concatenate((np.array([analytical_rewards_all_samples]).T, np.array([reward_model_mean_rewards_all_samples]).T, np.array([expert_rewards_all_samples]).T),axis=1)
                elif num_expert_rews_each_sample == 2:                  
                    new_analytRew_GPmodelRew_ExpertRew = np.concatenate((np.array([analytical_rewards_all_samples]).T, np.array([reward_model_mean_rewards_all_samples]).T, np.array(expert_rewards_all_samples)),axis=1)
                
                np.save(os.path.join(work_dir + '/' + 'all_polParamRew_data', 'polParamsRews_' + 'epoch_'+str(epoch) + '_ep_'+str(sample) + '.npy'), \
                    np.concatenate((np.array(policy_params_all_samples), new_analytRew_GPmodelRew_ExpertRew), axis=1))
            
            # save relevant data to npy files 
            reward_learner.save_reward_learning_data_to_np_arrays(work_dir, time_to_complete_cut, task_success, task_success_more_granular, \
                reward_features_all_samples, expert_rewards_all_epochs, training_data_list)
            
            import pdb; pdb.set_trace()
            # reset to starting cut position            
            new_position = copy.deepcopy(starting_position)
            #new_position.translation[1] = fa.get_pose().translation[1]
            new_position.translation[1] = fa.get_pose().translation[1] - 0.003
            fa.goto_pose(new_position, duration=5, use_impedance=False)

            # move over a bit (y dir)       
            y_shift = 0.006 #0.004 #float(input('enter how far to shift in y dir (m): '))
            move_over_slice_thickness = RigidTransform(translation=np.array([0.0, y_shift, 0.0]),
                from_frame='world', to_frame='world')       
            fa.goto_pose_delta(move_over_slice_thickness, duration=3, use_impedance=False)

            # move down to contact
            import pdb; pdb.set_trace()
            if not args.debug:
                move_down_to_contact = RigidTransform(translation=np.array([0.0, 0.0, -0.1]),
                from_frame='world', to_frame='world')   
                if args.food_name == 'banana': 
                    fa.goto_pose_delta(move_down_to_contact, duration=5, use_impedance=False, force_thresholds=[10.0, 10.0, 2.0, 10.0, 10.0, 10.0], ignore_virtual_walls=True)
                else:
                    fa.goto_pose_delta(move_down_to_contact, duration=5, use_impedance=False, force_thresholds=[10.0, 10.0, 3.0, 10.0, 10.0, 10.0], ignore_virtual_walls=True)
                        
        # save reward 
        if num_expert_rews_each_sample == 1:
            new_analytRew_GPmodelRew_ExpertRew = np.concatenate((np.array([analytical_rewards_all_samples]).T, np.array([reward_model_mean_rewards_all_samples]).T, np.array([expert_rewards_all_samples]).T),axis=1)
        elif num_expert_rews_each_sample == 2:                  
            new_analytRew_GPmodelRew_ExpertRew = np.concatenate((np.array([analytical_rewards_all_samples]).T, np.array([reward_model_mean_rewards_all_samples]).T, np.array(expert_rewards_all_samples)),axis=1)               
               
        np.save(os.path.join(work_dir, 'polParamsRews_' + 'epoch_'+str(epoch) +'.npy'), \
            np.concatenate((np.array(policy_params_all_samples), new_analytRew_GPmodelRew_ExpertRew), axis=1))
                   
        # Train initial GP model with args.num_GP_training_samples      
        '''only compute EPD if epoch!=0 (i.e. reward model has been trained on initial set of data)'''
        if epoch == 0: 
            # updating this to train w/ just 1st sample (instead of x number of samples) ---> then iteratively update based on sample w/ max KLD 
            samples_to_query = np.arange(1)
            queried_outcomes = np.squeeze(np.array(outcomes_all))[0:1] 
            import pdb; pdb.set_trace()

            # query expert rewards and update GP training x and y with new queried samples - HERE WE ARE UPDATING GP_training_data_x_all, GP_training_data_y_all
            if samples_to_query != []:
                GP_training_data_x_all, GP_training_data_y_all, queried_expert_rewards = reward_learner.query_expert_rewards_and_update_GP_training_data(epoch, \
                    GP_training_data_x_all, GP_training_data_y_all, samples_to_query, queried_outcomes, expert_rewards_all_epochs)
            else:
                print('no samples to query - reward model will not be updated')
            
            print('new GP_training_data_x_all shape', GP_training_data_x_all.shape)
            print('new GP_training_data_y_all shape', GP_training_data_y_all.shape)
            import pdb; pdb.set_trace()                                          

            # --------------train with initial number of training samples
            print('initializing and training GP reward model')
            likelihood = gpytorch.likelihoods.GaussianLikelihood()      
            # import pdb; pdb.set_trace()      
            train_x = torch.from_numpy(queried_outcomes)
            train_x = train_x.float()
            train_y = torch.from_numpy(queried_expert_rewards)
            train_y = train_y.float()
            import pdb; pdb.set_trace()

            # instantiate reward model
            gpr_reward_model = GPRegressionModel(train_x, train_y, likelihood) 

            # Set prior on GP model hyperparams
            gpr_reward_model.covar_module.outputscale = args.GPsignal_var_initial

            # adding to initialize hyperparams based on prior data!
            if args.init_GP_lengthscale_hyperparam_from_prior:
                gpr_reward_model.covar_module.base_kernel.lengthscale = torch.from_numpy(np.array([[5.1391, 5.0817, 5.1031, 5.1450, 5.1920, 5.1543, 2.8334]]))
            import pdb; pdb.set_trace()

            optimizer = torch.optim.Adam([                
                {'params': gpr_reward_model.covar_module.parameters()},
                {'params': gpr_reward_model.mean_module.parameters()},
                {'params': gpr_reward_model.likelihood.parameters()},
            ], lr=0.01) # lr = 0.01 originally 

            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gpr_reward_model)
            # train reward GP model given initial train_x and train_y data (voxel_data, queried human rewards)
            gpr_reward_model = reward_learner.train_GPmodel(work_dir, GP_training_epochs_initial, optimizer, gpr_reward_model, likelihood, mll, train_x, train_y)

            print('CHECK size GP_mean_rews_all_data_current_reward_model', len(GP_mean_rews_all_data_current_reward_model))
            print('shape train_x', train_x.shape)
            print('shape train_y', train_y.shape)        
            import pdb; pdb.set_trace()

            # compute pi_tilda_wts (current reward model)
            reps_agent = reps.Reps(rel_entropy_bound=1.5, min_temperature=0.001) #Create REPS object

            # iteratively compute EPD and update reward model with sample w/ highest KLD --> recompute EPD, continue until max KLD val < kappa
            max_KLD_val = np.inf
            max_KLD_vals_all = []
            mse_from_human_rews = []
            iteration = 0
            # plot to keep track of GP model updates 
            fig, axs = plt.subplots(2, 1, sharey=True, tight_layout=True)                
            # TODO: START WHILE LOOP HERE (while max_KLD_val > thresh:)
            while max_KLD_val > kappa:
                print('max_KLD_val, start of iter %i'%iteration, max_KLD_val)
                # calculate GP model rewards for ALL samples in training set under current reward model
                GP_mean_rews_all_data_current_reward_model, GP_var_rews_all_data_current_reward_model = reward_learner.calc_expected_reward_for_observed_outcome_w_GPmodel \
                    (gpr_reward_model, likelihood, np.array(np.array(training_data_list)[:,0].tolist()))                    

                #calc pi_tilda_wts
                pi_tilda_wts, temp = reps_agent.weights_from_rewards(GP_mean_rews_all_data_current_reward_model)

                #calc MSE from human rews
                mse_from_human_rews.append(np.linalg.norm(np.array(expert_rewards_all_epochs)- np.array(GP_mean_rews_all_data_current_reward_model)))

                # determine outcomes to query using EPD
                if iteration ==0:
                    GP_training_data_x_all = train_x #GP_training_data_x_all[0:args.num_GP_training_samples,:]
                    GP_training_data_y_all = train_y #GP_training_data_y_all[0:args.num_GP_training_samples]
                    queried_samples_all = [0] #queried_samples_all[0:args.num_GP_training_samples]   

                print('shape GP_training_data_x_all, epoch %i, iter %i: '%(epoch, iteration), GP_training_data_x_all.shape)
                print('shape GP_training_data_y_all, epoch %i, iter %i: '%(epoch, iteration), GP_training_data_y_all.shape)
                print('shape queried_samples_all, epoch %i, iter %i: '%(epoch, iteration), len(queried_samples_all))        
                import pdb; pdb.set_trace() 

                # plot every time we update the reward model with a sample
                if iteration % 2 == 0:
                    axs[0].plot(GP_mean_rews_all_data_current_reward_model,'-o', label='num_queries=%i'%len(queried_samples_all))    
                    axs[1].plot(GP_var_rews_all_data_current_reward_model,'-o', label='num_queries=%i'%len(queried_samples_all))             
                
                # compute EPD for each sample - NOTE: ONLY 1 SAMPLE BEING QUERIED at a time 
                current_epoch = epoch
                samples_to_query, queried_outcomes, max_KLD_val  = reward_learner.compute_EPD_for_each_sample_iterative\
                    (GP_mean_rews_all_data_current_reward_model, GP_var_rews_all_data_current_reward_model, \
                current_epoch, work_dir, num_EPD_epochs, optimizer, gpr_reward_model, likelihood, mll, agent, pi_tilda_wts, \
                    training_data_list, queried_samples_all, GP_training_data_x_all, GP_training_data_y_all, args.cut_type)

                if max_KLD_val < kappa:
                    print('max_KLD_val < kappa threshold')
                    break
                print('max_KLD_val', max_KLD_val)
                max_KLD_vals_all.append(max_KLD_val)

                # update queried samples all
                queried_samples_all = queried_samples_all + [samples_to_query] # running list of queried samples 
                import pdb; pdb.set_trace()

                # update GP training data
                #queried_expert_rewards = np.array([np.array(expert_rewards_all_epochs)[samples_to_query]])
                #GP_training_data_x_all = np.vstack((GP_training_data_x_all, queried_outcomes))
                #GP_training_data_y_all = np.concatenate((GP_training_data_y_all, queried_expert_rewards))
                
                GP_training_data_x_all, GP_training_data_y_all, queried_expert_rewards = reward_learner.query_expert_rewards_and_update_GP_training_data\
                    (epoch, GP_training_data_x_all, GP_training_data_y_all, samples_to_query, queried_outcomes, expert_rewards_all_epochs)

                print('shape GP_training_data_x_all', GP_training_data_x_all.shape)
                print('shape GP_training_data_y_all', GP_training_data_y_all.shape)
                print('shape queried_samples_all', len(queried_samples_all))  
                #import pdb; pdb.set_trace()  

                if queried_outcomes.size!=0:                 
                    print('updating reward model')
                    #import pdb; pdb.set_trace()  
                    updated_train_x = GP_training_data_x_all
                    updated_train_y = GP_training_data_y_all                
                    continue_training = False 
                    gpr_reward_model = reward_learner.update_reward_GPmodel(work_dir, continue_training, GP_training_epochs_later, optimizer, gpr_reward_model, likelihood, mll, updated_train_x, updated_train_y)                                       
                
                iteration += 1

            axs[0].plot(expert_rewards_all_epochs,'-o', linestyle = 'dashed', label= 'human rewards')
            axs[0].set_xlabel('samples')
            axs[0].set_ylabel('rewards')
            axs[0].legend()
            #axs[0].legend(('human rewards', 'GP model rews'))
            axs[0].set_title('human rewards vs. GP model rewards, init training size = %i, epoch %i, kappa = %s'%(args.num_GP_training_samples, epoch, str(kappa)))

            axs[1].set_xlabel('samples')
            axs[1].set_ylabel('variance - GP rewards')
            #axs[1].legend(('GP model variance','GP model variance'))
            axs[1].legend()
            axs[1].set_title('GP model reward variance, init training size = %i, epoch %i, kappa = %s'%(args.num_GP_training_samples, epoch, str(kappa)))
            
            print('max_KLD_vals_all', max_KLD_vals_all)
            print('mse_from_human_rews', mse_from_human_rews)
            plt.figure()
            plt.plot(np.arange(1,len(mse_from_human_rews)+1), mse_from_human_rews,'-o')
            plt.xlabel('number of human queries')
            plt.xticks(np.arange(1,len(mse_from_human_rews)+1))
            plt.ylabel('MSE from human rewards')
            plt.title('MSE from human rewards vs. number of queries, epoch %i, kappa = %s'%(epoch, str(kappa)))
            plt.show() 
            import pdb; pdb.set_trace()

            # Save queried_samples_all   
            print('size queried_samples_all', len(queried_samples_all))        
            if args.desired_cutting_behavior == 'slow':
                np.save(work_dir + '/' + 'GP_reward_model_data/' + 'queried_samples_all_slowCut_epoch_'+str(epoch)+'.npy', np.array(queried_samples_all))

            elif args.desired_cutting_behavior == 'fast':
                np.save(work_dir + '/' + 'GP_reward_model_data/' + 'queried_samples_all_fastCut_epoch_'+str(epoch)+'.npy', np.array(queried_samples_all))

            else:
                np.save(work_dir + '/' + 'GP_reward_model_data/' + 'queried_samples_all_qualityCut_epoch_'+str(epoch)+'.npy', np.array(queried_samples_all))
            

        else:  # do this for all epoch > 0
            print('computing EPD and iteratively updating reward model')
            # iteratively compute EPD and update reward model with sample w/ highest KLD --> recompute EPD, continue until max KLD val < kappa
            max_KLD_val = np.inf
            max_KLD_vals_all = []
            mse_from_human_rews = []
            iteration = 0
            # plot to keep track of GP model updates 
            fig, axs = plt.subplots(2, 1, sharey=True, tight_layout=True)                
            reps_agent = reps.Reps(rel_entropy_bound=1.5, min_temperature=0.001) #Create REPS object
            while max_KLD_val > kappa:
                print('max_KLD_val, start of iter %i'%iteration, max_KLD_val)
                # calculate GP model rewards for ALL samples in training set under current reward model
                GP_mean_rews_all_data_current_reward_model, GP_var_rews_all_data_current_reward_model = reward_learner.calc_expected_reward_for_observed_outcome_w_GPmodel \
                    (gpr_reward_model, likelihood, np.array(np.array(training_data_list)[:,0].tolist()))                    

                #calc pi_tilda_wts
                pi_tilda_wts, temp = reps_agent.weights_from_rewards(GP_mean_rews_all_data_current_reward_model)

                #calc MSE from human rews
                mse_from_human_rews.append(np.linalg.norm(np.array(expert_rewards_all_epochs)- np.array(GP_mean_rews_all_data_current_reward_model)))               

                print('shape GP_training_data_x_all, epoch %i, iter %i: '%(epoch, iteration), GP_training_data_x_all.shape)
                print('shape GP_training_data_y_all, epoch %i, iter %i: '%(epoch, iteration), GP_training_data_y_all.shape)
                print('shape queried_samples_all, epoch %i, iter %i: '%(epoch, iteration), len(queried_samples_all))        
                #import pdb; pdb.set_trace() 

                # plot every time we update the reward model with a sample
                if iteration % 2 == 0:
                    axs[0].plot(GP_mean_rews_all_data_current_reward_model,'-o', label='num_queries=%i'%len(queried_samples_all))    
                    axs[1].plot(GP_var_rews_all_data_current_reward_model,'-o', label='num_queries=%i'%len(queried_samples_all))             
                
                # compute EPD for each sample - NOTE: ONLY 1 SAMPLE BEING QUERIED at a time 
                current_epoch = epoch
                samples_to_query, queried_outcomes, max_KLD_val  = reward_learner.compute_EPD_for_each_sample_iterative\
                    (GP_mean_rews_all_data_current_reward_model, GP_var_rews_all_data_current_reward_model, \
                current_epoch, work_dir, num_EPD_epochs, optimizer, gpr_reward_model, likelihood, mll, agent, pi_tilda_wts, \
                    training_data_list, queried_samples_all, GP_training_data_x_all, GP_training_data_y_all, args.cut_type)

                if max_KLD_val < kappa:
                    print('max_KLD_val < kappa threshold')
                    break
                print('max_KLD_val', max_KLD_val)
                max_KLD_vals_all.append(max_KLD_val)

                # update queried samples all
                queried_samples_all = queried_samples_all + [samples_to_query] # running list of queried samples                 
                
                GP_training_data_x_all, GP_training_data_y_all, queried_expert_rewards = reward_learner.query_expert_rewards_and_update_GP_training_data\
                    (epoch, GP_training_data_x_all, GP_training_data_y_all, samples_to_query, queried_outcomes, expert_rewards_all_epochs)

                print('shape GP_training_data_x_all', GP_training_data_x_all.shape)
                print('shape GP_training_data_y_all', GP_training_data_y_all.shape)
                print('shape queried_samples_all', len(queried_samples_all))  
                #import pdb; pdb.set_trace()  

                if queried_outcomes.size!=0:                 
                    print('updating reward model')
                    #import pdb; pdb.set_trace()  
                    updated_train_x = GP_training_data_x_all
                    updated_train_y = GP_training_data_y_all                
                    continue_training = False 
                    gpr_reward_model = reward_learner.update_reward_GPmodel(work_dir, continue_training, GP_training_epochs_later, optimizer, gpr_reward_model, likelihood, mll, updated_train_x, updated_train_y)                                       
                
                iteration += 1

            axs[0].plot(expert_rewards_all_epochs,'-o', linestyle = 'dashed', label= 'human rewards')
            axs[0].set_xlabel('samples')
            axs[0].set_ylabel('rewards')
            axs[0].legend()
            #axs[0].legend(('human rewards', 'GP model rews'))
            axs[0].set_title('human rewards vs. GP model rewards, init training size = %i, epoch %i, kappa = %s'%(args.num_GP_training_samples, epoch, str(kappa)))

            axs[1].set_xlabel('samples')
            axs[1].set_ylabel('variance - GP rewards')
            #axs[1].legend(('GP model variance','GP model variance'))
            axs[1].legend()
            axs[1].set_title('GP model reward variance, init training size = %i, epoch %i, kappa = %s'%(args.num_GP_training_samples, epoch, str(kappa)))
            
            print('max_KLD_vals_all', max_KLD_vals_all)
            print('mse_from_human_rews', mse_from_human_rews)
            plt.figure()
            plt.plot(np.arange(1,len(mse_from_human_rews)+1), mse_from_human_rews,'-o')
            plt.xlabel('number of human queries')
            plt.xticks(np.arange(1,len(mse_from_human_rews)+1))
            plt.ylabel('MSE from human rewards')
            plt.title('MSE from human rewards vs. number of queries, epoch %i, kappa = %s'%(epoch, str(kappa)))
            plt.show() 
            import pdb; pdb.set_trace()

            # Save queried_samples_all   
            print('size queried_samples_all', len(queried_samples_all))         
            if args.desired_cutting_behavior == 'slow':
                np.save(work_dir + '/' + 'GP_reward_model_data/' + 'queried_samples_all_slowCut_epoch_'+str(epoch)+'.npy', np.array(queried_samples_all))

            elif args.desired_cutting_behavior == 'fast':
                np.save(work_dir + '/' + 'GP_reward_model_data/' + 'queried_samples_all_fastCut_epoch_'+str(epoch)+'.npy', np.array(queried_samples_all))

            else:
                np.save(work_dir + '/' + 'GP_reward_model_data/' + 'queried_samples_all_qualityCut_epoch_'+str(epoch)+'.npy', np.array(queried_samples_all))               

           
        print('new GP_training_data_x_all shape', GP_training_data_x_all.shape)
        print('new GP_training_data_y_all shape', GP_training_data_y_all.shape)
        import pdb; pdb.set_trace() 
        
        # #Keep track of number of queried samples 
        if epoch > 0:
            num_prev_queried = total_queried_samples_each_epoch[epoch-1]
            #total_queried_samples_each_epoch.append(num_prev_queried + len(samples_to_query))    
            total_queried_samples_each_epoch.append(num_prev_queried + len(max_KLD_vals_all))    
        else:
            total_queried_samples_each_epoch.append(len(max_KLD_vals_all))
            #total_queried_samples_each_epoch.append(len(samples_to_query)) 
        
        if args.desired_cutting_behavior == 'slow':
            np.save(work_dir + '/' 'GP_reward_model_data/' + 'total_queried_samples_each_epoch_slowCut.npy', np.array(total_queried_samples_each_epoch))

        elif args.desired_cutting_behavior == 'fast':
            np.save(work_dir + '/' 'GP_reward_model_data/' + 'total_queried_samples_each_epoch_fastCut.npy', np.array(total_queried_samples_each_epoch))
        
        else:
            np.save(work_dir + '/' 'GP_reward_model_data/' + 'total_queried_samples_each_epoch_qualityCut.npy', np.array(total_queried_samples_each_epoch))
        import pdb; pdb.set_trace()       
        
        # calculate GP model rewards for ALL samples in training set under current reward model
        GP_mean_rews_all_data_current_reward_model, GP_var_rews_all_data_current_reward_model = reward_learner.calc_expected_reward_for_observed_outcome_w_GPmodel \
            (gpr_reward_model, likelihood, np.array(np.array(training_data_list)[:,0].tolist()))         
        
        # save all training data rewards under current reward model
        np.save(work_dir + '/' + 'GP_reward_model_data/all_training_data_rewards_under_latest_reward_model.npy', np.array(GP_mean_rews_all_data_current_reward_model))          

        # plot post-EPD rewards 
        fig, axs = plt.subplots(2, 1, sharey=True, tight_layout=True) 
        axs[0].plot(expert_rewards_all_epochs,'-o') 
        axs[0].plot(GP_mean_rews_all_data_current_reward_model,'-o')
        axs[1].plot(GP_var_rews_all_data_current_reward_model,'-o')

        axs[0].set_xlabel('samples')
        axs[0].set_ylabel('rewards')
        axs[0].set_title('human vs. GP model rewards - epoch %i'%epoch)

        axs[1].set_xlabel('samples') 
        axs[1].set_ylabel('variance')
        axs[1].set_title('GP model variances - epoch %i'%epoch)

        axs[0].legend(('human rewards', 'GP rewards post-EPD update'))
        plt.show()

        # get reward model rewards for samples from only THIS epoch for REPS - NOTE: updated this from previous loading them in from saved npy to prevent discrepancies in already saved data w/ diff reward model 
        import pdb; pdb.set_trace()
        if epoch == 0:
            reward_model_mean_rewards_all_samples = GP_mean_rews_all_data_current_reward_model[0:25]
        elif epoch == 1:
            reward_model_mean_rewards_all_samples = GP_mean_rews_all_data_current_reward_model[25:50]
        elif epoch == 2:
            reward_model_mean_rewards_all_samples = GP_mean_rews_all_data_current_reward_model[50:75]
        elif epoch == 3:
            reward_model_mean_rewards_all_samples = GP_mean_rews_all_data_current_reward_model[75:95]
        elif epoch == 4:
            reward_model_mean_rewards_all_samples = GP_mean_rews_all_data_current_reward_model[95:]
        
        # get policy parameters for all samples in this epoch if not starting from sample = 0
        if args.starting_sample_num !=0:
            all_data = work_dir + '/all_polParamRew_data/' +'polParamsRews_epoch_' + str(epoch) + '_ep_'+ str(sample) + '.npy'
            if num_expert_rews_each_sample == 1:       
                #import pdb; pdb.set_trace()         
                policy_params_all_samples = (np.load(all_data)[:,0:-3]).tolist()
                # reward_model_mean_rewards_all_samples = (np.load(all_data)[:,-2]).tolist()
                expert_rewards_all_samples = (np.load(all_data)[:,-1]).tolist()

            elif num_expert_rews_each_sample == 2: 
                #import pdb; pdb.set_trace()               
                policy_params_all_samples = (np.load(all_data)[:,0:-4]).tolist()
                # reward_model_mean_rewards_all_samples = (np.load(all_data)[:,-3]).tolist()
                expert_rewards_all_samples = (np.load(all_data)[:,-2:]).tolist()  # 2 expert reward values (desired fast and slow rewards)     
        import pdb; pdb.set_trace()                       

        print('updating policy w/ REPS')
        reps_agent = reps.Reps(rel_entropy_bound=rel_entropy_bound,min_temperature=0.001) #Create REPS object
        import pdb; pdb.set_trace() 
        
        # unscaled
        policy_params_mean, policy_params_sigma, reps_info = \
            reps_agent.policy_from_samples_and_rewards(policy_params_all_samples, reward_model_mean_rewards_all_samples)            
        print('updated policy params mean')
        print(policy_params_mean)
        print('updated policy cov')
        print(policy_params_sigma)
        # import pdb; pdb.set_trace()            
        mu, sigma = policy_params_mean, policy_params_sigma
        
        # scaled for KLD calc
        policy_params_all_samples_scaled = agent.scale_pol_params(policy_params_all_samples)
        policy_params_mean_scaled, policy_params_sigma_scaled, reps_info_scaled = \
            reps_agent.policy_from_samples_and_rewards(policy_params_all_samples_scaled, reward_model_mean_rewards_all_samples)

        # pi_tilda is the new policy under the current reward model -- will use this to compute KLD
        if args.scale_pol_params_for_KLD:    # scaled
            pi_tilda_mean = policy_params_mean_scaled   
            pi_tilda_cov = policy_params_sigma_scaled  
        else:  # unscaled
            pi_tilda_mean = policy_params_mean
            pi_tilda_cov = policy_params_sigma           
                
        # save policy mean and cov to buffers
        mean_params_each_epoch.append(policy_params_mean)
        cov_each_epoch.append(policy_params_sigma)
        np.save(os.path.join(work_dir, 'policy_mean_each_epoch.npy'),np.array(mean_params_each_epoch))
        np.save(os.path.join(work_dir, 'policy_cov_each_epoch.npy'),np.array(cov_each_epoch))

        # calculate pi_tilda wts based on current reward model rewards - TODO - take this out (already done above)
        pi_tilda_wts, temp = reps_agent.weights_from_rewards(GP_mean_rews_all_data_current_reward_model) 

        # plot updated policy mean trajectory to visualize
        plot_updated_policy_mean_traject_HIL_ARL(work_dir, fa, args.cut_type, dmp_wts_file, epoch, args.dmp_traject_time, control_type_z_axis, init_dmp_info_dict,\
            initial_wts, policy_params_mean)

        # save new policy params mean and cov   
        np.savez(os.path.join(work_dir, 'REPSupdatedMean_' + 'epoch_'+str(epoch) +'.npz'), \
            updated_mean = policy_params_mean, updated_cov = policy_params_sigma, pi_tilda_wts = pi_tilda_wts)
                
        # after epoch is complete, reset start_sample to 0
        args.starting_sample_num = 0
        plt.figure()
        plt.plot(task_success)
        plt.title('epoch %i task_success'%epoch)
        plt.xlabel('samples')
        plt.ylabel('task_success')
        plt.show()

        print('Starting new epoch')
    
    print('total samples', total_samples)   
    print('cumulative queried samples', total_queried_samples_each_epoch)
    plt.plot(np.arange(epoch+1), total_queried_samples_each_epoch,'-o')
    plt.xlabel('epochs', fontsize=22)
    plt.xticks(np.arange((epoch+1)),fontsize=22)
    plt.tick_params(axis="x", labelsize=15)
    plt.tick_params(axis="y", labelsize=15)
    plt.ylabel('cumulative human queried samples', fontsize=22)
    plt.ylim([0, np.max(total_queried_samples_each_epoch)+3])
    plt.title('cumulative human queries vs epochs, total queries = %i, total samples = %i, %s, %s'%(len(queried_samples_all),115, args.cut_type, args.food_name), fontsize=22)
    plt.show()  
    import pdb; pdb.set_trace()       

    fa.goto_joints(reset_joint_positions)
