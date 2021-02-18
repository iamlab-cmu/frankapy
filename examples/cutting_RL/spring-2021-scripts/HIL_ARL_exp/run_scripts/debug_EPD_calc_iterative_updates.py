'''
UPDATED: iterative reward model updates instead of batch updates
ex. CL args:
python -m pdb debug_EPD_calc.py --exp_num 4 --food_type hard --food_name potato --cut_type normal --start_from_previous True --previous_datadir /home/sony/Documents/cutting_RL_experiments/data/Jan-2021-HIL-ARL-exps/normal/potato/exp_4/ --starting_epoch_num 1 --num_samples 25 --desired_cutting_behavior fast --num_GP_training_samples 15
'''

import argparse

import torch
import gpytorch
from matplotlib import cm
import os
import sys
sys.path.append('/home/sony/Documents/frankapy/examples/cutting_RL/spring-2021-scripts/HIL_ARL_exp/')

from reward_learner import RewardLearner
from policy_learner import REPSPolicyLearner
from gp_regression_model import GPRegressionModel
from data_utils import *
import sklearn.gaussian_process as gp

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
    parser.add_argument('--kappa', type=float, default = 0.01) #5) # 0.005
    parser.add_argument('--rel_entropy_bound', type=float, default = 1.2)
    parser.add_argument('--num_EPD_epochs', type=int, default = 5)
    parser.add_argument('--GP_training_epochs_initial', type=int, default = 120)
    parser.add_argument('--GP_training_epochs_later', type=int, default = 11)
    parser.add_argument('--desired_cutting_behavior', type=str, help='options: fast, slow, quality_cut') # fast, slow, quality_cut
    parser.add_argument('--standardize_reward_feats', type=bool, default = True)
    parser.add_argument('--scale_pol_params_for_KLD', type=bool, default = True)
    parser.add_argument('--add_ridge_to_pol_cov_for_KLD', type=bool, default = False)
    parser.add_argument('--sampl_or_weight_kld_calc', type=str, default = 'weight', help = 'sampling or weight')

    # for GP evaluation/debugging    
    parser.add_argument('--use_all_GP_training_samples', type=bool, default = False)
    parser.add_argument('--num_GP_training_samples', type=int, default = 25)
    parser.add_argument('--save_var_mean_GP_rews', type=bool, default = False)
    parser.add_argument('--GPsignal_var_initial', type=int, default = 3) #4)
    parser.add_argument('--init_GP_lengthscale_hyperparam_from_prior', type=bool, default = False) #4)
    parser.add_argument('--plot_GP_model_comparisons', type=bool, default = False)
    parser.add_argument('--add_noise_to_expert_rews', type=bool, default = False)
    
    
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
 
    # load dmp weights and starting knife orientation
    dmp_wts_file, knife_orientation = load_dmp_wts_and_knife_orientation(args.cut_type)
    
    position_dmp_pkl = open(dmp_wts_file,"rb")
    init_dmp_info_dict = pickle.load(position_dmp_pkl)

    # Initialize Gaussian policy params (DMP weights) - mean and sigma
    # initial_wts, initial_mu, initial_sigma, S, control_type_z_axis = agent.initialize_gaussian_policy(init_dmp_info_dict, work_dir, dmp_wts_file)
    # print('initial mu', initial_mu)        
    # mu, sigma = initial_mu, initial_sigma

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

    import pdb; pdb.set_trace()    
        
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
        if args.desired_cutting_behavior == 'slow':
            prev_queried_samples_all = np.load(work_dir + '/' + 'GP_reward_model_data/' + 'queried_samples_all_slowCut_epoch_'+str(args.starting_epoch_num-1)+'.npy')
            queried_samples_all = prev_queried_samples_all.tolist()

            prev_total_queried_samples_each_epoch = np.load(work_dir + '/' 'GP_reward_model_data/' + 'total_queried_samples_each_epoch_slowCut.npy')
            total_queried_samples_each_epoch = prev_total_queried_samples_each_epoch.tolist()    

        elif args.desired_cutting_behavior == 'fast':
            prev_queried_samples_all = np.load(work_dir + '/' + 'GP_reward_model_data/' + 'queried_samples_all_fastCut_epoch_'+str(args.starting_epoch_num-1)+'.npy')
            queried_samples_all = prev_queried_samples_all.tolist()

            prev_total_queried_samples_each_epoch = np.load(work_dir + '/' 'GP_reward_model_data/' + 'total_queried_samples_each_epoch_fastCut.npy')
            total_queried_samples_each_epoch = prev_total_queried_samples_each_epoch.tolist()    

        else:    
            prev_queried_samples_all = np.load(work_dir + '/' + 'GP_reward_model_data/' + 'queried_samples_all_qualityCut_epoch_'+str(args.starting_epoch_num-1)+'.npy')
            queried_samples_all = prev_queried_samples_all.tolist()

            prev_total_queried_samples_each_epoch = np.load(work_dir + '/' 'GP_reward_model_data/' + 'total_queried_samples_each_epoch_qualityCut.npy')
            total_queried_samples_each_epoch = prev_total_queried_samples_each_epoch.tolist()    
        
        # import pdb; pdb.set_trace()
    
    # if not starting from 0-th epoch - need to load/train GP reward model based on previous data
    # TODO: pull this out to a separate function
    if args.starting_epoch_num > 0:
        print('training GP reward model from previously saved data')
        # import pdb; pdb.set_trace()    
            
        if args.desired_cutting_behavior == 'slow':
            prev_GP_training_data = np.load(work_dir + '/' 'GP_reward_model_data/' + 'GP_reward_model_training_data_slowCut_epoch_' +str(args.starting_epoch_num-1) + '.npz')
            GP_training_data_x_all = prev_GP_training_data['GP_training_data_x_all']
            GP_training_data_y_all = prev_GP_training_data['GP_training_data_y_all']
            GP_training_data_y_all_slow = GP_training_data_y_all            

        elif args.desired_cutting_behavior == 'fast': 
            prev_GP_training_data = np.load(work_dir + '/' 'GP_reward_model_data/' + 'GP_reward_model_training_data_fastCut_epoch_' +str(args.starting_epoch_num-1) + '.npz')
            GP_training_data_x_all = prev_GP_training_data['GP_training_data_x_all']
            GP_training_data_y_all = prev_GP_training_data['GP_training_data_y_all']
            GP_training_data_y_all_fast = GP_training_data_y_all
        
        else:
            prev_GP_training_data = np.load(work_dir + '/' 'GP_reward_model_data/' + 'GP_reward_model_training_data_qualityCut_epoch_' +str(args.starting_epoch_num-1) + '.npz')
            GP_training_data_x_all = prev_GP_training_data['GP_training_data_x_all']
            GP_training_data_y_all = prev_GP_training_data['GP_training_data_y_all']

        # train with previous data
        print('initializing and training GP reward model from previous data')             
        # import pdb; pdb.set_trace()      
        train_x = torch.from_numpy(GP_training_data_x_all)
        train_x = train_x.float()
        train_y = torch.from_numpy(GP_training_data_y_all)
        train_y = train_y.float()
        print('train_y variance', train_y.var())

        # adding to eval different training set sizes
        if not args.use_all_GP_training_samples:        
            train_x = train_x[0:args.num_GP_training_samples]
            train_y = train_y[0:args.num_GP_training_samples]
        # import pdb; pdb.set_trace()
        
        if args.add_noise_to_expert_rews:
            train_y = train_y + np.random.normal(0, 0.2)
        likelihood = gpytorch.likelihoods.GaussianLikelihood() 
        # add white noise 
        #likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_prior = gpytorch.priors.NormalPrior(0,0.5)) 
        #likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(torch.ones(train_x.shape[0]) * beta)
        gpr_reward_model = GPRegressionModel(train_x, train_y, likelihood) 
        gpr_reward_model.covar_module.outputscale = args.GPsignal_var_initial
        # adding to initialize differently based on prior data!
        if args.init_GP_lengthscale_hyperparam_from_prior:
            gpr_reward_model.covar_module.base_kernel.lengthscale = torch.from_numpy(np.array([[5.1391, 5.0817, 5.1031, 5.1450, 5.1920, 5.1543, 2.8334]]))
        # import pdb; pdb.set_trace()

        optimizer = torch.optim.Adam([                
            {'params': gpr_reward_model.covar_module.parameters()},
            {'params': gpr_reward_model.mean_module.parameters()},
            {'params': gpr_reward_model.likelihood.parameters()},
        ], lr=0.01) # lr = 0.01 originally 
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gpr_reward_model)
        # train reward GP model given initial train_x and train_y data (voxel_data, queried human rewards)
        # using gpytorch 
        gpr_reward_model = reward_learner.train_GPmodel(work_dir, GP_training_epochs_initial, optimizer, gpr_reward_model, likelihood, mll, train_x, train_y)
                        
        # calculate GP model rewards for ALL samples in training set under current reward model
        GP_mean_rews_all_data_current_reward_model, GP_var_rews_all_data_current_reward_model = reward_learner.calc_expected_reward_for_observed_outcome_w_GPmodel \
            (gpr_reward_model, likelihood, np.array(np.array(training_data_list)[:,0].tolist()))     
        agent.GP_mean_rews_all_data_current_reward_model = GP_mean_rews_all_data_current_reward_model
        
        agent.GP_mean_rews_all_data_current_reward_model = GP_mean_rews_all_data_current_reward_model
        print('CHECK size GP_mean_rews_all_data_current_reward_model', len(GP_mean_rews_all_data_current_reward_model))
        print('shape train_x', train_x.shape)
        print('shape train_y', train_y.shape)        
        import pdb; pdb.set_trace()
        # compute pi_tilda_wts (current reward model)
        reps_agent = reps.Reps(rel_entropy_bound=1.5, min_temperature=0.001) #Create REPS object

        max_KLD_val = np.inf
        max_KLD_vals_all = []
        mse_from_human_rews = []
        iteration = 0
        #import pdb; pdb.set_trace()
        # plot to keep track of GP model updates 
        fig, axs = plt.subplots(2, 1, sharey=True, tight_layout=True)    
        
        # TODO: START WHILE LOOP HERE (while max_KLD_val > thresh:)
        while max_KLD_val > kappa:
            print('max_KLD_val, start of iter %i'%iteration, max_KLD_val)
            #import pdb; pdb.set_trace()
            # calculate GP model rewards for ALL samples in training set under current reward model
            GP_mean_rews_all_data_current_reward_model, GP_var_rews_all_data_current_reward_model = reward_learner.calc_expected_reward_for_observed_outcome_w_GPmodel \
                (gpr_reward_model, likelihood, np.array(np.array(training_data_list)[:,0].tolist()))                    
            agent.GP_mean_rews_all_data_current_reward_model = GP_mean_rews_all_data_current_reward_model

            #calc pi_tilda_wts
            pi_tilda_wts, temp = reps_agent.weights_from_rewards(GP_mean_rews_all_data_current_reward_model)

            #calc MSE from human rews
            mse_from_human_rews.append(np.linalg.norm(np.array(expert_rewards_all_epochs)- np.array(GP_mean_rews_all_data_current_reward_model)))

            # determine outcomes to query using EPD
            # adding this to account for trying out different amounts of training data
            if not args.use_all_GP_training_samples and iteration ==0:
                GP_training_data_x_all = GP_training_data_x_all[0:args.num_GP_training_samples,:]
                GP_training_data_y_all = GP_training_data_y_all[0:args.num_GP_training_samples]
                queried_samples_all = queried_samples_all[0:args.num_GP_training_samples]                
            print('shape GP_training_data_x_all, iter %i: '%iteration, GP_training_data_x_all.shape)
            print('shape GP_training_data_y_all, iter %i: '%iteration, GP_training_data_y_all.shape)
            print('shape queried_samples_all, iter %i: '%iteration, len(queried_samples_all))        
            #import pdb; pdb.set_trace() 

            # plot every time we update the reward model with a sample
            if iteration%2 == 0:
                axs[0].plot(GP_mean_rews_all_data_current_reward_model,'-o', label='num_queries=%i'%len(queried_samples_all))    
                axs[1].plot(GP_var_rews_all_data_current_reward_model,'-o', label='num_queries=%i'%len(queried_samples_all))             
            #import pdb; pdb.set_trace()
            
            # compute EPD for each sample - NOTE: ONLY 1 SAMPLE BEING QUERIED
            current_epoch = 0
            samples_to_query, queried_outcomes, max_KLD_val  = reward_learner.compute_EPD_for_each_sample_iterative\
                (GP_mean_rews_all_data_current_reward_model, GP_var_rews_all_data_current_reward_model, \
            current_epoch, work_dir, num_EPD_epochs, optimizer, gpr_reward_model, likelihood, mll, agent, pi_tilda_wts, \
                training_data_list, queried_samples_all, GP_training_data_x_all, GP_training_data_y_all, args.cut_type)

            if max_KLD_val < kappa:
                break
            print('max_KLD_val', max_KLD_val)
            max_KLD_vals_all.append(max_KLD_val)

            # update queried samples all
            queried_samples_all = queried_samples_all + [samples_to_query] # running list of queried samples 
            #import pdb; pdb.set_trace()

            # update GP training data
            queried_expert_rewards = np.array([np.array(expert_rewards_all_epochs)[samples_to_query]])
            GP_training_data_x_all = np.vstack((GP_training_data_x_all, queried_outcomes))
            GP_training_data_y_all = np.concatenate((GP_training_data_y_all, queried_expert_rewards))
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
        axs[0].set_title('human rewards vs. GP model rewards, init training size = %i, kappa = %s'%(args.num_GP_training_samples,str(kappa)))

        axs[1].set_xlabel('samples')
        axs[1].set_ylabel('variance - GP rewards')
        #axs[1].legend(('GP model variance','GP model variance'))
        axs[1].legend()
        axs[1].set_title('GP model reward variance, init training size = %i, kappa = %s'%(args.num_GP_training_samples,str(kappa)))  
        

        print('max_KLD_vals_all', max_KLD_vals_all)
        print('mse_from_human_rews', mse_from_human_rews)
        plt.figure()
        plt.plot(np.arange(1,len(mse_from_human_rews)+1), mse_from_human_rews,'-o')
        plt.xlabel('number of human queries')
        plt.xticks(np.arange(1,len(mse_from_human_rews)+1))
        plt.ylabel('MSE from human rewards')
        plt.title('MSE from human rewards vs. number of queries, kappa = %s'%str(kappa))
        plt.show() 
        import pdb; pdb.set_trace()

    import pdb; pdb.set_trace()
    
    ########################## ORIGINAL CODE BELOW

    # Initialize Gaussian policy params (DMP weights) - mean and sigma
    initial_wts, initial_mu, initial_sigma, S, control_type_z_axis = agent.initialize_gaussian_policy(init_dmp_info_dict, work_dir, dmp_wts_file)
    print('initial mu', initial_mu)        
    mu, sigma = initial_mu, initial_sigma   
    import pdb; pdb.set_trace()

    pi_current_mean = mu
    pi_current_cov = sigma

    mean_params_each_epoch, cov_each_epoch = [], []
    # if starting from a later epoch: load previous data    
    if os.path.isfile(os.path.join(work_dir, 'policy_mean_each_epoch.npy')):
        test = np.load(os.path.join(work_dir, 'policy_mean_each_epoch.npy'))
        if len(test.shape) == 1:            
            mean_params_each_epoch.append(np.load(os.path.join(work_dir, 'policy_mean_each_epoch.npy')))
            cov_each_epoch.append(np.load(os.path.join(work_dir, 'policy_cov_each_epoch.npy')))
        else:
            mean_params_each_epoch = np.load(os.path.join(work_dir, 'policy_mean_each_epoch.npy')).tolist()
            cov_each_epoch = np.load(os.path.join(work_dir, 'policy_cov_each_epoch.npy')).tolist()
    else:   
        mean_params_each_epoch.append(initial_mu)   
        cov_each_epoch.append(initial_sigma) 
    
    # if not starting from initial mean and sigma, load these and save them to the policy learner for scaling params later
    if args.starting_epoch_num > 0:    
        # import pdb; pdb.set_trace()
        if len(np.array(mean_params_each_epoch).shape) == 1: 
            agent.init_mu_0 = np.array(mean_params_each_epoch)
            agent.init_cov_0 = np.array(cov_each_epoch)   
        else:
            agent.init_mu_0 = np.array(mean_params_each_epoch[0])
            agent.init_cov_0 = np.array(cov_each_epoch[0])

    reps_agent = reps.Reps(rel_entropy_bound=1.5, min_temperature=0.001) #Create REPS object

    # load prev pol params
    #training_data_list = training_data_list[0:75]
    prev_epoch_data = np.load(work_dir + '/polParamsRews_epoch_0.npy')

    #prev_epoch_data = np.load(work_dir + '/polParamsRews_epoch_2.npy')
    policy_params_all_samples = prev_epoch_data[:,0:9]
    policy_params_all_samples_scaled = agent.scale_pol_params(policy_params_all_samples)

    # calculate weights!    
    import pdb; pdb.set_trace()
    GP_mean_rews_all_data_current_reward_model, GP_var_rews_all_data_current_reward_model = reward_learner.calc_expected_reward_for_observed_outcome_w_GPmodel(gpr_reward_model, likelihood, \
        np.array(np.array(training_data_list)[:,0].tolist()))
    pi_tilda_wts, temp = reps_agent.weights_from_rewards(GP_mean_rews_all_data_current_reward_model)

    if args.starting_epoch_num == 1:
            reward_model_mean_rewards_all_samples = GP_mean_rews_all_data_current_reward_model[0:25]
    elif args.starting_epoch_num  == 2:
        reward_model_mean_rewards_all_samples = GP_mean_rews_all_data_current_reward_model[25:50]
    elif args.starting_epoch_num  == 3:
        reward_model_mean_rewards_all_samples = GP_mean_rews_all_data_current_reward_model[50:75]
    elif args.starting_epoch_num  == 4:
        reward_model_mean_rewards_all_samples = GP_mean_rews_all_data_current_reward_model[75:95]
    import pdb; pdb.set_trace()        
    # if args.desired_cutting_behavior == 'quality_cut':
    #     reward_model_mean_rewards_all_samples = prev_epoch_data[:,-2]   
    # else:
    #     reward_model_mean_rewards_all_samples = prev_epoch_data[:,-3]  
    # import pdb; pdb.set_trace()

    # scaled
    policy_params_mean_scaled, policy_params_sigma_scaled, reps_info_scaled = \
        reps_agent.policy_from_samples_and_rewards(policy_params_all_samples_scaled, reward_model_mean_rewards_all_samples)
    # unscaled
    policy_params_mean, policy_params_sigma, reps_info = \
        reps_agent.policy_from_samples_and_rewards(policy_params_all_samples, reward_model_mean_rewards_all_samples)
    
    # SCALED PI_TILDA MEAN AND COV
    if args.scale_pol_params_for_KLD:
        pi_tilda_mean = policy_params_mean_scaled #scaled
        pi_tilda_cov = policy_params_sigma_scaled #scaled
        #pi_tilda_wts = reps_info_scaled['weights']
        #import pdb; pdb.set_trace()

    else:
        pi_tilda_mean = policy_params_mean
        pi_tilda_cov = policy_params_sigma
    
        
    test_dir = work_dir + '/'+'GP_reward_model_data/'+'2-15-21-GP-training-size-eval/' 
    if args.save_var_mean_GP_rews:
        np.save(test_dir + '/gp_rewsVars_trainingSize_'+ str(args.num_GP_training_samples) + '_sigVar_'+str(args.GPsignal_var_initial)+'.npy', \
            np.concatenate((np.expand_dims(np.array(GP_mean_rews_all_data_current_reward_model),axis=1),np.expand_dims(np.array(GP_var_rews_all_data_current_reward_model), axis=1)),axis=1)) 


    import pdb; pdb.set_trace()    
    if args.plot_GP_model_comparisons:
        print('GP signal var initial', args.GPsignal_var_initial)
        s5 = np.load(test_dir + 'gp_rewsVars_trainingSize_5' + '_sigVar_'+str(args.GPsignal_var_initial)+'.npy')
        s10 = np.load(test_dir + 'gp_rewsVars_trainingSize_10' + '_sigVar_'+str(args.GPsignal_var_initial)+'.npy')
        s15 = np.load(test_dir + 'gp_rewsVars_trainingSize_15' + '_sigVar_'+str(args.GPsignal_var_initial)+'.npy')
        s20 = np.load(test_dir + 'gp_rewsVars_trainingSize_20' + '_sigVar_'+str(args.GPsignal_var_initial)+'.npy')
        s25 = np.load(test_dir + 'gp_rewsVars_trainingSize_25' + '_sigVar_'+str(args.GPsignal_var_initial)+'.npy')

        human_reward = np.load(test_dir + 'human_rewards.npy')
        fig, axs = plt.subplots(2, 1, sharey=True, tight_layout=True)        
        axs[0].plot(s5[:,0],'-o')
        axs[0].plot(s10[:,0],'-o')
        axs[0].plot(s15[:,0],'-o')
        axs[0].plot(s20[:,0],'-o')
        axs[0].plot(s25[:,0],'-o')

        if len(human_reward.shape) == 1:
            axs[0].plot(human_reward, linestyle='dashed', marker='o', linewidth=4)

        elif len(human_reward.shape) == 2:
            axs[0].plot(human_reward[:,1],linestyle='dashed', marker='o', linewidth=4)

        axs[0].set_xlabel('samples')
        axs[0].set_xticks(np.arange(s10.shape[0]))
        axs[0].set_ylabel('rewards')
        axs[0].legend(('GP model rews, s=5', 'GP model rews, s=10', 'GP model rews, s=15','GP model rews, s=20','GP model rews, s=25', 'human rews'))
        axs[0].set_title('human rewards vs. GP model rewards with different training set sizes')
        
        #axs[1].vlines(np.array(queried),ymin=0,ymax=3,colors = ['r']*len(queried))
        axs[1].plot(s5[:,1],'-o')
        axs[1].plot(s10[:,1],'-o')
        axs[1].plot(s15[:,1],'-o')
        axs[1].plot(s20[:,1],'-o')
        axs[1].plot(s25[:,1],'-o')
        axs[1].set_xlabel('samples')
        axs[1].set_xticks(np.arange(s10.shape[0]))
        axs[1].set_ylabel('variance - GP rewards')
        axs[1].legend(('GP model rews, s=5', 'GP model, s=10', 'GP model, s=15','GP model, s=20','GP model, s=25'))
        axs[1].set_title('GP model reward variance with different training set sizes')        
        plt.show()

        #import pdb; pdb.set_trace()
        ###############--------------------- plot queried samples on same axis w/ variance 

        # - POTATO NORMAL CUT
        # queried_5 = [16, 20, 30, 41, 42, 44]
        # queried_10 = [16, 20, 30, 39, 41, 42, 44]
        # queried_15 = [15, 16, 20, 29, 30, 39, 41, 42, 44, 45, 49]
        # queried_20 = [20, 29, 30, 39, 41, 42, 44, 45, 49]
        # queried_25 = [29, 30, 39, 41, 42, 44, 45, 49]

        # TOMATO SCORING
        # queried_5 = [7, 10, 11, 13, 14, 15, 16, 17, 18, 20, 21, 24]
        # queried_10 = [13, 14, 17, 21]
        # queried_15 = [16, 17, 18, 22]
        # queried_20 = [23]
        # queried_25 = []

        fig, axs = plt.subplots(2, 5, sharey=True, tight_layout=True)

        axs[0,0].plot(s5[:,0],'-o')
        axs[0,1].plot(s10[:,0],'-o')
        axs[0,2].plot(s15[:,0],'-o')
        axs[0,3].plot(s20[:,0],'-o')
        axs[0,4].plot(s25[:,0],'-o')

        if len(human_reward.shape) == 1:
            # axs[0].plot(human_reward, linestyle='dashed', marker='o', linewidth=2)
            axs[0,0].plot(human_reward,linestyle='dashed', marker='o', linewidth=2)
            axs[0,1].plot(human_reward,linestyle='dashed', marker='o', linewidth=2)
            axs[0,2].plot(human_reward,linestyle='dashed', marker='o', linewidth=2)
            axs[0,3].plot(human_reward,linestyle='dashed', marker='o', linewidth=2)
            axs[0,4].plot(human_reward,linestyle='dashed', marker='o', linewidth=2)

        elif len(human_reward.shape) == 2:
            #axs[0].plot(human_reward[:,1],linestyle='dashed', marker='o', linewidth=2)

            axs[0,0].plot(human_reward[:,1],linestyle='dashed', marker='o', linewidth=2)
            axs[0,1].plot(human_reward[:,1],linestyle='dashed', marker='o', linewidth=2)
            axs[0,2].plot(human_reward[:,1],linestyle='dashed', marker='o', linewidth=2)
            axs[0,3].plot(human_reward[:,1],linestyle='dashed', marker='o', linewidth=2)
            axs[0,4].plot(human_reward[:,1],linestyle='dashed', marker='o', linewidth=2)

        
        for i in range(5):
            axs[0,i].set_xlabel('samples')
            axs[0,i].set_xticks(np.arange(s10.shape[0]))
        
        axs[0,0].set_ylabel('rewards')        
        axs[0,2].set_title('human rewards vs. GP model rewards with different training set sizes')     

        axs[0,0].legend(('GP model rews, s=5', 'human rews'))
        axs[0,1].legend(('GP model rews, s=10', 'human rews'))
        axs[0,2].legend(('GP model rews, s=15', 'human rews'))
        axs[0,3].legend(('GP model rews, s=20', 'human rews'))
        axs[0,4].legend(('GP model rews, s=25', 'human rews'))
         

        axs[1,0].plot(s5[:,1],'-o')
        axs[1,1].plot(s10[:,1],'-o')
        axs[1,2].plot(s15[:,1],'-o')
        axs[1,3].plot(s20[:,1],'-o')
        axs[1,4].plot(s25[:,1],'-o')

        for i in range(5):
            axs[1,i].set_xlabel('samples')
            axs[1,i].set_xticks(np.arange(s10.shape[0]))
        
        axs[1,0].set_ylabel('variance - GP rewards')        
        axs[1,2].set_title('GP model reward variance with different training set sizes')

        axs[1,0].legend(('GP model variance, s=5', 'human rews'))
        axs[1,1].legend(('GP model variance, s=10', 'human rews'))
        axs[1,2].legend(('GP model variance, s=15', 'human rews'))
        axs[1,3].legend(('GP model variance, s=20', 'human rews'))
        axs[1,4].legend(('GP model variance, s=25', 'human rews'))
        
        axs[1,0].vlines(np.array(queried_5),ymin=-2,ymax=5,colors = ['r']*len(queried_5),linestyles='dashed')
        axs[1,1].vlines(np.array(queried_10),ymin=-2,ymax=5,colors = ['r']*len(queried_10),linestyles='dashed')
        axs[1,2].vlines(np.array(queried_15),ymin=-2,ymax=5,colors = ['r']*len(queried_15),linestyles='dashed')
        axs[1,3].vlines(np.array(queried_20),ymin=-2,ymax=5,colors = ['r']*len(queried_20),linestyles='dashed')
        axs[1,4].vlines(np.array(queried_25),ymin=-2,ymax=5,colors = ['r']*len(queried_25),linestyles='dashed')
        
        plt.show()

        import pdb; pdb.set_trace()    

    # # PLOT REWARD MODEL INPUT FEATURES distributions
    # x_outcomes = np.array(np.array(training_data_list)[:,0].tolist())
    # fig, axs = plt.subplots(2, 7, sharey=True, tight_layout=True)
    # for dim in range(7):
    #     if dim == 6:
    #         axs[0,dim].hist(x_outcomes[0:25,dim], bins=15, color = 'green')
    #         axs[1,dim].hist(x_outcomes[25:,dim], bins=15, color = 'blue')
    #     else:
    #         axs[0,dim].hist(x_outcomes[0:25,dim], bins='auto', color = 'green')
    #         axs[1,dim].hist(x_outcomes[25:,dim], bins='auto', color = 'blue')
    #     axs[0,dim].set_title('std reward feat %i - training distr' %dim)
    #     axs[1,dim].set_title('std reward feat %i - test distr' %dim)
    #     axs[1,dim].set_xlabel('dim %i value' %dim)
    #     axs[0,dim].set_ylabel('freq')  
    #     axs[1,dim].set_ylabel('freq')  
    # plt.show()

    # import pdb; pdb.set_trace()

    # determine outcomes to query using EPD
    current_epoch = args.starting_epoch_num # 0
    # adding this to account for trying out different amounts of training data
    if not args.use_all_GP_training_samples:
        GP_training_data_x_all = GP_training_data_x_all[0:args.num_GP_training_samples,:]
        GP_training_data_y_all = GP_training_data_y_all[0:args.num_GP_training_samples]
        queried_samples_all = queried_samples_all[0:args.num_GP_training_samples]
    import pdb; pdb.set_trace()
    
    print('shape GP_training_data_x_all', GP_training_data_x_all.shape)
    print('shape GP_training_data_y_all', GP_training_data_y_all.shape)
    print('shape queried_samples_all', len(queried_samples_all))
    import pdb; pdb.set_trace()    

    # compute EPD for each sample
    samples_to_query, queried_outcomes  = reward_learner.compute_EPD_for_each_sample_updated \
        (GP_mean_rews_all_data_current_reward_model, GP_var_rews_all_data_current_reward_model, current_epoch, args.num_samples, work_dir, num_EPD_epochs, optimizer, \
            gpr_reward_model, likelihood, mll, agent, pi_tilda_mean, pi_tilda_cov, pi_tilda_wts, training_data_list, queried_samples_all, \
                GP_training_data_x_all, GP_training_data_y_all, beta, initial_wts, args.cut_type, S) 
    print('CHECK SAMPLES TO QUERY')
    import pdb; pdb.set_trace()

    #-----------------ONLY DO BELOW IF WANT TO UPDATE REWARD MODEL!!!!!!!!!!!!!!!!!!
    # query expert rewards and update GP training x and y with new queried samples - HERE WE ARE UPDATING GP_training_data_x_all, GP_training_data_y_all
    if samples_to_query != []:
        GP_training_data_x_all, GP_training_data_y_all, queried_expert_rewards = reward_learner.query_expert_rewards_and_update_GP_training_data(current_epoch, \
            GP_training_data_x_all, GP_training_data_y_all, samples_to_query, queried_outcomes, expert_rewards_all_epochs)
        import pdb; pdb.set_trace()  
    else:
        print('no samples to query - will not update reward model')   

    if queried_outcomes.size!=0:                 
        print('updating reward model')
        import pdb; pdb.set_trace()  
        updated_train_x = GP_training_data_x_all
        updated_train_y = GP_training_data_y_all                
        continue_training = False 
        gpr_reward_model = reward_learner.update_reward_GPmodel(work_dir, continue_training, GP_training_epochs_later, optimizer, gpr_reward_model, likelihood, mll, updated_train_x, updated_train_y)                                       
    import pdb; pdb.set_trace()
    
    # save new rewards under updated reward model from EPD queries
    GP_mean_rews_all_data_current_reward_model_after_EPD, GP_var_rews_all_data_current_reward_model_after_EPD = reward_learner.calc_expected_reward_for_observed_outcome_w_GPmodel(gpr_reward_model, likelihood, \
        np.array(np.array(training_data_list)[:,0].tolist()))

    if args.save_var_mean_GP_rews:
        np.save(test_dir + '/gp_rewsVars_trainingSize_'+ str(args.num_GP_training_samples) + '_sigVar_'+str(args.GPsignal_var_initial)+'_after_EPD_query_update.npy', \
            np.concatenate((np.expand_dims(np.array(GP_mean_rews_all_data_current_reward_model_after_EPD),axis=1),np.expand_dims(np.array(GP_var_rews_all_data_current_reward_model_after_EPD), axis=1)),axis=1)) 
    
    if args.plot_GP_model_comparisons:
        s5_EPDupdated = np.load(test_dir + 'gp_rewsVars_trainingSize_5' + '_sigVar_'+str(args.GPsignal_var_initial)+'_after_EPD_query_update.npy')
        s10_EPDupdated = np.load(test_dir + 'gp_rewsVars_trainingSize_10' + '_sigVar_'+str(args.GPsignal_var_initial)+'_after_EPD_query_update.npy')
        s15_EPDupdated = np.load(test_dir + 'gp_rewsVars_trainingSize_15' + '_sigVar_'+str(args.GPsignal_var_initial)+'_after_EPD_query_update.npy')
        s20_EPDupdated = np.load(test_dir + 'gp_rewsVars_trainingSize_20' + '_sigVar_'+str(args.GPsignal_var_initial)+'_after_EPD_query_update.npy')
        s25_EPDupdated = np.load(test_dir + 'gp_rewsVars_trainingSize_25' + '_sigVar_'+str(args.GPsignal_var_initial)+'_after_EPD_query_update.npy')

        fig, axs = plt.subplots(2, 5, sharey=True, tight_layout=True)

        axs[0,0].plot(s5[:,0],'-o')
        axs[0,1].plot(s10[:,0],'-o')
        axs[0,2].plot(s15[:,0],'-o')
        axs[0,3].plot(s20[:,0],'-o')
        axs[0,4].plot(s25[:,0],'-o')

        axs[0,0].plot(s5_EPDupdated[:,0],'-o')
        axs[0,1].plot(s10_EPDupdated[:,0],'-o')
        axs[0,2].plot(s15_EPDupdated[:,0],'-o')
        axs[0,3].plot(s20_EPDupdated[:,0],'-o')
        axs[0,4].plot(s25_EPDupdated[:,0],'-o')


        if len(human_reward.shape) == 1:
            axs[0,0].plot(human_reward,linestyle='dashed', marker='o', linewidth=2)
            axs[0,1].plot(human_reward,linestyle='dashed', marker='o', linewidth=2)
            axs[0,2].plot(human_reward,linestyle='dashed', marker='o', linewidth=2)
            axs[0,3].plot(human_reward,linestyle='dashed', marker='o', linewidth=2)
            axs[0,4].plot(human_reward,linestyle='dashed', marker='o', linewidth=2)

        elif len(human_reward.shape) == 2:
            axs[0,0].plot(human_reward[:,1],linestyle='dashed', marker='o', linewidth=2)
            axs[0,1].plot(human_reward[:,1],linestyle='dashed', marker='o', linewidth=2)
            axs[0,2].plot(human_reward[:,1],linestyle='dashed', marker='o', linewidth=2)
            axs[0,3].plot(human_reward[:,1],linestyle='dashed', marker='o', linewidth=2)
            axs[0,4].plot(human_reward[:,1],linestyle='dashed', marker='o', linewidth=2)

        
        for i in range(5):
            axs[0,i].set_xlabel('samples')
            axs[0,i].set_xticks(np.arange(s10.shape[0]))
        
        axs[0,0].set_ylabel('rewards')        
        axs[0,2].set_title('human rewards vs. GP model rewards with different training set sizes')     

        axs[0,0].legend(('GP model rews, s=5', 'GP model rews, s=17 post EPD update', 'human rews'))
        axs[0,1].legend(('GP model rews, s=10', 'GP model rews, s=14 post EPD update', 'human rews'))
        axs[0,2].legend(('GP model rews, s=15', 'GP model rews, s=19 post EPD update','human rews'))
        axs[0,3].legend(('GP model rews, s=20','GP model rews, s=21 post EPD update', 'human rews'))
        axs[0,4].legend(('GP model rews, s=25', 'GP model rews, s=25 post EPD update','human rews'))
         

        axs[1,0].plot(s5[:,1],'-o')
        axs[1,1].plot(s10[:,1],'-o')
        axs[1,2].plot(s15[:,1],'-o')
        axs[1,3].plot(s20[:,1],'-o')
        axs[1,4].plot(s25[:,1],'-o')

        axs[1,0].plot(s5_EPDupdated[:,1],'-o')
        axs[1,1].plot(s10_EPDupdated[:,1],'-o')
        axs[1,2].plot(s15_EPDupdated[:,1],'-o')
        axs[1,3].plot(s20_EPDupdated[:,1],'-o')
        axs[1,4].plot(s25_EPDupdated[:,1],'-o')

        for i in range(5):
            axs[1,i].set_xlabel('samples')
            axs[1,i].set_xticks(np.arange(s10.shape[0]))
        
        axs[1,0].set_ylabel('variance - GP rewards')        
        axs[1,2].set_title('GP model reward variance with different training set sizes')

        axs[1,0].legend(('s=5', 's=17 post EPD update'))
        axs[1,1].legend(('s=10', 's=14 post EPD update'))
        axs[1,2].legend(('s=15', 's=19 post EPD update'))
        axs[1,3].legend(('s=20', 's=21 post EPD update'))
        axs[1,4].legend(('s=25', 's=25 post EPD update'))

        plt.show()

    import pdb; pdb.set_trace()