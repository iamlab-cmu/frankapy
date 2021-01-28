import gym
from rl_utils import reps
import numpy as np 
import matplotlib.pyplot as plt 
from frankapy.utils import *

from tqdm import trange

class REPSPolicyLearner: 
    '''REPS policy learner for ARL 
    '''
    def __init__(self):
        self.reps_pol_updates_mean = []       
        self.reps_pol_updates_sigma = [] 
        self.rewards_all = []
        self.policy_params_all = []
        self.training_samples_all = []
        self.outcomes_all = []     

    def initialize_gaussian_policy(self, cut_type, food_type, dmp_wt_sampling_var, start_from_previous, previous_datadir, prev_epochs_to_calc_pol_update, \
            init_dmp_info_dict, work_dir, position_dmp_weights_file_path, starting_epoch_num, dmp_traject_time):

        use_all_dmp_dims = False
        if start_from_previous: # load previous data collected and start from updated policy and/or sample/epoch        
            prev_data_dir = previous_datadir
            if use_all_dmp_dims:
                policy_params_mean, policy_params_sigma = parse_policy_params_and_rews_from_file(prev_data_dir, prev_epochs_to_calc_pol_update, hfpc = False)
            else:
                policy_params_mean, policy_params_sigma = parse_policy_params_and_rews_from_file(prev_data_dir, prev_epochs_to_calc_pol_update)

            initial_mu, initial_sigma = policy_params_mean, policy_params_sigma
            mu, sigma = initial_mu, initial_sigma
            print('starting from updated policy - mean', policy_params_mean)
            initial_wts = np.array(init_dmp_info_dict['weights'])

            if cut_type == 'normal':
                if food_type == 'hard':
                    S = [1,1,0,1,1,1] 
                elif food_type == 'soft':
                    S = [1,1,0,1,1,1]
            
            elif cut_type == 'pivchop':
                if food_type == 'hard':
                    S = [0,1,1,1,1,1]
                elif food_type == 'soft':
                    S = [1,1,1,1,1,1]

            elif cut_type == 'scoring':
                if food_type == 'hard':
                    S = [1,0,0,1,1,1] 
                elif food_type == 'soft':
                    S = [1,1,0,1,1,1]
            
            if S[2] == 0:
                control_type_z_axis = 'force'
            elif S[2] == 1:
                control_type_z_axis = 'position'

            # plot updated policy mean trajectory to visualize
            print('plotting REPS updated mean trajectory')
            plot_updated_policy_mean_traject(work_dir, cut_type, position_dmp_weights_file_path, starting_epoch_num, dmp_traject_time, control_type_z_axis,\
                init_dmp_info_dict, initial_wts, mu)
            import pdb; pdb.set_trace()

        else: # start w/ initial DMP weights from IL
            initial_wts = np.array(init_dmp_info_dict['weights'])
            if cut_type == 'normal' or cut_type == 'scoring':
                f_initial = -10
                cart_pitch_stiffness_initial = 200  
                
            elif cut_type == 'pivchop':
                cart_pitch_stiffness_initial = 20 

            if use_all_dmp_dims: # use position control in dims (use all wt dims (x/y/z))
                initial_mu = initial_wts.flatten() 
                initial_sigma = np.diag(np.repeat(dmp_wt_sampling_var, initial_mu.shape[0]))

            else: # use only x wts, z-force, cart pitch stiffness 
                if cut_type == 'normal':
                    if food_type == 'hard':
                        S = [1,1,0,1,1,1] 
                    elif food_type == 'soft':
                        S = [1,1,0,1,1,1]
                    
                elif cut_type == 'pivchop':
                    if food_type == 'hard':
                        S = [0,1,1,1,1,1]
                    elif food_type == 'soft':
                        S = [1,1,1,1,1,1]

                elif cut_type == 'scoring':
                    if food_type == 'hard':
                        S = [1,0,0,1,1,1] 
                    elif food_type == 'soft':
                        S = [1,1,0,1,1,1]

                if cut_type == 'normal' or cut_type == 'scoring':
                    if S[0] == 1 and S[2] == 0: # position control x axis, force control z axis
                        initial_mu = np.append(initial_wts[0,:,:], [f_initial, cart_pitch_stiffness_initial]) 
                        initial_sigma = np.diag(np.repeat(dmp_wt_sampling_var, initial_mu.shape[0]))
                        initial_sigma[-2,-2] = 120 # change exploration variance for force parameter 
                        initial_sigma[-1,-1] = 800
                    
                    elif S[0] == 1 and S[2] == 1: # position control x axis, position control z axis
                        initial_mu = np.concatenate((initial_wts[0,:,:],initial_wts[2,:,:]),axis = 0)
                        initial_mu = np.append(initial_mu, cart_pitch_stiffness_initial) 
                        initial_sigma = np.diag(np.repeat(dmp_wt_sampling_var, initial_mu.shape[0]))
                        initial_sigma[-1,-1] = 800            
                
                elif cut_type == 'pivchop': 
                    if S[2] == 1: # z axis position control + var pitch stiffness
                        initial_mu = np.append(initial_wts[2,:,:], cart_pitch_stiffness_initial)  
                        initial_sigma = np.diag(np.repeat(dmp_wt_sampling_var, initial_mu.shape[0]))
                        initial_sigma[-1,-1] = 500 # change exploration variance for force parameter - TODO: increase
    
                    elif S[2] == 0: # no position control, only z axis force control + var pitch stiffness
                        f_initial = -10
                        initial_mu = np.append(f_initial, cart_pitch_stiffness_initial)  
                        initial_sigma = np.diag([120, 500])   
                        S = [1,1,0,1,1,1]

        return initial_mu, initial_sigma, S, control_type_z_axis

    def sample_params_from_policy(self, u, sigma):
        #import pdb; pdb.set_trace()
        sampled_new_params = np.random.multivariate_normal(u,sigma)  
        while sampled_new_params[1] < 0: #make sure height param is not < 0 to prevent from trying to place block below table
            sampled_new_params = np.random.multivariate_normal(u,sigma)  
        self.policy_params_all.append(sampled_new_params)        
        return sampled_new_params
    
    def update_policy_REPS(self, rewards_all, policy_params_all, rel_entropy_bound, min_temperature):
        REPS = reps.Reps(rel_entropy_bound, min_temperature) #Create REPS object
        policy_params_mean, policy_params_sigma, reps_info = REPS.policy_from_samples_and_rewards(policy_params_all, rewards_all)
        reps_wts = reps_info['weights']
        self.reps_pol_updates_mean.append(policy_params_mean)
        self.reps_pol_updates_sigma.append(policy_params_sigma)
        # NOTE: updates this function to return reps_wts - need to update previous scripts that use this method
        return policy_params_mean, policy_params_sigma, reps_wts

    def calculate_REPS_wts(self, rewards_all, rel_entropy_bound, min_temperature):
        reps_wts, temp = reps.reps_weights_from_rewards(rewards_all, rel_entropy_bound, min_temperature)
        return reps_wts

    
    