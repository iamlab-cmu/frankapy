from rl_utils import reps
import numpy as np 
import matplotlib.pyplot as plt 
from frankapy.utils import *

from tqdm import trange

class REPSPolicyLearner: 
    '''REPS policy learner for ARL 
    '''
    def __init__(self, num_expert_rews_each_sample, cut_type, food_type, dmp_wt_sampling_var, start_from_previous, previous_datadir,\
            prev_epochs_to_calc_pol_update, starting_epoch_num, dmp_traject_time):
        self.reps_pol_updates_mean = []       
        self.reps_pol_updates_sigma = [] 
        self.rewards_all = []
        self.policy_params_all = []
        self.training_samples_all = []
        self.outcomes_all = []     
        self.num_expert_rews_each_sample = num_expert_rews_each_sample
        self.cut_type = cut_type
        self.food_type = food_type
        self.dmp_wt_sampling_var = dmp_wt_sampling_var
        self.start_from_previous = start_from_previous
        self.previous_datadir = previous_datadir
        self.prev_epochs_to_calc_pol_update = prev_epochs_to_calc_pol_update
        self.starting_epoch_num = starting_epoch_num
        self.dmp_traject_time = dmp_traject_time
        self.GP_mean_rews_all_data_current_reward_model = None
    
    def scale_pol_params(self, pol_params):
        pol_params_scaled = (pol_params-self.init_mu_0)/np.sqrt(np.diag(self.init_cov_0))        
        return pol_params_scaled

    def initialize_gaussian_policy(self, init_dmp_info_dict, work_dir, position_dmp_weights_file_path):

        use_all_dmp_dims = False
        if self.start_from_previous and self.starting_epoch_num > 0:
            prev_data_dir = self.previous_datadir
            if use_all_dmp_dims:
                policy_params_mean, policy_params_sigma = parse_policy_params_and_rews_from_file_HIL_ARL(self.GP_mean_rews_all_data_current_reward_model, self.num_expert_rews_each_sample, prev_data_dir, self.prev_epochs_to_calc_pol_update, hfpc = False)
            else:
                policy_params_mean, policy_params_sigma = parse_policy_params_and_rews_from_file_HIL_ARL(self.GP_mean_rews_all_data_current_reward_model, self.num_expert_rews_each_sample, prev_data_dir, self.prev_epochs_to_calc_pol_update)

            initial_mu, initial_sigma = policy_params_mean, policy_params_sigma
            mu, sigma = initial_mu, initial_sigma
            print('starting from updated policy - mean', policy_params_mean)
            initial_wts = np.array(init_dmp_info_dict['weights'])

            if self.cut_type == 'normal':
                if self.food_type == 'hard':
                    S = [1,1,0,1,1,1] 
                elif self.food_type == 'soft':
                    S = [1,1,0,1,1,1]
            
            elif self.cut_type == 'pivchop':
                if self.food_type == 'hard':
                    S = [0,1,1,1,1,1]
                elif self.food_type == 'soft':
                    S = [1,1,1,1,1,1]

            elif self.cut_type == 'scoring':
                if self.food_type == 'hard':
                    S = [1,0,0,1,1,1] 
                elif self.food_type == 'soft':
                    S = [1,1,0,1,1,1]         
                     
        else: # start w/ initial DMP weights from IL
            initial_wts = np.array(init_dmp_info_dict['weights'])
            if self.cut_type == 'normal' or self.cut_type == 'scoring':
                f_initial = -10
                cart_pitch_stiffness_initial = 200  
                
            elif self.cut_type == 'pivchop':
                cart_pitch_stiffness_initial = 20 

            if use_all_dmp_dims: # use position control in dims (use all wt dims (x/y/z))
                initial_mu = initial_wts.flatten() 
                initial_sigma = np.diag(np.repeat(self.dmp_wt_sampling_var, initial_mu.shape[0]))

            else: # use only x wts, z-force, cart pitch stiffness 
                if self.cut_type == 'normal':
                    if self.food_type == 'hard':
                        S = [1,1,0,1,1,1] 
                    elif food_type == 'soft':
                        S = [1,1,0,1,1,1]
                    
                elif self.cut_type == 'pivchop':
                    if self.food_type == 'hard':
                        S = [0,1,1,1,1,1]
                    elif self.food_type == 'soft':
                        S = [1,1,1,1,1,1]

                elif self.cut_type == 'scoring':
                    if self.food_type == 'hard':
                        S = [1,0,0,1,1,1] 
                    elif self.food_type == 'soft':
                        S = [1,1,0,1,1,1]

                if self.cut_type == 'normal' or self.cut_type == 'scoring':
                    if S[0] == 1 and S[2] == 0: # position control x axis, force control z axis
                        initial_mu = np.append(initial_wts[0,:,:], [f_initial, cart_pitch_stiffness_initial]) 
                        initial_sigma = np.diag(np.repeat(self.dmp_wt_sampling_var, initial_mu.shape[0]))
                        initial_sigma[-2,-2] = 120 # change exploration variance for force parameter 
                        initial_sigma[-1,-1] = 800
                    
                    elif S[0] == 1 and S[2] == 1: # position control x axis, position control z axis
                        initial_mu = np.concatenate((initial_wts[0,:,:],initial_wts[2,:,:]),axis = 0)
                        initial_mu = np.append(initial_mu, cart_pitch_stiffness_initial) 
                        initial_sigma = np.diag(np.repeat(self.dmp_wt_sampling_var, initial_mu.shape[0]))
                        initial_sigma[-1,-1] = 800            
                
                elif self.cut_type == 'pivchop': 
                    if S[2] == 1: # z axis position control + var pitch stiffness
                        initial_mu = np.append(initial_wts[2,:,:], cart_pitch_stiffness_initial)  
                        initial_sigma = np.diag(np.repeat(self.dmp_wt_sampling_var, initial_mu.shape[0]))
                        initial_sigma[-1,-1] = 500 # change exploration variance for force parameter - TODO: increase
    
                    elif S[2] == 0: # no position control, only z axis force control + var pitch stiffness
                        f_initial = -10
                        initial_mu = np.append(f_initial, cart_pitch_stiffness_initial)  
                        initial_sigma = np.diag([120, 500])   
                        S = [1,1,0,1,1,1]

        if S[2] == 0:
            control_type_z_axis = 'force'
        elif S[2] == 1:
            control_type_z_axis = 'position'
        
        if self.starting_epoch_num == 0: # in [0,1]:
            self.init_mu_0 = initial_mu
            self.init_cov_0 = initial_sigma
        return initial_wts, initial_mu, initial_sigma, S, control_type_z_axis
    
    def sample_new_params_from_policy(self, mu, sigma, use_all_dmp_dims, initial_wts, cut_type, S):
        new_params = np.random.multivariate_normal(mu, sigma)    

        if use_all_dmp_dims:
            #new_z_force = new_params[-1] -- not using z force!
            new_weights = new_params.reshape(initial_wts.shape)
            new_z_force = 'NA'                                
        else:    
            if cut_type == 'normal' or cut_type == 'scoring':
                if S[2] == 0:
                    new_x_weights = new_params[0:-2]
                    new_z_force = new_params[-2]
                    new_cart_pitch_stiffness = new_params[-1]
                    # cap force value
                    new_z_force = np.clip(new_z_force, -40, -3)    # TODO: up force to -50N        
                    new_params[-2] = int(new_z_force)  
                    print('clipped sampled z force', new_z_force)  

                    new_cart_pitch_stiffness = np.clip(new_cart_pitch_stiffness, 5, 600)           
                    new_params[-1] = int(new_cart_pitch_stiffness)  
                    print('clipped sampled new_cart_pitch_stiffness', new_cart_pitch_stiffness)         
                
                elif S[2] == 1:
                    #import pdb; pdb.set_trace()
                    num_weights = initial_wts.shape[2]
                    new_x_weights = new_params[0:num_weights]
                    new_z_weights = new_params[num_weights:(2*num_weights)]
                    new_cart_pitch_stiffness = new_params[-1]
                    # cap value                   
                    new_cart_pitch_stiffness = np.clip(new_cart_pitch_stiffness, 5, 600)           
                    new_params[-1] = int(new_cart_pitch_stiffness)  
                    print('clipped sampled new_cart_pitch_stiffness', new_cart_pitch_stiffness)      
                    new_z_force = 'NA'   
            
            elif cut_type == 'pivchop':                    
                if S[2] == 0:  
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
                
                elif S[2] == 1:
                    new_z_weights = new_params[0:-1]
                    new_cart_pitch_stiffness = new_params[-1]
                    # cap value
                    new_cart_pitch_stiffness = np.clip(new_cart_pitch_stiffness, 5, 600)           
                    new_params[-1] = int(new_cart_pitch_stiffness)  
                    print('clipped sampled new_cart_pitch_stiffness', new_cart_pitch_stiffness)
                    new_z_force = 'NA'  

        # concat new sampled x weights w/ old y (zero's) and z weights if we're only sampling x weights
        if not use_all_dmp_dims: 
            if cut_type == 'normal' or cut_type == 'scoring':
                if S[2] == 0:
                    new_weights = np.expand_dims(np.vstack((new_x_weights,initial_wts[1,:,:],initial_wts[2,:,:])),axis=1)
                
                elif S[2] == 1:
                    new_weights = np.expand_dims(np.vstack((new_x_weights,initial_wts[1,:,:],new_z_weights)),axis=1)
            
            elif cut_type == 'pivchop':     
                new_weights = np.expand_dims(np.vstack((initial_wts[0,:,:],initial_wts[1,:,:], new_z_weights)),axis=1)

        return new_params, new_weights, new_z_force, new_cart_pitch_stiffness 

    def sample_new_params_from_policy_only_mu_sigma(self, scaled_pol_params, mu, sigma, initial_wts, cut_type, S):
        new_params = np.random.multivariate_normal(mu, sigma)    
 
        if cut_type == 'normal' or cut_type == 'scoring':
            if S[2] == 0:
                new_x_weights = new_params[0:-2]
                new_z_force = new_params[-2]
                new_cart_pitch_stiffness = new_params[-1]
                # cap force value
                if not scaled_pol_params:
                    new_z_force = np.clip(new_z_force, -40, -3)    # TODO: up force to -50N        
                    new_params[-2] = int(new_z_force)  
                    print('clipped sampled z force', new_z_force)  

                    new_cart_pitch_stiffness = np.clip(new_cart_pitch_stiffness, 5, 600)           
                    new_params[-1] = int(new_cart_pitch_stiffness)  
                    print('clipped sampled new_cart_pitch_stiffness', new_cart_pitch_stiffness)         
            
            elif S[2] == 1:
                #import pdb; pdb.set_trace()
                num_weights = initial_wts.shape[2]
                new_x_weights = new_params[0:num_weights]
                new_z_weights = new_params[num_weights:(2*num_weights)]
                new_cart_pitch_stiffness = new_params[-1]
                # cap value if we're sampling and not in scaled space                   
                if not scaled_pol_params:
                    new_cart_pitch_stiffness = np.clip(new_cart_pitch_stiffness, 5, 600)           
                    new_params[-1] = int(new_cart_pitch_stiffness)  
                    print('clipped sampled new_cart_pitch_stiffness', new_cart_pitch_stiffness)      
                
                new_z_force = 'NA'   
        
        elif cut_type == 'pivchop':                    
            if S[2] == 0:  
                new_z_weights = initial_wts[2,:,:] # not actually using this in this case b/c no position control in z
                new_z_force = new_params[0]
                new_cart_pitch_stiffness = new_params[1]
                # cap force value
                if not scaled_pol_params:
                    new_z_force = np.clip(new_z_force, -40, -3)    # TODO: up force to -50N        
                    new_params[0] = int(new_z_force)  
                    print('clipped sampled z force', new_z_force)  
                    new_cart_pitch_stiffness = np.clip(new_cart_pitch_stiffness, 5, 600)           
                    new_params[1] = int(new_cart_pitch_stiffness)  
                    print('clipped sampled new_cart_pitch_stiffness', new_cart_pitch_stiffness)  
            
            elif S[2] == 1:
                new_z_weights = new_params[0:-1]
                new_cart_pitch_stiffness = new_params[-1]
                # cap value
                if not scaled_pol_params:
                    new_cart_pitch_stiffness = np.clip(new_cart_pitch_stiffness, 5, 600)           
                    new_params[-1] = int(new_cart_pitch_stiffness)  
                    print('clipped sampled new_cart_pitch_stiffness', new_cart_pitch_stiffness)
                new_z_force = 'NA'        

        return new_params
    
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

    
    