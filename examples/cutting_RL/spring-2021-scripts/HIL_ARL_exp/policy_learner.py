import gym
from rl_utils import reps
import numpy as np 
import matplotlib.pyplot as plt 

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

    # def initialize_linear_policy_params(self, num_params, initial_var, scaling_factor): #Initialize using a single Gaussian with random mean
    #     '''
    #     not using anymore - only used for randomly initializing policy params 
    #     '''
    #     #import pdb; pdb.set_trace()
    #     initial_pol_mean = [(np.random.rand()*2-1) * scaling_factor, (np.random.rand()*2-1) * scaling_factor, \
    #         np.abs((np.random.rand()*2-1) *scaling_factor)]

    #     initial_var = np.repeat(initial_var, num_params)         
    #     initial_pol_cov = np.diag(initial_var)
    #     return initial_pol_mean, initial_pol_cov

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
    
    