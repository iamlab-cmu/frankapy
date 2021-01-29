

from rl_utils import reps
import numpy as np 
import matplotlib.pyplot as plt 
import sklearn.gaussian_process as gp
from scipy.stats import multivariate_normal
import sklearn
import copy 

import sys
import os
import pprint
import torch.nn as nn
# from torchvision.transforms import functional as F
import torch.optim as optim
import gpytorch
import torch
import time
import json
import copy
from tqdm import tqdm
from sklearn.metrics import f1_score

     
class RewardLearner:
    '''Reward_Learner for ARL using GPytorch for GP implementation 
    '''
    def __init__(self, kappa):
        self.kappa = kappa #EPD sampling threshold

    def plot_rewardModel_vs_oracle_rewards(self, reward_model_rewards_all_mean_buffer, automated_expert_rewards_all):
        plt.plot(reward_model_rewards_all_mean_buffer)
        plt.plot(automated_expert_rewards_all)
        plt.title('expert rewards vs reward model rewards')
        plt.legend(('GP reward model rewards', 'oracle expert rewards'))
        plt.show()

    def remove_already_queried_samples_from_list(self, samples_to_query, queried_samples_all):
        num_samples = len(samples_to_query)
        samples_to_query_new = copy.deepcopy(samples_to_query)        
        for i in range(0, num_samples):
            if samples_to_query[i] in queried_samples_all:
                samples_to_query_new.remove(samples_to_query[i])
        return samples_to_query_new

    def convert_list_outcomes_to_array(self, outcomes_list):
        import pdb; pdb.set_trace()
        num_samples = len(outcomes_list)
        num_features = len(outcomes_list[0])
        outcomes_arr = np.zeros((num_samples, num_features))
        for i in range(0,len(outcomes_list)):
            outcomes_arr[i,:] = np.array(outcomes_list[i])
        return outcomes_arr

    def compute_KL_div_sampling_updated(self, agent, num_samples, pi_tilda_mean, pi_tilda_cov, \
        pi_star_mean, pi_star_cov, pi_current_mean, pi_current_cov): #taking samples from policies pi_star and pi_tilda
        sampled_params_pi_tilda, sampled_params_pi_star, sampled_params_pi_current = [], [], []
        # sample params from each of the three policies
        for i in range(0, num_samples):
            sampled_params_pi_tilda.append(agent.sample_params_from_policy(pi_tilda_mean, pi_tilda_cov))
            sampled_params_pi_star.append(agent.sample_params_from_policy(pi_star_mean, pi_star_cov))
            sampled_params_pi_current.append(agent.sample_params_from_policy(pi_current_mean, pi_current_cov))

        sampled_params_pi_tilda = np.array(sampled_params_pi_tilda)
        sampled_params_pi_star = np.array(sampled_params_pi_star)
        sampled_params_pi_current = np.array(sampled_params_pi_current)

        pi_star_wi = multivariate_normal.pdf(sampled_params_pi_star, mean=pi_star_mean, cov=pi_star_cov, allow_singular=True)
        pi_tilda_wi = multivariate_normal.pdf(sampled_params_pi_tilda, mean=pi_tilda_mean, cov=pi_tilda_cov, allow_singular=True)
        pi_current_wi = multivariate_normal.pdf(sampled_params_pi_current, mean=pi_current_mean, cov=pi_current_cov, allow_singular=True)

        star_div_current = (pi_star_wi/pi_current_wi)
        star_div_tilda = (pi_star_wi/pi_tilda_wi)
        
        approx_sampling_KLdiv = (1/num_samples)*np.sum(star_div_current*np.log(star_div_tilda))

        #import pdb; pdb.set_trace()
        return approx_sampling_KLdiv
        
    def train_GPmodel(self, num_epochs, optimizer, model, likelihood, mll, train_x, train_y):
        print('training model')    
        model.train()
        likelihood.train()
        for epoch in range(num_epochs):
            # Zero backprop gradients
            optimizer.zero_grad()
            # Get output from model
            output = model(train_x) #output.mean and output.variance returns mean and var of model
            # Calc loss and backprop derivatives
            loss = -mll(output, train_y)        
            loss.backward()

            # only print this info if we're training/updating GP model, not when computing EPD
            if num_epochs > 1:
                print('Epoch%d, Loss:%.3f, scale:%.3f' % (epoch, loss.item(), model.covar_module.outputscale.item()))
                # print updated covar matrix 
                #if epoch % 10 == 0:
                print('updated covariance matrix', output.lazy_covariance_matrix.evaluate())
                print('model noise', model.likelihood.noise.item())
                # save updated covariance_matrix
                covmat_np = output.lazy_covariance_matrix.evaluate().detach().numpy()
                # np.savetxt('/home/test2/Documents/obj-rel-embeddings/AS_data/test_DKL_GP/Isaac_exps/epoch%i_numTrainSamples%i.txt'%(epoch, train_x.shape[0]), covmat_np)

            optimizer.step() #updates lengthscale, signal variance, AND g NN weights
        print('updated lengthscale: ', model.covar_module.base_kernel.lengthscale)
        print('updated outputscale: ', model.covar_module.outputscale)
        print('updated covariance matrix', output.lazy_covariance_matrix.evaluate())
        print('done training')

        self.num_reward_features = model.num_features
        return model

    def calc_expected_reward_for_observed_outcome_w_GPmodel(self, model, likelihood, new_outcomes):
        # convert new_outcomes data to torch tensor
        if type(new_outcomes)==np.ndarray:
            new_outcomes = torch.from_numpy(new_outcomes)
            new_outcomes = new_outcomes.float()
        model.eval()
        likelihood.eval()
        print('evaluating model')
        mean_expected_rewards, var_expected_rewards =[], [] 
        with torch.no_grad(), gpytorch.settings.use_toeplitz(False):              
            preds = model(new_outcomes)
            mean_expected_rewards = preds.mean.numpy().tolist()
            var_expected_rewards = preds.variance.numpy().tolist()
        
        import pdb; pdb.set_trace()
        return mean_expected_rewards, var_expected_rewards
    
    def compute_EPD_for_each_sample_updated(self, num_training_epochs, optimizer, current_reward_model, likelihood, mll, \
            agent, pi_tilda_mean, pi_tilda_cov, pi_current_mean, pi_current_cov, prior_training_data, \
                queried_samples_all, GP_training_data_x_all, GP_training_data_y_all, beta):

        agent = REPS_policy_learner() 
        
        prior_training_data_expect_rewards_mean, prior_training_data_policy_params, \
            prior_training_data_expect_rewards_sig = [], [], []
        
        prior_training_data_o = np.empty([0,self.num_reward_features])
        
        for i in range(len(prior_training_data)):
            prior_training_data_o = np.vstack((prior_training_data_o, prior_training_data[i][0]))
            prior_training_data_expect_rewards_mean.append(prior_training_data[i][1])
            prior_training_data_expect_rewards_sig.append(prior_training_data[i][2])
            prior_training_data_policy_params.append(prior_training_data[i][3])      
       
        prior_training_data_expect_rewards_mean = np.array(prior_training_data_expect_rewards_mean)
        prior_training_data_expect_rewards_sig = np.array(prior_training_data_expect_rewards_sig)
        prior_training_data_policy_params = np.array(prior_training_data_policy_params)

        '''
        prior_training_data_o: nx3x50x50x50 arr
        prior_training_data_expect_rewards_mean: (n,) arr
        prior_training_data_expect_rewards_sig: (n,) arr
        prior_training_data_policy_params: nx3 arr
        '''

        num_samples = len(prior_training_data)       
        samples_to_query, KL_div_all = [], []
        # iterate through all samples (i.e. outcomes) in training data set
        for i in range(0, num_samples):       
            # TODO - don't iterate through all samples, skip already queried and ones from previous rollouts (?)     
            if i in queried_samples_all: # total_samples - samples_in_current_epoch
                continue            
            else:
                outcome = np.expand_dims(prior_training_data_o[i,:],axis=0)
                #outcome = np.expand_dims(prior_training_data_o[i,:,:,:,:],axis=0)
                mean_expect_reward = prior_training_data_expect_rewards_mean[i]
                sigma_expect_reward = prior_training_data_expect_rewards_sig[i]            

                sigma_pt_1 = mean_expect_reward + sigma_expect_reward
                sigma_pt_2 = mean_expect_reward - sigma_expect_reward

                #### SHOULD be using sigma points to estimate UPDATED reward model!
                # TODO: should be updating reward model separately for each sigma pt??
                outcomes_to_update = np.vstack((outcome, outcome))
                rewards_to_update = np.array([sigma_pt_1, sigma_pt_2])
                
                #updating hypoth_reward_model for this sample instead of actual model           
                hypoth_reward_model = copy.deepcopy(current_reward_model)
                hypoth_likelihood = copy.deepcopy(likelihood)
                hypoth_optimizer = copy.deepcopy(optimizer) #TODO: need to redefine this using hypoth_reward_model?
                hypoth_mll = copy.deepcopy(mll)            
                
                # GP_training_data_x_all and GP_training_data_y_all are previous training data for 
                og_train_x = copy.deepcopy(GP_training_data_x_all)
                og_train_y = copy.deepcopy(GP_training_data_y_all)            
                updated_train_x = np.vstack((og_train_x, outcomes_to_update))
                updated_train_y = np.concatenate((og_train_y, rewards_to_update))            
                
                # NOTE: MIGHT need to update likelihood here b/c of added noise params 
                #hypoth_likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(torch.ones(updated_train_x.shape[0]) * beta)

                #update hypoth reward model with this outcome
                continue_training = False
                hypoth_reward_model = self.update_reward_GPmodel(continue_training, num_training_epochs, hypoth_optimizer, hypoth_reward_model, hypoth_likelihood, hypoth_mll, updated_train_x, updated_train_y)
                
                #calculate rewards for training data under updated reward model                   
                mean_exp_rewards, var_exp_rewards = self.calc_expected_reward_for_observed_outcome_w_GPmodel(hypoth_reward_model, \
                    hypoth_likelihood, prior_training_data_o)
                
                #Calculate policy update under updated reward model            
                pi_star_mean, pi_star_cov, reps_wts = agent.update_policy_REPS(mean_exp_rewards, \
                    prior_training_data_policy_params, rel_entropy_bound=1.5, min_temperature=0.001) 
                
                # note - 40 is higher # samples (used to be 10)
                KL_div = self.compute_KL_div_sampling_updated(agent, 40, pi_tilda_mean, pi_tilda_cov, \
                    pi_star_mean, pi_star_cov, pi_current_mean, pi_current_cov)
                
                print('KLdiv_sampling', KL_div)
                KL_div_all.append(KL_div)           
                
                if (np.all(np.isnan(KL_div)==True))==False and np.any(KL_div >= self.kappa):
                    samples_to_query.append(i)

        #Check if we've already queried these samples. If yes, remove from list:
        import pdb; pdb.set_trace()
        print('KL divs', KL_div_all)
        print('median KL DIV', np.median(KL_div_all))
        samples_to_query_new = self.remove_already_queried_samples_from_list(samples_to_query,\
            queried_samples_all)
        print('new samples_to_query', samples_to_query_new)
        import pdb; pdb.set_trace()
        queried_outcomes_arr = prior_training_data_o[samples_to_query_new]              
        import pdb; pdb.set_trace()
        return samples_to_query_new, queried_outcomes_arr #indexes of samples to query from expert
    
    def update_reward_GPmodel(self, continue_training, num_training_epochs, optimizer, model, likelihood, mll, updated_train_x, updated_train_y):
        # if updated_train data are np arrays, convert to torch float tensors
        #import pdb; pdb.set_trace()
        if type(updated_train_x)==np.ndarray:
            updated_train_x = torch.from_numpy(updated_train_x)
            updated_train_x = updated_train_x.float()

        if type(updated_train_y)==np.ndarray:
            updated_train_y = torch.from_numpy(updated_train_y)
            updated_train_y = updated_train_y.float()
        
        model.set_train_data(inputs = updated_train_x, targets = updated_train_y, strict = False)

        epochs_to_cont_training = num_training_epochs        
        if epochs_to_cont_training != 0:
            model = self.train_GPmodel(epochs_to_cont_training, optimizer, model, likelihood, mll, updated_train_x, updated_train_y)
        
        return model

   
# class Reward_Learner:
#     '''Reward_Learner for original block grasping task
#     '''
#     def __init__(self, lambd, beta, kappa):
#         self.lambd = lambd #tuning parameter that affects how often agent queries the expert for reward
#         self.beta = beta #noise added to observations (to model human imprecision)
#         self.kappa = kappa #EPD sampling threshold

#     def remove_already_queried_samples_from_list(self, samples_to_query, queried_samples_all):
#         num_samples = len(samples_to_query)
#         samples_to_query_new = copy.deepcopy(samples_to_query)
        
#         for i in range(0, num_samples):
#             #import pdb; pdb.set_trace()
#             if samples_to_query[i] in queried_samples_all:
#                 samples_to_query_new.remove(samples_to_query[i])
#         return samples_to_query_new

#     def convert_list_outcomes_to_array(self, outcomes_list):
#         num_samples = len(outcomes_list)
#         num_features = len(outcomes_list[0])
#         outcomes_arr = np.zeros((num_samples, num_features))
#         for i in range(0,len(outcomes_list)):
#             outcomes_arr[i,:] = np.array(outcomes_list[i])
#         return outcomes_arr


#     def compute_kl_divergence(self, pm, pv, qm, qv):    
#         """
#         Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
#         Also computes KL divergence from a single Gaussian pm,pv to a set
#         of Gaussians qm,qv.
#         Diagonal covariances are assumed.  Divergence is expressed in nats.
#         """
#         if (len(qm.shape) == 2):
#             axis = 1
#         else:
#             axis = 0
#         # Determinants of diagonal covariances pv, qv
#         dpv = pv.prod()
#         dqv = qv.prod(axis)
#         # Inverse of diagonal covariance qv
#         iqv = 1./qv
#         # Difference between means pm, qm
#         diff = qm - pm

#         #import pdb; pdb.set_trace()
#         return (0.5 *(np.log(dqv / dpv) + (iqv * pv).sum(axis) + (diff * iqv * diff).sum(axis) - len(pm))) 

#     def compute_KL_div_sampling(self, agent, num_samples, pi_tilda_mean, pi_tilda_cov, pi_star_mean, pi_star_cov): #taking samples from policies pi_star and pi_tilda
#         #print('computing numerical sampling-based KL_div')
#         sampled_params_pi_tilda = []
#         sampled_params_pi_star = []
#         #import pdb; pdb.set_trace()
#         for i in range(0, num_samples):
#             sampled_params_pi_tilda.append(agent.sample_params_from_policy(pi_tilda_mean, pi_tilda_cov))
#             sampled_params_pi_star.append(agent.sample_params_from_policy(pi_star_mean, pi_star_cov))

#         sampled_params_pi_tilda = np.array(sampled_params_pi_tilda)
#         sampled_params_pi_star = np.array(sampled_params_pi_star)
#         #import pdb; pdb.set_trace() 

#         div = (sampled_params_pi_star/sampled_params_pi_tilda)
#         div[div<=0]=.01
#         approx_sampling_KLdiv = (1/num_samples)*np.sum(div*np.log(div))

#         return approx_sampling_KLdiv

#     def initialize_reward_model(self, signal_var_initial, length_scale_initial):
#         print('initializing reward model')
#          #initialize GP with zero mean prior 
#         #kernel = gp.kernels.RBF(length_scale_initial, (1e-3, 1e3))

#         kernel = gp.kernels.ConstantKernel(signal_var_initial, (1, 20)) \
#             * gp.kernels.RBF(length_scale_initial, (40, 150))
#         #import pdb; pdb.set_trace()
#         gpr_reward_model = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15, alpha=0.0001)
#         #import pdb; pdb.set_trace()
#         return gpr_reward_model
        
#     def compute_EPD_for_each_sample(self, agent, pi_tilda_mean, pi_tilda_cov, current_reward_model,\
#         prior_training_data, queried_samples_all): #output o+ = argmax u(o)
#         agent = REPS_policy_learner() 
        
#         prior_training_data = np.array(prior_training_data)
#         prior_training_data_o = prior_training_data[:,0]
#         prior_training_data_expect_rewards_mean = prior_training_data[:,1]
#         prior_training_data_expect_rewards_sig = prior_training_data[:,2]
#         prior_training_data_policy_params = prior_training_data[:, 3]
        
#         num_samples = len(prior_training_data)       
#         #import pdb; pdb.set_trace()
#         samples_to_query = []
#         for i in range(0, num_samples):
#             outcome = prior_training_data_o[i]
#             mean_expect_reward = prior_training_data_expect_rewards_mean[i]
#             sigma_expect_reward = prior_training_data_expect_rewards_sig[i]            

#             sigma_pt_1 = mean_expect_reward + sigma_expect_reward
#             sigma_pt_2 = mean_expect_reward - sigma_expect_reward

#             #### SHOULD be using sigma points to estimate UPDATED reward model!!!!! ()
#             outcomes_to_update = np.array([outcome, outcome])
#             rewards_to_update = np.array([sigma_pt_1, sigma_pt_2])
            
#             #updating hypoth_reward_model for this sample instead of actual model           
#             hypoth_reward_model = copy.deepcopy(current_reward_model)

            
#             #update hypoth reward model with this outcome
#             hypoth_reward_model = self.update_reward_model(hypoth_reward_model, outcomes_to_update, rewards_to_update)
            
#             #calculate rewards for training data under updated reward model                   
#             #import pdb; pdb.set_trace()
#             mean_exp_rewards, var_exp_rewards = \
#                 self.calc_expected_reward_for_an_observed_outcome(agent, \
#                     hypoth_reward_model, prior_training_data_o)      
            
#             #Calculate policy update under updated reward model
#             pi_star_mean, pi_star_cov, reps_wts = agent.update_policy_REPS(mean_exp_rewards, \
#                 prior_training_data_policy_params, rel_entropy_bound=1.5, min_temperature=0.001) 
            
           
#             KL_div = self.compute_KL_div_sampling(agent, \
#                 10, pi_tilda_mean, pi_tilda_cov, pi_star_mean, pi_star_cov)
            
#             print('KLdiv_sampling', KL_div)
            
#             #Calculate KL-diverg b/w the two policies
#             # KL_div = self.compute_kl_divergence(pi_star_mean, pi_star_cov, pi_tilda_mean, pi_tilda_cov)
             
#             if (np.all(np.isnan(KL_div)==True))==False and np.any(KL_div >= self.kappa):
#                 samples_to_query.append(i)
        
#         #Check if we've already queried these samples. If yes, remove from list:
#         samples_to_query_new = self.remove_already_queried_samples_from_list(samples_to_query,\
#             queried_samples_all)
#         print('new samples_to_query', samples_to_query_new)
#         #import pdb; pdb.set_trace()

#         queried_outcomes = prior_training_data_o[samples_to_query_new]       
#         queried_outcomes_arr = self.convert_list_outcomes_to_array(queried_outcomes)

#         return samples_to_query_new, queried_outcomes_arr #indexes of samples to query from expert


#     def calc_expected_reward_for_an_observed_outcome(self, agent, gpr_reward_model, new_outcomes): #provided to policy learner
#         mean_expected_rewards, var_expected_rewards =[], []        
#         #import pdb; pdb.set_trace()
#         if type(new_outcomes[0])==np.float64: #single new outcome (1x5 list)
#             #print('outcome', new_outcomes)
#             X=np.atleast_2d(new_outcomes)
#             mean_expected_reward, var_expected_reward = gpr_reward_model.predict(X, return_std=True)
#             mean_expected_rewards.append(mean_expected_reward[0])
#             var_expected_rewards.append(var_expected_reward[0])
        
#         else: #list of new outcomes
            
#             #import pdb; pdb.set_trace()
#             for outcome in new_outcomes:
#                 #print('outcome = ', outcome)
#                 X=np.atleast_2d(outcome)
#                 mean_expected_reward, var_expected_reward = \
#                     gpr_reward_model.predict(X, return_std=True)
                
#                 mean_expected_rewards.append(mean_expected_reward[0][0])
#                 var_expected_rewards.append(var_expected_reward[0])

#         return mean_expected_rewards, var_expected_rewards

#     def update_reward_model(self, gpr_reward_model, outcomes, rewards): #update GP for reward model p(R|o,D)
#         outcomes = np.atleast_2d(outcomes)
#         rewards = np.atleast_2d(rewards).T
#         #import pdb; pdb.set_trace()
#         gpr_reward_model.fit(outcomes, rewards) #fit Gaussian process regression model to data
#         return gpr_reward_model

#     #def plot_GRP_reward_function(self,):

    def plot_kernel_length_scale(self,epoch,gpr_reward_model):
        #print('kernel length scale', gpr_reward_model.kernel_.get_params()['length_scale'])
        length_scale = gpr_reward_model.kernel_.get_params()['k2__length_scale']
        signal_var = gpr_reward_model.kernel_.get_params()['k1__constant_value']
        print('kernel length scale', length_scale)
        print('kernel signal variance', signal_var)

        #plt.figure()c
        plt.scatter(epoch, length_scale, color='green')
        #plt.title('kernel length scale vs epochs')
        #plt.xlabel('num epochs')
        #plt.ylabel('length scale')
        #plt.axis([0, 21, 0, 150])
        plt.pause(0.05)

    def plot_rewards_all_episodes(self,training_data_list, kappa):
        plt.plot(np.array(training_data_list)[:,1])
        plt.plot(np.array(training_data_list)[:,3])
        plt.xlabel('total episodes')
        plt.ylabel('rewards')
        plt.title('rewards vs. episodes, kappa = %f'%kappa)
        plt.legend(('mean_expected_reward from reward learner model', 'expert reward'))
        plt.show()

    def plot_mean_rewards_each_epoch(self, epoch,mean_reward_model_rewards_all_epochs,mean_expert_rewards_all_epochs,kappa):
        plt.plot(np.arange(epoch), mean_reward_model_rewards_all_epochs)
        plt.plot(np.arange(epoch), mean_expert_rewards_all_epochs)
        plt.xlabel('epochs')
        plt.ylabel('rewards')
        plt.title('rewards vs. epochs, kappa = %f'%kappa)
        plt.legend(('mean_expected_reward from reward learner model', 'mean expert reward'))
        plt.show()

    def plot_cumulative_queried_samples_vs_epochs(self, epoch,total_queried_samples_each_epoch,kappa):
        plt.plot(np.arange(epoch), total_queried_samples_each_epoch)
        plt.xlabel('epochs')
        plt.ylabel('total (cumululative) queried samples')
        plt.title('total (cumulative) queried samples vs. epochs, kappa = %f'%kappa)
        plt.show()
