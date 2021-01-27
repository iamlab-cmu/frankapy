'''To do: refactor code so there's a base rewardlearner class and the other ones inherit from this class
'''

import gym
from rl_utils import reps
import numpy as np 
import matplotlib.pyplot as plt 
import sklearn.gaussian_process as gp
from scipy.stats import multivariate_normal
import sklearn
from .policy_learner_blockGrasp import REPS_policy_learner
from carbongym_utils.policy import GraspBlockPolicy_ARLTest
import copy 

import sys
import os
sys.path.append(os.path.abspath("/home/test2/Documents/carbongym-utils/carbongym_utils/robot_interface_utils"))
import pprint
import torch.nn as nn
from torchvision.transforms import functional as F
import torch.optim as optim
import gpytorch
import torch
import time
import json
import copy
from tqdm import tqdm
from sklearn.metrics import f1_score

from action_relation.trainer.train_voxels_online_contrastive import create_model_with_checkpoint
from action_relation.trainer.train_voxels_online_contrastive import create_voxel_trainer_with_checkpoint

from vae.config.base_config import BaseVAEConfig
from vae.trainer.base_train import BaseVAETrainer

from carbongym_utils.voxelData_utils import load_eigvec, project_original_data_to_PC_space, use_k_PCs_from_projected_data, get_lower_dimens_obj_embed


class GPRegressionModel_MultBlocks(gpytorch.models.ExactGP):
    '''
    modified forward method to accomodate multiple blocks in scene
    '''
    def __init__(self, train_x, train_y, likelihood):
        super(GPRegressionModel_MultBlocks, self).__init__(train_x, train_y, likelihood)
        print('creating GPRModel')      
        self.use_embeds_as_features = True  # whether or not to use embeddings as input to GP
        self.use_lower_dim_embed = False # whether or not to use PCA embeds 

        self.mean_module = gpytorch.means.ZeroMean()
        signal_var_initial = 2        
        
        if self.use_lower_dim_embed == True: # use lower dimensional PCA embedd
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims = 50)) 
        else: 
            if self.use_embeds_as_features == True: # use full embedd
                self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims = 128))
            else: # use baseline xyz as features
                #self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims = 6))
                self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims = 3))
          
        checkpoint_path ='/home/test2/Documents/obj-rel-embeddings/emb-checkpoints/checkpoint/cp_24000.pth'
        emb_trainer = create_voxel_trainer_with_checkpoint(checkpoint_path, cuda = 1)    
        embedd_extractor = emb_trainer.model
        self.embedd_extractor = embedd_extractor 

         # initialize lengthscale and outputscale kernel hyperparameters
        self.covar_module.outputscale = signal_var_initial
        # try initializing lengthscale based on SD of embed data
        if self.use_embeds_as_features == True:
            #temp_x = train_x.reshape([train_x.shape[0], 2, 3, 50, 50, 50]) # TODO: update this to work w/ new shape
            #temp_proj_x = self.embedd_extractor.forward_image(temp_x, relu_on_emb=True) 
            # initialize length scale based on std dev of embeddings from data
            #length_scale_initial = (torch.mean(torch.std(temp_proj_x,axis=0))*100).detach().numpy()
            length_scale_initial = 3 # TODO: this is temporary, need to update this 
            self.covar_module.base_kernel.lengthscale = length_scale_initial

    def forward(self, x):
        # TODO: update this to work with multiple pairwise voxel data inputs(from 3 blocks in scene)
        '''
        first putting our data through a deep net (feature extractor)       
        x is voxel data shape (n, 3, 50, 50, 50), projected_x is 128 embedding vector
        '''
        #print('in forward method of GPRModel')  
        num_samples = x.shape[0]

        if self.use_embeds_as_features == True:
            #import pdb; pdb.set_trace()
            x = x.reshape([num_samples, 2, 3, 50, 50, 50])
            # get the embeddings separately and sum?
            # TODO: need to figure this out! - summing and stacking
            projected_x = torch.empty(0,128)
            for i in range(num_samples):
                single_pair_voxels = x[i, :, :, :, :, :]
                projected_x_temp = self.embedd_extractor.forward_image(single_pair_voxels, relu_on_emb=True) 
                # sum the two embeddings
                summed_projected_x_temp = torch.unsqueeze(torch.sum(projected_x_temp,axis=0),0) 
                projected_x = torch.cat((projected_x, summed_projected_x_temp), 0)         
            
        else: # use baseline xyz positions of both objects as input features
            projected_x = x
            #import pdb; pdb.set_trace()

        # input to GP kernel is outputted embedd
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# define the GPR model
class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        print('creating GPRModel')      
        self.use_embeds_as_features = True  # whether or not to use embeddings as input to GP
        self.use_lower_dim_embed = False # whether or not to use PCA embeds 

        self.mean_module = gpytorch.means.ZeroMean()
        signal_var_initial = 2        
        
        if self.use_lower_dim_embed == True: # use lower dimensional PCA embedd
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims = 50)) 
        else: 
            if self.use_embeds_as_features == True: # use full embedd
                self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims = 128))
            else: # use baseline xyz as features
                #self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims = 6))
                self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims = 3))
          
        checkpoint_path ='/home/test2/Documents/obj-rel-embeddings/emb-checkpoints/checkpoint/cp_24000.pth'
        emb_trainer = create_voxel_trainer_with_checkpoint(checkpoint_path, cuda = 1)    
        embedd_extractor = emb_trainer.model
        self.embedd_extractor = embedd_extractor 

         # initialize lengthscale and outputscale kernel hyperparameters
        self.covar_module.outputscale = signal_var_initial
        # try initializing lengthscale based on SD of embed data
        if self.use_embeds_as_features == True:
            temp_x = train_x.reshape([train_x.shape[0], 3, 50, 50, 50])
            temp_proj_x = self.embedd_extractor.forward_image(temp_x, relu_on_emb=True) 
            # initialize length scale based on std dev of embeddings from data
            length_scale_initial = (torch.mean(torch.std(temp_proj_x,axis=0))*100).detach().numpy()
            self.covar_module.base_kernel.lengthscale = length_scale_initial


    def forward(self, x):
        '''
        first putting our data through a deep net (feature extractor)       
        x is voxel data shape (n, 3, 50, 50, 50), projected_x is 128 embedding vector
        '''
        #print('in forward method of GPRModel')  
        num_samples = x.shape[0]

        if self.use_embeds_as_features == True:
            x = x.reshape([num_samples, 3, 50, 50, 50])
            projected_x = self.embedd_extractor.forward_image(x, relu_on_emb=True) 
            
            # project embeds down to lower dimensional PCA space if self.use_lower_dim_embed = True:
            if self.use_lower_dim_embed == True:
                eigvec_filepath = '/home/test2/Documents/obj-rel-embeddings/AS_data/emb_deep_dive/PCA/obj_emb_eig_vecs.txt'
                projected_x = get_lower_dimens_obj_embed(eigvec_filepath, projected_x.detach().numpy(), num_PCs=50) 
                projected_x = torch.from_numpy(projected_x)
                #import pdb; pdb.set_trace()

        else: # use baseline xyz positions of both objects as input features
            projected_x = x
            #import pdb; pdb.set_trace()

        # input to GP kernel is outputted embedd
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    # def reset_embed_model_to_initial_network(self):
    #     checkpoint_path ='/home/test2/Documents/obj-rel-embeddings/emb-checkpoints/checkpoint/cp_24000.pth'
    #     emb_trainer = create_voxel_trainer_with_checkpoint(checkpoint_path, cuda = 1)    
    #     embedd_extractor = emb_trainer.model
    #     self.embedd_extractor = embedd_extractor 
       

class RewardLearnerObjEmbedDKL:
    '''Reward_Learner for ARL with object embeddings using GPytorch for GP implementation and DKL
    '''
    def __init__(self, kappa):
        self.kappa = kappa #EPD sampling threshold
        self.mult_blocks = None
        '''
        TODO: instantiate GRP reward model here?
        '''
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
        #import pdb; pdb.set_trace()
        sampled_params_pi_tilda = np.array(sampled_params_pi_tilda)
        sampled_params_pi_star = np.array(sampled_params_pi_star)
        sampled_params_pi_current = np.array(sampled_params_pi_current)
        #import pdb; pdb.set_trace()

        pi_star_wi = multivariate_normal.pdf(sampled_params_pi_star, mean=pi_star_mean, cov=pi_star_cov, allow_singular=True)
        pi_tilda_wi = multivariate_normal.pdf(sampled_params_pi_tilda, mean=pi_tilda_mean, cov=pi_tilda_cov, allow_singular=True)
        pi_current_wi = multivariate_normal.pdf(sampled_params_pi_current, mean=pi_current_mean, cov=pi_current_cov, allow_singular=True)

        star_div_current = (pi_star_wi/pi_current_wi)
        star_div_tilda = (pi_star_wi/pi_tilda_wi)
        
        #import pdb; pdb.set_trace()        
        approx_sampling_KLdiv = (1/num_samples)*np.sum(star_div_current*np.log(star_div_tilda))

        #import pdb; pdb.set_trace()
        return approx_sampling_KLdiv
    
    def select_initial_hyperparams(self, update_embed, train_x, train_y, lr):
        print('selecting initial kernel hyperparameters')        
        '''
        # these are hyperparam ranges for block stacking task w/ block sizes being the same!!
        sigVar_all = [2, 2.5, 3, 3.5, 4, 4.5, 5]               
        ls_all = [4, 5, 6]
        '''
        # block stacking task w/ multiple block sizes
        sigVar_all = [0.25, 0.5, 1, 1.25]               
        ls_all = [1, 2, 3]

        best_loss = np.inf
        initial_sigVar, initial_ls = None, None
        for i in range(0,len(sigVar_all)):
            for j in range(0, len(ls_all)):
                temp_likelihood = gpytorch.likelihoods.GaussianLikelihood()  
                if self.mult_blocks == 'True':
                    temp_gpr_reward_model =  GPRegressionModel_MultBlocks(train_x, train_y, temp_likelihood) 
                else:
                    temp_gpr_reward_model = GPRegressionModel(train_x, train_y, temp_likelihood) 

                # set kernel hyperparams
                temp_gpr_reward_model.covar_module.base_kernel.lengthscale = ls_all[j]
                temp_gpr_reward_model.covar_module.outputscale = sigVar_all[i]

                if update_embed == 'no':
                    temp_optimizer = torch.optim.Adam([
                        {'params': temp_gpr_reward_model.covar_module.parameters()},
                        {'params': temp_gpr_reward_model.mean_module.parameters()},
                        {'params': temp_gpr_reward_model.likelihood.parameters()},
                    ], lr=lr)

                else:
                    temp_optimizer = torch.optim.Adam([
                        {'params': temp_gpr_reward_model.embedd_extractor.parameters()},
                        {'params': temp_gpr_reward_model.covar_module.parameters()},
                        {'params': temp_gpr_reward_model.mean_module.parameters()},
                        {'params': temp_gpr_reward_model.likelihood.parameters()},
                    ], lr=lr)  # lr originally 0.01 

                temp_mll = gpytorch.mlls.ExactMarginalLogLikelihood(temp_likelihood, temp_gpr_reward_model)
                temp_gpr_reward_model.train()
                temp_likelihood.train()
                
                # train for 3 epochs and pick hyperparams with lowest loss after 3 epochs
                for epoch in range(0,4):
                    # Zero backprop gradients
                    temp_optimizer.zero_grad()                    
                    # Get output from model
                    output = temp_gpr_reward_model(train_x) #output.mean and output.variance returns mean and var of model
                    # Calc loss and backprop derivatives
                    loss = -temp_mll(output, train_y)
                    print('loss', loss)
                    loss.backward()
                    temp_optimizer.step()
                if loss < best_loss:
                    best_loss = loss
                    initial_sigVar = sigVar_all[i]
                    initial_ls = ls_all[j]            
            print('best_loss', best_loss)
        #import pdb; pdb.set_trace()
        print('initial_sigVar', initial_sigVar)
        print('initial_ls', initial_ls)

        return initial_sigVar, initial_ls
    
    def train_with_LOO_crossVal(self, update_embed, train_x, train_y, lr, sigVar, ls):
        '''
        determine how long to train initial model for (goal is to reduce overfitting).
        train for x epochs --> perform LOO cross val and calc mean cross val loss --> continue training and 
        performing cross eval 
        return num_epochs to train for (w/ lowest cross eval loss)
        '''
        print('performing LOO cross val')
        temp_likelihood = gpytorch.likelihoods.GaussianLikelihood()  
        temp_gpr_reward_model = GPRegressionModel(train_x, train_y, temp_likelihood) 
        # set kernel hyperparams
        temp_gpr_reward_model.covar_module.base_kernel.lengthscale = ls
        temp_gpr_reward_model.covar_module.outputscale = sigVar

        if update_embed == 'no':
            temp_optimizer = torch.optim.Adam([
                {'params': temp_gpr_reward_model.covar_module.parameters()},
                {'params': temp_gpr_reward_model.mean_module.parameters()},
                {'params': temp_gpr_reward_model.likelihood.parameters()},
            ], lr=lr) 

        else:             
            temp_optimizer = torch.optim.Adam([
                {'params': temp_gpr_reward_model.embedd_extractor.parameters()},
                {'params': temp_gpr_reward_model.covar_module.parameters()},
                {'params': temp_gpr_reward_model.mean_module.parameters()},
                {'params': temp_gpr_reward_model.likelihood.parameters()},
            ], lr=lr)  # lr originally 0.01 

        temp_mll = gpytorch.mlls.ExactMarginalLogLikelihood(temp_likelihood, temp_gpr_reward_model)
        
        mean_cross_val_loss = {} # dict with key:value - num_epochs training: mean cross val loss
        for epoch in range(0, 21):
            # train on full x/y dataset
            temp_gpr_reward_model.set_train_data(inputs = train_x, targets = train_y, strict = False)
            temp_gpr_reward_model.train()
            temp_likelihood.train()
            # Zero backprop gradients
            temp_optimizer.zero_grad()                    
            # Get output from model
            output = temp_gpr_reward_model(train_x) #output.mean and output.variance returns mean and var of model
            # Calc loss and backprop derivatives
            loss = -temp_mll(output, train_y)
            print('loss', loss)
            loss.backward()
            temp_optimizer.step()

            # perform cross-val every x epochs
            cross_val_loss_all = []
            if epoch %5 == 0 and epoch !=0:
                # remove 1 sample at a time from train set 
                for sample in range(0, train_x.shape[0]):
                    #import pdb; pdb.set_trace()
                    train_x_removed1 = torch.from_numpy(np.delete(train_x.numpy(), sample, 0))
                    train_y_removed1 = torch.from_numpy(np.delete(train_y.numpy(), sample, 0))
                    temp_gpr_reward_model.set_train_data(inputs = train_x_removed1, targets = train_y_removed1, strict = False)

                    # evaluate model on sample removed from training set
                    temp_gpr_reward_model.eval()
                    temp_likelihood.eval()
                    #print('evaluating model')
                    with torch.no_grad(), gpytorch.settings.use_toeplitz(False):  
                        test_sample = train_x[sample, :]    
                        if len(test_sample.shape) == 1:
                            test_sample = torch.unsqueeze(test_sample, 0)

                        preds = temp_gpr_reward_model(test_sample)
                        #print('model predicted output', preds.mean.numpy())
                        #print('ground truth output', train_y[sample].numpy())
                        cross_val_loss = np.abs(preds.mean.numpy() - train_y[sample].numpy())
                        #print('cross val loss', cross_val_loss)
                        cross_val_loss_all.append(cross_val_loss)

                #print('mean cross val loss ', np.mean(cross_val_loss_all))
                mean_cross_val_loss[epoch] = np.mean(cross_val_loss_all)
                print('mean_cross_val_loss dict', mean_cross_val_loss)
                #import pdb; pdb.set_trace()


        # select num_epochs to train with lowest cross val loss
        key_list = list(mean_cross_val_loss.keys())
        val_list = list(mean_cross_val_loss.values())
        # plot cross val loss vs epochs
        plt.plot(key_list, val_list)
        plt.title('mean leave-one-out cross valid loss vs training epochs')
        plt.xlabel('num training epochs')
        plt.ylabel('mean LOO cross valid loss')
        plt.show()
        min_cross_val_loss = min(mean_cross_val_loss.values())
        num_epochs_to_train_model = key_list[val_list.index(min_cross_val_loss)]

        #import pdb; pdb.set_trace()
        print('num_epochs_to_train_model', num_epochs_to_train_model)
        return num_epochs_to_train_model

    def determine_num_epochs_to_update_model_with_LOO_crossVal(self, optimizer, model, likelihood, \
        mll, updated_train_x, updated_train_y):        
        temp_gpr_reward_model = copy.deepcopy(model)
        temp_likelihood = copy.deepcopy(likelihood)
        temp_optimizer = copy.deepcopy(optimizer) 
        temp_mll = copy.deepcopy(mll)     

        mean_cross_val_loss = {} # dict with key:value - num_epochs training: mean cross val loss
        for epoch in range(0, 6):
            if epoch!=0:
                temp_gpr_reward_model.set_train_data(inputs = updated_train_x, targets = updated_train_y, strict = False)
            # train on full x/y dataset
            temp_gpr_reward_model.train()
            temp_likelihood.train()
            # Zero backprop gradients
            temp_optimizer.zero_grad()                    
            # Get output from model
            output = temp_gpr_reward_model(updated_train_x) #output.mean and output.variance returns mean and var of model
            # Calc loss and backprop derivatives
            loss = -temp_mll(output, updated_train_y)
            print('loss', loss)
            loss.backward()
            temp_optimizer.step()

            # perform cross-val every x epochs
            cross_val_loss_all = []            
            # remove 1 sample at a time from train set 
            for sample in range(0, updated_train_x.shape[0]):
                import pdb; pdb.set_trace()
                train_x_removed1 = torch.from_numpy(np.delete(updated_train_x.numpy(), sample, 0))
                train_y_removed1 = torch.from_numpy(np.delete(updated_train_y.numpy(), sample, 0))
                temp_gpr_reward_model.set_train_data(inputs = train_x_removed1, targets = train_y_removed1, strict = False)

                # evaluate model on sample removed from training set
                temp_gpr_reward_model.eval()
                temp_likelihood.eval()
                #print('evaluating model')
                with torch.no_grad(), gpytorch.settings.use_toeplitz(False):  
                    test_sample = updated_train_x[sample, :]    
                    if len(test_sample.shape) == 1:
                        test_sample = torch.unsqueeze(test_sample, 0)

                    preds = temp_gpr_reward_model(test_sample)
                    #print('model predicted output', preds.mean.numpy())
                    #print('ground truth output', train_y[sample].numpy())
                    cross_val_loss = np.abs(preds.mean.numpy() - updated_train_y[sample].numpy())
                    #print('cross val loss', cross_val_loss)
                    cross_val_loss_all.append(cross_val_loss)

            #print('mean cross val loss ', np.mean(cross_val_loss_all))
            mean_cross_val_loss[epoch] = np.mean(cross_val_loss_all)
            print('mean_cross_val_loss dict', mean_cross_val_loss)

        # select num_epochs to train with lowest cross val loss
        key_list = list(mean_cross_val_loss.keys())
        val_list = list(mean_cross_val_loss.values())
        # plot cross val loss vs epochs
        plt.plot(key_list, val_list)
        plt.title('mean leave-one-out cross valid loss vs training epochs')
        plt.xlabel('num training epochs')
        plt.ylabel('mean LOO cross valid loss')
        plt.show()
        min_cross_val_loss = min(mean_cross_val_loss.values())
        num_epochs_to_continue_training_GPmodel = key_list[val_list.index(min_cross_val_loss)]
        import pdb; pdb.set_trace()
        return num_epochs_to_continue_training_GPmodel

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
                np.savetxt('/home/test2/Documents/obj-rel-embeddings/AS_data/test_DKL_GP/Isaac_exps/epoch%i_numTrainSamples%i.txt'%(epoch, train_x.shape[0]), covmat_np)

            optimizer.step() #updates lengthscale, signal variance, AND g NN weights
        print('updated lengthscale: ', model.covar_module.base_kernel.lengthscale)
        print('updated outputscale: ', model.covar_module.outputscale)
        print('updated covariance matrix', output.lazy_covariance_matrix.evaluate())
        print('done training')
        return model

    def calc_expected_reward_for_observed_outcome_w_GPmodel(self, model, likelihood, new_outcomes):
        # convert new_outcomes data to torch tensor
        if type(new_outcomes)==np.ndarray:
            new_outcomes = torch.from_numpy(new_outcomes)
            new_outcomes = new_outcomes.float()

        # if new_outcomes is [n,3,50,50,50] or [n,2,3,50,50,50], reshape to nxD
        #import pdb; pdb.set_trace()
        if len(new_outcomes.shape) > 2:
            new_outcomes = self.reshape_voxel_data_to_2D(new_outcomes)

        model.eval()
        likelihood.eval()
        print('evaluating model')
        mean_expected_rewards, var_expected_rewards =[], [] 
        with torch.no_grad(), gpytorch.settings.use_toeplitz(False):              
            preds = model(new_outcomes)
            #import pdb; pdb.set_trace()
            mean_expected_rewards = preds.mean.numpy().tolist()
            var_expected_rewards = preds.variance.numpy().tolist()

        return mean_expected_rewards, var_expected_rewards
    
    def compute_EPD_for_each_sample_updated(self, num_training_epochs, optimizer, current_reward_model, likelihood, mll, \
            agent, pi_tilda_mean, pi_tilda_cov, pi_current_mean, pi_current_cov, prior_training_data, \
                queried_samples_all, GP_training_data_x_all, GP_training_data_y_all, beta):

        agent = REPS_policy_learner() 
        prior_training_data_expect_rewards_mean, prior_training_data_policy_params, \
            prior_training_data_expect_rewards_sig = [], [], []
        
        if self.mult_blocks == 'True':
            prior_training_data_o = np.empty([0,2,3,50,50,50])

        else:
            prior_training_data_o = np.empty([0,3,50,50,50])
        #prior_training_data_o = np.empty([0,6])
        #prior_training_data_o = np.empty([0,3])
        
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
        print('KL divs', KL_div_all)
        print('median KL DIV', np.median(KL_div_all))
        samples_to_query_new = self.remove_already_queried_samples_from_list(samples_to_query,\
            queried_samples_all)
        print('new samples_to_query', samples_to_query_new)
        #import pdb; pdb.set_trace()
        queried_outcomes_arr = prior_training_data_o[samples_to_query_new]              

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
        
        if len(updated_train_x.shape) > 2:
            updated_train_x = self.reshape_voxel_data_to_2D(updated_train_x)
        
        model.set_train_data(inputs = updated_train_x, targets = updated_train_y, strict = False)
        
        # TODO: only continue to train if num queried samples is large (i/e dont update if only a few updates)
        if continue_training == True:
            # determine how many epochs to contine training for based on LOO cross-val loss            
            # epochs_to_cont_training = self.determine_num_epochs_to_update_model_with_LOO_crossVal(optimizer, \
            #     model, likelihood, mll, updated_train_x, updated_train_y)
            # print('epochs_to_cont_training', epochs_to_cont_training)
            
            #import pdb; pdb.set_trace()
            epochs_to_cont_training = num_training_epochs
            
            if epochs_to_cont_training != 0:
                model = self.train_GPmodel(epochs_to_cont_training, optimizer, model, likelihood, mll, updated_train_x, updated_train_y)
        
        return model

    def reshape_voxel_data_to_2D(self, input_voxel_data):
        num_samples = input_voxel_data.shape[0]

        if self.mult_blocks == 'False':
            input_voxel_data = input_voxel_data.reshape([num_samples, 375000])
        else: #multiple blocks in scene w/ 2 embeds for each samples
            input_voxel_data = input_voxel_data.reshape([num_samples, 750000])
        return input_voxel_data

class RewardLearnerObjEmbedNNModel(RewardLearnerObjEmbedDKL):    
    '''
    this class inherits from RewardLearnerObjEmbedDKL, updating compute EPD method to work for NN policy (not just linear policy w/ REPS)
    '''
    def __init__(self, kappa):
        super().__init__(kappa)

    def compute_EPD_for_each_sample_updated(self, num_training_epochs, optimizer, current_reward_model, likelihood, mll, \
            agent, pi_tilda_mean, pi_tilda_cov, pi_current_mean, pi_current_cov, prior_training_data, \
                queried_samples_all, GP_training_data_x_all, GP_training_data_y_all, beta):
        
        return None


class RewardLearnerObjEmbed:
    '''
    NOTE: this class is deprecated. No longer using sklearn GaussianProcessRegressor implementation. Switched to gpytorch implementation.
    Reward_Learner for ARL with object embeddings
    '''
    def __init__(self, lambd, beta, kappa):
        self.lambd = lambd #tuning parameter that affects how often agent queries the expert for reward
        self.beta = beta #noise added to observations (to model human imprecision)
        self.kappa = kappa #EPD sampling threshold

    def remove_already_queried_samples_from_list(self, samples_to_query, queried_samples_all):
        num_samples = len(samples_to_query)
        samples_to_query_new = copy.deepcopy(samples_to_query)
        
        for i in range(0, num_samples):
            #import pdb; pdb.set_trace()
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

    def compute_KL_div_sampling(self, agent, num_samples, pi_tilda_mean, pi_tilda_cov, pi_star_mean, pi_star_cov): #taking samples from policies pi_star and pi_tilda
        #print('computing numerical sampling-based KL_div')
        sampled_params_pi_tilda = []
        sampled_params_pi_star = []
        #import pdb; pdb.set_trace()
        for i in range(0, num_samples):
            sampled_params_pi_tilda.append(agent.sample_params_from_policy(pi_tilda_mean, pi_tilda_cov))
            sampled_params_pi_star.append(agent.sample_params_from_policy(pi_star_mean, pi_star_cov))

        sampled_params_pi_tilda = np.array(sampled_params_pi_tilda)
        sampled_params_pi_star = np.array(sampled_params_pi_star)
        #import pdb; pdb.set_trace() 

        div = (sampled_params_pi_star/sampled_params_pi_tilda)
        div[div<=0]=.01
        approx_sampling_KLdiv = (1/num_samples)*np.sum(div*np.log(div))

        return approx_sampling_KLdiv

    def initialize_reward_model(self, signal_var_initial, length_scale_initial):
        print('initializing reward model')
         #initialize GP with zero mean prior 
        #kernel = gp.kernels.RBF(length_scale_initial, (1e-3, 1e3))

        kernel = gp.kernels.ConstantKernel(signal_var_initial, (1, 20)) \
            * gp.kernels.RBF(length_scale_initial, (1, 150)) #20
        #import pdb; pdb.set_trace()
        gpr_reward_model = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=30, alpha=0.0001)
        #n_restarts_optimizer = 15 originally
        
        #import pdb; pdb.set_trace()
        return gpr_reward_model
        
    def compute_EPD_for_each_sample(self, agent, pi_tilda_mean, pi_tilda_cov, current_reward_model,\
        prior_training_data, queried_samples_all): #output o+ = argmax u(o)
        '''
        prior_training_data is a list of all training data, with each samples containing: 
        [outcomes_from_sample, mean_expected_reward, var_expected_reward, sampled_new_params]
        '''
        agent = REPS_policy_learner() 
        
        prior_training_data_o, prior_training_data_expect_rewards_mean, \
            prior_training_data_policy_params, prior_training_data_expect_rewards_sig = [], [], [], []
                       
        for i in range(len(prior_training_data)):
            prior_training_data_o.append(prior_training_data[i][0])
            prior_training_data_expect_rewards_mean.append(prior_training_data[i][1])
            prior_training_data_expect_rewards_sig.append(prior_training_data[i][2])
            prior_training_data_policy_params.append(prior_training_data[i][3])

        prior_training_data_o = np.squeeze(np.array(prior_training_data_o))
        prior_training_data_expect_rewards_mean = np.array(prior_training_data_expect_rewards_mean)
        prior_training_data_expect_rewards_sig = np.array(prior_training_data_expect_rewards_sig)
        prior_training_data_policy_params = np.array(prior_training_data_policy_params)

        #import pdb; pdb.set_trace()

        #prior_training_data_o = prior_training_data[:,0]
        #prior_training_data_expect_rewards_mean = prior_training_data[:,1]
        #prior_training_data_expect_rewards_sig = prior_training_data[:,2]
        #prior_training_data_policy_params = prior_training_data[:, 3]
        
        num_samples = len(prior_training_data)       
        #import pdb; pdb.set_trace()
        samples_to_query = []
        KL_div_all = []
        for i in range(0, num_samples):
            outcome = prior_training_data_o[i]
            mean_expect_reward = prior_training_data_expect_rewards_mean[i]
            sigma_expect_reward = prior_training_data_expect_rewards_sig[i]            

            sigma_pt_1 = mean_expect_reward + sigma_expect_reward
            sigma_pt_2 = mean_expect_reward - sigma_expect_reward

            #### SHOULD be using sigma points to estimate UPDATED reward model!!!!! ()
            outcomes_to_update = np.array([outcome, outcome])
            rewards_to_update = np.array([sigma_pt_1, sigma_pt_2])

            if len(outcomes_to_update.shape)==3:
                outcomes_to_update = np.squeeze(outcomes_to_update)
            
            #updating hypoth_reward_model for this sample instead of actual model           
            hypoth_reward_model = copy.deepcopy(current_reward_model)
            
            #update hypoth reward model with this outcome
            #import pdb; pdb.set_trace()
            hypoth_reward_model = self.update_reward_model(hypoth_reward_model, outcomes_to_update, rewards_to_update)
            
            #calculate rewards for training data under updated reward model                   
            #import pdb; pdb.set_trace()
            mean_exp_rewards, var_exp_rewards = \
                self.calc_expected_reward_for_an_observed_outcome(agent, \
                    hypoth_reward_model, prior_training_data_o)      
            
            #Calculate policy update under updated reward model            
            pi_star_mean, pi_star_cov, reps_wts = agent.update_policy_REPS(mean_exp_rewards, \
                prior_training_data_policy_params, rel_entropy_bound=1.5, min_temperature=0.001) 
            #import pdb; pdb.set_trace()
           
            KL_div = self.compute_KL_div_sampling(agent, \
                10, pi_tilda_mean, pi_tilda_cov, pi_star_mean, pi_star_cov)
            
            print('KLdiv_sampling', KL_div)
            KL_div_all.append(KL_div)

            #Calculate KL-diverg b/w the two policies
            # KL_div = self.compute_kl_divergence(pi_star_mean, pi_star_cov, pi_tilda_mean, pi_tilda_cov)
             
            if (np.all(np.isnan(KL_div)==True))==False and np.any(KL_div >= self.kappa):
                samples_to_query.append(i)
        
        #Check if we've already queried these samples. If yes, remove from list:
        print('median KL DIV', np.median(KL_div_all))
        samples_to_query_new = self.remove_already_queried_samples_from_list(samples_to_query,\
            queried_samples_all)
        print('new samples_to_query', samples_to_query_new)
        #import pdb; pdb.set_trace()

        queried_outcomes_arr = prior_training_data_o[samples_to_query_new]       
        #import pdb; pdb.set_trace()
        #queried_outcomes_arr = self.convert_list_outcomes_to_array(queried_outcomes)

        import pdb; pdb.set_trace()
        '''Outputs:
        samples_to_query_new: list of indices of samples to query from full training dataset
        queried_outcomes_arr: nxD array of outcomes to query (where n is num samples, D is size of outcome data) 
        '''
        return samples_to_query_new, queried_outcomes_arr #indexes of samples to query from expert


    def calc_expected_reward_for_an_observed_outcome(self, agent, gpr_reward_model, new_outcomes): #provided to policy learner
        mean_expected_rewards, var_expected_rewards =[], []        
        #import pdb; pdb.set_trace()
        if type(new_outcomes[0])==np.float64: #single new outcome (1x5 list)
            #print('outcome', new_outcomes)
            X=np.atleast_2d(new_outcomes)
            mean_expected_reward, var_expected_reward = gpr_reward_model.predict(X, return_std=True)
            mean_expected_rewards.append(mean_expected_reward[0])
            var_expected_rewards.append(var_expected_reward[0])
        
        else: #list of new outcomes            
            #import pdb; pdb.set_trace()
            for outcome in new_outcomes:
                #print('outcome = ', outcome)
                X=np.atleast_2d(outcome)
                mean_expected_reward, var_expected_reward = gpr_reward_model.predict(X, return_std=True)
                #import pdb; pdb.set_trace
                if len(mean_expected_reward.shape) == 2:
                    mean_expected_rewards.append(mean_expected_reward[0][0])
                else:
                    mean_expected_rewards.append(mean_expected_reward[0])
                var_expected_rewards.append(var_expected_reward[0])

        return mean_expected_rewards, var_expected_rewards

    def update_reward_model(self, gpr_reward_model, outcomes, rewards): #update GP for reward model p(R|o,D)
        outcomes = np.atleast_2d(outcomes)
        rewards = np.atleast_2d(rewards).T
        #import pdb; pdb.set_trace()
        gpr_reward_model.fit(outcomes, rewards) #fit Gaussian process regression model to data
        return gpr_reward_model



class Reward_Learner:
    '''Reward_Learner for original block grasping task
    '''
    def __init__(self, lambd, beta, kappa):
        self.lambd = lambd #tuning parameter that affects how often agent queries the expert for reward
        self.beta = beta #noise added to observations (to model human imprecision)
        self.kappa = kappa #EPD sampling threshold

    def remove_already_queried_samples_from_list(self, samples_to_query, queried_samples_all):
        num_samples = len(samples_to_query)
        samples_to_query_new = copy.deepcopy(samples_to_query)
        
        for i in range(0, num_samples):
            #import pdb; pdb.set_trace()
            if samples_to_query[i] in queried_samples_all:
                samples_to_query_new.remove(samples_to_query[i])
        return samples_to_query_new

    def convert_list_outcomes_to_array(self, outcomes_list):
        num_samples = len(outcomes_list)
        num_features = len(outcomes_list[0])
        outcomes_arr = np.zeros((num_samples, num_features))
        for i in range(0,len(outcomes_list)):
            outcomes_arr[i,:] = np.array(outcomes_list[i])
        return outcomes_arr


    def compute_kl_divergence(self, pm, pv, qm, qv):    
        """
        Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
        Also computes KL divergence from a single Gaussian pm,pv to a set
        of Gaussians qm,qv.
        Diagonal covariances are assumed.  Divergence is expressed in nats.
        """
        if (len(qm.shape) == 2):
            axis = 1
        else:
            axis = 0
        # Determinants of diagonal covariances pv, qv
        dpv = pv.prod()
        dqv = qv.prod(axis)
        # Inverse of diagonal covariance qv
        iqv = 1./qv
        # Difference between means pm, qm
        diff = qm - pm

        #import pdb; pdb.set_trace()
        return (0.5 *(np.log(dqv / dpv) + (iqv * pv).sum(axis) + (diff * iqv * diff).sum(axis) - len(pm))) 

    def compute_KL_div_sampling(self, agent, num_samples, pi_tilda_mean, pi_tilda_cov, pi_star_mean, pi_star_cov): #taking samples from policies pi_star and pi_tilda
        #print('computing numerical sampling-based KL_div')
        sampled_params_pi_tilda = []
        sampled_params_pi_star = []
        #import pdb; pdb.set_trace()
        for i in range(0, num_samples):
            sampled_params_pi_tilda.append(agent.sample_params_from_policy(pi_tilda_mean, pi_tilda_cov))
            sampled_params_pi_star.append(agent.sample_params_from_policy(pi_star_mean, pi_star_cov))

        sampled_params_pi_tilda = np.array(sampled_params_pi_tilda)
        sampled_params_pi_star = np.array(sampled_params_pi_star)
        #import pdb; pdb.set_trace() 

        div = (sampled_params_pi_star/sampled_params_pi_tilda)
        div[div<=0]=.01
        approx_sampling_KLdiv = (1/num_samples)*np.sum(div*np.log(div))

        return approx_sampling_KLdiv

    def initialize_reward_model(self, signal_var_initial, length_scale_initial):
        print('initializing reward model')
         #initialize GP with zero mean prior 
        #kernel = gp.kernels.RBF(length_scale_initial, (1e-3, 1e3))

        kernel = gp.kernels.ConstantKernel(signal_var_initial, (1, 20)) \
            * gp.kernels.RBF(length_scale_initial, (40, 150))
        #import pdb; pdb.set_trace()
        gpr_reward_model = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15, alpha=0.0001)
        #import pdb; pdb.set_trace()
        return gpr_reward_model
        
    def compute_EPD_for_each_sample(self, agent, pi_tilda_mean, pi_tilda_cov, current_reward_model,\
        prior_training_data, queried_samples_all): #output o+ = argmax u(o)
        agent = REPS_policy_learner() 
        
        prior_training_data = np.array(prior_training_data)
        prior_training_data_o = prior_training_data[:,0]
        prior_training_data_expect_rewards_mean = prior_training_data[:,1]
        prior_training_data_expect_rewards_sig = prior_training_data[:,2]
        prior_training_data_policy_params = prior_training_data[:, 3]
        
        num_samples = len(prior_training_data)       
        #import pdb; pdb.set_trace()
        samples_to_query = []
        for i in range(0, num_samples):
            outcome = prior_training_data_o[i]
            mean_expect_reward = prior_training_data_expect_rewards_mean[i]
            sigma_expect_reward = prior_training_data_expect_rewards_sig[i]            

            sigma_pt_1 = mean_expect_reward + sigma_expect_reward
            sigma_pt_2 = mean_expect_reward - sigma_expect_reward

            #### SHOULD be using sigma points to estimate UPDATED reward model!!!!! ()
            outcomes_to_update = np.array([outcome, outcome])
            rewards_to_update = np.array([sigma_pt_1, sigma_pt_2])
            
            #updating hypoth_reward_model for this sample instead of actual model           
            hypoth_reward_model = copy.deepcopy(current_reward_model)

            
            #update hypoth reward model with this outcome
            hypoth_reward_model = self.update_reward_model(hypoth_reward_model, outcomes_to_update, rewards_to_update)
            
            #calculate rewards for training data under updated reward model                   
            #import pdb; pdb.set_trace()
            mean_exp_rewards, var_exp_rewards = \
                self.calc_expected_reward_for_an_observed_outcome(agent, \
                    hypoth_reward_model, prior_training_data_o)      
            
            #Calculate policy update under updated reward model
            pi_star_mean, pi_star_cov, reps_wts = agent.update_policy_REPS(mean_exp_rewards, \
                prior_training_data_policy_params, rel_entropy_bound=1.5, min_temperature=0.001) 
            
           
            KL_div = self.compute_KL_div_sampling(agent, \
                10, pi_tilda_mean, pi_tilda_cov, pi_star_mean, pi_star_cov)
            
            print('KLdiv_sampling', KL_div)
            
            #Calculate KL-diverg b/w the two policies
            # KL_div = self.compute_kl_divergence(pi_star_mean, pi_star_cov, pi_tilda_mean, pi_tilda_cov)
             
            if (np.all(np.isnan(KL_div)==True))==False and np.any(KL_div >= self.kappa):
                samples_to_query.append(i)
        
        #Check if we've already queried these samples. If yes, remove from list:
        samples_to_query_new = self.remove_already_queried_samples_from_list(samples_to_query,\
            queried_samples_all)
        print('new samples_to_query', samples_to_query_new)
        #import pdb; pdb.set_trace()

        queried_outcomes = prior_training_data_o[samples_to_query_new]       
        queried_outcomes_arr = self.convert_list_outcomes_to_array(queried_outcomes)

        return samples_to_query_new, queried_outcomes_arr #indexes of samples to query from expert


    def calc_expected_reward_for_an_observed_outcome(self, agent, gpr_reward_model, new_outcomes): #provided to policy learner
        mean_expected_rewards, var_expected_rewards =[], []        
        #import pdb; pdb.set_trace()
        if type(new_outcomes[0])==np.float64: #single new outcome (1x5 list)
            #print('outcome', new_outcomes)
            X=np.atleast_2d(new_outcomes)
            mean_expected_reward, var_expected_reward = gpr_reward_model.predict(X, return_std=True)
            mean_expected_rewards.append(mean_expected_reward[0])
            var_expected_rewards.append(var_expected_reward[0])
        
        else: #list of new outcomes
            
            #import pdb; pdb.set_trace()
            for outcome in new_outcomes:
                #print('outcome = ', outcome)
                X=np.atleast_2d(outcome)
                mean_expected_reward, var_expected_reward = \
                    gpr_reward_model.predict(X, return_std=True)
                
                mean_expected_rewards.append(mean_expected_reward[0][0])
                var_expected_rewards.append(var_expected_reward[0])

        return mean_expected_rewards, var_expected_rewards

    def update_reward_model(self, gpr_reward_model, outcomes, rewards): #update GP for reward model p(R|o,D)
        outcomes = np.atleast_2d(outcomes)
        rewards = np.atleast_2d(rewards).T
        #import pdb; pdb.set_trace()
        gpr_reward_model.fit(outcomes, rewards) #fit Gaussian process regression model to data
        return gpr_reward_model

    #def plot_GRP_reward_function(self,):

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
