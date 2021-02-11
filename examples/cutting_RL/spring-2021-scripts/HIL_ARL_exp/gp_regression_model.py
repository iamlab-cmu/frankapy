# code for implementing GP and DKL w/ gpytorch 

import math
import torch
import gpytorch
from matplotlib import pyplot as plt

import numpy as np
import argparse
import pickle
import h5py
import sys
import os
import pprint
import torch.nn as nn
import torch.optim as optim
import time
import json
import copy
from tqdm import tqdm
from sklearn.metrics import f1_score

                                  
# define the GPR model
class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        print('creating GPRModel')    
        self.train_x = train_x
        self.train_y = train_y
         # define likelihood
        self.likelihood = likelihood  

        self.mean_module = gpytorch.means.ZeroMean()
        length_scale_initial = 4 
        signal_var_initial = 4 
        self.num_features = 7 # TODO: update this if adding more reward features
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims = self.num_features))

        # initialize lengthscale and outputscale hyperparams
        self.covar_module.base_kernel.lengthscale = length_scale_initial
        self.covar_module.outputscale = signal_var_initial                   

    def forward(self, x):
        #print('in forward method of GPRModel')      
        # input to GP kernel is  feature vector
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x) 
        #print('exiting forward method of GPRModel')
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
# def train(num_epochs, model):
#     print('begin training')  
#     # Use adam optimizer
#     optimizer = torch.optim.Adam([           
#         {'params': model.covar_module.parameters()},
#         {'params': model.mean_module.parameters()},
#         {'params': model.likelihood.parameters()},
#     ], lr=0.01)

#     # # "Loss" for GPs - the marginal log likelihood
#     mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
        
#     for epoch in range(num_epochs):
#         print('epoch', epoch)
#         # Zero backprop gradients
#         optimizer.zero_grad()
#         # Get output from model
#         output = model(train_x) #output.mean and output.variance returns mean and var of model
#         # Calc loss and backprop derivatives
#         loss = -mll(output, train_y)    
#         print('loss', loss)
#         loss.backward()
#         print('Epoch%d, Loss:%.3f, lengthscale:%.3f, scale:%.3f' % (epoch, loss.item(),model.covar_module.base_kernel.lengthscale.item(),model.covar_module.outputscale.item()))
#         optimizer.step() #updates lengthscale, signal variance, AND g NN weights
#     print('done training')

#     return model

# def evaluate(model, test_x):
#     model.eval()
#     model.likelihood.eval()
#     print('evaluating model')
#     with torch.no_grad(), gpytorch.settings.use_toeplitz(False):  
#         preds = model(test_x)
#         observed_pred = model.likelihood(model(test_x))
    
#     print('pred mean', preds.mean)
#     print('pred var', preds.variance)

#     return preds.mean, preds.variance





