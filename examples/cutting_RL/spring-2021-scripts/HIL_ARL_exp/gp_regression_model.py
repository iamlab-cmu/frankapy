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
        self.length_scale_initial = 4 
        self.signal_var_initial = 4 
        self.num_features = 7 # update this if adding more reward features
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims = self.num_features))

        # initialize lengthscale and outputscale hyperparams
        self.covar_module.base_kernel.lengthscale = self.length_scale_initial
        self.covar_module.outputscale = self.signal_var_initial                   

    def forward(self, x):
        # input to GP kernel is  feature vector
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x) 
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    





