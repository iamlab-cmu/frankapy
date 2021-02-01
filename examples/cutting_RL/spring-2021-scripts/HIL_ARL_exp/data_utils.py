import numpy as np
import matplotlib.pyplot as plt 
import os
import glob

# def calc_mean_std_reward_features(cut_type, save_mean_std):
#     '''
#     save_mean_std: True, False
#     cut_type: normal, pivchop, scoring
#     '''
#     base_dir = '/home/sony/Documents/cutting_RL_experiments/data/Jan-2021-LL-param-exps/' + cut_type + '/'
#     food_types = ['potato', 'celery', 'carrot', 'banana', 'tomato', 'mozz']
#     reward_feats_all_foods = np.empty((0,7))
#     for food in food_types:
#         #import pdb; pdb.set_trace()
#         work_dir = base_dir + food + '/'
#         exp_folder = os.listdir(work_dir)
#         work_dir = work_dir + exp_folder[0] + '/'

#         reward_feats = np.load(work_dir + '/reward_features_all_samples.npy')
#         # import pdb; pdb.set_trace()
#         if len(np.where(reward_feats==np.inf)[0])!= 0: # there are inf's in array
#             inf_inds = np.unique(np.where(reward_feats==np.inf)[0])
#             last_inf_ind = inf_inds.max()
#             reward_feats = reward_feats[last_inf_ind+1:]
#         reward_feats_all_foods = np.vstack((reward_feats_all_foods, reward_feats))
    
#     #import pdb; pdb.set_trace()
#     reward_feats_mean = np.mean(reward_feats_all_foods, axis=0)
#     reward_feats_std = np.std(reward_feats_all_foods, axis=0)

#     if save_mean_std:
#         np.save(base_dir + '/' + cut_type + '_reward_feats_mean_std' + '.npy', np.array([reward_feats_mean, reward_feats_std]))
    
#     return reward_feats_mean, reward_feats_std

# def standardize_reward_feature(cut_type, current_reward_feat):
#     base_dir = '/home/sony/Documents/cutting_RL_experiments/data/Jan-2021-LL-param-exps/' + cut_type + '/'
#     mean_std_reward_feats = np.load(base_dir + '/' + cut_type + '_reward_feats_mean_std' + '.npy')
#     mean_reward_feats = mean_std_reward_feats[0,:]
#     std_reward_feats = mean_std_reward_feats[1,:]
#     #import pdb; pdb.set_trace()
#     standardized_reward_feat = (current_reward_feat - mean_reward_feats)/std_reward_feats

#     return standardized_reward_feat

# def unstandardize_reward_feature(cut_type, standardized_reward_feat):
#     base_dir = '/home/sony/Documents/cutting_RL_experiments/data/Jan-2021-LL-param-exps/' + cut_type + '/'
#     mean_std_reward_feats = np.load(base_dir + '/' + cut_type + '_reward_feats_mean_std' + '.npy')
#     mean_reward_feats = mean_std_reward_feats[0,:]
#     std_reward_feats = mean_std_reward_feats[1,:]

#     unstandardized_reward_feat = (standardized_reward_feat*std_reward_feats) + mean_reward_feats

#     return unstandardized_reward_feat


# save_mean_std = False
# cut_type = 'scoring'
# reward_feats_mean, reward_feats_std = calc_mean_std_reward_features(cut_type, save_mean_std)
# import pdb; pdb.set_trace()

