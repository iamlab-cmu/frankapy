'''OLD CODE - moved these functions to reward_learner.py
'''
import numpy as np
import matplotlib.pyplot as plt 
import os
import glob

def plot_analytical_human_GPmodel_rewards_one_file(pol_param_data_filepath):
    data = np.load(pol_param_data_filepath)
    plt.figure()
    plt.plot(data[:,-3])
    plt.plot(data[:,-2])
    plt.plot(data[:,-1])
    plt.legend(('analytical reward','GP reward model reward','human reward'))
    plt.xlabel('sample')
    plt.ylabel('reward')
    plt.title('scoring-tomato-HIL-RL, comparison of rewards')

    plt.figure()
    plt.plot(data[:,-2])
    plt.plot(data[:,-1])
    plt.legend(('GP reward model reward','human reward'))
    plt.xlabel('sample')
    plt.ylabel('reward')
    plt.title('scoring-tomato-HIL-RL, comparison of rewards')

    plt.show()

def plot_analytical_human_GPmodel_rewards_all_prev_epochs(work_dir, desired_cutting_behavior):
    data_files = glob.glob(work_dir + "*epoch_*.npy")
    num_prev_epochs = len(data_files)
    # get num policy params
    first_file = np.load(data_files[0])

    analyt_rews_all_epochs = np.empty((0))
    GP_rews_all_epochs = np.empty((0))
    human_rews_all_epochs = np.empty((0))
    avg_GP_rews_each_epoch = []
    avg_human_rews_each_epoch = []
    num_samples_each_epoch = []
    num_samples = 0
    for i in range(num_prev_epochs):
        data_file = glob.glob(work_dir + "*epoch_%s*.npy"%str(i))[0]
        data = np.load(data_file)
        # import pdb; pdb.set_trace()
        if desired_cutting_behavior == 'quality_cut':
            analyt_rews = data[:,-3]
            GP_model_rews = data[:,-2]
            human_rews = data[:,-1]

        elif desired_cutting_behavior == 'slow':
            analyt_rews = data[:,-4]
            GP_model_rews = data[:,-3]
            human_rews = data[:,-2]

        elif desired_cutting_behavior == 'fast':
            analyt_rews = data[:,-4]
            GP_model_rews = data[:,-3]
            human_rews = data[:,-1]

        avg_GP_rews_each_epoch.append(np.mean(GP_model_rews))
        avg_human_rews_each_epoch.append(np.mean(human_rews))

        analyt_rews_all_epochs = np.concatenate((analyt_rews_all_epochs, analyt_rews),axis=0)
        GP_rews_all_epochs = np.concatenate((GP_rews_all_epochs, GP_model_rews),axis=0)
        human_rews_all_epochs = np.concatenate((human_rews_all_epochs, human_rews),axis=0)

        num_samples+=data.shape[0]
        num_samples_each_epoch.append(num_samples)
    
    plt.figure()
    plt.plot(analyt_rews_all_epochs, '-o')
    plt.plot(GP_rews_all_epochs, '-o')
    plt.plot(human_rews_all_epochs, '-o')
    plt.legend(('analytical reward','GP reward model reward','human reward'))
    plt.xlabel('sample')
    plt.ylabel('reward')
    plt.vlines(np.array(num_samples_each_epoch),np.min(analyt_rews_all_epochs)-1,2, colors = ['r','r','r'], linestyles={'dashed', 'dashed', 'dashed'})
    plt.title('scoring-tomato-HIL-RL, comparison of rewards')

    plt.figure()
    plt.plot(GP_rews_all_epochs, '-o')
    plt.plot(human_rews_all_epochs, '-o')
    plt.legend(('GP reward model reward','human reward'))
    plt.xlabel('sample')
    plt.ylabel('reward')
    plt.vlines(np.array(num_samples_each_epoch),np.min(GP_rews_all_epochs)-1,2, colors = ['r','r','r'], linestyles={'dashed', 'dashed', 'dashed'})
    plt.title('scoring-tomato-HIL-RL, comparison of rewards')

    # plot average rewards each epoch
    plt.figure()
    avg_GP_rews_each_epoch = [np.mean(GP_rews_all_epochs[0:25]), np.mean(GP_rews_all_epochs[25:50]), np.mean(GP_rews_all_epochs[50:75]), np.mean(GP_rews_all_epochs[75:95]),np.mean(GP_rews_all_epochs[95:])]
    avg_human_rews_each_epoch = [np.mean(human_rews_all_epochs[0:25]), np.mean(human_rews_all_epochs[25:50]), np.mean(human_rews_all_epochs[50:75]),np.mean(human_rews_all_epochs[75:95]),np.mean(human_rews_all_epochs[95:])]
    plt.plot(avg_GP_rews_each_epoch, '-o', linestyle='dashed', color='blue')
    plt.plot(avg_human_rews_each_epoch, '-o', color='blue')
    plt.xlabel('epoch', fontsize = 22)
    plt.xticks(np.arange(5),fontsize=22)
    plt.ylabel('avg rewards each epoch',fontsize = 22)
    plt.title('avg reward vs epochs',fontsize = 22)
    plt.tick_params(axis='both', which='major', labelsize=22)
    plt.legend(('GP reward model reward','human reward'),fontsize = 22)

    plt.show()
    #import pdb; pdb.set_trace()

def plot_task_success_analyticalRew_vs_HIL(analyt_work_dir, HIL_work_dir):
    task_success_HIL = np.load(HIL_work_dir + '/task_success_all_samples.npy')
    task_success_analyt = np.load(analyt_work_dir + '/task_success_all_samples.npy')

    epochs = [0,1,2,3,4]
    mean_task_success_HIL = [np.mean(task_success_HIL[0:25]), np.mean(task_success_HIL[25:50]), np.mean(task_success_HIL[50:75]), np.mean(task_success_HIL[75:95]),np.mean(task_success_HIL[95:])]
    mean_task_success_analyt = [np.mean(task_success_analyt[0:25]), np.mean(task_success_analyt[25:50]), np.mean(task_success_analyt[50:75]), np.mean(task_success_analyt[75:95]),np.mean(task_success_analyt[95:])]

    plt.figure()
    plt.plot(epochs,mean_task_success_HIL,'-o')
    plt.plot(epochs,mean_task_success_analyt,'-o')
    plt.xlabel('epoch', fontsize=20)
    plt.ylabel('average task success each epoch', fontsize=20)
    plt.ylim([0,2.02])
    plt.xticks(np.arange(5),fontsize=22)
    plt.tick_params(axis="x", labelsize=15)
    plt.tick_params(axis="y", labelsize=15)
    plt.title('average task success (0,1,2) vs. epoch - analytical reward experiment vs. HIL RL - scoring, tomato', fontsize=20)
    plt.legend(('HIL RL', 'analytical reward'), fontsize=20)
    #import pdb; pdb.set_trace()
    perc_succ_HIL = [(25 - float(np.where(np.array(task_success_HIL[0:25])==0)[0].shape[0]))/25, 
        (25 - float(np.where(np.array(task_success_HIL[25:50])==0)[0].shape[0]))/25,
        (25 - float(np.where(np.array(task_success_HIL[50:75])==0)[0].shape[0]))/25,
            (20 - float(np.where(np.array(task_success_HIL[75:95])==0)[0].shape[0]))/20,
                (20 - float(np.where(np.array(task_success_HIL[95:])==0)[0].shape[0]))/20]
    
    perc_succ_analyt = [(25 -  float(np.where(np.array(task_success_analyt[0:25])==0)[0].shape[0]))/25, \
        (25 -  float(np.where(np.array(task_success_analyt[25:50])==0)[0].shape[0]))/25,
        (25 -  float(np.where(np.array(task_success_analyt[50:75])==0)[0].shape[0]))/25,
            (20 -  float(np.where(np.array(task_success_analyt[75:95])==0)[0].shape[0]))/20,
                (20 -  float(np.where(np.array(task_success_analyt[95:])==0)[0].shape[0]))/20]
    
    plt.figure()
    plt.plot(epochs,100*np.array(perc_succ_HIL),'-o')
    plt.plot(epochs,100*np.array(perc_succ_analyt),'-o')
    plt.xlabel('epoch', fontsize=20)
    plt.ylabel('% task success each epoch', fontsize=20)
    plt.ylim([0,102])
    plt.xticks(np.arange(5),fontsize=22)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.title('% task success vs. epoch - analytical reward experiment vs. HIL RL - scoring, tomato', fontsize=20)
    plt.legend(('HIL RL', 'analytical reward'), fontsize=20)

    plt.show()

    print('HIL task success', 100*np.array(perc_succ_HIL))
    print('analyt task success', 100*np.array(perc_succ_analyt))

def plot_task_success_analyticalRew_vs_HIL_mult_HIL_exps(analyt_work_dir, HIL_work_dir1, HIL_work_dir2):
    task_success_HIL_1 = np.load(HIL_work_dir1 + '/task_success_all_samples.npy')
    task_success_HIL_2 = np.load(HIL_work_dir2 + '/task_success_all_samples.npy')
    task_success_analyt = np.load(analyt_work_dir + '/task_success_all_samples.npy')

    epochs = [0,1,2,3,4]
    mean_task_success_HIL_1 = [np.mean(task_success_HIL_1[0:25]), np.mean(task_success_HIL_1[25:50]), np.mean(task_success_HIL_1[50:75]), np.mean(task_success_HIL_1[75:95]),np.mean(task_success_HIL_1[95:])]
    mean_task_success_HIL_2 = [np.mean(task_success_HIL_2[0:25]), np.mean(task_success_HIL_2[25:50]), np.mean(task_success_HIL_2[50:75]), np.mean(task_success_HIL_2[75:95]),np.mean(task_success_HIL_2[95:])]

    mean_task_success_analyt = [np.mean(task_success_analyt[0:25]), np.mean(task_success_analyt[25:50]), np.mean(task_success_analyt[50:75]), np.mean(task_success_analyt[75:95]),np.mean(task_success_analyt[95:])]

    plt.figure()
    plt.plot(epochs,mean_task_success_HIL_1,'-o')
    plt.plot(epochs,mean_task_success_HIL_2,'-o')
    plt.plot(epochs,mean_task_success_analyt,'-o')
    plt.xlabel('epoch', fontsize=20)
    plt.ylabel('average task success each epoch', fontsize=20)
    plt.ylim([0,2.02])
    plt.xticks(np.arange(5),fontsize=22)
    plt.tick_params(axis="x", labelsize=15)
    plt.tick_params(axis="y", labelsize=15)
    plt.title('average task success (0,1,2) vs. epoch - analytical reward vs. HIL RL experiments - scoring, tomato', fontsize=20)
    plt.legend(('HIL RL - kappa = 0.1', 'HIL RL - kappa = 0.3', 'analytical reward'), fontsize=20)
    #import pdb; pdb.set_trace()
    perc_succ_HIL_1 = [(25 - float(np.where(np.array(task_success_HIL_1[0:25])==0)[0].shape[0]))/25, 
        (25 - float(np.where(np.array(task_success_HIL_1[25:50])==0)[0].shape[0]))/25,
        (25 - float(np.where(np.array(task_success_HIL_1[50:75])==0)[0].shape[0]))/25,
            (20 - float(np.where(np.array(task_success_HIL_1[75:95])==0)[0].shape[0]))/20,
                (20 - float(np.where(np.array(task_success_HIL_1[95:])==0)[0].shape[0]))/20]

    perc_succ_HIL_2 = [(25 - float(np.where(np.array(task_success_HIL_2[0:25])==0)[0].shape[0]))/25, 
        (25 - float(np.where(np.array(task_success_HIL_2[25:50])==0)[0].shape[0]))/25,
        (25 - float(np.where(np.array(task_success_HIL_2[50:75])==0)[0].shape[0]))/25,
            (20 - float(np.where(np.array(task_success_HIL_2[75:95])==0)[0].shape[0]))/20,
                (20 - float(np.where(np.array(task_success_HIL_2[95:])==0)[0].shape[0]))/20]
    
    perc_succ_analyt = [(25 -  float(np.where(np.array(task_success_analyt[0:25])==0)[0].shape[0]))/25, \
        (25 -  float(np.where(np.array(task_success_analyt[25:50])==0)[0].shape[0]))/25,
        (25 -  float(np.where(np.array(task_success_analyt[50:75])==0)[0].shape[0]))/25,
            (20 -  float(np.where(np.array(task_success_analyt[75:95])==0)[0].shape[0]))/20,
                (20 -  float(np.where(np.array(task_success_analyt[95:])==0)[0].shape[0]))/20]
    
    plt.figure()
    plt.plot(epochs,100*np.array(perc_succ_HIL_1),'-o')
    plt.plot(epochs,100*np.array(perc_succ_HIL_2),'-o')
    plt.plot(epochs,100*np.array(perc_succ_analyt),'-o')
    plt.xlabel('epoch', fontsize=20)
    plt.ylabel('% task success each epoch', fontsize=20)
    plt.ylim([0,102])
    plt.xticks(np.arange(5),fontsize=22)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.title('% task success vs. epoch - analytical reward vs. HIL RL experiments - scoring, tomato', fontsize=20)
    plt.legend(('HIL RL - kappa = 0.1', 'HIL RL - kappa = 0.3', 'analytical reward'), fontsize=20)

    plt.show()

    #print('HIL task success', 100*np.array(perc_succ_HIL))
    #print('analyt task success', 100*np.array(perc_succ_analyt))
   
# analyt_work_dir = '/home/sony/Documents/cutting_RL_experiments/data/Jan-2021-LL-param-exps/scoring/tomato/exp_18/'
# HIL_work_dir_1 = '/home/sony/Documents/cutting_RL_experiments/data/Jan-2021-HIL-ARL-exps/scoring/tomato/exp_5/'
# HIL_work_dir_2 = '/home/sony/Documents/cutting_RL_experiments/data/Jan-2021-HIL-ARL-exps/scoring/tomato/exp_6/'
# plot_task_success_analyticalRew_vs_HIL_mult_HIL_exps(analyt_work_dir, HIL_work_dir_1, HIL_work_dir_2)


#analyt_work_dir = '/home/sony/Documents/cutting_RL_experiments/data/Jan-2021-LL-param-exps/scoring/tomato/exp_18/'
#HIL_work_dir = '/home/sony/Documents/cutting_RL_experiments/data/Jan-2021-HIL-ARL-exps/scoring/tomato/exp_6/'
#plot_task_success_analyticalRew_vs_HIL(analyt_work_dir, HIL_work_dir)

#work_dir = '/home/sony/Documents/cutting_RL_experiments/data/Jan-2021-HIL-ARL-exps/scoring/tomato/exp_5/'
#desired_cutting_behavior = 'quality_cut'
#plot_analytical_human_GPmodel_rewards_all_prev_epochs(work_dir, desired_cutting_behavior)


#work_dir = '/home/sony/Documents/cutting_RL_experiments/data/Jan-2021-HIL-ARL-exps/scoring/tomato/exp_6/'
#plot_analytical_human_GPmodel_rewards_all_prev_epochs(work_dir, desired_cutting_behavior)
#import pdb; pdb.set_trace()

#pol_param_data_filepath = '/home/sony/Documents/cutting_RL_experiments/data/Jan-2021-HIL-ARL-exps/scoring/tomato/exp_1/all_polParamRew_data/polParamsRews_epoch_1_ep_24.npy'
#plot_analytical_human_GPmodel_rewards_one_file(pol_param_data_filepath)


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
    
#     # for i in range(reward_feats_all_foods.shape[0]):
#     #     plt.plot(reward_feats_all_foods[i,:],'o-')
#     #     plt.xlabel('reward feature dimension num')
#     #     plt.ylabel('reward feature value - unstandardized features (training data)')
#     #     plt.title('distrib of reward feature values for each dimension - unstandardized features (training data)')
#     # plt.show()
        
#     fig, axs = plt.subplots(1, 7, sharey=True, tight_layout=True)
#     for dim in range(7):
#         if dim == 6:
#             axs[dim].hist(reward_feats_all_foods[:,dim], bins=15)
#         else:
#             axs[dim].hist(reward_feats_all_foods[:,dim], bins='auto')
#         axs[dim].set_title('reward feat %i - unstandardized' %dim)
#         axs[dim].set_xlabel('reward feat dim %i value' %dim)
#         axs[dim].set_ylabel('reward feat dim %i frequency' %dim)            
#     plt.show()   
    
#     import pdb; pdb.set_trace()
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
# cut_type = 'normal'
# #reward_feats_mean, reward_feats_std = calc_mean_std_reward_features(cut_type, save_mean_std)
# import pdb; pdb.set_trace()

# '''
# save_mean_std: True, False
# cut_type: normal, pivchop, scoring
# '''
# base_dir = '/home/sony/Documents/cutting_RL_experiments/data/Jan-2021-LL-param-exps/' + cut_type + '/'
# food_types = ['potato', 'celery', 'carrot', 'banana', 'tomato', 'mozz']
# reward_feats_all_foods = np.empty((0,7))
# for food in food_types:
#     #import pdb; pdb.set_trace()
#     work_dir = base_dir + food + '/'
#     exp_folder = os.listdir(work_dir)
#     work_dir = work_dir + exp_folder[0] + '/'

#     reward_feats = np.load(work_dir + '/reward_features_all_samples.npy')
#     # import pdb; pdb.set_trace()
#     if len(np.where(reward_feats==np.inf)[0])!= 0: # there are inf's in array
#         inf_inds = np.unique(np.where(reward_feats==np.inf)[0])
#         last_inf_ind = inf_inds.max()
#         reward_feats = reward_feats[last_inf_ind+1:]
#     reward_feats_all_foods = np.vstack((reward_feats_all_foods, reward_feats))

# standardized_reward_feats_all_food = np.empty((0,7))
# for i in range(reward_feats_all_foods.shape[0]):
#     standardized_reward_feats_all_food = np.vstack((standardized_reward_feats_all_food, standardize_reward_feature(cut_type, reward_feats_all_foods[i,:])))

# fig, axs = plt.subplots(1, 7, sharey=True, tight_layout=True)
# for dim in range(7):
#     if dim == 6:
#         axs[dim].hist(standardized_reward_feats_all_food[:,dim], bins=15)
#     else:
#         axs[dim].hist(standardized_reward_feats_all_food[:,dim], bins='auto')
#     axs[dim].set_title('reward feat %i - standardized' %dim)
#     axs[dim].set_xlabel('reward feat dim %i value' %dim)
#     axs[dim].set_ylabel('reward feat dim %i frequency' %dim)            
# plt.show()

