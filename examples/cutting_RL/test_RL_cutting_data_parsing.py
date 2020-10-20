from frankapy.utils import *

# load previous data and calculate updated policy and plot rewards
# data_dir = '/home/sony/Documents/cutting_RL_experiments/data/celery/normalCut/exp_9/'
# #policy_params_mean, policy_params_sigma = parse_policy_params_and_rews_from_file(data_dir,prev_epochs_to_calc_pol_update, hfpc = False)
# policy_params_mean, policy_params_sigma = parse_policy_params_and_rews_from_file(data_dir, prev_epochs_to_calc_pol_update=10, hfpc = True)

# plot multiple experiment rewards together on same plots
work_dirs = ['/home/sony/Documents/cutting_RL_experiments/data/celery/normalCut/exp_9_posXZ_varStiff/',\
    '/home/sony/Documents/cutting_RL_experiments/data/celery/normalCut/exp_8_posXforceZvarStiff/',
        '/home/sony/Documents/cutting_RL_experiments/data/celery/normalCut/exp_7_genericRF/']
rews_or_avg_rews = 'rews'
plot_rewards_mult_experiments(work_dirs, rews_or_avg_rews)

rews_or_avg_rews = 'avg_rews'
plot_rewards_mult_experiments(work_dirs, rews_or_avg_rews)
import pdb; pdb.set_trace()

# num_epochs = 3
# plot_rewards_mult_epochs(data_dir, num_epochs)