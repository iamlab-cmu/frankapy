from frankapy.utils import *

data_dir = '/home/sony/Documents/cutting_RL_experiments/data/celery/normalCut/exp_7/'
#policy_params_mean, policy_params_sigma = parse_policy_params_and_rews_from_file(data_dir,prev_epochs_to_calc_pol_update, hfpc = False)
policy_params_mean, policy_params_sigma = parse_policy_params_and_rews_from_file(data_dir, prev_epochs_to_calc_pol_update=10, hfpc = True)

import pdb; pdb.set_trace()

# num_epochs = 3
# plot_rewards_mult_epochs(data_dir, num_epochs)