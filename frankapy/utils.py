import numpy as np 
import matplotlib.pyplot as plt 
from numba import jit
from autolab_core import RigidTransform, Point 
import pickle 

import os
import subprocess
import rospy
import glob
from sensor_msgs.msg import Image
from perception import CameraIntrinsics
import cv2
from cv_bridge import CvBridge, CvBridgeError

from rl_utils import reps

class DMPPositionTrajectoryGenerator:
    '''
    Generate DMP trajectory on Python side and pass via ROS to control PC side -
    goal is to use w/ HFPC on control PC. Note: currently this just loads in parameters from 
    an already trained DMP (saved in pkl file). TODO: incorporate end to end DMP training here 
    '''
    def __init__(self, traject_time):
        self.traject_time = int(traject_time/0.001)
    
    def load_saved_dmp_params_from_pkl_file(self, pkl_filepath):
        dmp_params_pkl_file = open(pkl_filepath, 'rb')
        dmp_params_dict = pickle.load(dmp_params_pkl_file)
        self.dmp_params_dict = dmp_params_dict 

    def parse_dmp_params_dict(self):
        self.tau = self.dmp_params_dict['tau']
        self.alpha = self.dmp_params_dict['alpha']
        self.beta = self.dmp_params_dict['beta']
        self.phi = self.dmp_params_dict['phi_j']
        
        self.std = self.dmp_params_dict['h'][1:] # NOTE: skipping 0 at beginning
        self.mean = self.dmp_params_dict['mu'][1:]  # NOTE: skipping 0 at beginning
        self.num_dims = self.dmp_params_dict['num_dims']
        self.weights_list = self.dmp_params_dict['weights']       
        self.num_sensors = self.dmp_params_dict['num_sensors']
        self.num_basis = self.dmp_params_dict['num_basis']

        # Get mu and h for all parameters separately
        self.mu_all = np.zeros((self.num_dims, self.num_sensors, self.num_basis-1))
        self.h_all = np.zeros((self.num_dims, self.num_sensors, self.num_basis-1))
        
        for i in range(self.num_dims):
            for j in range(self.num_sensors):
                self.mu_all[i, j] = self.mean
                self.h_all[i, j] = self.std

        self.phi_j = np.ones((self.num_sensors))
        # self.phi_j = self.phi*np.ones((self.num_sensors))

        # parse list of weights and save as np array: 
        self.weights = np.zeros((self.num_dims, self.num_sensors, self.num_basis))
        for i in range(self.num_dims):
            for j in range(self.num_sensors):
                self.weights[i,j,:] =  self.weights_list[i][j][:]   

    def run_dmp_with_weights(self, y0, dt=0.001, phi_j=None):
        '''
        NOTE: taken from dmp_class.py in dmp repository 

        Run DMP with given weights.
        weights: array of weights. size: (N*M*K, 1) i.e. 
            (num_dims*num_sensors*num_basis, 1)
        y0: Start location for dmps. Array of size (N,)
        dt: Time step to use. Float.
        traj_time: Time length to sample trajectories. Integer
        '''
        x = 1.0
        y  = np.zeros((self.traject_time, self.num_dims))
        dy = np.zeros((self.traject_time, self.num_dims))
        y[0] = y0
        # This reshape happens along the vector i.e. the first (M*K) values 
        # belong to dimension 0 (i.e. N = 0). Following (M*K) values belong to
        # dimension 1 (i.e. N = 1), and so forth.
        # NOTE: We add 1 for the weights of the jerk basis function

        min_jerk_arr = np.zeros((self.traject_time))

        x_log = []
        psi_log = []
        min_jerk_log = []
        for i in range(self.traject_time - 1):
            # psi_ijk is of shape (N, M, K)
            psi_ijk = np.exp(-self.h_all * (x-self.mu_all)**2)
            psi_ij_sum = np.sum(psi_ijk, axis=2, keepdims=True)

            #import pdb; pdb.set_trace()
            '''NOTE: this is expecting self.std and self.mean to be size (num_basis - 1). But
            when loading these from a saved pkl file, self.mean and self.std are padded with an extra
            0 at the beginning. So, if loading dmp data from an already saved pkl file, need to modify to 
            skip 0 at beginning of self.std and self.mean
            '''
            f = (psi_ijk * self.weights[:, :, 1:] * x).sum(
                    axis=2, keepdims=True) / (psi_ij_sum + 1e-10)
            # f_min_jerk = (i * dt)/(traj_time * dt)
            f_min_jerk = min(-np.log(x)*2, 1)
            f_min_jerk = (f_min_jerk**3)*(6*(f_min_jerk**2) - 15*f_min_jerk+ 10)
            psi_ij_jerk = self.weights[:, :, 0:1] * f_min_jerk

            # for debug
            min_jerk_arr[i] = f_min_jerk

            # calculate f(x; w_j)l -- shape (N, M)
            all_f_ij = self.alpha * self.beta * (f + psi_ij_jerk).squeeze()

            # Calculate sum_j(phi_j * f(x; w_j) -- shape (N,)
            
            if phi_j is None:
                phi_j = self.phi_j

            if phi_j.shape == (1,):
                #all_f_i = np.dot(all_f_ij, phi_j) #Uncomment this if num sensors =2
                all_f_i = np.dot((self.alpha * self.beta * (f + psi_ij_jerk)), phi_j) #comment out if num sensors=2 (updated to make matrix dims work for num sensors=1)
                all_f_i=all_f_i.squeeze() #comment out if num sensors=2
            
            elif phi_j.shape == (2,):
                #all_f_i = np.sum(all_f_ij * phi_j, axis=1)
                all_f_i = np.dot(all_f_ij, phi_j) #Uncomment this if num sensors =2
            else:
                raise ValueError("Incorrect shape for phi_j")
            
            ddy = self.alpha*(self.beta*(y0 - y[i]) - dy[i]/self.tau) + all_f_i
            ddy = ddy * (self.tau ** 2)
            dy[i+1] = dy[i] + ddy * dt
            y[i+1] = y[i] + dy[i+1] * dt

            x_log.append(x)
            psi_log.append(psi_ijk)
            min_jerk_log.append(f_min_jerk)

            x += ((-self.tau * x) * dt)
            if (x < self.mean[-1] - 3.0*np.sqrt(1.0/(self.std[-1]))):
                x = 1e-7   
                            
        # visualize xyz trajectory
        # for i in range(3):
        #     plt.plot(np.arange(0, self.traject_time/1000, 0.001), y[:,i])
        # plt.legend(('x','y','z'))
        # plt.title('Cartesian Position')
        # plt.xlabel('Time (s)')
        # plt.ylabel('Position (m)')
        # plt.show()
        return y, dy, x_log, np.array(psi_log), np.array(min_jerk_log)

def save_weights(save_path, data_dict):
    with open(save_path, 'wb') as pkl_f:
        pickle.dump(data_dict, pkl_f, protocol=2)
        print("Did save dmp params: {}".format(save_path))

def get_dmp_traj_poses_reformatted(y, starting_rotation):
    '''
    this converts y (xyz position traject) to list of 1x16 
    '''
    target_poses = []
    last_row = np.array([0, 0, 0, 1])
    for t in range(y.shape[0]):
        transl = np.array([y[t,:]]).T
        r_t = np.hstack((starting_rotation, transl))
        TF_matrix = np.vstack((r_t,last_row)) # TF matrix
        flattened_TF_matrix = TF_matrix.T.flatten().tolist()
        target_poses.append(flattened_TF_matrix)    
    return target_poses

def downsample_dmp_traject(original_dmp_traject, og_dt, new_downsampled_dt):
    '''
    downsample original dmp_traject (default is dt = 0.001) (e.g. new dmp_traject dt = 0.01)
    '''
    downsmpled_dmp_traject = np.empty((0,original_dmp_traject.shape[1]))
    samples_to_skip = int(new_downsampled_dt/og_dt)

    inds_og_traject = np.arange(0,original_dmp_traject.shape[0],samples_to_skip)
    for i in inds_og_traject:
        downsmpled_dmp_traject = np.vstack((downsmpled_dmp_traject, original_dmp_traject[i,:]))
    
    return downsmpled_dmp_traject

# def plot_updated_policy_mean_traject(work_dir, position_dmp_weights_file_path, epoch, dmp_traject_time, control_type_z_axis, init_dmp_info_dict, \
#     initial_wts, REPS_updated_mean):
#     '''
#     plot updated policy mean dmp trajectories
#     '''
#     if control_type_z_axis == 'force':                
#         new_weights = np.expand_dims(np.vstack((REPS_updated_mean[0:7],initial_wts[1,:,:],initial_wts[2,:,:])),axis=1)
#         if REPS_updated_mean[-2] < 0: 
#             new_z_force = REPS_updated_mean[-2] #REPS_updated_mean[-1] #
#         else: 
#             new_z_force = REPS_updated_mean[-1]

#     elif control_type_z_axis == 'position':
#         new_weights = np.expand_dims(np.vstack((REPS_updated_mean[0:7],initial_wts[1,:,:],REPS_updated_mean[7:14])),axis=1)
#         new_z_force = 0

#     # Save new weights to dict
#     data_dict = {
#         'tau':           init_dmp_info_dict['tau'],
#         'alpha':         init_dmp_info_dict['alpha'],
#         'beta':          init_dmp_info_dict['beta'],
#         'num_dims':      init_dmp_info_dict['num_dims'],
#         'num_basis':     init_dmp_info_dict['num_basis'],
#         'num_sensors':   init_dmp_info_dict['num_sensors'],
#         'mu':            init_dmp_info_dict['mu'],
#         'h':             init_dmp_info_dict['h'],
#         'phi_j':         init_dmp_info_dict['phi_j'],
#         'weights':       new_weights.tolist(),                
#         }

#     # save new sampled weights to pkl file
#     weight_save_file = os.path.join(work_dir, 'meanWeightsUpdatedPol' + '.pkl')
#     save_weights(weight_save_file, data_dict)

#     # Calculate dmp trajectory             
#     traject_time = dmp_traject_time   # define length of dmp trajectory  
#     # Load dmp traject params
#     dmp_traj = DMPPositionTrajectoryGenerator(traject_time)
#     dmp_traj.load_saved_dmp_params_from_pkl_file(weight_save_file)
#     dmp_traj.parse_dmp_params_dict()

#     # Define starting position 
#     #start_pose = fa.get_pose()
#     #starting_rotation = start_pose.rotation
#     y0 = 0 #start_pose.translation 
#     # calculate dmp position trajectory - NOTE: this assumes a 0.001 dt for calc the dmp traject
#     dmp_traject, dy, _, _, _ = dmp_traj.run_dmp_with_weights(y0) # y: np array(tx3)
    
#     # check new dmp sampled wt trajectory vs original
#     sample = 0
#     plot_sampled_new_dmp_traject_and_original_dmp(epoch, sample, work_dir, new_z_force, traject_time, \
#         position_dmp_weights_file_path, dmp_traject, y0)
#     import pdb; pdb.set_trace()

def load_dmp_wts_and_knife_orientation(cut_type):
    # load dmp weights
    if cut_type == 'normal':
        dmp_wts_file = '/home/sony/092420_normal_cut_dmp_weights_zeroY.pkl'
        # more angled to sharp knife - normal cut
        knife_orientation = np.array([[0.0,   0.9805069,  -0.19648464],
                                  [ 1.0,   0.0,  0.0],
                                  [ 0.0, -0.19648464,  -0.9805069]])

    elif cut_type == 'pivchop':
        dmp_wts_file = '/home/sony/raw_IL_trajects/Jan-2021/011321_piv_chop_potato_position_weights_zeroXY.pkl' 
        # dmp_wts_file = '/home/sony/raw_IL_trajects/100220_piv_chop_position_dmp_weights_zeroXY_2.pkl'
        # metal knife (26 deg tilt forward) - pivchop
        knife_orientation = np.array([[0.0,   0.8988,  -0.4384],
                                  [ 1.0,   0.0,  0.0],
                                  [ 0.0, -0.4384,  -0.8988]])

    elif cut_type == 'scoring':
        dmp_wts_file = '/home/sony/raw_IL_trajects/Jan-2021/011321_scoring_potato_position_weights_zeroYZ.pkl' 
        # metal knife (26 deg tilt forward) - pivchop
        knife_orientation = np.array([[0.0,   0.8988,  -0.4384],
                                  [ 1.0,   0.0,  0.0],
                                  [ 0.0, -0.4384,  -0.8988]])    
    return dmp_wts_file, knife_orientation

def load_prev_task_success_data(work_dir):
    prev_data_time = np.load(os.path.join(work_dir, 'cut_times_all_samples.npy'))
    prev_data_task_succ = np.load(os.path.join(work_dir, 'task_success_all_samples.npy'))
    time_to_complete_cut = prev_data_time.tolist()
    task_success = prev_data_task_succ.tolist()
    
    # load previous sample granular task success and reward features
    if os.path.isfile(os.path.join(work_dir, 'task_success_more_granular_all_samples.npy')):
        prev_data_task_succ_more_granular = np.load(os.path.join(work_dir, 'task_success_more_granular_all_samples.npy'))
        task_success_more_granular = prev_data_task_succ_more_granular.tolist() 
    else:
        for i in range(len(task_success)):
            task_success_more_granular.append([np.inf, np.inf, np.inf])

    if os.path.isfile(os.path.join(work_dir, 'reward_features_all_samples.npy')):
        prev_data_reward_feats_all_samples = np.load(os.path.join(work_dir, 'reward_features_all_samples.npy'))        
        reward_features_all_samples = prev_data_reward_feats_all_samples.tolist()
    else: 
        for i in range(len(task_success)):
            reward_features_all_samples.append([np.inf]*7)
    
    return reward_features_all_samples, time_to_complete_cut, task_success, task_success_more_granular

def plot_updated_policy_mean_traject_HIL_ARL(work_dir, fa, cut_type, position_dmp_weights_file_path, epoch, dmp_traject_time, control_type_z_axis, init_dmp_info_dict, \
    initial_wts, REPS_updated_mean):
    if cut_type == 'normal' or cut_type =='scoring':
        if control_type_z_axis == 'force':                
            new_weights = np.expand_dims(np.vstack((REPS_updated_mean[0:7],initial_wts[1,:,:],initial_wts[2,:,:])),axis=1)
            new_z_force = REPS_updated_mean[-2]
        elif control_type_z_axis == 'position':
            new_weights = np.expand_dims(np.vstack((REPS_updated_mean[0:7],initial_wts[1,:,:],REPS_updated_mean[7:14])),axis=1)
            new_z_force = 0
    
    elif cut_type == 'pivchop':
        if control_type_z_axis == 'force':                
            new_weights = np.expand_dims(np.vstack((REPS_updated_mean[0:7],initial_wts[1,:,:],initial_wts[2,:,:])),axis=1)
            new_z_force = REPS_updated_mean[-2]
        elif control_type_z_axis == 'position':
            new_weights = np.expand_dims(np.vstack((initial_wts[0,:,:],initial_wts[1,:,:],REPS_updated_mean[0:7])),axis=1)
            new_z_force = REPS_updated_mean[-1] #0

    # Save new weights to dict
    data_dict = {
        'tau':           init_dmp_info_dict['tau'],
        'alpha':         init_dmp_info_dict['alpha'],
        'beta':          init_dmp_info_dict['beta'],
        'num_dims':      init_dmp_info_dict['num_dims'],
        'num_basis':     init_dmp_info_dict['num_basis'],
        'num_sensors':   init_dmp_info_dict['num_sensors'],
        'mu':            init_dmp_info_dict['mu'],
        'h':             init_dmp_info_dict['h'],
        'phi_j':         init_dmp_info_dict['phi_j'],
        'weights':       new_weights.tolist(),                
        }

    # save new sampled weights to pkl file
    weight_save_file = os.path.join(work_dir, 'meanWeightsUpdatedPol' + '.pkl')
    save_weights(weight_save_file, data_dict)

    # Calculate dmp trajectory             
    traject_time = dmp_traject_time   # define length of dmp trajectory  
    # Load dmp traject params
    dmp_traj = DMPPositionTrajectoryGenerator(traject_time)
    dmp_traj.load_saved_dmp_params_from_pkl_file(weight_save_file)
    dmp_traj.parse_dmp_params_dict()

    # Define starting position 
    start_pose = fa.get_pose()
    starting_rotation = start_pose.rotation
    y0 = start_pose.translation 
    # calculate dmp position trajectory - NOTE: this assumes a 0.001 dt for calc the dmp traject
    dmp_traject, dy, _, _, _ = dmp_traj.run_dmp_with_weights(y0) # y: np array(tx3)
    
    # check new dmp sampled wt trajectory vs original
    sample = 0
    plot_sampled_new_dmp_traject_and_original_dmp(epoch, work_dir, sample, new_z_force, traject_time, \
        position_dmp_weights_file_path, dmp_traject, y0)
    import pdb; pdb.set_trace()

def plot_sampled_new_dmp_traject_and_original_dmp(epoch, work_dir, sample, new_z_force, traject_time, \
    initial_dmp_weights_pkl_file, new_dmp_traject, y0):
    #original_dmp_wts_pkl_filepath = '/home/sony/Desktop/debug_dmp_wts.pkl'
    dmp_traj = DMPPositionTrajectoryGenerator(traject_time)
    dmp_traj.load_saved_dmp_params_from_pkl_file(initial_dmp_weights_pkl_file)
    dmp_traj.parse_dmp_params_dict()
    # calculate dmp position trajectory - NOTE: this assumes a 0.001 dt for calc the dmp traject
    original_traject, dy, _, _, _ = dmp_traj.run_dmp_with_weights(y0) # y: np array(tx3)
    
    axes = ['x', 'y','z']   
    fig, ax = plt.subplots(3,1) 
    for i in range(3):
        ax[i].plot(np.arange(0, traject_time, 0.001), original_traject[:,i])
        ax[i].plot(np.arange(0, traject_time, 0.001), new_dmp_traject[:,i])           
        if i!=0:
            ax[i].set_title('Cartesian Position - '+str(axes[i]))
        else:     
            if new_z_force == 'NA':
                ax[i].set_title('Cartesian Position - '+str(axes[i]))
            else:
                ax[i].set_title('Cartesian Position - '+str(axes[i]) + ' ' + 'Downward z-force (N): '+str(new_z_force))
        ax[i].set_ylabel('Position (m)')
        ax[i].legend((axes[i] + '-original traject', axes[i] + '-new sampled traject'))
    
    ax[2].set_xlabel('Time (s)')
    plt.show()
    # save figure to working dir
    fig.savefig(work_dir + '/' + 'dmp_traject_plots' + '/sampledDMP_' + 'epoch_'+str(epoch) + '_ep_'+str(sample)+'.png')

# OLD
# def plot_sampled_new_dmp_traject_and_original_dmp(epoch, sample, save_dir, new_z_force, traject_time, \
#     initial_dmp_weights_pkl_file, new_dmp_traject, y0):
#     '''
#     plot original IL trajectory overlayed with new sampled dmp trajectory
#     '''
#     #original_dmp_wts_pkl_filepath = '/home/sony/Desktop/debug_dmp_wts.pkl'
#     dmp_traj = DMPPositionTrajectoryGenerator(traject_time)
#     dmp_traj.load_saved_dmp_params_from_pkl_file(initial_dmp_weights_pkl_file)
#     dmp_traj.parse_dmp_params_dict()
#     # calculate dmp position trajectory - NOTE: this assumes a 0.001 dt for calc the dmp traject
#     original_traject, dy, _, _, _ = dmp_traj.run_dmp_with_weights(y0) # y: np array(tx3)
    
#     axes = ['x', 'y','z']   
#     fig, ax = plt.subplots(3,1) 
#     for i in range(3):
#         ax[i].plot(np.arange(0, traject_time, 0.001), original_traject[:,i])
#         ax[i].plot(np.arange(0, traject_time, 0.001), new_dmp_traject[:,i])           
#         if i!=0:
#             ax[i].set_title('Cartesian Position - '+str(axes[i]))
#         else:     
#             if new_z_force == 'NA':
#                 ax[i].set_title('Cartesian Position - '+str(axes[i]))
#             else:
#                 ax[i].set_title('Cartesian Position - '+str(axes[i]) + ' ' + 'Downward z-force (N): '+str(new_z_force))
#         ax[i].set_ylabel('Position (m)')
#         ax[i].legend((axes[i] + '-original traject', axes[i] + '-new sampled traject'))
    
#     ax[2].set_xlabel('Time (s)')
#     plt.show()
#     # save figure to working dir
#     fig.savefig(work_dir + '/' + 'dmp_traject_plots' + '/sampledDMP_' + 'epoch_'+str(epoch) + '_ep_'+str(sample)+'.png')

def parse_policy_params_and_rews_from_file(work_dir, prev_epochs_to_calc_pol_update, hfpc=True):
    '''
    calculate updated policy from previous data and plot rewards from previous data

    prev_epochs_to_calc_pol_update: how many prev epochs' data to use to calculate policy update
    '''
    data_files = glob.glob(work_dir + "*epoch_*.npy")
    #import pdb; pdb.set_trace()
    num_prev_epochs = len(data_files)
    #import pdb; pdb.set_trace()
    # get num policy params
    first_file = np.load(data_files[0])
    num_pol_params = first_file.shape[1]-1 # subtract 1 b/c last dim is reward

    rews_all_epochs = np.empty((0))
    avg_rews_each_epoch = []
    num_samples_each_epoch = []
    num_samples = 0
    if hfpc:
        pol_params_all_epochs = np.empty((0,num_pol_params))
    else:
        pol_params_all_epochs = np.empty((0,num_pol_params))
    for i in range(num_prev_epochs):
        data_file = glob.glob(work_dir + "*epoch_%s*.npy"%str(i))[0]
        data = np.load(data_file)
        pol_params = data[:,0:-1]
        rewards = data[:,-1]  
        avg_rews_each_epoch.append(np.mean(rewards))
        rews_all_epochs = np.concatenate((rews_all_epochs, rewards),axis=0)
        pol_params_all_epochs = np.concatenate((pol_params_all_epochs, pol_params),axis=0)
        num_samples+=data.shape[0]
        num_samples_each_epoch.append(num_samples)
        
    #import pdb; pdb.set_trace()
    # plot rewards
    plt.plot(np.arange(rews_all_epochs.shape[0]),rews_all_epochs,'-o')
    # add in labels for epochs
    plt.vlines(np.array(num_samples_each_epoch)-1,np.min(rews_all_epochs)-5,0, colors = ['r','r','r'], linestyles={'dashed', 'dashed', 'dashed'})

    counter = 0
    if hfpc:
        for x,y in zip(np.arange(rews_all_epochs.shape[0]), pol_params_all_epochs[:,-1]):
            label = 'f_z = %i'%np.round(y) #'rStiff = %i'%np.round(y) 
            plt.annotate(label, (x,rews_all_epochs[counter]), textcoords = 'offset points', xytext = (0,10), ha='center')
            counter += 1
    plt.xlabel('sample num')
    plt.ylabel('reward - average across all dmps for each slice')
    plt.ylim(np.min(rews_all_epochs)-5, 0)
    plt.title('reward vs. sample - normalCut, potato')
    plt.xticks(np.arange(rews_all_epochs.shape[0]))
    plt.show()

    # plot average rewards each epoch
    plt.plot(avg_rews_each_epoch, '-o')
    plt.xlabel('epoch')
    plt.ylabel('avg reward each epoch - average across all dmps for each slice')
    plt.ylim(-60, 0)
    plt.title('avg reward vs epochs')
    plt.xticks(np.arange(3))
    plt.show()
    
    # update policy mean and cov (REPS)       
    reps_agent = reps.Reps(rel_entropy_bound=1.5,min_temperature=0.001) #Create REPS object
    if np.abs(-1-prev_epochs_to_calc_pol_update) > len(num_samples_each_epoch): # use all data from all epochs
        #import pdb; pdb.set_trace()
        policy_params_mean, policy_params_sigma, reps_info = reps_agent.policy_from_samples_and_rewards(pol_params_all_epochs, rews_all_epochs)
    else:
        print('using previous %i epochs to calc policy update'%prev_epochs_to_calc_pol_update)
        #import pdb; pdb.set_trace()   
        pol_params_desired_epochs = pol_params_all_epochs[num_samples_each_epoch[-1-prev_epochs_to_calc_pol_update]:]
        rews_desired_epochs = rews_all_epochs[num_samples_each_epoch[-1-prev_epochs_to_calc_pol_update]:]
        print('shape of prev data is ', rews_desired_epochs.shape)
        policy_params_mean, policy_params_sigma, reps_info = reps_agent.policy_from_samples_and_rewards(pol_params_desired_epochs, rews_desired_epochs)

    # np.savez(os.path.join(work_dir, 'REPSupdatedMean_' + 'epoch_'+str(x) +'.npz'), \
    #         updated_mean = policy_params_mean, updated_cov = policy_params_sigma)

    return policy_params_mean, policy_params_sigma

def parse_policy_params_and_rews_from_file_HIL_ARL(num_expert_rews_each_sample, work_dir, prev_epochs_to_calc_pol_update, hfpc=True):
    '''
    calculate updated policy from previous data and plot rewards from previous data

    prev_epochs_to_calc_pol_update: how many prev epochs' data to use to calculate policy update

    NOTE: modified this function from "parse_policy_params_and_rews_from_file()" to take into account different reward indices depending on type of human reward/etc
    '''
    data_files = glob.glob(work_dir + "*epoch_*.npy")
    #import pdb; pdb.set_trace()
    num_prev_epochs = len(data_files)
    #import pdb; pdb.set_trace()
    # get num policy params
    first_file = np.load(data_files[0])

    if num_expert_rews_each_sample == 1:   
        num_pol_params = first_file[:,0:-3].shape[1]          

    elif num_expert_rews_each_sample == 2:  
        num_pol_params = first_file[:,0:-4].shape[1]                                  

    rews_all_epochs = np.empty((0))
    avg_rews_each_epoch = []
    num_samples_each_epoch = []
    num_samples = 0
    if hfpc:
        pol_params_all_epochs = np.empty((0,num_pol_params))
    else:
        pol_params_all_epochs = np.empty((0,num_pol_params))
    for i in range(num_prev_epochs):
        data_file = glob.glob(work_dir + "*epoch_%s*.npy"%str(i))[0]
        data = np.load(data_file)
        if num_expert_rews_each_sample == 1:  
            pol_params = data[:,0:-3] 
            rewards = data[:,-2]# reward model rewards
            
        elif num_expert_rews_each_sample == 2:  
            pol_params = data[:,0:-4] 
            rewards = data[:,-3]# reward model rewards

        # pol_params = data[:,0:-1]
        # rewards = data[:,-1]  
        avg_rews_each_epoch.append(np.mean(rewards))
        rews_all_epochs = np.concatenate((rews_all_epochs, rewards),axis=0)
        pol_params_all_epochs = np.concatenate((pol_params_all_epochs, pol_params),axis=0)
        num_samples+=data.shape[0]
        num_samples_each_epoch.append(num_samples)
        
    #import pdb; pdb.set_trace()
    # plot rewards
    plt.plot(np.arange(rews_all_epochs.shape[0]),rews_all_epochs,'-o')
    # add in labels for epochs
    plt.vlines(np.array(num_samples_each_epoch)-1,np.min(rews_all_epochs)-5,0, colors = ['r','r','r'], linestyles={'dashed', 'dashed', 'dashed'})

    counter = 0
    if hfpc:
        for x,y in zip(np.arange(rews_all_epochs.shape[0]), pol_params_all_epochs[:,-1]):
            label = 'f_z = %i'%np.round(y) #'rStiff = %i'%np.round(y) 
            plt.annotate(label, (x,rews_all_epochs[counter]), textcoords = 'offset points', xytext = (0,10), ha='center')
            counter += 1
    plt.xlabel('sample num')
    plt.ylabel('reward - average across all dmps for each slice')
    plt.ylim(np.min(rews_all_epochs)-5, 0)
    plt.title('reward vs. sample - normalCut, potato')
    plt.xticks(np.arange(rews_all_epochs.shape[0]))
    plt.show()

    # plot average rewards each epoch
    plt.plot(avg_rews_each_epoch, '-o')
    plt.xlabel('epoch')
    plt.ylabel('avg reward each epoch - average across all dmps for each slice')
    plt.ylim(-60, 0)
    plt.title('avg reward vs epochs')
    plt.xticks(np.arange(3))
    plt.show()
    
    # update policy mean and cov (REPS)       
    reps_agent = reps.Reps(rel_entropy_bound=1.5,min_temperature=0.001) #Create REPS object
    if np.abs(-1-prev_epochs_to_calc_pol_update) > len(num_samples_each_epoch): # use all data from all epochs
        #import pdb; pdb.set_trace()
        policy_params_mean, policy_params_sigma, reps_info = reps_agent.policy_from_samples_and_rewards(pol_params_all_epochs, rews_all_epochs)
    else:
        print('using previous %i epochs to calc policy update'%prev_epochs_to_calc_pol_update)
        #import pdb; pdb.set_trace()   
        pol_params_desired_epochs = pol_params_all_epochs[num_samples_each_epoch[-1-prev_epochs_to_calc_pol_update]:]
        rews_desired_epochs = rews_all_epochs[num_samples_each_epoch[-1-prev_epochs_to_calc_pol_update]:]
        print('shape of prev data is ', rews_desired_epochs.shape)
        policy_params_mean, policy_params_sigma, reps_info = reps_agent.policy_from_samples_and_rewards(pol_params_desired_epochs, rews_desired_epochs)

    # np.savez(os.path.join(work_dir, 'REPSupdatedMean_' + 'epoch_'+str(x) +'.npz'), \
    #         updated_mean = policy_params_mean, updated_cov = policy_params_sigma)

    return policy_params_mean, policy_params_sigma

def plot_rewards_mult_experiments(work_dirs, rews_or_avg_rews, cut_type, hfpc=True):
    '''
    plot rewards from multiple experiments on the same plot for comparison
    '''
    for work_dir in work_dirs:
        data_files = glob.glob(work_dir + "*epoch_*.npy")
        num_prev_epochs = len(data_files)
        # get num policy params
        first_file = np.load(data_files[0])
        num_pol_params = first_file.shape[1]-1 # subtract 1 b/c last dim is reward

        rews_all_epochs = np.empty((0))
        avg_rews_each_epoch = []
        num_samples_each_epoch = []
        num_samples = 0
        if hfpc:
            #pol_params_all_epochs = np.empty((0,8))
            pol_params_all_epochs = np.empty((0,num_pol_params))
        else:
            pol_params_all_epochs = np.empty((0,num_pol_params))
        for i in range(num_prev_epochs):
            data_file = glob.glob(work_dir + "*epoch_%s*.npy"%str(i))[0]
            data = np.load(data_file)
            pol_params = data[:,0:-1]
            rewards = data[:,-1]  
            avg_rews_each_epoch.append(np.mean(rewards))
            rews_all_epochs = np.concatenate((rews_all_epochs, rewards),axis=0)
            pol_params_all_epochs = np.concatenate((pol_params_all_epochs, pol_params),axis=0)
            num_samples+=data.shape[0]
            num_samples_each_epoch.append(num_samples)
        
        #import pdb; pdb.set_trace()
        
        if rews_or_avg_rews == 'rews':
        # plot rewards
            plt.plot(np.arange(rews_all_epochs.shape[0]),rews_all_epochs,'-o')
            # add in labels for epochs
            plt.vlines(np.array(num_samples_each_epoch)-1,np.min(rews_all_epochs)-5,0, colors = ['r','r','r'], linestyles={'dashed', 'dashed', 'dashed'})

            plt.xlabel('sample num')
            plt.ylabel('reward - average across all dmps for each slice')
            #plt.ylim(np.min(rews_all_epochs)-5, 0)
            plt.title('reward vs. sample - %s'%cut_type)
            plt.xticks(np.arange(rews_all_epochs.shape[0]))

        # plot average rewards each epoch
        elif rews_or_avg_rews == 'avg_rews':
            plt.plot(avg_rews_each_epoch, '-o')
            plt.xlabel('epoch')
            plt.ylabel('avg reward each epoch - average across all dmps for each slice - %s'%cut_type)
            plt.ylim(-60, 0)
            plt.title('avg reward vs epochs')
            plt.xticks(np.arange(3))
    
    #plt.legend(('exp9: posX_posZ_varStiff','exp8 :posX_forceZ_varStiff', 'exp7: posX_forceZ'))
    # plt.legend(('exp3: posZ_varStiff','exp4: forceZ_varStiff'))
    #plt.legend(('exp4: normal-banana','exp5: normal-tomato', 'exp6: normal-mozz'))
    #plt.legend(('exp1: normal-potato','exp2: normal-celery', 'exp3: normal-carrot','exp4: normal-banana','exp5: normal-tomato', 'exp6: normal-mozz'))
    # plt.legend(('exp1: pivchop-potato','exp2: pivchop-celery', 'exp3: pivchop-carrot','exp4: pivchop-banana','exp5: pivchop-tomato', 'exp6: pivchop-mozz'))
    # leg_str = 'exp1: %s-potato, exp2: %s-celery, exp3: %s-carrot, exp4: %s-banana, exp5: %s-tomato, exp6: %s-mozz'%(cut_type,cut_type,cut_type,cut_type,cut_type,cut_type)
    plt.legend(('exp1: %s-potato'%cut_type,'exp2: %s-celery'%cut_type, 'exp3: %s-carrot'%cut_type,'exp4: %s-banana'%cut_type,'exp5: %s-tomato'%cut_type, 'exp6: %s-mozz'%cut_type))
    plt.show()


def plot_rewards_mult_epochs(work_dir, num_epochs):
    pol_params_all_epochs = np.empty((0,8))
    forces_all_epochs = np.empty((0))
    rews_all_epochs = np.empty((0))
    avg_rews_each_epoch = []
    for i in range(num_epochs):
        data_file = glob.glob(work_dir + "*epoch_%s*.npy"%str(i))[0]
        data = np.load(data_file)
        pol_params = data[:,0:-1]
        forces = data[:,-2]
        rewards = data[:,-1]  
        avg_rews_each_epoch.append(np.mean(rewards))
        forces_all_epochs = np.concatenate((forces_all_epochs, forces), axis=0)
        rews_all_epochs = np.concatenate((rews_all_epochs, rewards),axis=0)
        pol_params_all_epochs = np.concatenate((pol_params_all_epochs, pol_params),axis=0)

    # plot rewards each sample
    plt.plot(np.arange(rews_all_epochs.shape[0]),rews_all_epochs,'-o')
    counter = 0
    for x,y in zip(np.arange(rews_all_epochs.shape[0]), pol_params_all_epochs[:,-1]):
        label = '%iN'%np.round(y)
        plt.annotate(label, (x, rews_all_epochs[counter]), textcoords = 'offset points', xytext = (0,7), ha='center')
        counter += 1
    plt.xlabel('sample num')
    plt.ylabel('reward - average across all dmps for each slice')
    plt.title('rewards vs. samples - epochs 0/1/2')
    plt.xticks(np.arange(rews_all_epochs.shape[0]))
    plt.show()

    # plot average rewards each epoch
    plt.plot(avg_rews_each_epoch, '-o')
    plt.xlabel('epoch')
    plt.ylabel('avg reward each epoch - average across all dmps for each slice')
    plt.title('avg reward vs epochs')
    plt.xticks(np.arange(3))
    plt.show()

def viz_force_data(force_data_dir, epoch, num_samples):
    data_files = glob.glob(force_data_dir + "epoch_" + str(epoch)+  "*.npz")
    max_z_forces_all = []
    for i in range(num_samples):
        data_file = glob.glob(force_data_dir + "epoch_" + str(epoch)+ '_ep_' + str(i)+ '_' + "*.npz") 
        if len(data_file) == 1:
            data_file = data_file[0]
            #import pdb; pdb.set_trace()
            data = np.load(data_file)
            data = data['robot_forces']
            max_z_force = np.min(data[:,2])
            max_z_forces_all.append(max_z_force)
        else:
            for j in range(len(data_file)):
                data_file = glob.glob(force_data_dir + "epoch_" + str(epoch)+ '_ep_' + str(i) + '_trial_info_' + str(j) + "*.npz") 
                #import pdb; pdb.set_trace()
                data_file = data_file[0]
                data = np.load(data_file)
                data = data['robot_forces']
                max_z_force = np.min(data[:,2])
                max_z_forces_all.append(max_z_force)
    plt.plot(max_z_forces_all)
    plt.xlabel('sample')
    plt.ylabel('max z forces')
    plt.title('max z forces vs samples - final epoch')
    plt.show()
    return max_z_forces_all

def plot_mean_task_success_and_percent_success_cuts(work_dir, food_type, cut_type, perc_succ_or_avg_succ):
    #import pdb; pdb.set_trace()
    data = np.load(work_dir + 'task_success_all_samples.npy')
    if type(data[0]) == np.str_:
        data = np.array([int(i) for i in data])
    #import pdb; pdb.set_trace()
    num_epochs = 5 #int((data.shape[0])/25)
    avg_task_success = []
    percent_success_cuts = []
    for i in range(num_epochs):
        #import pdb; pdb.set_trace()
        if i == 0 or i == 1 or i == 2:
            num_success_cut = 25 - np.where(data[25*i:(i+1)*25]==0)[0].shape[0]
            perc_succes = (num_success_cut/25.0)*100
        elif i == 3:
            num_success_cut = 20 - np.where(data[75:95]==0)[0].shape[0]
            perc_succes = (num_success_cut/20.0)*100
        elif i == 4:
            num_success_cut = 20 - np.where(data[95:]==0)[0].shape[0]
            perc_succes = (num_success_cut/20.0)*100

        percent_success_cuts.append(perc_succes)
        #import pdb; pdb.set_trace()
        avg_task_success.append(np.mean(data[25*i:(i+1)*25]))
    #import pdb; pdb.set_trace()
    label = cut_type 
    #plt.figure()
    if perc_succ_or_avg_succ == 'perc_succ':
        plt.plot(np.arange(num_epochs), percent_success_cuts,'-o')
        
        plt.title('percent successful cuts vs epochs - %s'%label)
        plt.xlabel('epoch')
        plt.ylim([0,105])
        plt.ylabel('percent successful cuts out of 25 samples')
    
    #plt.figure()
    elif perc_succ_or_avg_succ == 'avg_succ':
        plt.plot(np.arange(num_epochs), avg_task_success,'-o')
        plt.title('average task success (0,1,2) vs epochs - %s'%label)
        plt.xlabel('epoch')
        plt.ylabel('average task success (0,1,2)')
        plt.ylim([0,2.05])
        # plt.show()

def plot_UCB_experiment_results(work_dir):
    data = np.load(work_dir + 'all_polParamRew_data/avgRews.npy')

    plt.plot(np.arange(0,8),data,'-o')
    actions = ['111', '110', '101', '011', '000', '001', '010', '100']
    plt.xticks(range(len(actions)), actions)
    plt.xlabel('action (xyz axis controller combo)')
    plt.ylabel('mean expected reward')
    plt.title('UCB Sampling - Mean Expected Reward for Each High Level Action')

def franka_pose_to_rigid_transform(franka_pose, from_frame='franka_tool_base', to_frame='world'):
    np_franka_pose = np.array(franka_pose).reshape(4, 4).T
    pose = RigidTransform(
            rotation=np_franka_pose[:3, :3], 
            translation=np_franka_pose[:3, 3],
            from_frame=from_frame,
            to_frame=to_frame
        )
    return pose


@jit(nopython=True)
def min_jerk_weight(t, T):
    r = t/T
    return (10 * r ** 3 - 15 * r ** 4 + 6 * r ** 5)


@jit(nopython=True)
def min_jerk(xi, xf, t, T):
    return xi + (xf - xi) * min_jerk_weight(t, T)


@jit(nopython=True)
def min_jerk_delta(xi, xf, t, T, dt):
    r = t/T
    return (xf - xi) * (30 * r ** 2 - 60 * r ** 3 + 30 * r ** 4) / T * dt


def transform_to_list(T):
    return T.matrix.T.flatten().tolist()

'''
utility functions below taken from playing_with_food repository
'''
def get_azure_kinect_rgb_image(cv_bridge, topic='/rgb/image_raw'):
    """
    Grabs an RGB image for the topic as argument
    """
    rgb_image_msg = rospy.wait_for_message(topic, Image)
    try:
        rgb_cv_image = cv_bridge.imgmsg_to_cv2(rgb_image_msg)
    except CvBridgeError as e:
        print(e)
    
    return rgb_cv_image

def get_azure_kinect_depth_image(cv_bridge, topic='/depth_to_rgb/image_raw'):
    """
    Grabs an Depth image for the topic as argument
    """
    depth_image_msg = rospy.wait_for_message(topic, Image)
    try:
        depth_cv_image = cv_bridge.imgmsg_to_cv2(depth_image_msg)
    except CvBridgeError as e:
        print(e)
    
    return depth_cv_image

def get_realsense_rgb_image(cv_bridge, topic='/camera/color/image_raw'):
    """
    Grabs an RGB image for the topic as argument
    """
    rgb_image_msg = rospy.wait_for_message(topic, Image)
    try:
        cv_image = cv_bridge.imgmsg_to_cv2(rgb_image_msg)
        rgb_cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
    except CvBridgeError as e:
        print(e)
    
    return rgb_cv_image

def get_realsense_depth_image(cv_bridge, topic='/camera/depth/image_rect_raw'):
    """
    Grabs an Depth image for the topic as argument
    """
    depth_image_msg = rospy.wait_for_message(topic, Image)
    try:
        depth_cv_image = cv_bridge.imgmsg_to_cv2(depth_image_msg)
    except CvBridgeError as e:
        print(e)
    
    return depth_cv_image

def createFolder(directory, delete_previous=False):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
        elif os.path.exists(directory) and delete_previous:
            for file_path in glob.glob(directory + '*'):
                os.remove(file_path)
            os.rmdir(directory)
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


def get_object_center_point_in_world(object_image_center_x, object_image_center_y, depth_image, intrinsics, transform):    
    
    object_center = Point(np.array([object_image_center_x, object_image_center_y]), 'azure_kinect_overhead')
    object_depth = depth_image[object_image_center_y, object_image_center_x]
    print("x, y, z: ({:.4f}, {:.4f}, {:.4f})".format(
        object_image_center_x, object_image_center_y, object_depth))
    
    object_center_point_in_world = transform * intrinsics.deproject_pixel(object_depth, object_center)    
    print(object_center_point_in_world)

    return object_center_point_in_world 

def save_audio(dir_path, filename):
    cmd = "python scripts/sound_subscriber.py " + dir_path + filename 
    audio_p = subprocess.Popen("exec " + cmd, stdout=subprocess.PIPE, shell=True)
    return audio_p

def save_finger_vision(dir_path, filename):
    cmd = "python scripts/finger_vision_subscriber.py " + dir_path + filename 
    finger_vision_p = subprocess.Popen("exec " + cmd, stdout=subprocess.PIPE, shell=True)
    return finger_vision_p

def save_realsense(dir_path, filename, topic='/camera', use_depth=True):
    if use_depth:
        cmd = "python scripts/cutting_scripts/realsense_subscriber.py " + dir_path + filename + ' -d -t ' + topic
    else:
        cmd = "python scripts/cutting_scripts/realsense_subscriber.py " + dir_path + filename + ' -t ' + topic
    realsense_p = subprocess.Popen("exec " + cmd, stdout=subprocess.PIPE, shell=True)
    return realsense_p

def get_robot_positions_and_forces(franka, run_time):
    robot_positions = np.zeros((0,3))
    robot_forces = np.zeros((0,6))

    start_time = rospy.get_rostime()
    current_time = rospy.get_rostime()
    duration = rospy.Duration(run_time)
    while current_time - start_time < duration:
        robot_positions = np.vstack((robot_positions, franka.get_pose().translation.reshape(1,3)))
        robot_forces = np.vstack((robot_forces, franka.get_ee_force_torque().reshape(1,6)))
        rospy.sleep(0.01)
        current_time = rospy.get_rostime()

    return (robot_positions, robot_forces)

