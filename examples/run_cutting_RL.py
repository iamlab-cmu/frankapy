import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import math
import rospy
import argparse
import pickle
from autolab_core import RigidTransform, Point
from frankapy import FrankaArm
from cv_bridge import CvBridge

from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import ForcePositionSensorMessage, ForcePositionControllerSensorMessage
from franka_interface_msgs.msg import SensorDataGroup
from frankapy.utils import *

from tqdm import trange

from rl_utils import reps

def plot_sampled_new_dmp_traject_and_original_dmp(traject_time, initial_dmp_weights_pkl_file, new_dmp_traject, y0):
    #original_dmp_wts_pkl_filepath = '/home/sony/Desktop/debug_dmp_wts.pkl'
    dmp_traj = DMPPositionTrajectoryGenerator(traject_time)
    dmp_traj.load_saved_dmp_params_from_pkl_file(initial_dmp_weights_pkl_file)
    dmp_traj.parse_dmp_params_dict()
    # calculate dmp position trajectory - NOTE: this assumes a 0.001 dt for calc the dmp traject
    original_traject, dy, _, _, _ = dmp_traj.run_dmp_with_weights(y0) # y: np array(tx3)
    
    axes = ['x', 'y','z']   
    fig, ax = plt.subplots(3,1) 
    for i in range(3):
        #plt.figure()
        ax[i].plot(np.arange(0, traject_time, 0.001), original_traject[:,i])
        ax[i].plot(np.arange(0, traject_time, 0.001), new_dmp_traject[:,i])        
        ax[i].set_title('Cartesian Position - '+str(axes[i]))
        ax[i].set_ylabel('Position (m)')
        #ax[i].set_xlabel('Time (s)')
        ax[i].legend((axes[i] + '-original traject', axes[i] + '-new sampled traject'))
                        
        # plt.xlabel('Time (s)')
        # plt.ylabel('Position (m)')
    # plt.legend([ax[0],ax[1],ax[2]],(axes[i] + '-original traject', axes[i] + '-new sampled traject'))
    ax[2].set_xlabel('Time (s)')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--position_dmp_weights_file_path', '-w', type=str, default='/home/sony/092420_normal_cut_dmp_weights_zeroY.pkl')
    parser.add_argument('--dmp_traject_time', '-t', type=int, default = 5)  
    parser.add_argument('--num_epochs', '-e', type=int, default = 10)  
    parser.add_argument('--num_samples', '-s', type=int, default = 15)    
    parser.add_argument('--data_savedir', '-d', type=str, default='/home/sony/Documents/cutting_RL_experiments/data/')
    parser.add_argument('--exp_num', '-n', type=int)
    args = parser.parse_args()

    if not os.path.isdir(args.data_savedir + 'exp_' + str(args.exp_num)):
        createFolder(args.data_savedir + 'exp_' + str(args.exp_num))

    work_dir = args.data_savedir + 'exp_' + str(args.exp_num)

    position_dmp_pkl = open(args.position_dmp_weights_file_path,"rb")
    init_dmp_info_dict = pickle.load(position_dmp_pkl)

    print('Starting robot')
    fa = FrankaArm()
    # set tool delta pose and reset joints
    #tool_delta_pose = RigidTransform(translation=np.array([0.04, 0.16, 0.0]), from_frame='franka_tool', to_frame='franka_tool_base')
    #fa.set_tool_delta_pose(tool_delta_pose)
    
    reset_joint_positions = [ 0.02846037, -0.51649966, -0.12048514, -2.86642333, -0.05060268,  2.30209197, 0.7744833 ]
    fa.goto_joints(reset_joint_positions)    

    knife_orientation = np.array([[0.0,   0.9805069,  -0.19648464],
                                  [ 1.0,   0.0,  0.0],
                                  [ 0.0, -0.19648464,  -0.9805069]])
    
    # go to initial cutting pose
    starting_position = RigidTransform(rotation=knife_orientation, \
        translation=np.array([0.5, 0.01, 0.2]), #z=0.05
        from_frame='franka_tool', to_frame='world')    
    fa.goto_pose(starting_position, duration=5, use_impedance=False)

    # # move down to contact
    # move_down_to_contact = RigidTransform(translation=np.array([0.0, 0.0, -0.1]),
    # from_frame='world', to_frame='world')   

    # fa.goto_pose_delta(move_down_to_contact, duration=5, use_impedance=False, force_thresholds=[10.0, 10.0, 3.0, 10.0, 10.0, 10.0], ignore_virtual_walls=True)
    
    # Initialize Gaussian policy params (DMP weights) - mean and sigma
    initial_wts = np.array(init_dmp_info_dict['weights'])
    f_initial = -8 # TODO: maybe update?
    # NOTE: only using x dmp weights and z force value for policy params (not all 3 dims)
    initial_mu = np.append(initial_wts[0,:,:], f_initial)  #initial_wts.flatten()
    initial_sigma = np.diag(np.repeat(0.001, initial_mu.shape[0]))
    initial_sigma[-1,-1] = 120 # change exploration variance for force parameter
    mu, sigma = initial_mu, initial_sigma

    mean_params_each_epoch = []
    mean_params_each_epoch.append(initial_mu)
    for epoch in range(args.num_epochs):
        policy_params_all_samples, rewards_all_samples =[], []

        for sample in range(args.num_samples):
            print('Epoch: %i Sample: %i'%(epoch,sample))
            # Sample new policy params from mu and sigma - NOTE: cap force to be [-1, -40]
            new_params = np.random.multivariate_normal(mu, sigma)            
            new_x_weights = new_params[0:-1]
            new_z_force = new_params[-1]

            # cap force value
            new_z_force = np.clip(new_z_force, -20, -5)
            new_params[-1] = int(new_z_force)

            # save to policy params buffer
            policy_params_all_samples.append(new_params.tolist())
            
            # concat new sampled x weights w/ old y (zero's) and z weights
            new_weights = np.expand_dims(np.vstack((new_x_weights,initial_wts[1,:,:],initial_wts[2,:,:])),axis=1)
            #new_weights = new_weights.reshape(initial_wts.shape)
            
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
            weight_save_file = os.path.join(work_dir, 'weights_' + 'epoch'+str(epoch) + '_ep'+str(sample)+ '.pkl')
            save_weights(weight_save_file, data_dict)

            # Calculate dmp trajectory             
            traject_time = args.dmp_traject_time   # define length of dmp trajectory  
            # Load dmp traject params
            dmp_traj = DMPPositionTrajectoryGenerator(traject_time)
            dmp_traj.load_saved_dmp_params_from_pkl_file(weight_save_file)
            # DEBUG 
            #dmp_traj.load_saved_dmp_params_from_pkl_file(args.position_dmp_weights_file_path)
            dmp_traj.parse_dmp_params_dict()

            # Define starting position 
            start_pose = fa.get_pose()
            starting_rotation = start_pose.rotation
            y0 = start_pose.translation 
            # calculate dmp position trajectory - NOTE: this assumes a 0.001 dt for calc the dmp traject
            dmp_traject, dy, _, _, _ = dmp_traj.run_dmp_with_weights(y0) # y: np array(tx3)
            
            # check new dmp sampled wt trajectory vs original
            plot_sampled_new_dmp_traject_and_original_dmp(traject_time, \
                args.position_dmp_weights_file_path, dmp_traject, y0)
            import pdb; pdb.set_trace()

            # sampling info for sending msgs via ROS
            dt = 0.01 
            T = traject_time
            ts = np.arange(0, T, dt)
            N = len(ts)

            # downsample dmp traject 
            downsmpled_dmp_traject = downsample_dmp_traject(dmp_traject, 0.001, dt)
            target_poses = get_dmp_traj_poses_reformatted(downsmpled_dmp_traject, starting_rotation) # target_poses is a nx16 list of target poses at each time step
            import pdb; pdb.set_trace()

            target_force = [0, 0, 0, 0, 0, 0] #[0, 0, new_z_force, 0, 0, 0] #[0, 0, -40, 0, 0, 0]   
            S = [1, 1, 1, 1, 1, 1] #[1, 1, 0, 1, 1, 1]
            position_kps_cart = FC.DEFAULT_TRANSLATIONAL_STIFFNESSES + FC.DEFAULT_ROTATIONAL_STIFFNESSES
            force_kps_cart = [0.1] * 6
            position_kps_joint = FC.DEFAULT_K_GAINS
            force_kps_joint = [0.1] * 7

            rospy.loginfo('Initializing Sensor Publisher')
            pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1)
            rate = rospy.Rate(1 / dt)
            n_times = 1
            rospy.loginfo('Publishing HFPC trajectory w/ cartesian gains...')
            
            current_ht = fa.get_pose().translation[2]
            dmp_num = 0 

            # start FP skill
            fa.run_dynamic_force_position(duration=T * 10, buffer_time = 3, S=S,
                                            use_cartesian_gains=True,
                                            position_kps_cart=position_kps_cart,
                                            force_kps_cart=force_kps_cart, block=False)

            # sample from gaussian to get dmp weights for this execution            
            dmp_num = 0            
            peak_forces_all_dmps, x_mvmt_all_dmps = [], [] # sum of abs (+x/-x mvmt)            
            while current_ht > 0.023:   
                print('starting dmp', dmp_num) 
                robot_positions = np.zeros((0,3))
                robot_forces = np.zeros((0,6))       
                init_time = rospy.Time.now().to_time()
                for i in trange(N * n_times):
                    t = i % N                 
                    timestamp = rospy.Time.now().to_time() - init_time
                    #NOTE: format of pose sent is: 1x16 Transform matrix 
                    
                    traj_gen_proto_msg = ForcePositionSensorMessage(
                        id=i, timestamp=timestamp, seg_run_time=dt,
                        pose=target_poses[t],
                        force=target_force
                    )
                    fb_ctrlr_proto = ForcePositionControllerSensorMessage(
                        id=i, timestamp=timestamp,
                        position_kps_cart=position_kps_cart,
                        force_kps_cart=force_kps_cart,
                        selection=S
                    )
                    ros_msg = make_sensor_group_msg(
                        trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                            traj_gen_proto_msg, SensorDataMessageType.FORCE_POSITION),
                        feedback_controller_sensor_msg=sensor_proto2ros_msg(
                            fb_ctrlr_proto, SensorDataMessageType.FORCE_POSITION_GAINS)
                        )
                    
                    robot_positions = np.vstack((robot_positions, fa.get_pose().translation.reshape(1,3)))
                    robot_forces = np.vstack((robot_forces, fa.get_ee_force_torque().reshape(1,6)))
                    
                    pub.publish(ros_msg)
                    rate.sleep()                                            

                peak_force = np.max(np.abs(robot_forces[:,2]))
                total_x_mvmt = (np.max(np.abs(robot_positions[:,0]) - np.abs(robot_positions[0,0]))) + \
                    (np.max(np.abs(robot_positions[:,0]) - np.abs(robot_positions[-1,0])))
                peak_forces_all_dmps.append(peak_force)
                x_mvmt_all_dmps.append(total_x_mvmt)
                # np.savez(work_dir +'trial_info_'+str(dmp_num)+'.npz', robot_positions=robot_positions, \
                #     robot_forces=robot_forces)
                
                completed_cut = input('cut complete? (1 yes/0 n): ')

                while completed_cut not in ['0', '1']:
                    completed_cut = input('please enter valid answer. cut complete? (1/0): ') 

                if completed_cut == '1': 
                    break

                elif completed_cut == '0':
                    current_ht = fa.get_pose().translation[2]
                    print('current_ht', current_ht)
                    dmp_num += 1  

                    # calculate new dmp traject based on current position
                    y0 = fa.get_pose().translation
                    # calculate dmp position trajectory - NOTE: this assumes a 0.001 dt for calc the dmp traject
                    dmp_traject, dy, _, _, _ = dmp_traj.run_dmp_with_weights(y0) # y: np array(tx3)
                    # downsample dmp traject and reformat target_poses
                    downsmpled_dmp_traject = downsample_dmp_traject(dmp_traject, 0.001, dt)
                    target_poses = get_dmp_traj_poses_reformatted(downsmpled_dmp_traject, starting_rotation) # target_poses is a nx16 list of target poses at each time step
                
            
            # After finishing set of dmps for a full slice - calculate avg reward here
            #   avg peak force, avg back and forth mvmt ? 
            import pdb; pdb.set_trace()
            avg_peak_force = np.mean(peak_forces_all_dmps)
            avg_x_mvmt = np.mean(x_mvmt_all_dmps)
            reward = -avg_peak_force - 10*avg_x_mvmt

            # save reward to buffer
            print('Epoch: %i Sample: %i Reward: '%(epoch,sample), reward)
            rewards_all_samples.append(reward)

            # reset to starting cut position
            fa.goto_pose(starting_position, duration=5, use_impedance=False)
        
        # save reward 
        import pdb; pdb.set_trace()
        np.save(os.path.join(work_dir, 'polParams_rewards_' + 'epoch'+str(epoch) +'.npy', \
            np.concatenate((np.array(policy_params_all_samples), np.array([rewards_all_samples]).T), axis=1)))

        # update policy mean and cov (REPS)        
        reps = rl_utils.Reps(rel_entropy_bound=1.5,min_temperature=0.001) #Create REPS object
        policy_params_mean, policy_params_sigma, reps_info = \
            reps.policy_from_samples_and_rewards(policy_params_all_samples, rewards_all_samples)

        import pdb; pdb.set_trace()
    

    fa.goto_joints(reset_joint_positions)

    