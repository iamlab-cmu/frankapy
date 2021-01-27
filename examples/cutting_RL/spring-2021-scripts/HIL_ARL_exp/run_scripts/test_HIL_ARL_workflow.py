import argparse

import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt
from matplotlib import cm
from autolab_core import YamlConfig, RigidTransform
import open3d as o3d


from visualization.visualizer3d import Visualizer3D as vis3d

from carbongym import gymapi
from carbongym_utils.scene import GymScene
from carbongym_utils.camera import GymCamera, CameraZMQPublisher
from carbongym_utils.assets import GymFranka, GymBoxAsset
from carbongym_utils.policy import RandomDeltaJointPolicy, GraspBlockPolicy, StackObjectARLObjectEmbed
from perception.camera_intrinsics import CameraIntrinsics
from carbongym_utils.draw import draw_transforms
from carbongym_utils.math_utils import RigidTransform_to_transform, vec3_to_np, rot_from_np_quat, quat_to_np, quat_to_rpy, np_to_quat
from carbongym_utils.open3d_utils import *
from carbongym_utils.voxelData_utils import IsaacVoxelSceneLoader
from carbongym_utils.robot_interface_utils.action_relation.trainer.test_strip_emb_model import get_embeddings_for_isaac_data

from carbongym_utils.policy_learner_blockGrasp import REPSPolicyLearnerObjEmbed
from carbongym_utils.reward_learner_blockGrasp import RewardLearnerObjEmbedDKL, GPRegressionModel

def load_eigvec(eigvec_filepath):
    eigvecs = np.genfromtxt(eigvec_filepath)
    return eigvecs

def project_original_data_to_PC_space(eigvecs, obj_embed_raw):
    proj = eigvecs.T.dot(obj_embed_raw.T)
    proj = proj.T
    return proj

def use_k_PCs_from_projected_data(proj, num_PCs):
    proj_kPCs = proj[:, 0:num_PCs]
    return proj_kPCs #returns num_samples x k_PCs array (reduced dimensionality vector) 

def get_lower_dimens_obj_embed(eigvec_filepath, obj_embed_raw, num_PCs):
    eigvecs = load_eigvec(eigvec_filepath)
    proj_PCA_space = project_original_data_to_PC_space(eigvecs, obj_embed_raw)
    proj_kPCs = use_k_PCs_from_projected_data(proj_PCA_space, num_PCs)

    return proj_kPCs

def reshape_voxel_data_to_2D(input_voxel_data):
    num_samples = input_voxel_data.shape[0]
    input_voxel_data = input_voxel_data.reshape([num_samples, 375000])
    return input_voxel_data

def reshape_voxel_data_to_5D(input_voxel_data):
    num_samples = input_voxel_data.shape[0]
    input_voxel_data = input_voxel_data.reshape([num_samples, 3, 50, 50, 50])
    return input_voxel_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', '-c', type=str, default='cfg/run_franka_block_stack_w_voxel_emb.yaml')
    parser.add_argument('--kappa', type=int, default=5)
    parser.add_argument('--initial_policy_var', type=float, default= 0.02)
    parser.add_argument('--num_episodes_initial_epoch', type=int, default = 20)
    parser.add_argument('--num_episodes_later_epochs', type=int, default = 20)
    parser.add_argument('--rel_entropy_bound', type=float, default = 1.2)
    parser.add_argument('--num_EPD_epochs', type=int, default = 5)
    parser.add_argument('--GP_training_epochs_initial', type=int, default = 120)
    parser.add_argument('--GP_training_epochs_later', type=int, default = 11)
    args = parser.parse_args()

    kappa = args.kappa   
    initial_var = args.initial_policy_var   
    num_episodes_initial_epoch = args.num_episodes_initial_epoch     
    num_episodes_later_epochs = args.num_episodes_later_epochs
    rel_entropy_bound = args.rel_entropy_bound
    num_EPD_epochs = args.num_EPD_epochs
    GP_training_epochs_initial = args.GP_training_epochs_initial
    GP_training_epochs_later = args.GP_training_epochs_later
    cfg = YamlConfig(args.cfg)

    anchor_obj_poses, other_obj_poses = [], []    

    scene = GymScene(cfg['scene'])
    
    table = GymBoxAsset(scene.gym, scene.sim, **cfg['table']['dims'], 
                        shape_props=cfg['table']['shape_props'], 
                        asset_options=cfg['table']['asset_options']
                        )
    franka = GymFranka(cfg['franka'], scene.gym, scene.sim, actuation_mode='attractors')

    block = GymBoxAsset(scene.gym, scene.sim, **cfg['block']['dims'], 
                    shape_props=cfg['block']['shape_props'], 
                    rb_props=cfg['block']['rb_props'],
                    asset_options=cfg['block']['asset_options']
                    )

    table_pose = RigidTransform_to_transform(RigidTransform(
        translation=[cfg['table']['dims']['width']/3, cfg['table']['dims']['height']/2, 0]
    ))
    franka_pose = RigidTransform_to_transform(RigidTransform(
        translation=[0, cfg['table']['dims']['height'] + 0.01, 0],
        rotation=RigidTransform.quaternion_from_axis_angle([-np.pi/2, 0, 0])
    ))

    # add things to the scene
    scene.add_asset('table0', table, table_pose)
    scene.add_asset('franka0', franka, franka_pose, collision_filter=2) # avoid self-collision
    scene.add_asset('block0', block, gymapi.Transform()) # we'll sample block poses later
    scene.add_asset('block1', block, gymapi.Transform()) # we'll sample block poses later

    cam = GymCamera(scene.gym, scene.sim, cam_props=cfg['camera'])
    cam_offset_transform = RigidTransform_to_transform(RigidTransform(
        rotation=RigidTransform.x_axis_rotation(-np.pi/2) @ RigidTransform.z_axis_rotation(-np.pi/2),
        translation=np.array([-0.046490, -0.083270, 0])
    ))
    scene.attach_camera('hand_cam0', cam, 'franka0', 'panda_hand', offset_transform=cam_offset_transform)
    
    def custom_draws(scene):
        for env_idx, env_ptr in enumerate(scene.env_ptrs):
            ee_transform = franka.get_ee_transform(env_ptr, 'franka0')
            desired_ee_transform = franka.get_desired_ee_transform(env_idx, 'franka0')
            transforms = [ee_transform, desired_ee_transform]          
            draw_transforms(scene.gym, scene.viewer, [env_ptr], transforms)

    def cb(scene, t_step, t_sim):         
        scene.render_cameras()
        color, depth, seg = cam.frames(scene.ch_map[0]['hand_cam0'], 'hand_cam0')
        #cam_pub.pub(color, depth, seg)
        #if t_step == 695:        
        for i, env_ptr in enumerate(scene.env_ptrs):     
            #import pdb; pdb.set_trace()  
            #Get object pose         
            block0_ah = scene.ah_map[i]['block0']
            block1_ah = scene.ah_map[i]['block1']

            anchor_pose = scene._assets[0]['block0'].get_rb_transforms(env_ptr, block0_ah)[0]
            # anchor_pose_arr = np.array([[anchor_pose.p.x, anchor_pose.p.y, anchor_pose.p.z, \
            #     anchor_pose.r.x, anchor_pose.r.y, anchor_pose.r.z, anchor_pose.r.w]])
            anchor_pose_arr = np.array([[anchor_pose.p.x, anchor_pose.p.y, anchor_pose.p.z]]) #just xyz positions
            other_pose = scene._assets[0]['block1'].get_rb_transforms(env_ptr, block1_ah)[0]
            # other_pose_arr = np.array([[other_pose.p.x, other_pose.p.y, other_pose.p.z, \
            #     other_pose.r.x, other_pose.r.y, other_pose.r.z, other_pose.r.w]])
            other_pose_arr = np.array([[other_pose.p.x, other_pose.p.y, other_pose.p.z]]) #just xyz positions

            anchor_obj_poses.append(anchor_pose_arr)
            other_obj_poses.append(other_pose_arr)

        #import pdb; pdb.set_trace()

        return anchor_obj_poses, other_obj_poses
    
    #TO DO: need to save block poses from each episode
    def query_expert_for_reward(scene, agent, franka, num_dims, block, samples_to_query, block0_poses, block1_poses, policy_params_all):
        print('querying expert for rewards')
        expert_rewards_for_samples = []
        for sample in samples_to_query: 
            #set block poses for each env
            for i, env_ptr in enumerate(scene.env_ptrs):
                block.set_rb_rigid_transforms(env_ptr, scene.ah_map[i]['block0'], [block0_poses[0]])
                block.set_rb_rigid_transforms(env_ptr, scene.ah_map[i]['block1'], [reset_block1_poses[0]])

            #import pdb; pdb.set_trace()

            policy = StackObjectARLObjectEmbed(franka, 'franka0', num_dims, block, 'block0', 'block1', policy_params_all[sample])         
            policy.reset()
            scene.run(time_horizon=policy.time_horizon, policy=policy, custom_draws=custom_draws, cb=cb)
            #import pdb; pdb.set_trace()
            while True: 
                try:
                    expert_reward=input('enter expert reward for this execution: ')
                    expert_reward = float(expert_reward)
                    break
                except ValueError:
                    print("enter valid reward (float)")

            expert_rewards_for_samples.append(expert_reward)
        expert_rewards_for_samples= np.array(expert_rewards_for_samples)
        return expert_rewards_for_samples
    
    def check_policy_converged(scene, agent, franka, num_dims, block, block0_poses, reset_block1_poses, policy_params_mean):
        successful_grasps=0
        num_rollouts=2
        for i in range(0,num_rollouts):            
            
            #set block poses for each env
            for i, env_ptr in enumerate(scene.env_ptrs):
                block.set_rb_rigid_transforms(env_ptr, scene.ah_map[i]['block0'], [block0_poses[0]])
                block.set_rb_rigid_transforms(env_ptr, scene.ah_map[i]['block1'], [reset_block1_poses[0]])
            
            policy = StackObjectARLObjectEmbed(franka, 'franka0', num_dims, block, 'block0', 'block1', policy_params_mean)   
            policy.reset()
            scene.run(time_horizon=policy.time_horizon, policy=policy, custom_draws=custom_draws, cb=cb)
            grasp_success = str(input('grasp_successful (y/n)? '))
            if grasp_success=='y':
                successful_grasps+=1
            elif grasp_success=='n':
                break
        if successful_grasps==num_rollouts:
            policy_converged = True
        else:
            policy_converged = False
        return policy_converged

   
    # sample ANCHOR (block 0) poses for each env 
    block0_poses = [RigidTransform(
        translation=[(np.random.rand()*2 - 1) * 0.1 + 0.5, 
        cfg['table']['dims']['height'] + cfg['block']['dims']['height'] / 2 + 0.02, 
        (np.random.rand()*2 - 1) * 0.2]) for _ in range(scene.n_envs)]

    # set OTHER block (block 1) pose for now (reset position) - will change when robot grasps during policy    
    reset_block1_poses = [RigidTransform(
        translation=[0.5, 
        cfg['table']['dims']['height'] + cfg['block']['dims']['height'] / 2 + 0.02, 
        block0_poses[0].translation[2] + 0.15] #right of robot is -z (facing robot), left of robot is +z
    ) for _ in range(scene.n_envs)]
    
    # set ANCHOR block poses for each env and prelimin OTHER block poses
    for i, env_ptr in enumerate(scene.env_ptrs):
        block.set_rb_rigid_transforms(env_ptr, scene.ah_map[i]['block0'], [block0_poses[i]])
        block.set_rb_rigid_transforms(env_ptr, scene.ah_map[i]['block1'], [reset_block1_poses[i]])

    # Instantiate Policy Learner (agent), Reward Learner, voxel data loader
    voxelDataLoader = IsaacVoxelSceneLoader(0.05, 0.01, [50, 50, 50])
    agent = REPSPolicyLearnerObjEmbed()


    # instantiate reward learner - note: GPR model not instantiated yet
    # kappa = 5    # defined in args
    reward_learner = RewardLearnerObjEmbedDKL(kappa)
    beta = 0.001 # fixed gaussian noise likelihood

    # initialize policy
    # initial_var = 0.02 # 0.03 #0.01   # defined in args
    # ---------------- STARTING FROM SET INITIALIZED POLICY - FOR COMPARISON
    initial_u = [-0.07, 0.0, 0.07] #[-0.08, 0.0, 0.08]  #[-0.06, 0.0, 0.06]   #[-0.03, 0.0, 0.05] 
    initial_var = np.repeat(initial_var, len(initial_u))
    initial_sigma = np.diag(initial_var)
    # ---------------------------------------------------
    num_dims = 2 #x/y/z policy params

    u = initial_u
    sigma = initial_sigma
    print('initial policy mean', u)  
    
    #num_episodes_initial_epoch = 20  # defined in args
    num_episodes = num_episodes_initial_epoch 
    num_epochs = 10
    epoch = 0
    converged = False
    total_queried_samples_each_epoch, mean_expert_rewards_all_epochs, mean_reward_model_rewards_all_epochs = [], [], [] #track number of queries to expert for rewards and average rewards for each epoch
    training_data_list, queried_samples_all = [], []
    GP_training_data_x_all = np.empty([0,3,50,50,50])
    GP_training_data_y_all =  np.empty([0])
    queried_outcomes, queried_expert_rewards, policy_params_all_epochs, block_poses_all_epochs = [], [], [], []
    reward_model_rewards_all_mean_buffer = []

    anchor_obj_poses_all, other_obj_poses_all = [], [] 
    total_samples = 0
    for epoch in range(num_epochs):
        print('epoch', epoch)
        print('policy mean', u)
        #Initialize lists to save data           
        expert_rewards_all, reward_model_rewards_all_mean, reward_model_rewards_all_cov, \
            policy_params_all, outcomes_all=[],[],[],[],[]
        block_poses_all = []        
        
        if epoch > 0:  
            #increase query threshold and number of samples for later epochs
            #reward_learner.kappa = 30 
            num_episodes = num_episodes_later_epochs #20
        
        for episode in range(num_episodes):
            print('episode', episode)
            total_samples += 1
            
            #Sample policy params: these are delta_x, delta_y, delta_z of other block relative to anchor
            sampled_new_params = agent.sample_params_from_policy(u, sigma)    
            print('sampled new params', sampled_new_params)      
            policy_params_all.append(sampled_new_params) # NOTE: not currently using policy_params_all, only using policy_params_all_epochs
            policy_params_all_epochs.append(sampled_new_params)         

            #reset other block (1) poses            
            for i, env_ptr in enumerate(scene.env_ptrs):
                block.set_rb_rigid_transforms(env_ptr, scene.ah_map[i]['block1'], [reset_block1_poses[i]])   

            policy = StackObjectARLObjectEmbed(franka, 'franka0', num_dims, block, 'block0', 'block1', sampled_new_params)            
            policy.reset()
            scene.run(time_horizon=policy.time_horizon, policy=policy, custom_draws=custom_draws, cb=cb)

            # Get outcomes from sample: 
            # OBJ EMBED
            anchor_xyz_pos, other_xyz_pos = agent.get_obj_poses_pre_drop(anchor_obj_poses, other_obj_poses)
            anchor_obj_poses_all.append(anchor_obj_poses)
            other_obj_poses_all.append(other_obj_poses)   

            # # get 3D array voxel repr of scene (w/ both blocks - centered on anchor) - pull this out in a dataloader function
            voxel_data = voxelDataLoader.get_3d_voxel_repr_two_obj_one_env(anchor_xyz_pos, other_xyz_pos)                       
           
            #outcomes_from_sample = voxel_emb_arr_lower_dim
            outcomes_from_sample = voxel_data
            
            outcomes_all.append(outcomes_from_sample)      
            
            #clear state data buffers
            anchor_obj_poses, other_obj_poses = [], []

            #Get reward from reward learner model based on CURRENT reward model (GP model not trained yet if epoch 0)
            if epoch == 0: #if epoch = 0, GP model hasn't been trained so mean = 0
                '''
                might not need to add these since we don't have reward model rewards in first epoch
                since GP model hasn't been trained yet
                '''
                reward_model_rewards_all_mean.append(0)
                reward_model_rewards_all_cov.append(0)
                reward_model_rewards_all_mean_buffer.append(0)

                training_data_list.append([outcomes_from_sample, 0, 0,\
                sampled_new_params])

            elif epoch != 0: 
                mean_expected_reward, var_expected_reward = reward_learner.calc_expected_reward_for_observed_outcome_w_GPmodel(gpr_reward_model, \
                    likelihood, outcomes_from_sample)
                print('reward model reward = ', mean_expected_reward)
                
                #Save expected reward mean and var to lists and add to training_data_list of all training data
                if type(mean_expected_reward[0])==np.ndarray:
                    '''
                    probably don't need this if statement anymore
                    '''
                    reward_model_rewards_all_mean.append(mean_expected_reward[0][0])
                    reward_model_rewards_all_cov.append(var_expected_reward[0])
                    reward_model_rewards_all_mean_buffer.append(mean_expected_reward[0][0])

                    training_data_list.append([outcomes_from_sample, mean_expected_reward[0][0], var_expected_reward[0],\
                    sampled_new_params])

                else:
                    reward_model_rewards_all_mean.append(mean_expected_reward[0])
                    print('GP model rewards mean all eps', reward_model_rewards_all_mean)
                    reward_model_rewards_all_cov.append(var_expected_reward[0])
                    print('GP model rewards var all eps', reward_model_rewards_all_cov)
                    reward_model_rewards_all_mean_buffer.append(mean_expected_reward[0])

                    # adding voxel_data to training data list 
                    '''
                    outcomes_from_samples here are the voxel data from the scene 
                    '''
                    training_data_list.append([outcomes_from_sample, mean_expected_reward[0], var_expected_reward[0],\
                    sampled_new_params])    
        
        #Save mean expert rewards and reward model reward from this rollout of 15 episodes        
        mean_reward_model_rewards_all_epochs.append(np.mean(reward_model_rewards_all_mean))

        # current policy pi_current (before updating)
        pi_current_mean = u
        pi_current_cov = sigma

        # Update policy - REPS (after 1st epoch)        
        #import pdb; pdb.set_trace()
        if epoch!=0:
            #use previous samples to update policy as well            
            u, sigma = agent.update_policy_REPS(reward_model_rewards_all_mean_buffer[num_episodes_initial_epoch:], \
                policy_params_all_epochs[num_episodes_initial_epoch:], \
                rel_entropy_bound, min_temperature=0.001) #rel_entropy_bound = 1.2, now defined in CL args
            print('REPS policy update:')
            print('u updated', u)
            print('sigma updated', sigma)

            # pi_tilda is the new policy under the current reward model 
            pi_tilda_mean = u
            pi_tilda_cov = sigma
      
        #check if policy converged:
        if epoch!=0:
            policy_converged = check_policy_converged(scene, agent, franka, num_dims, block, block0_poses, reset_block1_poses, u)
            if policy_converged == True:
                print('policy_converged', policy_converged)
                print('policy_converged after %i epochs '%(epoch+1))
                print('policy converged. mean policy params: ', u)
                break

        # #Reward model: Evaluate sample outcomes from above set of iterations and determine outcomes to query from expert        
        # only compute EPD if epoch!=0 (i.e. reward model has been trained on initial set of data)
        if epoch == 0:
            # query all samples if epoch = 0
            samples_to_query = np.arange(num_episodes).tolist() # query all samples
            queried_outcomes = np.squeeze(np.array(outcomes_all)) # use all outcomes

        else:
            import pdb; pdb.set_trace()
            #num_EPD_epochs = 5 # define in CL args
            samples_to_query, queried_outcomes  = reward_learner.compute_EPD_for_each_sample_updated(num_EPD_epochs, optimizer, \
                gpr_reward_model, likelihood, mll, agent, pi_tilda_mean, pi_tilda_cov, pi_current_mean, pi_current_cov, \
                    training_data_list, queried_samples_all, GP_training_data_x_all, GP_training_data_y_all, beta) 
            
        if samples_to_query!=[]:
            #import pdb; pdb.set_trace()       
            queried_expert_rewards = query_expert_for_reward(scene, agent, franka, num_dims, block, samples_to_query, \
                block0_poses, reset_block1_poses, policy_params_all_epochs)

        # save all queried outcomes and queried rewards in buffer to send to GPytorch model as training data everytime it get updated:
        #   train_x is queried_outcomes ((nxD) arr), train_y is queried_expert_rewards ((n,) arr)
        GP_training_data_x_all = np.vstack((GP_training_data_x_all, queried_outcomes))
        GP_training_data_y_all = np.concatenate((GP_training_data_y_all, queried_expert_rewards))
        import pdb; pdb.set_trace()       

        # Add samples to query to running list of queried_samples
        queried_samples_all = queried_samples_all + samples_to_query # running list of queried samples

        # #Keep track of number of queried samples 
        #import pdb; pdb.set_trace()
        if epoch > 0:
            num_prev_queried = total_queried_samples_each_epoch[epoch-1]
            total_queried_samples_each_epoch.append(num_prev_queried + len(samples_to_query))    
        else:
            total_queried_samples_each_epoch.append(len(samples_to_query))   
        
        # initialize reward GP model 
        if epoch == 0:
            likelihood = gpytorch.likelihoods.GaussianLikelihood()            
            train_x = torch.from_numpy(queried_outcomes)
            train_x = train_x.float()
            train_x = reshape_voxel_data_to_2D(train_x)            
            train_y = torch.from_numpy(queried_expert_rewards)
            train_y = train_y.float()
            print('train_y variance', train_y.var())
            # add white noise 
            #likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(torch.ones(train_x.shape[0]) * beta)

            gpr_reward_model = GPRegressionModel(train_x, train_y, likelihood) 
            optimizer = torch.optim.Adam([
                {'params': gpr_reward_model.embedd_extractor.parameters()},
                {'params': gpr_reward_model.covar_module.parameters()},
                {'params': gpr_reward_model.mean_module.parameters()},
                {'params': gpr_reward_model.likelihood.parameters()},
            ], lr=0.01) # lr = 0.01 originally 
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gpr_reward_model)
            # train reward GP model given initial train_x and train_y data (voxel_data, queried human rewards)
            #GP_training_epochs_initial = 120 #300 # now defined in CL args
            reward_learner.train_GPmodel(GP_training_epochs_initial, optimizer, gpr_reward_model, likelihood, mll, train_x, train_y)
            import pdb; pdb.set_trace()            

        #Update Reward model GP if there are any outcomes to query [outcome/s, expert reward/s]
        else: # update reward model with new training data
            if queried_outcomes.size!=0:  
                print('updating reward model')
                updated_train_x = GP_training_data_x_all
                updated_train_y = GP_training_data_y_all                
                #GP_training_epochs_later = 11 #100  # how long to train during updates?? # now define in CL args
                reward_learner.update_reward_GPmodel(GP_training_epochs_later, optimizer, gpr_reward_model, likelihood, mll, updated_train_x, updated_train_y)
                                       
                import pdb; pdb.set_trace()
        import pdb; pdb.set_trace()
            
    print('total samples', total_samples)
    # last epoch did not query samples (if policy converged), so add 0 to queried samples list
    total_queried_samples_each_epoch.append(total_queried_samples_each_epoch[-1])
    # total_queried_samples_each_epoch is CUMULATIVE queries samples
    print('cumulative queried samples', total_queried_samples_each_epoch)
    import pdb; pdb.set_trace()
    plt.plot(np.arange(epoch+1), total_queried_samples_each_epoch)
    plt.xlabel('epochs')
    plt.xticks(np.arange((epoch+1)))
    plt.ylabel('cumulative human queried samples')
    plt.title('human queries vs epochs, total samples = %i'%total_samples)
    plt.show()

    # TODO: Save model if converged! --> so we can analyze the updated embedding model
    torch.save(gpr_reward_model.state_dict(),'/home/test2/Documents/obj-rel-embeddings/AS_data/test_DKL_GP/Isaac_exps/model_state_checkpt.pth')
    import pdb; pdb.set_trace()
    








