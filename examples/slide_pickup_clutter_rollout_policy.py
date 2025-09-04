import numpy as np
import copy
from omegaconf import OmegaConf
import cv2
import rospy
import random, time
import matplotlib.pyplot as plt
from std_msgs.msg import Float32MultiArray

from frankapy.franka_perception import Perception
from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from franka_interface_msgs.msg import SensorDataGroup

from frankapy.proto import ShouldTerminateSensorMessage
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
import gym
def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print('Chosen grasp position x = %d, y = %d'%(x, y))
        param[0] = x
        param[1] = y

def signed_dist_fn_rectangle(grid_x, x_target_min, x_target_max, obstacle=False, plot=False):
    # Compute distances to each edge of the rectangle
    dist_from_walls = np.maximum(x_target_min - grid_x, grid_x - x_target_max)
    signed_distance_grid = np.max(dist_from_walls, axis=-1)
    if obstacle:
        signed_distance_grid = -1*signed_distance_grid
    return signed_distance_grid

class FrankaSlidePickupClutterEnv:
    def __init__(self, cfg):
        self.env_cfg = cfg.env_cfg

        self.fa = FrankaArm()
        self.percept = Perception(cfg=cfg.sensor_cfg)

        self.drag_z_height = self.env_cfg.drag_z_height
        self.table_height = 0.0548 # 0.0548
        self.des_init_ee_height = 0.06
        self.shift_gripper_height_by = -self.table_height

        self.doneType = self.env_cfg.doneType
        self.costType = self.env_cfg.costType
        self.return_type = self.env_cfg.return_type
        self.scaling_target = self.env_cfg.scaling_target
        self.scaling_safety = self.env_cfg.scaling_safety
        self.observations = self.env_cfg.observations
        self.action_space = gym.spaces.Box(np.array(self.env_cfg.control_low), np.array(self.env_cfg.control_high))

        self.n_rel_objs = self.env_cfg.n_rel_objs
        self.constraint_type_repr = self.env_cfg.get('constraint_type_repr', 'int')
        self.normalize_pos = self.env_cfg.get('normalize_pos', False)
        self.multi_goal = self.env_cfg.get('multi_goal', False)
        self.randomize_constraint_types = self.env_cfg.get('randomize_constraint_types', True)

        self.all_blocks_object_names = [self.env_cfg.block_bottom.block_name, self.env_cfg.block_top.block_name] + self.env_cfg.objects.names

        self.N_all_objects = len(self.all_blocks_object_names)
        self.N_dist_objects = len(self.env_cfg.objects.names)

        self.object_type_to_int_mapping = {
            f'ee': 0,
            f'{self.env_cfg.block_bottom.block_name}': 1,
            f'{self.env_cfg.block_top.block_name}': 2
        }
        self.constraint_to_int_mapping = {
            'any_contact': 3,
            'soft_contact': 4,
            'no_contact': 5,
            'no_over': 6,
        }
        self.max_constraint_types = 7

        if self.constraint_type_repr == 'int':
            self.low_dim_sizes = {'robot0': 7, 'objects': 7, 'semantics': 1}
        elif self.constraint_type_repr == 'one_hot':
            self.low_dim_sizes = {'robot0': 6+self.max_constraint_types, 'objects': 6+self.max_constraint_types, 'semantics': self.max_constraint_types}

        if self.multi_goal:
            self.target_set_to_int_mapping = {t_name: t for t, t_name in enumerate(list(self.env_cfg.block_bottom.multi_goal_target_sets.keys()))}
            self.low_dim_sizes['robot0'] += 1
            self.low_dim_sizes['objects'] += 1
            self.low_dim_sizes['semantics'] += 1

        self.low_dim_sizes['full_size'] = self.low_dim_sizes['robot0'] + (2+self.n_rel_objs)*self.low_dim_sizes['objects']

        self.set_observation_shapes()

        self.obj_to_constraint_map = self.env_cfg.obj_to_constraint_map
        self.obj_to_constraint_map_gt = copy.deepcopy(self.obj_to_constraint_map)

        assert self.env_cfg.n_rel_objs <= len(self.env_cfg.objects.names)
        
        self.u_min = np.array(self.env_cfg.control_low)
        self.u_max = np.array(self.env_cfg.control_high)

        self.env_bound_low = np.array(self.env_cfg.env_bounds.low)
        self.env_bound_high = np.array(self.env_cfg.env_bounds.high)

        self.tracker_sub_callback_cnt = 0
        self.last_time = time.time()
        self.objects_pos_vel_t = {}
        self.update_ee_block_bot_pos()
        self.obj_pos_sub = rospy.Subscriber('objects_world_pos_topic', Float32MultiArray, self.tracker_sub_callback)

    def tracker_sub_callback(self, msg):
        if (self.tracker_sub_callback_cnt == 0):
            data_xyz = np.array(msg.data).reshape(1+self.n_rel_objs, 3)

            # self.objects_pos_vel_t = {}
            bt_pos = data_xyz[0]
            bt_pos[2] = bt_pos[2] - self.env_cfg.block_top.size[2]
            self.objects_pos_vel_t[self.env_cfg.block_top.block_name] = np.append(bt_pos, [0,0,0])
            for i, name in enumerate(self.env_cfg.objects.names):
                self.objects_pos_vel_t[name] = np.append(data_xyz[1+i], [0,0,0])
                self.objects_pos_vel_t[name][2] = 0
            self.objects_pos_vel_t[self.env_cfg.block_bottom.block_name] = self.block_bottom_pos_vel

            self.objects_pos_vel_tm1 = copy.deepcopy(self.objects_pos_vel_t)
            self.last_time = time.time()
        else:
            current_time = time.time()
            time_diff = current_time - self.last_time  # Time since last callback
            self.last_time = current_time  # Update the last time

            self.objects_pos_vel_tm1 = copy.deepcopy(self.objects_pos_vel_t)
            data_xyz = np.array(msg.data).reshape(1+self.n_rel_objs, 3)

            # block top pos
            bt_pos = data_xyz[0]
            bt_pos[2] = bt_pos[2] - self.env_cfg.block_top.size[2]
            vel = (bt_pos-self.objects_pos_vel_tm1[self.env_cfg.block_top.block_name][:3])/time_diff
            self.objects_pos_vel_t[self.env_cfg.block_top.block_name] = np.append(bt_pos, vel)

            # Other objects pos
            for i, name in enumerate(self.env_cfg.objects.names):
                # vel = (data_xyz[1+i]-self.objects_pos_vel_tm1[name][:3])/time_diff
                # self.objects_pos_vel_t[name] = np.append(data_xyz[1+i], vel)
                # self.objects_pos_vel_t[name][2] = self.env_cfg.objects[name].size[2]
                # self.objects_pos_vel_t[name][2] = 0
                self.objects_pos_vel_t[name] = copy.deepcopy(self.objects_pos_vel_tm1[name])
            
            # bottom block pos
            self.objects_pos_vel_t[self.env_cfg.block_bottom.block_name] = self.block_bottom_pos_vel
        self.tracker_sub_callback_cnt += 1

    def set_observation_shapes(self):
        # TODO(saumya): this function needs to be fixed so that observation shapes correspond to relevant objects not all objects in the env
        self.observation_shapes = {}
        self.observation_shapes['robot_state'] = (self.low_dim_sizes['robot0'],)
        if 'objects_state' in self.observations['low_dim']:
            self.observation_shapes['objects_state'] = (self.N_all_objects, self.low_dim_sizes['objects'])
        if 'objects_semantics' in self.observations['low_dim']:
            self.observation_shapes['objects_semantics'] = (self.N_all_objects+1, self.low_dim_sizes['semantics'])
        if 'objects_mask' in self.observations['low_dim']:
            self.observation_shapes['objects_mask'] = (self.N_all_objects+1, self.N_all_objects+1)

    @property
    def env_observation_shapes(self):
        return self.observation_shapes
    
    def reset(self):
        print("==========Resetting==========")
        print("Robot Task: Opening Gripper")
        self.fa.open_gripper()
        print("Robot Task: Resetting pose")
        # self.fa.reset_pose() 
        self.fa.reset_joints()
        print("Robot Task: Closing Gripper 0.02")
        self.fa.goto_gripper(0.02)

        # Choose new grasp position
        object_image_position = np.array([800, 800])
        rgb, depth, _ = self.percept.sensor.frames()
        cv2.namedWindow('Choose grasp point', cv2.WINDOW_NORMAL)
        cv2.imshow('Choose grasp point', rgb.rgb2bgr().data)
        cv2.setMouseCallback('Choose grasp point', onMouse, object_image_position)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        object_center_point_in_world = self.percept.get_object_center_point_in_world(object_image_position[0], object_image_position[1], depth.data)
        
        object_center_pose = self.fa.get_pose()
        object_z_height = 0.015
        object_center_pose.translation = [object_center_point_in_world[0], object_center_point_in_world[1], object_z_height]

        intermediate_robot_pose = object_center_pose.copy()
        intermediate_pose_z_height = 0.19
        # intermediate_pose_z_height = 0.25
        intermediate_robot_pose.translation = [object_center_point_in_world[0], object_center_point_in_world[1], intermediate_pose_z_height]

        lift_up_pose = object_center_pose.copy()
        lift_up_pose_z_height = self.drag_z_height
        lift_up_pose.translation = [object_center_point_in_world[0], object_center_point_in_world[1], lift_up_pose_z_height]

        print('Moving to intermediate grasp pose')
        self.fa.goto_pose(intermediate_robot_pose)

        print('Moving to grasp pose')
        self.fa.goto_pose(object_center_pose, 10, force_thresholds=[10, 10, 10, 15, 15, 15])

        print('Closing gripper')
        self.fa.close_gripper()

        print('Lifting up')
        self.fa.goto_pose(lift_up_pose, 5, force_thresholds=[10, 10, 10, 15, 15, 15])
        # # TODO: reset tracker, add it to percept

        self.failure_mode = None
        self.colliding_obj_name = None

        _gripper_z_height_on_reset = self.fa.get_pose().translation[2]
        self.shift_gripper_height_by = self.des_init_ee_height - _gripper_z_height_on_reset

        self.rel_obj_names = self.env_cfg.objects.names
        self.update_constraint_types()

        self.update_goal(self.env_cfg.block_bottom.chosen_target_set)

        self.update_ee_block_bot_pos()
        block_top_pos = self.objects_pos_vel_t[self.env_cfg.block_top.block_name][:3]
        self.safety_set_top_low = block_top_pos + self.env_cfg.block_top.safety_set.low
        self.safety_set_top_high = block_top_pos + self.env_cfg.block_top.safety_set.high

        if self.env_cfg.block_bottom.target_set_type == 'relative':
            block_bot_pos = self.objects_pos_vel_t[self.env_cfg.block_bottom.block_name][:3]
            self.target_set_low = block_bot_pos + self.sampled_target_set_low
            self.target_set_high = block_bot_pos + self.sampled_target_set_high
        elif self.env_cfg.block_bottom.target_set_type == 'absolute':
            self.target_set_low = self.sampled_target_set_low
            self.target_set_high = self.sampled_target_set_high
        else:
            raise NotImplementedError('Target set type not implemented.')

        return self.get_state()

    def update_constraint_types(self, const_types=None):
        if const_types is None: # Training: pred constraint types and GT constraint types are the same
            if self.randomize_constraint_types:
                self.pred_constraint_types = {obj_name: random.choice(self.env_cfg.constraint_types) for obj_name in self.env_cfg.objects.names}
            else:
                self.pred_constraint_types = copy.deepcopy(self.obj_to_constraint_map_gt)
            self.obj_to_constraint_map = copy.deepcopy(self.pred_constraint_types) # used for safety analysis
        else: # Eval: Use GT constraint types specified by user
            self.pred_constraint_types = const_types
            self.obj_to_constraint_map = copy.deepcopy(self.obj_to_constraint_map_gt) # used for safety analysis

    def update_ee_block_bot_pos(self):
        ee_pos = self.fa.get_pose().translation
        joint_vel = self.fa.get_joint_velocities()
        J = self.fa.get_jacobian(self.fa.get_joints())
        ee_vel = J @ joint_vel
        self.ee_pos_vel = np.append(ee_pos, ee_vel[:3])

        # self.ee_pos_vel[2] = self.ee_pos_vel[2] - self.table_height
        self.ee_pos_vel[2] = self.ee_pos_vel[2] + self.shift_gripper_height_by

        bb_pos = np.zeros(3)
        bb_pos[0] = self.ee_pos_vel[0]
        bb_pos[1] = self.ee_pos_vel[1] - 0.1 # half-dim in sim
        bb_pos[2] = self.ee_pos_vel[2] - 0.03 # half-dim in sim
        
        self.block_bottom_pos_vel = np.append(bb_pos, ee_vel[:3])
        self.objects_pos_vel_t[self.env_cfg.block_bottom.block_name] = self.block_bottom_pos_vel.copy()
    
    def get_xyz_action(self, action):
        
        action = np.clip(action, -1, 1)
        pos_delta = action * self.env_cfg.action_scale
        pos_delta = np.clip(
            pos_delta,
            np.array(self.env_cfg.mocap_del_low),
            np.array(self.env_cfg.mocap_del_high),
        )

        new_mocap_pos = np.array(self.fa.get_pose().translation) + pos_delta
        new_mocap_pos = np.clip(
            new_mocap_pos,
            np.array(self.env_cfg.mocap_low),
            np.array(self.env_cfg.mocap_high),
        )

        return new_mocap_pos, pos_delta

    def update_goal(self, target_set_name=None):
        if target_set_name is None:
            if self.multi_goal:
                self.target_set_name = random.choice(list(self.env_cfg.block_bottom.multi_goal_target_sets.keys()))
                # self.target_set_name = 'target_set1'
                self.sampled_target_set_low = np.array(self.env_cfg.block_bottom.multi_goal_target_sets[self.target_set_name].low)
                self.sampled_target_set_high = np.array(self.env_cfg.block_bottom.multi_goal_target_sets[self.target_set_name].high)
            else:
                self.target_set_name = 'target_set'
                self.sampled_target_set_low = np.array(self.env_cfg.block_bottom.target_set.low)
                self.sampled_target_set_high = np.array(self.env_cfg.block_bottom.target_set.high)
        else:
            self.target_set_name = target_set_name
            self.sampled_target_set_low = np.array(self.env_cfg.block_bottom.multi_goal_target_sets[self.target_set_name].low)
            self.sampled_target_set_high = np.array(self.env_cfg.block_bottom.multi_goal_target_sets[self.target_set_name].high)

    def get_state(self):
        """ 
            Returns states of self.observations for the environment:
        """
        obs = {} # observation dict
        self.update_ee_block_bot_pos()

        if self.constraint_type_repr == 'int':
            const_type_rep = [self.object_type_to_int_mapping['ee']]
        elif self.constraint_type_repr == 'one_hot':
            const_type_rep = np.zeros(self.max_constraint_types)
            const_type_rep[self.object_type_to_int_mapping['ee']] = 1

        ee_state = np.concatenate([self.ee_pos_vel, const_type_rep])

        semantics = [const_type_rep]

        if self.multi_goal:
            ee_state = np.append(ee_state, self.target_set_to_int_mapping[self.target_set_name])
            semantics = [np.append(const_type_rep, self.target_set_to_int_mapping[self.target_set_name])]

        if self.normalize_pos:
            ee_state[:3] = self.get_normalized_pos(ee_state[:3])

        obs['robot_state'] = ee_state

        if 'objects_state' in self.observations['low_dim']:
            xt_stack, semantics_stack = [], []
            for i, body_name in enumerate([self.env_cfg.block_bottom.block_name, self.env_cfg.block_top.block_name]):
                if self.constraint_type_repr == 'int':
                    const_type_rep = [self.object_type_to_int_mapping[body_name]]
                elif self.constraint_type_repr == 'one_hot':
                    const_type_rep = np.zeros(self.max_constraint_types)
                    const_type_rep[self.object_type_to_int_mapping[body_name]] = 1
                body_state = np.concatenate([self.objects_pos_vel_t[body_name], const_type_rep])
                if self.multi_goal:
                    body_state = np.append(body_state, self.target_set_to_int_mapping[self.target_set_name])
                    const_type_rep = np.append(const_type_rep, self.target_set_to_int_mapping[self.target_set_name])
                if self.normalize_pos:
                    body_state[:3] = self.get_normalized_pos(body_state[:3])
                xt_stack.append(body_state)
                semantics_stack.append(const_type_rep)

            xt_dist, semantics_dist = [], []
            for i, body_name in enumerate(self.rel_obj_names):
                if self.constraint_type_repr == 'int':
                    const_type_rep = [self.constraint_to_int_mapping[self.pred_constraint_types[body_name]]]
                elif self.constraint_type_repr == 'one_hot':
                    const_type_rep = np.zeros(self.max_constraint_types)
                    const_type_rep[self.constraint_to_int_mapping[self.pred_constraint_types[body_name]]] = 1
                body_state = np.concatenate([self.objects_pos_vel_t[body_name], const_type_rep])

                if self.multi_goal:
                    body_state = np.append(body_state, self.target_set_to_int_mapping[self.target_set_name])
                    const_type_rep = np.append(const_type_rep, self.target_set_to_int_mapping[self.target_set_name])

                if self.normalize_pos:
                    body_state[:3] = self.get_normalized_pos(body_state[:3])

                xt_dist.append(body_state)
                semantics_dist.append(const_type_rep)

            obs['objects_state'] = np.stack(xt_stack+xt_dist, axis=0)
            if 'objects_semantics' in self.observations['low_dim']:
                obs['objects_semantics'] = np.vstack(semantics + semantics_stack + semantics_dist)
            if 'objects_mask' in self.observations['low_dim']:
                obs['objects_mask'] = self.get_topk_obj_mask(ee_state[:3], np.vstack(xt_dist)[:, :3])

        if 'rgb_front_cam' in self.observations['rgb']:
            obs['rgb_front_cam'] = self.get_current_image(self.front_cam_name)
        if 'rgb_eye_in_hand_cam' in self.observations['rgb']:
            obs['rgb_eye_in_hand_cam'] = self.get_current_image(self.eye_in_hand_cam_name)

        return obs
    
    def get_topk_obj_mask(self, ee_pos, x_dist):
        dist = np.linalg.norm(ee_pos - x_dist, axis=1)
        sorted_indices = np.argsort(dist)[:self.env_cfg.tok_k_mask] + 3

        mask = np.zeros((self.N_all_objects+1,self.N_all_objects+1))
        mask[:3, sorted_indices] = 1
        mask[:3,:3] = 1 # ee, block bottom and top attend to one another
        mask[-self.N_dist_objects:,-self.N_dist_objects:] = np.eye(self.N_dist_objects) #np.eye(self.N_dist_objects) # objects attend to themselves
        mask[sorted_indices, :3] = 1

        any_contact_indices = [3+i for i, name in enumerate(self.rel_obj_names) if self.pred_constraint_types[name] == 'any_contact']
        mask[:3, any_contact_indices] = 0
        mask[any_contact_indices, :3] = 0

        return mask
    
    def get_done(self, state, success, fail):
        # state: shape(batch,n)
        if self.doneType == 'toEnd':
            done = self.check_within_env(state) # TODO: Implement
        elif self.doneType == 'fail':
            done = fail
        elif self.doneType == 'TF':
            done = np.logical_or(fail, success)
        elif self.doneType == 'all':
            done = np.logical_or(fail, success)
            if success:
                self.failure_mode = 'success'
        else:
            raise ValueError("invalid done type!")
        return done
    
    def get_cost(self, l_x, g_x, success, fail):
        if self.costType == 'dense_ell':
            cost = l_x
        elif self.costType == 'dense':
            cost = l_x + g_x
        elif self.costType == 'sparse':
            cost = 0.
        elif self.costType == 'max_ell_g':
            if 'reward' in self.return_type:
                cost = np.minimum(l_x, g_x)
            else:
                cost = np.maximum(l_x, g_x)
        else:
            raise ValueError("invalid cost type!")
        return cost, l_x, g_x
    
    def target_margin(self, s=None, target_set_low=None, target_set_high=None):
        """Computes the margin (e.g. distance) between the state and the target set.

        Args:
            s (np.ndarray): the state of the agent. shape (batch, n)

        Returns:
            float: negative numbers indicate reaching the target. If the target set
                is not specified, return None.
        """
        # l(x)<0 is target

        # TODO: fix
        if s is None:
            # self.update_ee_block_bot_pos()
            _ee_pos = self.ee_pos_vel[:3].reshape(1,-1)
            _bottom_block_pos = self.objects_pos_vel_t[self.env_cfg.block_bottom.block_name][:3].reshape(1,-1) # shape (batch, 3)
        else:
            _ee_pos = s[...,:3].copy()
            _bottom_block_pos = s[...,6:9].copy()

        if target_set_low is None:
            if self.env_cfg.block_bottom.target_set_type == 'relative':
                target_set_low = _bottom_block_pos + self.sampled_target_set_low
            elif self.env_cfg.block_bottom.target_set_type == 'absolute':
                target_set_low = self.sampled_target_set_low
            else:
                raise NotImplementedError('Target set type not implemented.')
        if target_set_high is None:
            if self.env_cfg.block_bottom.target_set_type == 'relative':
                target_set_high = _bottom_block_pos + self.sampled_target_set_high
            elif self.env_cfg.block_bottom.target_set_type == 'absolute':
                target_set_high = self.sampled_target_set_high
            else:
                raise NotImplementedError('Target set type not implemented.')
            
        # block bottom goal
        lx = signed_dist_fn_rectangle(
            _bottom_block_pos, 
            target_set_low, 
            target_set_high)

        lx = self.scaling_target * lx
        if 'reward' in self.return_type: # l(x)>0 is target
            lx = -1.*lx

        return lx

    def get_object_bounds(self, obj_name):
        obj_hor_rad = np.array(self.env_cfg.objects[obj_name].size)
        obj_low = -1*obj_hor_rad
        obj_high = 1*obj_hor_rad
        return obj_low, obj_high
    
    def collision_distance(self, obj1_name, obj1_state, obj2_name, obj2_state):
        # g(x)>0 is obstacle

        obj1_low, obj1_high = self.get_object_bounds(obj1_name)
        obj2_low, obj2_high = self.get_object_bounds(obj2_name)

        obstacle_high = obj1_state[...,:3] + obj1_high - obj2_low + self.env_cfg.thresh
        obstacle_low = obj1_state[...,:3] + obj1_low - obj2_high - self.env_cfg.thresh
        obstacle = signed_dist_fn_rectangle(
            obj2_state[...,:3], 
            obstacle_low, 
            obstacle_high,
            obstacle=True)
        # print(f'{obstacle=}')
        return obstacle
    
    def constraint_distance(
        self, 
        obj1_name, 
        obj1_pos,
        obj1_vel, 
        obj2_name, 
        obj2_pos,
        obj2_vel,
    ):
        # g(x)>0 is obstacle
                
        obstacle = self.collision_distance(obj1_name, obj1_pos, obj2_name, obj2_pos)

        if 'no_contact' in self.obj_to_constraint_map[obj2_name]:
            if obstacle[0]>0 and self.failure_mode is None:
                self.failure_mode = 'collision_no_contact'
                self.colliding_obj_name = obj2_name
        elif 'any_contact' in self.obj_to_constraint_map[obj2_name]:
            obstacle = -1e6*np.ones(obstacle.shape)
        elif 'soft_contact' in self.obj_to_constraint_map[obj2_name]:
            velocity_const =  np.linalg.norm(obj1_vel-obj2_vel, axis=-1) - self.env_cfg.vel_thresh
            obstacle = velocity_const*(obstacle>0) + -1e6*np.ones(obstacle.shape)*(obstacle<0)
            if obstacle[0]>0 and self.failure_mode is None:
                self.failure_mode = 'collision_soft_contact'
                self.colliding_obj_name = obj2_name
        elif 'no_over' in self.obj_to_constraint_map[obj2_name]:
            xy_dist =  np.linalg.norm(obj1_pos[...,:2]-obj2_pos[...,:2], axis=-1) - self.env_cfg.thresh
            if (obstacle[0]>0 or xy_dist[0]<0) and self.failure_mode is None:
                self.failure_mode = 'collision_no_over'
                self.colliding_obj_name = obj2_name
        else:
            raise NotImplementedError('Contraint type not implemented!')

        return obstacle

    def safety_margin(self, s=None, safety_set_top_low=None, safety_set_top_high=None):
        # Input:s top block position, shape (batch, 3)
        # g(x)>0 is obstacle

        if s is None:
            # self.update_ee_block_bot_pos()
            _ee_pos = self.ee_pos_vel[:3].reshape(1,-1)
            _bottom_block_pos = self.objects_pos_vel_t[self.env_cfg.block_bottom.block_name][:3].reshape(1,-1) # shape (batch, 3)
            _bottom_block_vel = self.objects_pos_vel_t[self.env_cfg.block_bottom.block_name][3:].reshape(1,-1) # shape (batch, 3)
            _top_block_pos = self.objects_pos_vel_t[self.env_cfg.block_top.block_name][:3].reshape(1,-1) # shape (batch, 3)
        else:
            _ee_pos = s[...,:3].copy()
            _bottom_block_pos = s[...,6:9].copy()
            _bottom_block_vel = s[...,9:12].copy()
            _top_block_pos = s[...,12:15].copy()
            
        if safety_set_top_low is None:
            safety_set_top_low = _top_block_pos + self.env_cfg.block_top.safety_set.low

        if safety_set_top_high is None:
            safety_set_top_high = _top_block_pos + self.env_cfg.block_top.safety_set.high

        # top block safety: stay within safety bounds
        gx = signed_dist_fn_rectangle(
            _top_block_pos, 
            safety_set_top_low, 
            safety_set_top_high)

        if gx[0]>0:
            self.failure_mode = 'top_block_oob'

        # obstacle avoidance between bottom block and distractors
        for i, obj_name in enumerate(self.env_cfg.objects.names):
            _obj_pos = self.objects_pos_vel_t[obj_name][:3] if s is None else s[...,18+i*6:18+i*6+3]
            _obj_vel = self.objects_pos_vel_t[obj_name][3:] if s is None else s[...,18+i*6+3:18+i*6+6]
            obstacle = self.constraint_distance(
                self.env_cfg.block_bottom.block_name, 
                _bottom_block_pos, 
                _bottom_block_vel,
                obj_name, 
                _obj_pos, 
                _obj_vel,)
            
            gx = np.maximum(gx, obstacle)
        
        gx = self.scaling_safety * gx

        if 'reward' in self.return_type: # g(x)<0 is obstacle
            gx = -1.*gx
        return gx


    def check_failure(self, state=None):
        g_x = self.safety_margin(state, self.safety_set_top_low, self.safety_set_top_high)
        if 'reward' in self.return_type: 
            return g_x<0, g_x
        else:
            return g_x>0, g_x # g(x)>0 is failure
    
    def check_success(self, state=None):
        l_x = self.target_margin(state, self.target_set_low, self.target_set_high)
        if 'reward' in self.return_type: 
            return l_x>0, l_x
        else:
            return l_x<0, l_x # l(x)<0 is target
        
    def get_normalized_pos(self, pos):
        return 2*(pos - self.env_bound_low) / (self.env_bound_high - self.env_bound_low) - 1.0

    def get_action(self):
        self.update_ee_block_bot_pos()
        target_pos = (self.target_set_high + self.target_set_low)/2 # [0., 0.3, 0.1]
        # print(f"Target pos: {target_pos}")
        action = target_pos - self.ee_pos_vel[:3]
        
        return self.get_xyz_action(action)

    def plot_trajectory(self, state, action, save_filename):
        """ 
            state_dict: list of dicts
            state shape = (T+1, self.n)
            action shape = (T, self.m)
        """
        _rows = 4
        fig, axes = plt.subplots(_rows, 3, figsize=(16, 16))
        
        ee_state = np.stack([s['robot_state'][:6] for s in state], axis=0)
        block_bottom_state = np.stack([s['objects_state'][0] for s in state], axis=0)
        block_top_state = np.stack([s['objects_state'][1] for s in state], axis=0)
        
        # plot position
        axes[0,0].plot(ee_state[:,0], label='EE x')
        axes[0,0].plot(block_bottom_state[:,0], label='Bottom block x')
        axes[0,0].plot(block_top_state[:,0], label='Top block x')

        axes[0,1].plot(ee_state[:,1], label='EE y')
        axes[0,1].plot(block_bottom_state[:,1], label='Bottom block y')
        axes[0,1].plot(block_top_state[:,1], label='Top block y')

        axes[0,2].plot(ee_state[:,2], label='EE z')
        axes[0,2].plot(block_bottom_state[:,2], label='Bottom block z')
        axes[0,2].plot(block_top_state[:,2], label='Top block z')

        # plot velocity
        axes[1,0].plot(ee_state[:,3], label='EE x vel')
        axes[1,0].plot(block_bottom_state[:,3], label='Bottom block x vel')
        axes[1,0].plot(block_top_state[:,3], label='Top block x vel')

        axes[1,1].plot(ee_state[:,4], label='EE y vel')
        axes[1,1].plot(block_bottom_state[:,4], label='Bottom block y vel')
        axes[1,1].plot(block_top_state[:,4], label='Top block y vel')

        axes[1,2].plot(ee_state[:,5], label='EE z vel')
        axes[1,2].plot(block_bottom_state[:,5], label='Bottom block z vel')
        axes[1,2].plot(block_top_state[:,5], label='Top block z vel')

        for i in range(self.env_cfg.n_rel_objs):
            block_state = np.stack([s['objects_state'][i+2] for s in state], axis=0)

            axes[0,0].plot(block_state[:,0], label=f'{self.rel_obj_names[i]} x')
            axes[0,1].plot(block_state[:,1], label=f'{self.rel_obj_names[i]} y')
            axes[0,2].plot(block_state[:,2], label=f'{self.rel_obj_names[i]} z')

            axes[1,0].plot(block_state[:,3], label=f'{self.rel_obj_names[i]} x vel')
            axes[1,1].plot(block_state[:,4], label=f'{self.rel_obj_names[i]} y vel')
            axes[1,2].plot(block_state[:,5], label=f'{self.rel_obj_names[i]} z vel')

            axes[3,0].plot(block_state[:,6], label=f'{self.rel_obj_names[i]} x')

        # control on bottom block
        axes[2,0].plot(action[:,0], label='u_x')
        axes[2,0].set_xlabel(f't')
        axes[2,0].set_ylabel(f'u_x')
        axes[2,0].set_title(f'Control x')
        axes[2,0].legend()

        axes[2,1].plot(action[:,1], label='u_y')
        axes[2,1].set_xlabel(f't')
        axes[2,1].set_ylabel(f'u_y')
        axes[2,1].set_title(f'Control y')
        axes[2,1].legend()

        axes[2,2].plot(action[:,2], label='u_z')
        axes[2,2].set_xlabel(f't')
        axes[2,2].set_ylabel(f'u_z')
        axes[2,2].set_title(f'Control z')
        axes[2,2].legend()

        axes[0,0].set_xlabel(f't')
        axes[0,0].set_ylabel(f'x')
        axes[0,0].set_title(f'x pos')
        axes[0,0].legend()
        
        axes[0,1].set_xlabel(f't')
        axes[0,1].set_ylabel(f'y')
        axes[0,1].set_title(f'y pos')
        axes[0,1].legend()

        axes[0,2].set_xlabel(f't')
        axes[0,2].set_ylabel(f'z')
        axes[0,2].set_title(f'z pos')
        axes[0,2].legend()

        axes[1,0].set_xlabel(f't')
        axes[1,0].set_ylabel(f'xdot')
        axes[1,0].set_title(f'x vel')
        axes[1,0].legend()

        axes[1,1].set_xlabel(f't')
        axes[1,1].set_ylabel(f'ydot')
        axes[1,1].set_title(f'y vel')
        axes[1,1].legend()

        axes[1,2].set_xlabel(f't')
        axes[1,2].set_ylabel(f'zdot')
        axes[1,2].set_title(f'z vel')
        axes[1,2].legend()

        axes[3,0].set_xlabel(f't')
        axes[3,0].set_ylabel(f'constraint type')
        axes[3,0].set_title(f'constraint type')
        axes[3,0].legend()

        plt.savefig(save_filename)
        plt.close()

def real_to_mujoco_obs(obs_real):
    obs_sim = copy.deepcopy(obs_real)
    # positions
    obs_sim['robot_state'][0] = -obs_real['robot_state'][1].copy()
    obs_sim['robot_state'][1] = obs_real['robot_state'][0].copy()
    obs_sim['objects_state'][:,0] = -obs_real['objects_state'][:,1].copy()
    obs_sim['objects_state'][:,1] = obs_real['objects_state'][:,0].copy()

    # velocity
    i=3
    obs_sim['robot_state'][0+i] = -obs_real['robot_state'][1+i].copy()
    obs_sim['robot_state'][1+i] = obs_real['robot_state'][0+i].copy()
    obs_sim['objects_state'][:,0+i] = -obs_real['objects_state'][:,1+i].copy()
    obs_sim['objects_state'][:,1+i] = obs_real['objects_state'][:,0+i].copy()
    return obs_sim

def sim_to_real_action(a_sim):
    a_real = a_sim.copy()
    a_real[0] = a_sim[1]
    a_real[1] = -a_sim[0]
    return a_real
    
import torch
from pathlib import Path
import os
import wandb
from safety_rl_manip.models.RARL.utils import ReplayBufferMultimodal
from safety_rl_manip.models.RARL.DDPG_multimodal_indep import DDPGMultimodalIndep
from safety_rl_manip.models.RARL.DDPG_core import TransformerIndepActorCriticSS
from frankapy.utils import save_video_from_images

def load_checkpoint(cfg, env):
    eval_cfg = cfg.eval_cfg
    out_folder = cfg.out_folder
    device = (
        torch.device("cuda", eval_cfg.gpu)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    eval_path = Path(out_folder)
    model_path = os.path.join(eval_path, 'ckpts')
    

    # Load checkpoint
    ckpt_file = wandb.restore(
        str(eval_cfg.wandb_load.file), 
        run_path=eval_cfg.wandb_load.run_path,
        root=str(model_path), replace=True,
    )
    ckpt = torch.load(ckpt_file.name, map_location=device)

    train_cfg = ckpt['train_cfg']
    ac = eval(train_cfg.ac_type)(
        env.env_observation_shapes, 
        env.action_space,
        device,
        ac_kwargs=train_cfg.ac_kwargs[f'{train_cfg.ac_type}'],
    )
    filtered_state_dict = {k[len('ac.'):]: v for k, v in ckpt["state_dict"].items() if k.startswith('ac.')}
    ac.load_state_dict(filtered_state_dict)

    replay_buffer = ReplayBufferMultimodal(
        env_observation_shapes=env.env_observation_shapes, 
        act_dim=env.action_space.shape[0], 
        size=1000, device=device
    )
    return ac, replay_buffer

def do_rollouts(cfg):
    out_folder = cfg.out_folder
    eval_cfg = cfg.eval_cfg
    time_str = time.strftime("%Y_%m_%d_%H_%M_%S")
    save_video = eval_cfg.save_video
    
    env_cfg = OmegaConf.load('/home/saumyas/franka/frankapy/cfg/slide_pickup_clutter_env.yaml')
    env = FrankaSlidePickupClutterEnv(env_cfg)

    if not eval_cfg.preset_rollouts:
        ac, replay_buffer = load_checkpoint(cfg, env)
        run_id = eval_cfg.wandb_load.run_path.split('/')[-1]
        time_str += f'ckpt_{run_id}_run'
    else:
        time_str += 'preset_actions_run'
    eval_dir = os.path.join(out_folder, time_str)
    os.makedirs(eval_dir, exist_ok=True)

    dt = env_cfg.env_cfg.dt
    T = dt*env_cfg.env_cfg.max_ep_len
    ts = np.arange(0, T, dt)

    rospy.loginfo('Initializing Sensor Publisher')
    pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1000)
    rate = rospy.Rate(1 / dt)

    num_episodes = 2
    done_for_steps = 3
    for n_ep in range(num_episodes):
        print(f'episode:{n_ep}')

        xt_all, at_all, imgs = [], [], []
        o, done = env.reset(), False
        xt_all.append(o.copy())
        print(f'initial position: {xt_all[0]}')

        p0 = env.fa.get_pose()
        goal_transform = p0.copy()
        # env.fa.goto_pose(p0, duration=T, dynamic=True, buffer_time=2, cartesian_impedances=[600, 600, 225] + FC.DEFAULT_ROTATIONAL_STIFFNESSES)
        env.fa.goto_pose(p0, duration=T, dynamic=True, buffer_time=2)
        init_time = rospy.Time.now().to_time()
        _dfs = 0
        for t in range(2, len(ts)):
            timestamp = rospy.Time.now().to_time() - init_time

            if eval_cfg.preset_rollouts:
                env.table_height = 0
                xtp1, pos_delta = env.get_action() 
            else:
                o_sim = replay_buffer.get_batch_from_obs(real_to_mujoco_obs(o))
                action_sim = ac.act(o_sim)[0]
                action = sim_to_real_action(action_sim)
                xtp1, pos_delta = env.get_xyz_action(action)

            fail, g_x = env.check_failure()
            success, l_x = env.check_success()
            done = env.get_done(None, success, fail)[0]

            goal_transform.translation = xtp1
            at_all.append(pos_delta)
            
            ros_msg, traj_gen_proto_msg = env.percept.get_pose_msg(timestamp, t, goal_transform)
            # rospy.loginfo(f'Robot position {env.ee_pos_vel[:3]}')
            # rospy.loginfo(f'Publishing: ID {traj_gen_proto_msg.id} position {xtp1}')
            pub.publish(ros_msg)
            rate.sleep()

            o = env.get_state()
            xt_all.append(o.copy())
            if save_video:
                imgs.append(env.percept.sensor.frames()[0].rgb2bgr().data)         

            if done :
                _dfs += 1
                rospy.loginfo(f'Done due to: {env.failure_mode=} {env.colliding_obj_name=}. DFS = {_dfs}')
                if _dfs > done_for_steps:
                    break
        
        # Stop the skill
        env.fa.stop_skill()
        term_proto_msg = ShouldTerminateSensorMessage(timestamp=rospy.Time.now().to_time() - init_time, should_terminate=True)
        ros_msg = make_sensor_group_msg(
            termination_handler_sensor_msg=sensor_proto2ros_msg(
                term_proto_msg, SensorDataMessageType.SHOULD_TERMINATE)
            )
        pub.publish(ros_msg)
        env.fa.goto_gripper(0.02)
        rospy.loginfo('Done')
        file_name = f'random_slide_pickup_{n_ep}_goal_{env.target_set_name}_{env.failure_mode}_{env.colliding_obj_name}'
        env.plot_trajectory(xt_all, np.stack(at_all,axis=0), os.path.join(eval_dir, f'{file_name}.png'))
        if save_video:
            save_video_from_images(imgs, os.path.join(eval_dir, f'{file_name}.mp4'), fps=10)
    env.percept.sensor.stop()


if __name__ == "__main__":
    cfg = OmegaConf.load('/home/saumyas/franka/frankapy/cfg/eval_rl_cfg.yaml')
    do_rollouts(cfg)