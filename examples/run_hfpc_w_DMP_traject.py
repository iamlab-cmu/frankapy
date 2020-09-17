import numpy as np

from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import ForcePositionSensorMessage, ForcePositionControllerSensorMessage
from franka_interface_msgs.msg import SensorDataGroup
from frankapy.utils import transform_to_list, min_jerk

from tqdm import trange

import rospy
import pickle 

class DMPPositionTrajectoryGenerator:
    '''
    Generate DMP trajectory on Python side and pass via ROS to control PC side -
    goal is to use w/ HFPC on control PC
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

def get_dmp_traj_poses_reformatted(y, starting_rotation):
    '''this converts y (xyz position traject) to list of 1x16 
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
    
if __name__ == "__main__":
    fa = FrankaArm()
    fa.reset_joints()
    #fa.close_gripper()

    # TODO: Move to desired cutting position/orientation prior to getting starting pose 

    rospy.loginfo('Generating Trajectory')
    start_pose = fa.get_pose()
    starting_rotation = start_pose.rotation

    # define length of dmp trajectory
    traject_time = 5 # TODO: make this a CL arg    
    # load dmp traject params
    dmp_wts_pkl_filepath = '/home/sony/Desktop/debug_dmp_wts.pkl'
    dmp_traj = DMPPositionTrajectoryGenerator(traject_time)
    dmp_traj.load_saved_dmp_params_from_pkl_file(dmp_wts_pkl_filepath)
    dmp_traj.parse_dmp_params_dict()

    # define starting position - need to change this based on where robot actually starts
    y0 = start_pose.translation
    # calculate dmp position trajectory
    y, dy, _, _, _ = dmp_traj.run_dmp_with_weights(y0) # y: np array(tx3)
    target_poses = get_dmp_traj_poses_reformatted(y, starting_rotation) # target_poses is a nx16 list of target poses at each time step
    import pdb; pdb.set_trace()

    # sampling info for sending msgs via ROS
    dt = 0.001
    T = traject_time
    ts = np.arange(0, T, dt)
    N = len(ts)

    target_force = [0, 0, 0, 0, 0, 0] #[0, 0, -10, 0, 0, 0] 
    S = [1, 1, 1, 1, 1, 1] # [1, 1, 0, 1, 1, 1]
    position_kps_cart = FC.DEFAULT_TRANSLATIONAL_STIFFNESSES + FC.DEFAULT_ROTATIONAL_STIFFNESSES
    force_kps_cart = [0.1] * 6
    position_kps_joint = FC.DEFAULT_K_GAINS
    force_kps_joint = [0.1] * 7

    rospy.loginfo('Initializing Sensor Publisher')
    pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1)
    rate = rospy.Rate(1 / dt)
    n_times = 1

    rospy.loginfo('Publishing HFPC trajectory w/ cartesian gains...')
    fa.run_dynamic_force_position(duration=T * n_times, buffer_time = 3, S=S,
                                use_cartesian_gains=True,
                                position_kps_cart=position_kps_cart,
                                force_kps_cart=force_kps_cart)
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
        pub.publish(ros_msg)
        rate.sleep()

        
    fa.stop_skill()
   
    fa.reset_joints()
    fa.open_gripper()
    
    rospy.loginfo('Done')
