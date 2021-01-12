import numpy as np
import matplotlib.pyplot as plt
plt.close("all")
import math
import rospy
import argparse
import pickle

from frankapy.utils import *

dmp_wts_file = '/home/sony/raw_IL_trajects/100220_piv_chop_position_dmp_weights_zeroXY_2.pkl'

# Calculate dmp trajectory  
traject_time = 5 # args.dmp_traject_time   # define length of dmp trajectory  
# Load dmp traject params
dmp_traj = DMPPositionTrajectoryGenerator(traject_time)
dmp_traj.load_saved_dmp_params_from_pkl_file(dmp_wts_file)
dmp_traj.parse_dmp_params_dict()

# Define starting position 
start_pose = np.array([0,0,0]) #fa.get_pose()
starting_rotation = np.array([[1,0,0],[0,1,0],[0,0,1]]) #start_pose.rotation
y0 = np.array([0,0,0]) #start_pose.translation 
# calculate dmp position trajectory - NOTE: this assumes a 0.001 dt for calc the dmp traject
dmp_traject, dy, _, _, _ = dmp_traj.run_dmp_with_weights(y0) # y: np array(tx3)
import pdb; pdb.set_trace()