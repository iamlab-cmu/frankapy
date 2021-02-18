import pickle as pkl
from frankapy import FrankaArm
import numpy as np
from autolab_core import RigidTransform

if __name__ == '__main__':

    recorded_pose = pkl.load(open('./data/franka_traj.pkl','rb'))

    print('Starting robot')
    fa = FrankaArm()

    fa.reset_joints()

    fa.goto_pose(recorded_pose, 10, use_impedance=True, force_thresholds=[10,10,10,10,10,10])
