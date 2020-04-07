import numpy as np 
from numba import jit
from autolab_core import RigidTransform


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
