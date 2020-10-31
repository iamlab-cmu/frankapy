import numpy as np
from autolab_core import RigidTransform, YamlConfig
import ipdb
class Circle:
    def __init__(self):
        pass
    @staticmethod
    def grasp_symmetries():
        return np.array([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    def placement_symmetries():
        return np.linspace(-np.pi,np.pi,5)
    @staticmethod
    def tforms_to_pose(ids, tforms,goal=False):
        if goal:
            shape_specific_ids = [12,13]
        else:
            shape_specific_ids = [10,11]
        relevant_ids = [ar_id for ar_id in ids if ar_id in shape_specific_ids]
        relevant_tforms = []
        for id_num in relevant_ids:
            tform_idx = ids.index(id_num)
            relevant_tforms.append(tforms[tform_idx])

        #avg_rotation = relevant_tforms[0]
        #for i in range(1, len(relevant_tforms)):
        #    avg_rotation = avg_rotation.interpolate_with(relevant_tforms[i], 0.5)

        if len(relevant_ids) == 2:
            translation = np.mean(np.vstack([T.translation for T in relevant_tforms]), axis=0)
        else:
            print(relevant_ids)
            print("Not enough detections to make accurate pose estimate for circle ")
            return None
        return RigidTransform(rotation = np.eye(3), translation = translation, to_frame = tforms[0].to_frame, from_frame = "peg_center")
class Obstacle:
    def __init__(self):
        pass
    @staticmethod
    def symmetries():
        return [-np.pi,-np.pi/2, 0, np.pi/2, np.pi]
    @staticmethod
    def tforms_to_pose(ids, tforms,goal=False):
        shape_specific_ids = [8,9]
        relevant_ids = [ar_id for ar_id in ids if ar_id in shape_specific_ids]
        relevant_tforms = []
        for id_num in relevant_ids:
            tform_idx = ids.index(id_num)
            relevant_tforms.append(tforms[tform_idx])
        
        avg_rotation = relevant_tforms[0]
        for i in range(1, len(relevant_tforms)):
            avg_rotation = avg_rotation.interpolate_with(relevant_tforms[i], 0.5)

        if len(relevant_ids) == 2:
            translation = np.mean(np.vstack([T.translation for T in relevant_tforms]), axis=0)
        else:
            print(relevant_ids)
            print("Not enough detections to make accurate pose estimate")
            return None
        return RigidTransform(rotation = avg_rotation.rotation, translation = translation, to_frame = tforms[0].to_frame, from_frame = "peg_center")

class Rectangle:
    def __init__(self):
        pass

    @staticmethod
    def symmetries():
        return np.array([-np.pi, 0, np.pi])
    def grasp_symmetries():
        return np.array([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    def placement_symmetries():
        return np.array([0, -np.pi, np.pi, 2*np.pi])

    """
    returns the intervals of rotation that are identical, at least enough to be useful. 
    The triangle can't be rotated but the rectangle can be inserted at any interval of 3.14
    """
    @staticmethod
    def tforms_to_pose(ids, tforms,goal=False):
        if goal:
            shape_specific_ids = [4,5,6,7]
        else:
            shape_specific_ids = [0,1,2,3]
        translation = None

        relevant_ids = [ar_id for ar_id in ids if ar_id in shape_specific_ids]
        print("Relevant IDs detected: ", len(relevant_ids))
        if len(relevant_ids) == 0:
            # print("No detections")
            return None
        
        """
        Assumes d distance from the center of the tag to the sticker. Then this 
                w
        ----------
        |0       1 |
        |          |
        |          |  l
        |3       2 |
        ----------
        """
        d = 0.00 #distance between center of tag and the edge
        w = 0.034163
        l = 0.034163

        relevant_tforms = []
        for id_num in relevant_ids:
            tform_idx = ids.index(id_num)
            relevant_tforms.append(tforms[tform_idx])

        avg_rotation = relevant_tforms[0]
        for i in range(1, len(relevant_tforms)):
            avg_rotation = avg_rotation.interpolate_with(relevant_tforms[i], 0.5)

        if len(relevant_ids) == 4:
            translation = np.mean(np.vstack([T.translation for T in relevant_tforms]), axis=0)

        if len(relevant_ids) == 2 or len(relevant_ids) == 3:
            corner_pairs = [(0,2), (1,3), (4,6), (5,7)]
            for pair in corner_pairs:
                if pair[0] in relevant_ids and pair[1] in relevant_ids:
                    first_idx = ids.index(pair[0])
                    second_idx = ids.index(pair[1])
                    relevant_tforms = [tforms[first_idx], tforms[second_idx]]
                    translation = np.mean(np.vstack([T.translation for T in relevant_tforms]), axis=0)

        if len(relevant_ids) == 2 or len(relevant_ids) == 3:
            corner_pairs = [(0, 1)]
            for pair in corner_pairs:
                if pair[0] in relevant_ids and pair[1] in relevant_ids:
                    first_idx = ids.index(pair[0])
                    second_idx = ids.index(pair[1])
                    T1 = tforms[first_idx].translation
                    T2 = tforms[second_idx].translation
                    translation = np.array([T1[0]+(0.5*l-d), (T1[1]+T2[1])/2, T1[2]])

        if len(relevant_ids) == 2 or len(relevant_ids) == 3:
            corner_pairs = [(3, 2)]
            for pair in corner_pairs:
                if pair[0] in relevant_ids and pair[1] in relevant_ids:
                    first_idx = ids.index(pair[0])
                    second_idx = ids.index(pair[1])
                    T1 = tforms[first_idx].translation
                    T2 = tforms[second_idx].translation
                    translation = np.array([T1[0]-(0.5*l-d), (T1[1]+T2[1])/2, T1[2]])

        if len(relevant_ids) == 2 or len(relevant_ids) == 3:
            corner_pairs = [(0, 3)]
            for pair in corner_pairs:
                if pair[0] in relevant_ids and pair[1] in relevant_ids:
                    first_idx = ids.index(pair[0])
                    second_idx = ids.index(pair[1])
                    T1 = tforms[first_idx].translation
                    T2 = tforms[second_idx].translation
                    translation = np.array([(T1[0]+T2[0])/2, T1[1]+(0.5*w-d), T1[2]])

        if len(relevant_ids) == 2 or len(relevant_ids) == 3:
            corner_pairs = [(1, 2)]
            for pair in corner_pairs:
                if pair[0] in relevant_ids and pair[1] in relevant_ids:
                    first_idx = ids.index(pair[0])
                    second_idx = ids.index(pair[1])
                    T1 = tforms[first_idx].translation
                    T2 = tforms[second_idx].translation
                    translation = np.array([(T1[0]+T2[0])/2, T1[1]-(0.5*w-d), T1[2]])

        if len(relevant_ids) == 1:

            pose_estimates = []
            for id_num in relevant_ids:
                tform_idx = relevant_ids.index(id_num)
                T = relevant_tforms[tform_idx].translation
                if id_num == 0:
                    pose_estimates.append([T[0]+(0.5*l-d), T[1]+(0.5*w-d), T[2]])
                elif id_num == 1:
                    pose_estimates.append([T[0]+(0.5*l-d), T[1]-(0.5*w-d), T[2]])
                elif id_num == 2:
                    pose_estimates.append([T[0]-(0.5*l-d), T[1]-(0.5*w-d), T[2]])
                elif id_num == 3:
                    pose_estimates.append([T[0]-(0.5*l-d), T[1]+(0.5*w-d), T[2]])
                else:
                    raise ValueError("Invalid id num. Should not come here")
            translation = np.mean(np.vstack(pose_estimates), axis=0)

        if translation is None:
            print(relevant_ids)
            # print("Not enough detections to make accurate pose estimate")
            return None
        return RigidTransform(rotation = relevant_tforms[0].rotation, translation = translation, to_frame = tforms[0].to_frame, from_frame = "peg_center")

class Square:
    def __init__(self):
        pass
    def grasp_symmetries():
        return np.array([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    def placement_symmetries():
        return np.array([0, -np.pi, np.pi, 2*np.pi, np.pi/2, -np.pi/2, 1.5*np.pi, -1.5*np.pi])

    """
    returns the intervals of rotation that are identical, at least enough to be useful. 
    The triangle can't be rotated but the rectangle can be inserted at any interval of 3.14
    """
    @staticmethod
    def tforms_to_pose(ids, tforms,goal=False):
        if goal:
            shape_specific_ids = [18,19, 20, 21]
        else:
            shape_specific_ids = [14,15,16,17]

        relevant_ids = [ar_id for ar_id in ids if ar_id in shape_specific_ids]
        
        relevant_tforms = []
        for id_num in relevant_ids:
            tform_idx = ids.index(id_num)
            relevant_tforms.append(tforms[tform_idx])

        avg_rotation = relevant_tforms[0]
        for i in range(1, len(relevant_tforms)):
            avg_rotation = avg_rotation.interpolate_with(relevant_tforms[i], 0.5)

        if len(relevant_ids) == 4:
            translation = np.mean(np.vstack([T.translation for T in relevant_tforms]), axis=0)
        corner_pairs = [(14,16), (15,17), (18,20),(19,21)]
        for pair in corner_pairs:
            if pair[0] in relevant_ids and pair[1] in relevant_ids:
                first_idx = ids.index(pair[0])
                second_idx = ids.index(pair[1])
                relevant_tforms = [tforms[first_idx], tforms[second_idx]]
                translation = np.mean(np.vstack([T.translation for T in relevant_tforms]), axis=0)

        if translation is None:
            print(relevant_ids)
            print("Not enough detections to make accurate pose estimate")
            return None
        return RigidTransform(rotation = relevant_tforms[0].rotation, translation = translation, to_frame = tforms[0].to_frame, from_frame = "peg_center")
