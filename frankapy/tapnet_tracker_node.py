import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

import rospy
from std_msgs.msg import Float32MultiArray, Int32MultiArray
from geometry_msgs.msg import Point, PointStamped

import jax
import jax.numpy as jnp
from tapnet.models import tapir_model
from tapnet.utils import model_utils

from franka_perception import Perception
from frankapy.utils import save_video_from_images

def get_colormap_colors(num_colors, cmap_name='tab10'):
    cmap = plt.get_cmap(cmap_name)
    colors = [tuple(int(c * 255) for c in cmap(i % cmap.N)[:3]) for i in range(num_colors)]
    return colors

class PointTrackingTapnet:
    def __init__(self, cfg, percept=None):
        tapnet_cfg = cfg.tapnet_cfg
        self.visualize = tapnet_cfg.visualize
        self.save_video = tapnet_cfg.save_video
        self.out_folder = tapnet_cfg.out_folder

        if percept is None:
            self.percept = Perception(cfg=cfg.sensor_cfg)

        self.n_top_block_points = 12 # 3 for each side
        self.n_points_per_obstacle = 3
        self.n_obstacles = tapnet_cfg.n_rel_objs

        self.total_num_points = self.n_top_block_points + self.n_points_per_obstacle*self.n_obstacles

        self.object_names = tapnet_cfg.object_names

        self.colors = get_colormap_colors(self.total_num_points, cmap_name='tab10')
        params, state = self.load_checkpoint(tapnet_cfg.checkpoint_path)
        self.tapir = tapir_model.ParameterizedTAPIR(
            params=params,
            state=state,
            tapir_kwargs=dict(
                use_causal_conv=True, bilinear_interp_with_depthwise_conv=False
            ),
        )
        print("Creating model...")
        self.online_init_apply = jax.jit(self.online_model_init)
        self.online_predict_apply = jax.jit(self.online_model_predict)

        self.reset_tracker()

        rgb, depth, _ = self.percept.sensor.frames()
        self.compile_functions(rgb.rgb2bgr().data)
    
    def reset_tracker(self):
        self.pos = tuple()
        self.query_frame = False
        self.have_point = [False] * self.total_num_points
        self.query_features = None
        self.causal_state = None
        self.next_query_idx = 0
        self.last_click_time = 0

    def compile_functions(self, frame):

        print("Compiling jax functions (this may take a while...)")
        # --------------------
        # Call one time to compile
        query_points = jnp.zeros([self.total_num_points, 3], dtype=jnp.float32)
        _ = self.online_init_apply(
            frames=model_utils.preprocess_frames(frame[None, None]),
            points=query_points[None, 0:1],
        )
        jax.block_until_ready(self.query_features)
        self.query_features = self.online_init_apply(
            frames=model_utils.preprocess_frames(frame[None, None]),
            points=query_points[None, :],
        )

        self.causal_state = self.tapir.construct_initial_causal_state(
            self.total_num_points, len(self.query_features.resolutions) - 1
        )

        prediction, self.causal_state = self.online_predict_apply(
            frames=model_utils.preprocess_frames(frame[None, None]),
            features=self.query_features,
            causal_context=self.causal_state,
        )

        jax.block_until_ready(prediction["tracks"])

        self.img_H = frame.shape[0]
        self.img_W = frame.shape[1]

    def load_checkpoint(self, checkpoint_path):
        ckpt_state = np.load(checkpoint_path, allow_pickle=True).item()
        return ckpt_state["params"], ckpt_state["state"]

    def online_model_init(self, frames, points):
        feature_grids = self.tapir.get_feature_grids(frames, is_training=False)
        features = self.tapir.get_query_features(
            frames,
            is_training=False,
            query_points=points,
            feature_grids=feature_grids,
        )
        return features
    
    def online_model_predict(self, frames, features, causal_context):
        """Compute point tracks and occlusions given frames and query points."""
        feature_grids = self.tapir.get_feature_grids(frames, is_training=False)
        trajectories = self.tapir.estimate_trajectories(
            frames.shape[-3:-1],
            is_training=False,
            feature_grids=feature_grids,
            query_features=features,
            query_points_in_video=None,
            query_chunk_size=64,
            causal_context=causal_context,
            get_causal_context=True,
        )
        causal_context = trajectories["causal_context"]
        del trajectories["causal_context"]
        return {k: v[-1] for k, v in trajectories.items()}, causal_context


    def mouse_click(self, event, x, y, flags, param):
        del flags, param

        # event fires multiple times per click sometimes??
        if (time.time() - self.last_click_time) < 0.5:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.pos = (y, self.img_W - x)
            self.query_frame = True
            self.last_click_time = time.time()
    
    def sanity_check(self, track, prev_track=None):
        # (720, 1280, 3)
        width_sanity = np.logical_or(track[:, 0] > self.img_W, track[:, 0] < 0)
        height_sanity = np.logical_or(track[:, 1] > self.img_H, track[:, 1] < 0)
        size_sanity = np.logical_or(width_sanity, height_sanity)
        track = track.at[size_sanity].set(prev_track[size_sanity])
        return track

    def start_tracker(self):

        cv2.namedWindow("Point Tracking", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Point Tracking", self.mouse_click)

        t = time.time()
        step_counter = 0
        if self.save_video:
            bgr_labeled_imgs = []
        all_tracks, all_visibles = [], []
        while True:
            rgb, _, _ = self.percept.sensor.frames()
            frame = rgb.rgb2bgr().data
            if self.query_frame:
                # print(f'pos: {pos}')
                query_points = jnp.array((0,) + self.pos, dtype=jnp.float32)
                init_query_features = self.online_init_apply(
                    frames=model_utils.preprocess_frames(frame[None, None]),
                    points=query_points[None, None],
                )
                self.query_frame = False
                self.query_features, self.causal_state = self.tapir.update_query_features(
                    query_features=self.query_features,
                    new_query_features=init_query_features,
                    idx_to_update=np.array([self.next_query_idx]),
                    causal_state=self.causal_state,
                )
                self.have_point[self.next_query_idx] = True
                self.next_query_idx = (self.next_query_idx + 1) % self.total_num_points
            if self.pos:
                prediction, self.causal_state = self.online_predict_apply(
                    frames=model_utils.preprocess_frames(frame[None, None]),
                    features=self.query_features,
                    causal_context=self.causal_state,
                )
                track = prediction["tracks"][0, :, 0]
                occlusion = prediction["occlusion"][0, :, 0]
                expected_dist = prediction["expected_dist"][0, :, 0]
                visibles = model_utils.postprocess_occlusions(occlusion, expected_dist)
                track = np.round(track)

                all_tracks.append(track)
                all_visibles.append(visibles)
                if self.visualize:
                    for i, _ in enumerate(self.have_point):
                        cv2.circle(frame, (int(track[i, 0]), int(track[i, 1])), 8, self.colors[i], thickness=2)
                        if visibles[i] and self.have_point[i]:
                            cv2.circle(frame, (int(track[i, 0]), int(track[i, 1])), 5, self.colors[i], -1)
            
            if self.visualize:
                cv2.imshow("Point Tracking", frame[:, ::-1])
                bgr_labeled_imgs.append(frame)
    
    def get_object_pos_from_tracks(self, tracks, visibles, all_tracks, all_visibles, depth):
        top_object_points = tracks[:self.n_top_block_points][visibles[:self.n_top_block_points]]
        if len(top_object_points)> 0:
            top_block_pos_pix = np.sum(top_object_points, axis=0)/len(top_object_points)
        else:
            found_visible = False
            indx = len(all_visibles) - 2
            while not found_visible:
                new_visibles = all_visibles[indx][:self.n_top_block_points]
                if np.sum(new_visibles) == 0:
                    indx -= 1
                else:
                    found_visible = True
            top_object_points = all_tracks[indx][:self.n_top_block_points][new_visibles]
            top_block_pos_pix = np.sum(top_object_points, axis=0)/len(top_object_points)

        top_block_pos_world = self.percept.get_object_center_point_in_world(int(top_block_pos_pix[0]), int(top_block_pos_pix[1]), depth)

        objs_pos_world = [top_block_pos_world.data]

        for i in range(self.n_obstacles):
            idx_low = self.n_top_block_points + i*self.n_points_per_obstacle
            idx_high = self.n_top_block_points + (i+1)*self.n_points_per_obstacle
            object_points = tracks[idx_low:idx_high][visibles[idx_low:idx_high]]
            if len(object_points) == 0:
                found_visible = False
                indx = len(all_visibles) - 2
                while not found_visible:
                    new_visibles = all_visibles[indx][idx_low:idx_high]
                    if np.sum(new_visibles) == 0:
                        indx -= 1
                    else:
                        found_visible = True
                object_points = all_tracks[indx][idx_low:idx_high][new_visibles]
            obj_pos_pix = np.sum(object_points, axis=0)/len(object_points)
            objs_pos_world.append(self.percept.get_object_center_point_in_world(int(obj_pos_pix[0]), int(obj_pos_pix[1]), depth).data)

        return np.array(objs_pos_world)

if __name__ == '__main__':
    cfg = OmegaConf.load('/home/saumyas/franka/frankapy/cfg/slide_pickup_clutter_env.yaml')

    rospy.init_node('tapnet_tracker', anonymous=True) # use when not using franka Arm

    pub = rospy.Publisher('tapnet_tracks_topic', Float32MultiArray, queue_size=10)
    objects_pub = rospy.Publisher('objects_world_pos_topic', Float32MultiArray, queue_size=10)

    rate = rospy.Rate(20)
    tracker = PointTrackingTapnet(cfg)

    cv2.namedWindow("Point Tracking", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Point Tracking", tracker.mouse_click)

    t = time.time()
    step_counter = 0
    if tracker.save_video:
        bgr_labeled_imgs = []
    all_tracks, all_visibles = [], []

    while not rospy.is_shutdown():
        rgb, depth, _ = tracker.percept.sensor.frames()
        frame = rgb.rgb2bgr().data
        if tracker.query_frame:
            query_points = jnp.array((0,) + tracker.pos, dtype=jnp.float32)
            init_query_features = tracker.online_init_apply(
                frames=model_utils.preprocess_frames(frame[None, None]),
                points=query_points[None, None],
            )
            tracker.query_frame = False
            tracker.query_features, tracker.causal_state = tracker.tapir.update_query_features(
                query_features=tracker.query_features,
                new_query_features=init_query_features,
                idx_to_update=np.array([tracker.next_query_idx]),
                causal_state=tracker.causal_state,
            )
            tracker.have_point[tracker.next_query_idx] = True
            rospy.loginfo(f"Chosen point {tracker.next_query_idx}: {tracker.pos}")
            tracker.next_query_idx = (tracker.next_query_idx + 1) % tracker.total_num_points
            
        if tracker.pos:
            prediction, tracker.causal_state = tracker.online_predict_apply(
                frames=model_utils.preprocess_frames(frame[None, None]),
                features=tracker.query_features,
                causal_context=tracker.causal_state,
            )
            track = prediction["tracks"][0, :, 0]
            occlusion = prediction["occlusion"][0, :, 0]
            expected_dist = prediction["expected_dist"][0, :, 0]
            visibles = model_utils.postprocess_occlusions(occlusion, expected_dist)
            track = np.round(track)

            if len(all_tracks) > 0:
                track = tracker.sanity_check(track, prev_track=all_tracks[-1])

            all_tracks.append(track)
            all_visibles.append(visibles)
            if tracker.visualize:
                for i, _ in enumerate(tracker.have_point):
                    cv2.circle(frame, (int(track[i, 0]), int(track[i, 1])), 8, tracker.colors[i], thickness=2)
                    if visibles[i] and tracker.have_point[i]:
                        cv2.circle(frame, (int(track[i, 0]), int(track[i, 1])), 5, tracker.colors[i], -1)

            if np.sum(tracker.have_point) == tracker.total_num_points:
                objects_xyz_world = tracker.get_object_pos_from_tracks(track, visibles, all_tracks, all_visibles, depth.data)
                msg = Float32MultiArray()
                msg.data = objects_xyz_world.flatten()
                objects_pub.publish(msg)

            msg = Float32MultiArray()
            msg.data = track.flatten()
            pub.publish(msg)
            # rospy.loginfo(f"Track np: {track}")
            # rospy.loginfo(f"Published array: {msg.data}")
        
        if tracker.visualize:
            cv2.imshow("Point Tracking", frame[:, ::-1])
        if tracker.save_video:
            bgr_labeled_imgs.append(frame)
        if tracker.pos:
            step_counter += 1
            if time.time() - t > 5:
                rospy.loginfo(f"{step_counter/(time.time()-t)} frames per second")
                t = time.time()
                step_counter = 0
        else:
            t = time.time()
        cv2.waitKey(1)
        
        rate.sleep()
    
    cv2.destroyWindow("Point Tracking")
    tracker.percept.sensor.stop()