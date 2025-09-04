import rospy
from omegaconf import OmegaConf

import numpy as np
import cv2, os, json, time
from cv_bridge import CvBridge, CvBridgeError

from perception import RgbdSensorFactory
from perception.camera_intrinsics import CameraIntrinsics

from autolab_core import RigidTransform

from pynput import keyboard

from frankapy import FrankaArm

stop_recording = False
def on_press(key):
    global stop_recording
    try:
        if key.char == 'q':
            print("Key 'q' pressed. Stopping recording...")
            stop_recording = True
            return False  # Stop listener
    except AttributeError:
        pass  # Special keys like ctrl, etc.

class Perception():
    def __init__(self, cfg=None):
        self.cfg = cfg
        self.bridge = CvBridge()
        self.sensor_name = list(cfg.sensors.keys())[0]
        self.sensor_config = cfg.sensors[self.sensor_name].sensor_config
        self.azure_kinect_intrinsics = CameraIntrinsics.load(cfg.sensors[self.sensor_name].intrinsics_file)
        self.azure_kinect_to_world_transform = RigidTransform.load(cfg.sensors[self.sensor_name].extrinsics_tf)

        self.setup_perception()

    def setup_perception(self):
        self.sensor_config['frame'] = self.sensor_name
        self.sensor = RgbdSensorFactory.sensor(self.sensor_config.type, self.sensor_config)
        self.sensor.start()

if __name__ == "__main__":
    cfg_path = '/home/saumyas/franka/frankapy/cfg/grounding_dino_pick.yaml'
    cfg = OmegaConf.load(cfg_path)

    out_path = '/home/saumyas/franka/frankapy/data/media'

    fa = FrankaArm()
    # rospy.init_node('save_video', anonymous=True) # use when not using franka Arm
    percept = Perception(cfg=cfg)

    
    # Start the listener in a non-blocking thread
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    print("Recording... Press 'q' to stop.")

    rgb_frames = []
    depth_frames = []
    step_counter = 0
    t = time.time()
    while not stop_recording:
        rgb, depth, _ = percept.sensor.frames()
        rgb_frames.append(rgb.rgb2bgr().data)

        depth_norm = cv2.normalize(depth.data, None, 0, 255, cv2.NORM_MINMAX)
        depth_uint8 = depth_norm.astype(np.uint8)
        depth_frames.append(depth_uint8)
        
        step_counter += 1
        if time.time() - t > 2:
            print(f"{step_counter/(time.time()-t)} frames per second")
            t = time.time()
            step_counter = 0
    
    listener.join()

    H, W = rgb_frames[0].shape[:2]
    fps = 5  # adjust if needed

    # percept.sensor.stop()
    # print('saving rgb video')
    # rgb_writer = cv2.VideoWriter(os.path.join(out_path, "rgb_output_redblue_tilt2.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    # for f in rgb_frames:
    #     rgb_writer.write(f)
    # rgb_writer.release()

    # # Save Depth video (grayscale)
    # print('saving depth video')
    # depth_writer = cv2.VideoWriter(os.path.join(out_path, "depth_output_redblue_tilt2.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H), isColor=False)
    # for f in depth_frames:
    #     depth_writer.write(f)
    # depth_writer.release()

    # print("Done!")