import numpy as np
import cv2, os, json, time
from cv_bridge import CvBridge, CvBridgeError
from PIL import Image as PilImage

from omegaconf import OmegaConf

import supervision as sv
from supervision.draw.color import ColorPalette

from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import PosePositionSensorMessage, ShouldTerminateSensorMessage, CartesianImpedanceSensorMessage
from franka_interface_msgs.msg import SensorDataGroup
from frankapy.utils import min_jerk, min_jerk_weight

from sensor_msgs.msg import Image

import rospy
import scipy.stats

# from shapes import Rectangle
from autolab_core import RigidTransform, YamlConfig, Point
from perception import RgbdSensorFactory


from perception.camera_intrinsics import CameraIntrinsics

from google import genai
from google.genai import types

import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from transformers import OwlViTProcessor, OwlViTForObjectDetection

CUSTOM_COLOR_MAP = [
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#0082c8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#d2f53c",
    "#fabebe",
    "#008080",
    "#e6beff",
    "#aa6e28",
    "#fffac8",
    "#800000",
    "#aaffc3",
]

def parse_json(json_output):
    # Parsing out the markdown fencing
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
            json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
            break  # Exit the loop once "```json" is found
    return json_output

class Perception():
    def __init__(self, cfg=None):
        self.cfg = cfg
        self.bridge = CvBridge()
        self.sensor_name = list(cfg.sensors.keys())[0]
        self.sensor_config = cfg.sensors[self.sensor_name].sensor_config
        self.azure_kinect_intrinsics = CameraIntrinsics.load(cfg.sensors[self.sensor_name].intrinsics_file)
        self.azure_kinect_to_world_transform = RigidTransform.load(cfg.sensors[self.sensor_name].extrinsics_tf)

        self.device = f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu"
        self.setup_perception()

        self.client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    def setup_perception(self):
        self.visualize = self.cfg.detection.visualize
        self.save_dir = self.cfg.detection.save_dir

        self.detector_type = self.cfg.detection.detector_type
        detection_kwargs = self.cfg.detection[self.detector_type]
        if self.detector_type == 'GroundedSAM2':
            self.processor = AutoProcessor.from_pretrained(detection_kwargs.model_id)
            self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(detection_kwargs.model_id).to(self.device)
        elif self.detector_type == 'OwlViT':
            self.processor = OwlViTProcessor.from_pretrained(detection_kwargs.model_id)
            self.grounding_model = OwlViTForObjectDetection.from_pretrained(detection_kwargs.model_id).to(self.device)

        self.labels = self.cfg.detection.labels
        self.labels = ['a coffee cup', 'a ball', 'a box']
        self.embed_labels = self.processor(text=[self.labels], return_tensors="pt", padding=True).to(self.device)

        self.sensor_config['frame'] = self.sensor_name
        self.sensor = RgbdSensorFactory.sensor(self.sensor_config.type, self.sensor_config)
        self.sensor.start()

    def get_azure_kinect_rgb_image(self, topic='/rgb/image_raw'):
        """
        Grabs an RGB image for the topic as argument
        """
        rgb_image_msg = rospy.wait_for_message(topic, Image)
        try:
            rgb_cv_image = self.bridge.imgmsg_to_cv2(rgb_image_msg)
        except CvBridgeError as e:
            print(e)
        return rgb_cv_image

    def get_azure_kinect_depth_image(self, topic='/depth_to_rgb/image_raw'):
        """
        Grabs an Depth image for the topic as argument
        """
        depth_image_msg = rospy.wait_for_message(topic, Image)
        try:
            depth_cv_image = self.bridge.imgmsg_to_cv2(depth_image_msg)
        except CvBridgeError as e:
            print(e)
        return depth_cv_image
    
    def get_points_for_text(self, rgb_img):
        # start = time.time()
        # (1536, 2048, 3)
        rgb_img = rgb_img[100:1100,750:1900]
        pil_img = PilImage.fromarray(rgb_img).convert("RGB")

        # Analyze the image using Gemini
        response = self.client.models.generate_content(
            model="gemini-2.5-pro-preview-03-25",
            contents=[
                pil_img,
                f"""
                Point to objects in the list {self.labels}.
                Also point to the silver handle attached to one of the boxes, corresponding to the point where it should be grasped to pick it up.
                The answer should follow the json format: [{"point": <point>, "label": <label1>}, ...]. The points are in [y, x] format normalized to 0-1000.
                """
            ],
            config = types.GenerateContentConfig(
                temperature=0.2
            )
        )
        
        # Check response
        print(response.text)
        parsed_json = parse_json(response.text)
        print(parsed_json)

        # results = [result for result in results if len(result["boxes"])>0]
        # if self.visualize:
        #     self.visualize_results(rgb_imgs, results)

    def visualize_results(self, rgb_imgs, results):
        # get the box prompt for SAM 2
        for idx in range(len(rgb_imgs)):
            img, result = rgb_imgs[idx], results[idx]

            input_boxes = result["boxes"].cpu().numpy()
            confidences = result["scores"].cpu().numpy().tolist()
            class_names = result["labels"]
            class_ids = np.array(list(range(len(class_names))))

            labels = [
                f"{class_name} {confidence:.2f}"
                for class_name, confidence
                in zip(class_names, confidences)
            ]

            """
            Visualize image with supervision useful API
            """
            bgr_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            detections = sv.Detections(
                xyxy=input_boxes,  # (n, 4)
                class_id=class_ids
            )

            """
            Note that if you want to use default color map,
            you can set color=ColorPalette.DEFAULT
            """
            cv2.imwrite(os.path.join(self.save_dir, f"owlVi_image_{idx}.jpg"), bgr_image)
            box_annotator = sv.BoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
            annotated_frame = box_annotator.annotate(scene=bgr_image.copy(), detections=detections)

            label_annotator = sv.LabelAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
            cv2.imwrite(os.path.join(self.save_dir, f"owlVit_annotated_image_{idx}.jpg"), annotated_frame)
            

    def get_object_center_point_in_world(self, object_image_center_x, object_image_center_y, depth_image):    
        object_center = Point(np.array([object_image_center_x, object_image_center_y]), self.sensor_name)
        object_depth = depth_image[object_image_center_y, object_image_center_x] * 0.001
        object_center_point_in_world = self.azure_kinect_to_world_transform * self.azure_kinect_intrinsics.deproject_pixel(object_depth, object_center)    
        return object_center_point_in_world 

    def get_pose_msg(self, timestamp, i, T_tag_world):
        traj_gen_proto_msg = PosePositionSensorMessage(
            id=i, timestamp=timestamp, 
            position=T_tag_world.translation, quaternion=T_tag_world.quaternion
        )
        return traj_gen_proto_msg

if __name__ == "__main__":

    cfg_path = '/home/saumyas/franka/frankapy/cfg/grounding_dino_pick.yaml'
    cfg = OmegaConf.load(cfg_path)

    # fa = FrankaArm()
    rospy.init_node('register_camera', anonymous=True) # use when not using franka Arm
    percept = Perception(cfg=cfg)

    rgb, depth, _ = percept.sensor.frames()

    aa = percept.get_points_for_text([rgb._data])
    
    import ipdb; ipdb.set_trace()

    fa.open_gripper()
    fa.goto_gripper(0.02)
    fa.reset_pose() 
    fa.reset_joints()

    p0 = fa.get_pose() # get_initial_pose

    T = 10
    dt = 0.02
    ts = np.arange(0, T, dt)

    rospy.loginfo('Initializing Sensor Publisher')
    pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1000)
    rate = rospy.Rate(1 / dt)

    rospy.loginfo('Publishing pose trajectory...')

    # To ensure skill doesn't end before completing trajectory, make the buffer time much longer than needed
    fa.goto_pose(p0, duration=T, dynamic=True, buffer_time=2)

    percept.sensor.stop()

