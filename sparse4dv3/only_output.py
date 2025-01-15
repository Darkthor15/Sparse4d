# Python Libraries
import os
import cv2
import torch
import copy
import sys
import time
import numpy as np
import pyquaternion
from mmcv import Config
from mmcv.runner import wrap_fp16_model, load_checkpoint
from mmdet.models import build_detector
from mmdet.datasets.pipelines import Compose
from mmcv.parallel import DataContainer as DC
from torch2trt import TRTModule
# ROS2 Libraries
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, CompressedImage
from message_filters import Subscriber, ApproximateTimeSynchronizer
from sparse_msgs.msg import CustomTFMessage, BoxInfo, BBoxes3D

def import_plugins(cfg: Config) -> None:
    """Import custom plugins.

    Args:
        cfg (Config): Configs loaded from config file.
    """
    if cfg.plugin:
        import importlib
        if hasattr(cfg, "plugin_dir"):
            sys.path.append(os.getcwd())
            plugin_dir = cfg.plugin_dir
            _module_dir = os.path.dirname(plugin_dir)
            _module_dir = _module_dir.split("/")
            _module_path = _module_dir[0]

            for m in _module_dir[1:]:
                _module_path = _module_path + "." + m
            importlib.import_module(_module_path)

def create_model(cfg: Config, ckpt: str) -> torch.nn.Module:
    """Create Sparse4Dv3 Model from checkpoint.

    Args:
        cfg (Config): Configs loaded from config file.
        ckpt (str): Path to PyTorch checkpoint(.pth) for model.

    Returns:
        model (nn.Module): Sparse4Dv3 Pytorch model.
    """
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    if cfg.get('fp16', None):
        wrap_fp16_model(model)
    load_checkpoint(model, ckpt, map_location="cpu")
    return model

def process_transforms(infos: list, tfs: list, cfg: Config) -> dict:
    """Process transformations for Sparse4Dv3.

    Args:
        infos (list): Camera intrinsic information for all cameras.
        tfs (list): List of transforms from /tf topic.
        cfg (Config): Configs loaded from config file.

    Returns:
        input_dict (Dict): Dictionary containing transformations for lidar and cameras.
    """
    lidar2global = np.eye(4)  # Mock transformation for simplicity

    input_dict = dict(
        lidar2global=lidar2global,
    )
    return input_dict

class Sparse4Dv3Node(Node):

    def __init__(self):
        super().__init__("sparse4dv3_node")

        # Configs and Necessary variables
        self.get_logger().info("Initializing...")
        self.cfg = Config.fromfile('projects/configs/sparse4dv3_temporal_r50_1x8_bs6_256x704.py')
        self.cams = self.cfg.get("cams", None)
        self.images = [None] * len(self.cams)
        self.cam_intrinsics = [None] * len(self.cams)
        self.tf_info = None
        self.count = 0

        # Detection Model and Preprocess Pipeline
        if hasattr(self.cfg, 'plugin'):
            import_plugins(self.cfg)
        self.model = create_model(self.cfg, 'ckpt/sparse4dv3_r50.pth')
        self.model.cuda().eval()
        self.pipeline = Compose(self.cfg.test_pipeline)

        # Subscribers
        self.img_subs = [
            Subscriber(
                self,
                CompressedImage,
                f'/{cam}/image_rect_compressed',
            ) for cam in self.cams
        ]
        self.cam_info_subs = [
            Subscriber(
                self,
                CameraInfo,
                f'/{cam}/camera_info',
            ) for cam in self.cams
        ]
        self.tf_subs = Subscriber(
            self,
            CustomTFMessage,
            '/tf_stamped',
        )
        all_subscribers = self.img_subs + self.cam_info_subs + [self.tf_subs]

        # Time Synchronizer
        self.time_sync = ApproximateTimeSynchronizer(all_subscribers, queue_size=10, slop=0.1)
        self.time_sync.registerCallback(self.sync_callback)

        # Publishers
        self.pub_boxes = self.create_publisher(BBoxes3D, '/bboxes3d', 10)

        self.get_logger().info("Node Ready.")

    def sync_callback(self, *msg):
        img_msgs = msg[:len(self.cams)]
        cam_info_msgs = msg[len(self.cams):-1]
        tf_msgs = msg[-1]
        self.get_logger().info("Received all info")

        for i in range(len(self.cams)):
            img_decode = cv2.imdecode(np.frombuffer(img_msgs[i].data, np.uint8), cv2.IMREAD_COLOR)
            self.images[i] = img_decode
            self.cam_intrinsics[i] = cam_info_msgs[i].k.reshape(3, 3)
        self.tf_info = tf_msgs.tf_message.transforms

        self.forward()

    def forward(self):
        if all(img is not None for img in self.images) and self.tf_info is not None:
            self.count += 1
            boxes_3d = BBoxes3D()
            self.get_logger().info(f"Preprocessing frame {self.count}...")
            input_dict = process_transforms(self.cam_intrinsics, self.tf_info, self.cfg)
            input_dict["img"] = self.images
            input_dict = self.pipeline(input_dict)
            with torch.no_grad():
                out = self.model(return_loss=False, rescale=True, **input_dict)
            for i in range(len(out[0]["img_bbox"]["labels_3d"])):
                box = BoxInfo()
                box.id = int(i)
                box.bbox = out[0]["img_bbox"]["boxes_3d"][i].cpu().flatten().tolist()
                box.score = float(out[0]["img_bbox"]["cls_scores"][i].cpu())
                box.label = int(out[0]["img_bbox"]["labels_3d"][i].cpu())
                boxes_3d.boxes3d.append(box)

            self.get_logger().info("Inferencing done. Publishing topics...")
            self.pub_boxes.publish(boxes_3d)
            self.get_logger().info(f"Number of boxes detected: {len(out[0]['img_bbox']['labels_3d'])}")

            # Reset
            self.images = [None] * len(self.cams)
            self.cam_intrinsics = [None] * len(self.cams)
            self.tf_info = None

def main(args=None):
    rclpy.init(args=args)
    sparse_node = Sparse4Dv3Node()
    rclpy.spin(sparse_node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()

