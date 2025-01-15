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
from mmdet.datasets import build_dataset
from mmdet.datasets.pipelines import Compose
from mmcv.parallel import DataContainer as DC
from torch2trt import TRTModule
# ROS2 Libraries
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, CompressedImage
from message_filters import Subscriber, ApproximateTimeSynchronizer
from sparse_msgs.msg import CustomTFMessage, BoxInfo, BBoxes3D

def import_plugins(cfg:Config) -> None:
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
            print(_module_path)
            plg_lib = importlib.import_module(_module_path)


def create_model(cfg: Config, ckpt:str) -> torch.nn.Module:
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
    checkpoint = load_checkpoint(model, ckpt, map_location="cpu")
    if cfg.use_tensorrt:
        for k, v in cfg.trt_paths.items():
            if k == 'backbone' and v is not None:
                model.img_backbone = TRTModule()
                model.img_backbone.load_state_dict(torch.load(v))
            elif k == 'neck' and v is not None:
                model.img_neck = TRTModule()
                model.img_neck.load_state_dict(torch.load(v))
            elif k == 'encoder' and v is not None:
                model.head.anchor_encoder = TRTModule()
                model.head.anchor_encoder.load_state_dict(torch.load(v))
            elif k == 'temp_encoder' and v is not None:
                model.head.temp_anchor_encoder = TRTModule()
                model.head.temp_anchor_encoder.load_state_dict(torch.load(v))

    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if "CLASSES" in checkpoint.get("meta", {}):
        model.CLASSES = checkpoint["meta"]["CLASSES"]
    # palette for visualization in segmentation tasks
    if "PALETTE" in checkpoint.get("meta", {}):
        model.PALETTE = checkpoint["meta"]["PALETTE"]
    return model


def obtain_sensor2lidar_rt(
    l2e_t:list, l2e_r:list, e2g_t:list, e2g_r:list,
    s2e_t:list, s2e_r:list, se2g_t:list, se2g_r:list
) -> tuple:
    """Obtain the info with RT from sensor to LiDAR.
    
    Args:
        l2e_t (list): Translation from LiDAR to ego in (x, y, z).
        l2e_r (list): Rotation quat from LiDAR to ego in (w, x, y, z).
        e2g_t (list): Translation from ego to global in (x, y, z).
        e2g_r (list): Rotation quat from ego to global in (w, x, y, z).
        s2e_t (list): Translation from sensor to ego in (x, y, z).
        s2e_r (list): Rotation quat from sensor to ego in (w, x, y, z).
        se2g_t (list): Translation from sensor ego to global in (x, y, z).
        se2g_r (list): Rotation quat from sensor ego to global in (w, x, y, z).
    
    Returns:
        s2l_r (np.ndarray): Sensor to LiDAR rotation matrix.
        s2l_t (np.ndarray): Sensor to LiDAR translation vector.
    """
    # sensor->ego->global->ego'->lidar
    l2e_r_mat = pyquaternion.Quaternion(l2e_r).rotation_matrix
    e2g_r_mat = pyquaternion.Quaternion(e2g_r).rotation_matrix
    s2e_r_mat = pyquaternion.Quaternion(s2e_r).rotation_matrix
    se2g_r_mat = pyquaternion.Quaternion(se2g_r).rotation_matrix

    R = (s2e_r_mat.T @ se2g_r_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T = (np.array(s2e_t) @ se2g_r_mat.T + np.array(se2g_t)) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T -= (
        np.array(e2g_t) @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
        + np.array(l2e_t) @ np.linalg.inv(l2e_r_mat).T
    )
    s2l_r = R.T  # points @ R.T + T
    s2l_t = T

    return s2l_r, s2l_t


def get_augmentation(cfg:Config) -> dict:
    """Get Image Augmentation parameters.
    
    Args:
        cfg (Config): Configs loaded from config file.
    
    Returns:
        aug_config (Dict): Dictionary with augmentation parameters.
    """
    if cfg.data_aug_conf is None:
        return None
    H, W = cfg.data_aug_conf["H"], cfg.data_aug_conf["W"]
    fH, fW = cfg.data_aug_conf["final_dim"]
    resize = max(fH / H, fW / W)
    resize_dims = (int(W * resize), int(H * resize))
    newW, newH = resize_dims
    crop_h = (
        int((1 - np.mean(cfg.data_aug_conf["bot_pct_lim"])) * newH)
        - fH
    )
    crop_w = int(max(0, newW - fW) / 2)
    crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
    flip = False
    rotate = 0
    rotate_3d = 0
    aug_config = {
        "resize": resize,
        "resize_dims": resize_dims,
        "crop": crop,
        "flip": flip,
        "rotate": rotate,
        "rotate_3d": rotate_3d,
    }
    return aug_config


def process_transforms(infos:list, tfs:list, cfg:Config) -> dict:
    """ Obtain lidar2global and lidar2image(if any) transforms in Dictionary format.

    Args:
        infos (list): Camera intrinsic information for all cameras.
        tfs (list): List of transforms from /tf topic.
        cfg (Config): Configs loaded from config file.
    
    Returns:
        input_dict (Dict): Dictionary containing tranforms for lidar and cameras.
    """
    timestamp = tfs[0].header.stamp.sec + (tfs[0].header.stamp.nanosec / 1e9)
    lidar2ego_translation = None
    lidar2ego_rotation = None
    ego2global_translation = None
    ego2global_rotation = None

    # Get LiDAR transforms first
    # (Needed for sensor2lidar_RT calculation)
    for tf in tfs:
        if tf.child_frame_id == "LIDAR_TOP":
            lidar2ego_translation = [
                tf.transform.translation.x,
                tf.transform.translation.y,
                tf.transform.translation.z
            ]
            lidar2ego_rotation = [
                tf.transform.rotation.w,
                tf.transform.rotation.x,
                tf.transform.rotation.y,
                tf.transform.rotation.z
            ]
        elif tf.child_frame_id == "LIDAR_TOP_GLOBAL":
            ego2global_translation = [
                tf.transform.translation.x,
                tf.transform.translation.y,
                tf.transform.translation.z
            ]
            ego2global_rotation = [
                tf.transform.rotation.w,
                tf.transform.rotation.x,
                tf.transform.rotation.y,
                tf.transform.rotation.z
            ]
    lidar2ego = np.eye(4)
    lidar2ego[:3, :3] = pyquaternion.Quaternion(
        lidar2ego_rotation).rotation_matrix
    lidar2ego[:3, 3] = np.array(lidar2ego_translation)
    ego2global = np.eye(4)
    ego2global[:3, :3] = pyquaternion.Quaternion(
        ego2global_rotation).rotation_matrix
    ego2global[:3, 3] = np.array(ego2global_translation)
    lidar2global = ego2global @ lidar2ego

    input_dict = dict(
        timestamp=timestamp,
        lidar2ego_translation=lidar2ego_translation,
        lidar2ego_rotation=lidar2ego_rotation,
        ego2global_translation=ego2global_translation,
        ego2global_rotation=ego2global_rotation,
        lidar2global=lidar2global,
    )

    # Get camera transforms next
    if cfg.input_modality["use_camera"]:
        cams = {k:v for v, k in enumerate(cfg.cams)}
        sensor2ego_translation = [None] * len(cams)
        sensor2ego_rotation = [None] * len(cams)
        sensor_ego2global_translation = [None] * len(cams)
        sensor_ego2global_rotation = [None] * len(cams)
        lidar2img_rts = [None] * len(cams)
        cam_intrinsic = [None] * len(cams)
        for tf in tfs:
            if tf.child_frame_id in cams:
                sensor2ego_translation[cams[tf.child_frame_id]] = [
                    tf.transform.translation.x,
                    tf.transform.translation.y,
                    tf.transform.translation.z
                ]
                sensor2ego_rotation[cams[tf.child_frame_id]] = [
                    tf.transform.rotation.w,
                    tf.transform.rotation.x,
                    tf.transform.rotation.y,
                    tf.transform.rotation.z
                ]
            elif tf.child_frame_id[:-7] in cams:
                sensor_ego2global_translation[cams[tf.child_frame_id[:-7]]] = [
                    tf.transform.translation.x,
                    tf.transform.translation.y,
                    tf.transform.translation.z
                ]
                sensor_ego2global_rotation[cams[tf.child_frame_id[:-7]]] = [
                    tf.transform.rotation.w,
                    tf.transform.rotation.x,
                    tf.transform.rotation.y,
                    tf.transform.rotation.z
                ]
        # Get LiDAR to Image projection matrices for all cameras
        for i in range(len(cams)):
            sensor2lidar_rotation, sensor2lidar_translation = obtain_sensor2lidar_rt(
                lidar2ego_translation, lidar2ego_rotation,
                ego2global_translation, ego2global_rotation,
                sensor2ego_translation[i], sensor2ego_rotation[i],
                sensor_ego2global_translation[i], sensor_ego2global_rotation[i]
            )
            lidar2cam_r = np.linalg.inv(sensor2lidar_rotation)
            lidar2cam_t = (sensor2lidar_translation @ lidar2cam_r.T)
            lidar2cam_rt = np.eye(4)
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3] = -lidar2cam_t
            intrinsic = copy.deepcopy(infos[i])
            cam_intrinsic[i] = intrinsic
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            lidar2img_rt = viewpad @ lidar2cam_rt.T
            lidar2img_rts[i] = lidar2img_rt
        input_dict.update(
            dict(
                lidar2img=lidar2img_rts,
                cam_intrinsic=cam_intrinsic,
            )
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
        self.cam_wh = [None] * len(self.cams)
        self.tf_info = None
        self.count = 0
        self.outs = []
        # Detection Model and Preprocess Pipeline
        if hasattr(self.cfg, 'plugin'):
            import_plugins(self.cfg)
        self.model = create_model(self.cfg, 'ckpt/sparse4dv3_r50.pth')
        self.model.cuda().eval()
        self.aug_config = get_augmentation(self.cfg)
        self.pipeline = Compose(self.cfg.test_pipeline)
        self.message_time_threshold = 30
        
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
        self.time_sync = ApproximateTimeSynchronizer(all_subscribers,
                                                    queue_size=10, slop=0.1)
        self.time_sync.registerCallback(self.sync_callback)

        # Publishers
        self.pub_boxes = self.create_publisher(
            BBoxes3D,
            '/bboxes3d',
            10,
        )

        self.message_time = time.time()
        self.dest_timer = self.create_timer(1.0, self.check_timeout)
        self.get_logger().info("Node Ready.")
        
    # Callbacks
    def sync_callback(self, *msg):
        img_msgs = msg[:len(self.cams)]
        cam_info_msgs = msg[len(self.cams):-1]
        tf_msgs = msg[-1]
        self.get_logger().info("Received all info")
        self.message_time = time.time()

        for i in range(len(self.cams)):
            img_decode = cv2.imdecode(np.frombuffer(img_msgs[i].data, np.uint8),
                                        cv2.IMREAD_COLOR)
            self.images[i] = img_decode
            self.cam_intrinsics[i] = cam_info_msgs[i].k.reshape(3, 3)
            self.cam_wh[i] = [cam_info_msgs[i].width, cam_info_msgs[i].height]
        self.tf_info = tf_msgs.tf_message.transforms

        self.forward()
    
    def check_timeout(self):
        curr_time = time.time()
        time_since_last = curr_time - self.message_time
        if time_since_last > self.message_time_threshold:
            self.get_logger().info(f"No messages from topics for {self.message_time_threshold}[sec].")
            self.destroy_node()
    
    # Forward to Sparse4Dv3
    def forward(self):
        if all(img is not None for img in self.images) \
        and all(cam_info is not None for cam_info in self.cam_intrinsics) \
        and (self.tf_info is not None):
            self.count += 1
            boxes_3d = BBoxes3D()
            self.get_logger().info(f"Preprocessing frame {self.count}...")
            input_dict = process_transforms(self.cam_intrinsics, self.tf_info, self.cfg)
            input_dict["img"] = self.images
            input_dict["aug_config"] = self.aug_config
            input_dict = self.pipeline(input_dict)
            for k, v in input_dict.items():
                if isinstance(v, DC):
                    if k == 'img':
                        input_dict[k] = v.data.unsqueeze(dim=0).cuda()
                    elif k == 'img_metas':
                        input_dict[k] = [v.data]
                        ts = str(input_dict[k][0]["timestamp"]).split(".")
                        boxes_3d.header.stamp.sec = int(ts[0])
                        boxes_3d.header.stamp.nanosec = int(ts[1])
                elif isinstance(v, np.ndarray):
                    input_dict[k] = torch.from_numpy(v).unsqueeze(dim=0).cuda()
                else:
                    input_dict[k] = torch.tensor([v], dtype=torch.float64).cuda()
            self.get_logger().info("Preprocessing done. Inferencing...")
            with torch.no_grad():
                out = self.model(return_loss=False, rescale=True, **input_dict)
                self.outs.append(out[0])
            #score_threshold = 0  # Set your desired threshold here
            #for i in range(len(out[0]["img_bbox"]["labels_3d"])):
             #   score = float(out[0]["img_bbox"]["cls_scores"][i].cpu())
              #  if score >= score_threshold:  # Only include boxes with scores above the threshold
               #     box = BoxInfo()
             #       box.id = int(i)
              #      box.bbox = out[0]["img_bbox"]["boxes_3d"][i].cpu().flatten().tolist()
               #     box.score = score
                #    box.label = int(out[0]["img_bbox"]["labels_3d"][i].cpu())
               #     box.instance = int(out[0]["img_bbox"]["instance_ids"][i].cpu())
                #    boxes_3d.boxes3d.append(box)
                 #   num_boxes = len(out[0]["img_bbox"]["labels_3d"])

            for i in range(len(out[0]["img_bbox"]["labels_3d"])):
                box = BoxInfo()
                box.id = int(i)
                box.bbox = out[0]["img_bbox"]["boxes_3d"][i].cpu().flatten().tolist()
                box.score = float(out[0]["img_bbox"]["cls_scores"][i].cpu())
                box.label = int(out[0]["img_bbox"]["labels_3d"][i].cpu())
                box.instance = int(out[0]["img_bbox"]["instance_ids"][i].cpu())
                boxes_3d.boxes3d.append(box)
          #      num_boxes = len(out[0]["img_bbox"]["labels_3d"])


            self.get_logger().info("Inferencing done. Publishing topics...")
            # Publish output
            self.pub_boxes.publish(boxes_3d)

            self.get_logger().info("Published successfully!")
            # Reset
            self.images = [None] * len(self.cams)
            self.cam_intrinsics = [None] * len(self.cams)
            self.cam_wh = [None] * len(self.cams)
            self.tf_info = None
    
    def print_eval_output(self, metrics_summary):
        self.get_logger().info('mAP: %.4f' % (metrics_summary['mean_ap']))
        err_name_mapping = {
            'trans_err': 'mATE',
            'scale_err': 'mASE',
            'orient_err': 'mAOE',
            'vel_err': 'mAVE',
            'attr_err': 'mAAE'
        }
        for tp_name, tp_val in metrics_summary['tp_errors'].items():
            self.get_logger().info('%s: %.4f' % (err_name_mapping[tp_name], tp_val))
        self.get_logger().info('NDS: %.4f' % (metrics_summary['nd_score']))
        self.get_logger().info('Eval time: %.1fs' % metrics_summary['eval_time'])

        # self.get_logger().info per-class metrics.
        self.get_logger().info('')
        self.get_logger().info('Per-class results:')
        self.get_logger().info('Object Class\tAP\tATE\tASE\tAOE\tAVE\tAAE')
        class_aps = metrics_summary['mean_dist_aps']
        class_tps = metrics_summary['label_tp_errors']
        for class_name in class_aps.keys():
            self.get_logger().info('%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
                  % (class_name, class_aps[class_name],
                     class_tps[class_name]['trans_err'],
                     class_tps[class_name]['scale_err'],
                     class_tps[class_name]['orient_err'],
                     class_tps[class_name]['vel_err'],
                     class_tps[class_name]['attr_err']))
    
    def destroy_node(self):
        if self.cfg.ros_eval:
            self.get_logger().info("Evaluating outputs...")
            dataset = build_dataset(self.cfg.data.test)
            eval_kwargs = self.cfg.get("evaluation", {}).copy()
            eval_kwargs.pop("interval", None)
            eval_kwargs.update(dict(metric='bbox'))
            eval_result, metrics = dataset.evaluate(self.outs, **eval_kwargs)
            self.print_eval_output(metrics)
        self.get_logger().info("Exiting.")
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    sparse_node = Sparse4Dv3Node()
    rclpy.spin(sparse_node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
