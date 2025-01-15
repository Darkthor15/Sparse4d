import torch
import torch.nn as nn
import numpy as np

from mmcv.cnn import Linear, Scale, bias_init_with_prob
from mmcv.runner.base_module import Sequential, BaseModule
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.registry import (
    PLUGIN_LAYERS,
    POSITIONAL_ENCODING,
)
from torch.profiler import record_function
from projects.mmdet3d_plugin.core.box3d import *
from ..blocks import linear_relu_ln, get_learnable_params
from ...ops import keypoints_generation_function as generate_keypoints
from ...ops import sparse_box3d_refinement_function

__all__ = [
    "SparseBox3DRefinementModule",
    "SparseBox3DKeyPointsGenerator",
    "SparseBox3DEncoder",
]

def sparse_box3d_refinement(instance_feature, anchor, anchor_embed, time_interval,
    layers, cls_layers, quality_layers, embed_dims, output_dims, num_cls, return_cls):
    weights, biases, scales = get_learnable_params([ layers, cls_layers, quality_layers ])
    output, cls, quality = sparse_box3d_refinement_function(instance_feature, anchor, anchor_embed, time_interval,
        weights, biases, scales[0], embed_dims, output_dims, num_cls, return_cls)
    return (output, cls, quality) if return_cls else (output, None, None)

@POSITIONAL_ENCODING.register_module()
class SparseBox3DEncoder(BaseModule):
    def __init__(
        self,
        embed_dims,
        vel_dims=3,
        mode="add",
        output_fc=True,
        in_loops=1,
        out_loops=2,
    ):
        super().__init__()
        assert mode in ["add", "cat"]
        self.embed_dims = embed_dims
        self.vel_dims = vel_dims
        self.mode = mode

        def embedding_layer(input_dims, output_dims):
            return nn.Sequential(
                *linear_relu_ln(output_dims, in_loops, out_loops, input_dims)
            )

        if not isinstance(embed_dims, (list, tuple)):
            embed_dims = [embed_dims] * 5
        self.pos_fc = embedding_layer(3, embed_dims[0])
        self.size_fc = embedding_layer(3, embed_dims[1])
        self.yaw_fc = embedding_layer(2, embed_dims[2])
        if vel_dims > 0:
            self.vel_fc = embedding_layer(self.vel_dims, embed_dims[3])
        if output_fc:
            self.output_fc = embedding_layer(embed_dims[-1], embed_dims[-1])
        else:
            self.output_fc = None

    def forward(self, box_3d: torch.Tensor):

        if self.vel_dims > 0:
            models = [self.pos_fc, self.size_fc, self.yaw_fc, self.vel_fc]
            inputs = [box_3d[..., X : X + 3], box_3d[..., W : W + 3],
                    box_3d[..., SIN_YAW : SIN_YAW + 2], box_3d[..., VX : VX + self.vel_dims]]
            outputs = [model(ins) for model, ins in zip(models, inputs)]
            if self.mode == "add":
                output = sum(outputs)
            elif self.mode == "cat":
                output = torch.cat(outputs, dim=-1)

        else:
            models = [self.pos_fc, self.size_fc, self.yaw_fc]
            inputs = [box_3d[..., X : X + 3], box_3d[..., W : W + 3], 
                    box_3d[..., SIN_YAW : SIN_YAW + 2]]
            outputs = [model(ins) for model, ins in zip(models, inputs)]
            if self.mode == "add":
                output = sum(outputs)
            elif self.mode == "cat":
                output = torch.cat(outputs, dim=-1)

        if self.output_fc is not None:
            output = self.output_fc(output)
        return output


@PLUGIN_LAYERS.register_module()
class SparseBox3DRefinementModule(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        output_dim=11,
        num_cls=10,
        normalize_yaw=False,
        refine_yaw=False,
        with_cls_branch=True,
        with_quality_estimation=False,
    ):
        super(SparseBox3DRefinementModule, self).__init__()
        self.embed_dims = embed_dims
        self.output_dim = output_dim
        self.num_cls = num_cls
        self.normalize_yaw = normalize_yaw
        self.refine_yaw = refine_yaw

        self.refine_state = [X, Y, Z, W, L, H]
        if self.refine_yaw:
            self.refine_state += [SIN_YAW, COS_YAW]

        self.layers = nn.Sequential(
            *linear_relu_ln(embed_dims, 2, 2),
            Linear(self.embed_dims, self.output_dim),
            Scale([1.0] * self.output_dim),
        )
        self.with_cls_branch = with_cls_branch
        if with_cls_branch:
            self.cls_layers = nn.Sequential(
                *linear_relu_ln(embed_dims, 1, 2),
                Linear(self.embed_dims, self.num_cls),
            )
        self.with_quality_estimation = with_quality_estimation
        if with_quality_estimation:
            self.quality_layers = nn.Sequential(
                *linear_relu_ln(embed_dims, 1, 2),
                Linear(self.embed_dims, 2),
            )

    def init_weight(self):
        if self.with_cls_branch:
            bias_init = bias_init_with_prob(0.01)
            nn.init.constant_(self.cls_layers[-1].bias, bias_init)

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        time_interval: torch.Tensor = 1.0,
        return_cls=True,
    ):
        enable_optimization = not self.normalize_yaw and self.output_dim > 8 and isinstance(time_interval, torch.Tensor)
        if enable_optimization:
            output, cls, quality = sparse_box3d_refinement(instance_feature, anchor, anchor_embed, time_interval,
                self.layers, self.cls_layers, self.quality_layers, self.embed_dims, self.output_dim, self.num_cls, return_cls)
            return output, cls, quality
        else:
            feature = instance_feature + anchor_embed
            output = self.layers(feature)
            output[..., self.refine_state] = (
                output[..., self.refine_state] + anchor[..., self.refine_state]
            )
            if self.normalize_yaw:
                output[..., [SIN_YAW, COS_YAW]] = torch.nn.functional.normalize(
                    output[..., [SIN_YAW, COS_YAW]], dim=-1
                )
            if self.output_dim > 8:
                if not isinstance(time_interval, torch.Tensor):
                    time_interval = instance_feature.new_tensor(time_interval)
                translation = torch.transpose(output[..., VX:], 0, -1)
                velocity = torch.transpose(translation / time_interval, 0, -1)
                output[..., VX:] = velocity + anchor[..., VX:]

        if return_cls:
            assert self.with_cls_branch, "Without classification layers !!!"
            cls = self.cls_layers(instance_feature)
        else:
            cls = None
        if return_cls and self.with_quality_estimation:
            quality = self.quality_layers(feature)
        else:
            quality = None
        return output, cls, quality

# @torch.compile
def get_learnable_scale(instance_feature, linear):
    return torch.sigmoid(linear(instance_feature)) - 0.5

@PLUGIN_LAYERS.register_module()
class SparseBox3DKeyPointsGenerator(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        num_learnable_pts=0,
        fix_scale=None,
    ):
        super(SparseBox3DKeyPointsGenerator, self).__init__()
        self.embed_dims = embed_dims
        self.num_learnable_pts = num_learnable_pts
        if fix_scale is None:
            fix_scale = ((0.0, 0.0, 0.0),)
        self.fix_scale = nn.Parameter(
            torch.tensor(fix_scale), requires_grad=False
        )
        self.num_pts = len(self.fix_scale) + num_learnable_pts
        if num_learnable_pts > 0:
            self.learnable_fc = Linear(self.embed_dims, num_learnable_pts * 3)

        self.use_keypoints_generation_kernel = True

    def init_weight(self):
        if self.num_learnable_pts > 0:
            xavier_init(self.learnable_fc, distribution="uniform", bias=0.0)

    def forward(
        self,
        anchor,
        instance_feature=None,
        T_cur2temp_list=None,
        cur_timestamp=None,
        temp_timestamps=None,
    ):
        with record_function("SparseKPGenerator"):
            static_generation = cur_timestamp is None or temp_timestamps is None or T_cur2temp_list is None or len(temp_timestamps) == 0
            if self.use_keypoints_generation_kernel and static_generation:
                fixed_scale = self.fix_scale
                learn_scale = get_learnable_scale(instance_feature, self.learnable_fc)
                keypoints = generate_keypoints(anchor, fixed_scale, learn_scale)
                return keypoints

            bs, num_anchor = anchor.shape[:2]
            size = anchor[..., None, [W, L, H]].exp()
            key_points = self.fix_scale * size
            if self.num_learnable_pts > 0 and instance_feature is not None:
                learnable_scale = (
                    self.learnable_fc(instance_feature)
                    .reshape(bs, num_anchor, self.num_learnable_pts, 3)
                    .sigmoid()
                    - 0.5
                )
                key_points = torch.cat(
                    [key_points, learnable_scale * size], dim=-2
                )

            rotation_mat = anchor.new_zeros([bs, num_anchor, 3, 3])

            rotation_mat[:, :, 0, 0] = anchor[:, :, COS_YAW]
            rotation_mat[:, :, 0, 1] = -anchor[:, :, SIN_YAW]
            rotation_mat[:, :, 1, 0] = anchor[:, :, SIN_YAW]
            rotation_mat[:, :, 1, 1] = anchor[:, :, COS_YAW]
            rotation_mat[:, :, 2, 2] = 1

            key_points = torch.matmul(
                rotation_mat[:, :, None], key_points[..., None]
            ).squeeze(-1)
            key_points = key_points + anchor[..., None, [X, Y, Z]]

            if static_generation:
                return key_points

            temp_key_points_list = []
            velocity = anchor[..., VX:]
            for i, t_time in enumerate(temp_timestamps):
                time_interval = cur_timestamp - t_time
                translation = (
                    velocity
                    * time_interval.to(dtype=velocity.dtype)[:, None, None]
                )
                temp_key_points = key_points - translation[:, :, None]
                T_cur2temp = T_cur2temp_list[i].to(dtype=key_points.dtype)
                temp_key_points = (
                    T_cur2temp[:, None, None, :3]
                    @ torch.cat(
                        [
                            temp_key_points,
                            torch.ones_like(temp_key_points[..., :1]),
                        ],
                        dim=-1,
                    ).unsqueeze(-1)
                )
                temp_key_points = temp_key_points.squeeze(-1)
                temp_key_points_list.append(temp_key_points)
        return key_points, temp_key_points_list

    @staticmethod
    def anchor_projection(
        anchor,
        T_src2dst_list,
        src_timestamp=None,
        dst_timestamps=None,
        time_intervals=None,
    ):
        dst_anchors = []
        for i in range(len(T_src2dst_list)):
            vel = anchor[..., VX:]
            vel_dim = vel.shape[-1]
            T_src2dst = torch.unsqueeze(
                T_src2dst_list[i].to(dtype=anchor.dtype), dim=1
            )

            center = anchor[..., [X, Y, Z]]
            if time_intervals is not None:
                time_interval = time_intervals[i]
            elif src_timestamp is not None and dst_timestamps is not None:
                time_interval = (src_timestamp - dst_timestamps[i]).to(
                    dtype=vel.dtype
                )
            else:
                time_interval = None
            if time_interval is not None:
                translation = vel.transpose(0, -1) * time_interval
                translation = translation.transpose(0, -1)
                center = center - translation
            center = (
                torch.matmul(
                    T_src2dst[..., :3, :3], center[..., None]
                ).squeeze(dim=-1)
                + T_src2dst[..., :3, 3]
            )
            size = anchor[..., [W, L, H]]
            yaw = torch.matmul(
                T_src2dst[..., :2, :2],
                anchor[..., [COS_YAW, SIN_YAW], None],
            ).squeeze(-1)
            vel = torch.matmul(
                T_src2dst[..., :vel_dim, :vel_dim], vel[..., None]
            ).squeeze(-1)
            dst_anchor = torch.cat([center, size, yaw, vel], dim=-1)
            # TODO: Fix bug
            # index = [X, Y, Z, W, L, H, COS_YAW, SIN_YAW] + [VX, VY, VZ][:vel_dim]
            # index = torch.tensor(index, device=dst_anchor.device)
            # index = torch.argsort(index)
            # dst_anchor = dst_anchor.index_select(dim=-1, index=index)
            dst_anchors.append(dst_anchor)
        return dst_anchors

    @staticmethod
    def distance(anchor):
        return torch.norm(anchor[..., :2], p=2, dim=-1)