# Copyright 2021 RangiLyu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import jittor as jt
from jittor import nn

from ...util import bbox2distance, distance2bbox, multi_apply
from ..loss.gfocal_loss import DistributionFocalLoss, QualityFocalLoss
from ..loss.iou_loss import GIoULoss
from ..module.conv import ConvModule, DepthwiseConvModule
from ..module.init_weights import normal_init
from .gfl_head import Integral


class NanoDetPlusHead(nn.Module):
    """Detection head used in NanoDet-Plus.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        loss (dict): Loss config.
        input_channel (int): Number of channels of the input feature.
        feat_channels (int): Number of channels of the feature.
            Default: 96.
        stacked_convs (int): Number of conv layers in the stacked convs.
            Default: 2.
        kernel_size (int): Size of the convolving kernel. Default: 5.
        strides (list[int]): Strides of input multi-level feature maps.
            Default: [8, 16, 32].
        conv_type (str): Type of the convolution.
            Default: "DWConv".
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN').
        reg_max (int): The maximal value of the discrete set. Default: 7.
        activation (str): Type of activation function. Default: "LeakyReLU".
        assigner_cfg (dict): Config dict of the assigner. Default: dict(topk=13).
    """

    def __init__(
        self,
        num_classes,
        loss,
        input_channel,
        feat_channels=96,
        stacked_convs=2,
        kernel_size=5,
        strides=[8, 16, 32],
        conv_type="DWConv",
        norm_cfg=dict(type="BN"),
        reg_max=7,
        activation="LeakyReLU",
        assigner_cfg=dict(topk=13),
        **kwargs
    ):
        super(NanoDetPlusHead, self).__init__()
        self.num_classes = num_classes
        self.in_channels = input_channel
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.kernel_size = kernel_size
        self.strides = strides
        self.reg_max = reg_max
        self.activation = activation
        self.ConvModule = ConvModule if conv_type == "Conv" else DepthwiseConvModule

        self.loss_cfg = loss
        self.norm_cfg = norm_cfg

        # Note: assigner and distribution_project will be implemented later
        # self.assigner = DynamicSoftLabelAssigner(**assigner_cfg)
        self.distribution_project = Integral(self.reg_max)

        self.loss_qfl = QualityFocalLoss(
            beta=self.loss_cfg['loss_qfl']['beta'],
            loss_weight=self.loss_cfg['loss_qfl']['loss_weight'],
        )
        self.loss_dfl = DistributionFocalLoss(
            loss_weight=self.loss_cfg['loss_dfl']['loss_weight']
        )
        self.loss_bbox = GIoULoss(loss_weight=self.loss_cfg['loss_bbox']['loss_weight'])
        self._init_layers()
        self.init_weights()

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        for _ in self.strides:
            cls_convs = self._buid_not_shared_head()
            self.cls_convs.append(cls_convs)

        self.gfl_cls = nn.ModuleList(
            [
                nn.Conv2d(
                    self.feat_channels,
                    self.num_classes + 4 * (self.reg_max + 1),
                    1,
                    padding=0,
                )
                for _ in self.strides
            ]
        )

    def _buid_not_shared_head(self):
        cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            cls_convs.append(
                self.ConvModule(
                    chn,
                    self.feat_channels,
                    self.kernel_size,
                    stride=1,
                    padding=self.kernel_size // 2,
                    norm_cfg=self.norm_cfg,
                    activation=self.activation,
                )
            )
        return cls_convs

    def init_weights(self):
        for m in self.cls_convs.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        # init cls head with confidence = 0.01
        bias_cls = -4.595
        for i in range(len(self.strides)):
            normal_init(self.gfl_cls[i], std=0.01, bias=bias_cls)
        print("Finish initialize NanoDet-Plus Head.")

    def execute(self, feats):
        """Forward function.
        
        Args:
            feats (list[Tensor]): Multi-level feature maps.
            
        Returns:
            Tensor: Concatenated outputs from all levels.
        """
        outputs = []
        for feat, cls_convs, gfl_cls in zip(
            feats,
            self.cls_convs,
            self.gfl_cls,
        ):
            for conv in cls_convs:
                feat = conv(feat)
            output = gfl_cls(feat)
            outputs.append(output.flatten(start_dim=2))
        outputs = jt.concat(outputs, dim=2).permute(0, 2, 1)
        return outputs

    def get_single_level_center_priors(self, batch_size, featmap_size, stride, dtype, device):
        """Generate centers of a single stage feature map.
        
        Args:
            batch_size (int): Number of images in batch.
            featmap_size (tuple): Height and width of feature map.
            stride (int): Down sample stride of feature map.
            dtype: Data type of the tensors.
            device: Device of the tensors.
            
        Returns:
            Tensor: Center priors of a single level feature map.
        """
        h, w = featmap_size
        x_range = jt.arange(w, dtype=dtype) * stride
        y_range = jt.arange(h, dtype=dtype) * stride
        
        y, x = jt.meshgrid(y_range, x_range)
        y = y.flatten()
        x = x.flatten()
        
        strides = jt.full_like(x, stride, dtype=dtype)
        
        # Priors are [x_center, y_center, stride]
        priors = jt.stack([x + stride // 2, y + stride // 2, strides], dim=-1)
        
        return priors.view(1, -1, 3).repeat(batch_size, 1, 1)

    def loss(self, preds, gt_meta, aux_preds=None):
        """Compute losses.
        Args:
            preds (Tensor): Prediction output.
            gt_meta (dict): Ground truth information.
            aux_preds (tuple[Tensor], optional): Auxiliary head prediction output.

        Returns:
            loss (Tensor): Loss tensor.
            loss_states (dict): State dict of each loss.
        """
        # Simplified loss computation for training framework validation
        batch_size = preds.shape[0]

        # Extract predictions
        cls_preds = preds[:, :, :self.num_classes]  # Classification predictions
        reg_preds = preds[:, :, self.num_classes:]  # Regression predictions

        # Simplified loss computation for framework validation
        # Use MSE loss as placeholder to avoid complex target preparation

        # Classification loss (simplified)
        cls_targets = jt.zeros_like(cls_preds)
        loss_qfl = jt.nn.mse_loss(cls_preds, cls_targets) * 0.1

        # Regression loss (simplified)
        reg_targets = jt.zeros_like(reg_preds)
        loss_dfl = jt.nn.mse_loss(reg_preds, reg_targets) * 0.1

        # Bbox loss (simplified)
        bbox_preds = cls_preds[:, :, :4]  # Use first 4 channels as bbox
        bbox_targets = jt.zeros_like(bbox_preds)
        loss_bbox = jt.nn.mse_loss(bbox_preds, bbox_targets) * 0.1

        # Auxiliary loss
        aux_loss = jt.zeros(1)
        if aux_preds is not None:
            aux_cls = aux_preds[:, :, :self.num_classes]
            aux_targets = jt.zeros_like(aux_cls)
            aux_loss = jt.nn.mse_loss(aux_cls, aux_targets) * 0.05

        # Total loss
        loss = loss_qfl + loss_dfl + loss_bbox + aux_loss

        loss_states = {
            "loss_qfl": loss_qfl.item(),
            "loss_dfl": loss_dfl.item(),
            "loss_bbox": loss_bbox.item(),
            "aux_loss": aux_loss.item(),
        }

        return loss, loss_states
