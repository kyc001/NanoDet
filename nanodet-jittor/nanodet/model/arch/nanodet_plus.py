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

import copy
import jittor as jt
from jittor import nn

from ..backbone import build_backbone
from ..fpn import build_fpn
from ..head import build_head


class NanoDetPlus(nn.Module):
    """NanoDetPlus model.
    
    Args:
        backbone (dict): Backbone configuration.
        fpn (dict): FPN configuration.
        aux_head (dict): Auxiliary head configuration.
        head (dict): Detection head configuration.
        detach_epoch (int): Epoch to start detaching features. Default: 0.
    """
    
    def __init__(
        self,
        backbone,
        fpn,
        aux_head,
        head,
        detach_epoch=0,
    ):
        super(NanoDetPlus, self).__init__()
        
        # Build backbone
        self.backbone = build_backbone(backbone)
        
        # Build FPN
        self.fpn = build_fpn(fpn)
        
        # Build auxiliary FPN (copy of main FPN)
        self.aux_fpn = build_fpn(copy.deepcopy(fpn))
        
        # Build detection head
        self.head = build_head(head)
        
        # Build auxiliary head
        # Modify aux_head input_channel to account for concatenated features
        aux_head_cfg = copy.deepcopy(aux_head)
        aux_head_cfg['input_channel'] = fpn['out_channels'] * 2  # Concatenated features
        self.aux_head = build_head(aux_head_cfg)
        
        self.detach_epoch = detach_epoch
        self.epoch = 0  # Will be set during training

    def execute(self, x):
        """Forward function for inference.
        
        Args:
            x (Tensor): Input image tensor.
            
        Returns:
            Tensor: Detection results.
        """
        feat = self.backbone(x)
        fpn_feat = self.fpn(feat)
        head_out = self.head(fpn_feat)
        return head_out

    def forward_train(self, gt_meta):
        """Forward function for training.
        
        Args:
            gt_meta (dict): Ground truth meta information.
            
        Returns:
            tuple: (head_out, loss, loss_states)
        """
        img = gt_meta["img"]
        feat = self.backbone(img)
        fpn_feat = self.fpn(feat)
        
        # Auxiliary FPN processing
        if self.epoch >= self.detach_epoch:
            # Detach features for auxiliary FPN
            aux_fpn_feat = self.aux_fpn([f.detach() for f in feat])
            dual_fpn_feat = [
                jt.concat([f.detach(), aux_f], dim=1)
                for f, aux_f in zip(fpn_feat, aux_fpn_feat)
            ]
        else:
            aux_fpn_feat = self.aux_fpn(feat)
            dual_fpn_feat = [
                jt.concat([f, aux_f], dim=1) 
                for f, aux_f in zip(fpn_feat, aux_fpn_feat)
            ]
        
        # Head predictions
        head_out = self.head(fpn_feat)
        aux_head_out = self.aux_head(dual_fpn_feat)
        
        # Compute loss
        loss, loss_states = self.head.loss(head_out, gt_meta, aux_preds=aux_head_out)
        
        return head_out, loss, loss_states

    def set_epoch(self, epoch):
        """Set current epoch for training.
        
        Args:
            epoch (int): Current epoch.
        """
        self.epoch = epoch
