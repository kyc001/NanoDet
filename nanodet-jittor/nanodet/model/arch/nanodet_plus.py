import jittor as jt
from jittor import init
from jittor import nn
import copy
from ..head import build_head
from .one_stage_detector import OneStageDetector

class NanoDetPlus(OneStageDetector):

    def __init__(self, backbone, fpn, head, aux_head=None, detach_epoch=0):
        super(NanoDetPlus, self).__init__(backbone_cfg=backbone, fpn_cfg=fpn, head_cfg=head)
        # 仅当提供了 aux_head 配置时，才构建辅助分支，避免额外显存占用
        if aux_head is not None:
            self.aux_fpn = copy.deepcopy(self.fpn)
            self.aux_head = build_head(aux_head)
        else:
            self.aux_fpn = None
            self.aux_head = None
        self.detach_epoch = detach_epoch

    def forward_train(self, gt_meta):
        img = gt_meta['img']
        feat = self.backbone(img)
        fpn_feat = self.fpn(feat)
        # 支持全部 FPN 层级输出（自适应 strides 个数）
        # 不再截断到前三层，保持与 head.strides 一致
        aux_head_out = None
        if self.aux_head is not None and self.aux_fpn is not None:
            if (self.epoch >= self.detach_epoch):
                aux_fpn_feat = self.aux_fpn([f.detach() for f in feat])
                # 不再截断到前三层，保持与 head.strides 一致
                dual_fpn_feat = [jt.contrib.concat([f.detach(), aux_f], dim=1) for (f, aux_f) in zip(fpn_feat, aux_fpn_feat)]
            else:
                aux_fpn_feat = self.aux_fpn(feat)
                # 不再截断到前三层，保持与 head.strides 一致
                dual_fpn_feat = [jt.contrib.concat([f, aux_f], dim=1) for (f, aux_f) in zip(fpn_feat, aux_fpn_feat)]
            aux_head_out = self.aux_head(dual_fpn_feat)
        head_out = self.head(fpn_feat)
        (loss, loss_states) = self.head.loss(head_out, gt_meta, aux_preds=aux_head_out)
        return (head_out, loss, loss_states)