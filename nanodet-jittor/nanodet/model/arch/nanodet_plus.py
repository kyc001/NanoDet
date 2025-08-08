import jittor as jt
from jittor import init
from jittor import nn
import copy
from ..head import build_head
from .one_stage_detector import OneStageDetector

class NanoDetPlus(OneStageDetector):

    def __init__(self, backbone, fpn, aux_head, head, detach_epoch=0):
        super(NanoDetPlus, self).__init__(backbone_cfg=backbone, fpn_cfg=fpn, head_cfg=head)
        self.aux_fpn = copy.deepcopy(self.fpn)
        self.aux_head = build_head(aux_head)
        self.detach_epoch = detach_epoch

    def forward_train(self, gt_meta):
        img = gt_meta['img']
        feat = self.backbone(img)
        fpn_feat = self.fpn(feat)
        if (self.epoch >= self.detach_epoch):
            aux_fpn_feat = self.aux_fpn([f.detach() for f in feat])
            dual_fpn_feat = [jt.contrib.concat([f.detach(), aux_f], dim=1) for (f, aux_f) in zip(fpn_feat, aux_fpn_feat)]
        else:
            aux_fpn_feat = self.aux_fpn(feat)
            # 使用列表而非生成器，避免后续多次迭代导致的意外行为
            dual_fpn_feat = [jt.contrib.concat([f, aux_f], dim=1) for (f, aux_f) in zip(fpn_feat, aux_fpn_feat)]
        head_out = self.head(fpn_feat)
        aux_head_out = self.aux_head(dual_fpn_feat)
        (loss, loss_states) = self.head.loss(head_out, gt_meta, aux_preds=aux_head_out)
        return (head_out, loss, loss_states)