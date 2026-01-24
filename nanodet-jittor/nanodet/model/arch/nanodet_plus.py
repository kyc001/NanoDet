import os
import time
import jittor as jt
from jittor import init
from jittor import nn
from ..head import build_head
from ..fpn import build_fpn
from .one_stage_detector import OneStageDetector

class NanoDetPlus(OneStageDetector):

    def __init__(self, backbone, fpn, head, aux_head=None, detach_epoch=0):
        super(NanoDetPlus, self).__init__(backbone_cfg=backbone, fpn_cfg=fpn, head_cfg=head)
        # 仅当提供了 aux_head 配置时，才构建辅助分支，避免额外显存占用
        if aux_head is not None:
            # 避免 Jittor deepcopy 的巨额开销：重新构建并拷贝权重
            self.aux_fpn = build_fpn(fpn)
            try:
                self.aux_fpn.load_state_dict(self.fpn.state_dict())
            except Exception:
                pass
            self.aux_head = build_head(aux_head)
        else:
            self.aux_fpn = None
            self.aux_head = None
        self.detach_epoch = detach_epoch
        self._timing_enabled = os.getenv("NANODET_TIMING", "") != ""
        self._timing_sync = os.getenv("NANODET_TIMING_SYNC", "1").lower() in ("1", "true", "yes")
        self.last_timing = None

    def _timing_mark(self):
        if self._timing_enabled and self._timing_sync:
            jt.sync_all()

    def forward_train(self, gt_meta):
        img = gt_meta['img']
        if not self._timing_enabled:
            feat = self.backbone(img)
            fpn_feat = self.fpn(feat)
            # 支持全部 FPN 层级输出（自适应 strides 个数）
            # 不再截断到前三层，保持与 head.strides 一致
            aux_head_out = None
            if self.aux_head is not None and self.aux_fpn is not None:
                if (self.epoch >= self.detach_epoch):
                    aux_fpn_feat = self.aux_fpn([f.detach() for f in feat])
                    # 不再截断到前三层，保持与 head.strides 一致
                    # 使用生成器，避免一次性物化所有 concat 特征，降低峰值显存
                    dual_fpn_feat = (
                        jt.contrib.concat([f.detach(), aux_f], dim=1)
                        for (f, aux_f) in zip(fpn_feat, aux_fpn_feat)
                    )
                else:
                    aux_fpn_feat = self.aux_fpn(feat)
                    # 不再截断到前三层，保持与 head.strides 一致
                    # 使用生成器，避免一次性物化所有 concat 特征，降低峰值显存
                    dual_fpn_feat = (
                        jt.contrib.concat([f, aux_f], dim=1)
                        for (f, aux_f) in zip(fpn_feat, aux_fpn_feat)
                    )
                aux_head_out = self.aux_head(dual_fpn_feat)
            head_out = self.head(fpn_feat)
            (loss, loss_states) = self.head.loss(head_out, gt_meta, aux_preds=aux_head_out)
            return (head_out, loss, loss_states)

        # 计时模式
        timing = {}
        self._timing_mark()
        t0 = time.time()
        feat = self.backbone(img)
        self._timing_mark()
        timing["backbone"] = time.time() - t0

        t1 = time.time()
        fpn_feat = self.fpn(feat)
        self._timing_mark()
        timing["fpn"] = time.time() - t1

        # 支持全部 FPN 层级输出（自适应 strides 个数）
        # 不再截断到前三层，保持与 head.strides 一致
        aux_head_out = None
        if self.aux_head is not None and self.aux_fpn is not None:
            if (self.epoch >= self.detach_epoch):
                t2 = time.time()
                aux_fpn_feat = self.aux_fpn([f.detach() for f in feat])
                self._timing_mark()
                timing["aux_fpn"] = time.time() - t2
                # 不再截断到前三层，保持与 head.strides 一致
                # 使用生成器，避免一次性物化所有 concat 特征，降低峰值显存
                dual_fpn_feat = (
                    jt.contrib.concat([f.detach(), aux_f], dim=1)
                    for (f, aux_f) in zip(fpn_feat, aux_fpn_feat)
                )
            else:
                t2 = time.time()
                aux_fpn_feat = self.aux_fpn(feat)
                self._timing_mark()
                timing["aux_fpn"] = time.time() - t2
                # 不再截断到前三层，保持与 head.strides 一致
                # 使用生成器，避免一次性物化所有 concat 特征，降低峰值显存
                dual_fpn_feat = (
                    jt.contrib.concat([f, aux_f], dim=1)
                    for (f, aux_f) in zip(fpn_feat, aux_fpn_feat)
                )
            t3 = time.time()
            aux_head_out = self.aux_head(dual_fpn_feat)
            self._timing_mark()
            timing["aux_head"] = time.time() - t3
        t4 = time.time()
        head_out = self.head(fpn_feat)
        self._timing_mark()
        timing["head"] = time.time() - t4
        t5 = time.time()
        (loss, loss_states) = self.head.loss(head_out, gt_meta, aux_preds=aux_head_out)
        self._timing_mark()
        timing["loss"] = time.time() - t5
        timing["total"] = sum(timing.values())
        self.last_timing = timing
        return (head_out, loss, loss_states)
