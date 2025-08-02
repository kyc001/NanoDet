import jittor as jt
from jittor import init
import time
from jittor import nn
from ..backbone import build_backbone
from ..fpn import build_fpn
from ..head import build_head

class OneStageDetector(nn.Module):

    def __init__(self, backbone_cfg, fpn_cfg=None, head_cfg=None):
        super(OneStageDetector, self).__init__()
        self.backbone = build_backbone(backbone_cfg)
        if (fpn_cfg is not None):
            self.fpn = build_fpn(fpn_cfg)
        if (head_cfg is not None):
            self.head = build_head(head_cfg)
        self.epoch = 0

    def execute(self, x):
        x = self.backbone(x)
        if hasattr(self, 'fpn'):
            x = self.fpn(x)
        if hasattr(self, 'head'):
            x = self.head(x)
        return x

    def inference(self, meta):
        with jt.no_grad():
            jt.sync_all(True)	
            time1 = time.time()
            preds = self(meta['img'])
            jt.sync_all(True)	
            time2 = time.time()
            print('forward time: {:.3f}s'.format((time2 - time1)), end=' | ')
            results = self.head.post_process(preds, meta)
            jt.sync_all(True)	
            print('decode time: {:.3f}s'.format((time.time() - time2)), end=' | ')
        return results

    def forward_train(self, gt_meta):
        preds = self(gt_meta['img'])
        (loss, loss_states) = self.head.loss(preds, gt_meta)
        return (preds, loss, loss_states)

    def set_epoch(self, epoch):
        self.epoch = epoch