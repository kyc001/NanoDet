import jittor as jt
from jittor import init
from jittor import nn
from .fpn import FPN

class PAN(FPN):

    def __init__(self, in_channels, out_channels, num_outs, start_level=0, end_level=(- 1), conv_cfg=None, norm_cfg=None, activation=None):
        super(PAN, self).__init__(in_channels, out_channels, num_outs, start_level, end_level, conv_cfg, norm_cfg, activation)
        self.init_weights()

    def execute(self, inputs):
        assert (len(inputs) == len(self.in_channels))
        laterals = [lateral_conv(inputs[(i + self.start_level)]) for (i, lateral_conv) in enumerate(self.lateral_convs)]
        used_backbone_levels = len(laterals)
        for i in range((used_backbone_levels - 1), 0, (- 1)):
            laterals[(i - 1)] += jt.interpolate(laterals[i], scale_factor=2, mode='bilinear')
        inter_outs = [laterals[i] for i in range(used_backbone_levels)]
        for i in range(0, (used_backbone_levels - 1)):
            inter_outs[(i + 1)] += jt.interpolate(inter_outs[i], scale_factor=0.5, mode='bilinear')
        outs = []
        outs.append(inter_outs[0])
        outs.extend([inter_outs[i] for i in range(1, used_backbone_levels)])
        return tuple(outs)
