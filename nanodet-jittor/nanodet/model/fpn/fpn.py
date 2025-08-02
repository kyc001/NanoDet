import jittor as jt
from jittor import init
from jittor import nn
from ..module.conv import ConvModule
from ..module.init_weights import xavier_init

class FPN(nn.Module):

    def __init__(self, in_channels, out_channels, num_outs, start_level=0, end_level=(- 1), conv_cfg=None, norm_cfg=None, activation=None):
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.fp16_enabled = False
        if (end_level == (- 1)):
            self.backbone_end_level = self.num_ins
            assert (num_outs >= (self.num_ins - start_level))
        else:
            self.backbone_end_level = end_level
            assert (end_level <= len(in_channels))
            assert (num_outs == (end_level - start_level))
        self.start_level = start_level
        self.end_level = end_level
        self.lateral_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(in_channels[i], out_channels, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, activation=activation, inplace=False)
            self.lateral_convs.append(l_conv)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv):
                xavier_init(m, distribution='uniform')

    def execute(self, inputs):
        assert (len(inputs) == len(self.in_channels))
        laterals = [lateral_conv(inputs[(i + self.start_level)]) for (i, lateral_conv) in enumerate(self.lateral_convs)]
        used_backbone_levels = len(laterals)
        for i in range((used_backbone_levels - 1), 0, (- 1)):
            laterals[(i - 1)] += jt.nn.interpolate(laterals[i], scale_factor=2, mode='bilinear')
        outs = [laterals[i] for i in range(used_backbone_levels)]
        return tuple(outs)