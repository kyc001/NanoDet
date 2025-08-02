import jittor as jt
from jittor import init
from jittor import nn
from ..module.conv import ConvModule
from ..module.init_weights import normal_init
from ..module.transformer import TransformerBlock

class TAN(nn.Module):

    def __init__(self, in_channels, out_channels, feature_hw, num_heads, num_encoders, mlp_ratio, dropout_ratio, activation='LeakyReLU'):
        super(TAN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        assert (self.num_ins == 3)
        self.lateral_convs = nn.ModuleList()
        for i in range(self.num_ins):
            l_conv = ConvModule(in_channels[i], out_channels, 1, norm_cfg=dict(type='BN'), activation=activation, inplace=False)
            self.lateral_convs.append(l_conv)
        self.transformer = TransformerBlock((out_channels * self.num_ins), out_channels, num_heads, num_encoders, mlp_ratio, dropout_ratio, activation=activation)
        self.pos_embed = jt.array(jt.zeros((feature_hw[0] * feature_hw[1]), 1, out_channels))
        self.init_weights()

    def init_weights(self):
        jt.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                jt.nn.init.trunc_normal_(m.weight, std=0.02)
                if (isinstance(m, nn.Linear) and (m.bias is not None)):
                    init.constant_(m.bias, value=0)
            elif isinstance(m, nn.LayerNorm):
                init.constant_(m.bias, value=0)
                init.constant_(m.weight, value=1.0)
            elif isinstance(m, nn.Conv):
                normal_init(m, 0.01)

    def execute(self, inputs):
        assert (len(inputs) == len(self.in_channels))
        laterals = [lateral_conv(inputs[i]) for (i, lateral_conv) in enumerate(self.lateral_convs)]
        mid_shape = laterals[1].shape[2:]
        mid_lvl = jt.contrib.concat((jt.nn.interpolate(laterals[0], size=mid_shape, mode='bilinear'), laterals[1], jt.nn.interpolate(laterals[2], size=mid_shape, mode='bilinear')), dim=1)
        mid_lvl = self.transformer(mid_lvl, self.pos_embed)
        outs = [(laterals[0] + jt.nn.interpolate(mid_lvl, size=laterals[0].shape[2:], mode='bilinear')), (laterals[1] + mid_lvl), (laterals[2] + jt.nn.interpolate(mid_lvl, size=laterals[2].shape[2:], mode='bilinear'))]
        return tuple(outs)