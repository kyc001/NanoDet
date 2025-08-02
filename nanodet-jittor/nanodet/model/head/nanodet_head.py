import jittor as jt
import jittor.nn as nn

from ..module.conv import ConvModule, DepthwiseConvModule
from ..module.init_weights import normal_init
from .gfl_head import GFLHead


class NanoDetHead(GFLHead):
    """
    Modified from GFL, use same loss functions but much lightweight convolution heads.
    (Jittor Version)
    """
    def __init__(
        self, num_classes, loss, input_channel, stacked_convs=2,
        octave_base_scale=5, conv_type="DWConv", conv_cfg=None,
        norm_cfg=dict(type="BN"), reg_max=16, share_cls_reg=False,
        activation="LeakyReLU", feat_channels=256, strides=[8, 16, 32],
        **kwargs
    ):
        self.share_cls_reg = share_cls_reg
        self.activation = activation
        # 假设 ConvModule 和 DepthwiseConvModule 已被迁移
        self.ConvModule = ConvModule if conv_type == "Conv" else DepthwiseConvModule
        
        # [遷移] super 調用保持不變
        super(NanoDetHead, self).__init__(
            num_classes, loss, input_channel, feat_channels,
            stacked_convs, octave_base_scale, strides, conv_cfg,
            norm_cfg, reg_max, **kwargs
        )

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for _ in self.strides:
            cls_convs, reg_convs = self._buid_not_shared_head()
            self.cls_convs.append(cls_convs)
            self.reg_convs.append(reg_convs)

        self.gfl_cls = nn.ModuleList([
            # [遷移] nn.Conv2d -> nn.Conv
            nn.Conv(
                self.feat_channels,
                self.cls_out_channels + 4 * (self.reg_max + 1)
                if self.share_cls_reg else self.cls_out_channels,
                1, padding=0)
            for _ in self.strides
        ])
        self.gfl_reg = nn.ModuleList([
            # [遷移] nn.Conv2d -> nn.Conv
            nn.Conv(self.feat_channels, 4 * (self.reg_max + 1), 1, padding=0)
            for _ in self.strides
        ])

    def _buid_not_shared_head(self):
        cls_convs = nn.ModuleList()
        reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            cls_convs.append(
                self.ConvModule(
                    chn, self.feat_channels, 3, stride=1, padding=1,
                    norm_cfg=self.norm_cfg, bias=self.norm_cfg is None,
                    activation=self.activation,
                )
            )
            if not self.share_cls_reg:
                reg_convs.append(
                    self.ConvModule(
                        chn, self.feat_channels, 3, stride=1, padding=1,
                        norm_cfg=self.norm_cfg, bias=self.norm_cfg is None,
                        activation=self.activation,
                    )
                )
        return cls_convs, reg_convs

    def init_weights(self):
        # [遷移] .modules() 在 Jittor 中用法相同
        for m in self.cls_convs.modules():
            # [遷移] nn.Conv2d -> nn.Conv
            if isinstance(m, nn.Conv):
                normal_init(m, std=0.01)
        for m in self.reg_convs.modules():
            if isinstance(m, nn.Conv):
                normal_init(m, std=0.01)
        
        bias_cls = -4.595
        for i in range(len(self.strides)):
            normal_init(self.gfl_cls[i], std=0.01, bias=bias_cls)
            normal_init(self.gfl_reg[i], std=0.01)
        print("Finish initialize NanoDet Head.")

    # [遷移] forward -> execute
    def execute(self, feats):
        # [遷移] 移除 ONNX 相關邏輯
        outputs = []
        for x, cls_convs, reg_convs, gfl_cls, gfl_reg in zip(
            feats, self.cls_convs, self.reg_convs, self.gfl_cls, self.gfl_reg
        ):
            cls_feat, reg_feat = x, x
            for cls_conv in cls_convs:
                cls_feat = cls_conv(cls_feat)
            for reg_conv in reg_convs:
                reg_feat = reg_conv(reg_feat)
            
            if self.share_cls_reg:
                output = gfl_cls(cls_feat)
            else:
                cls_score = gfl_cls(cls_feat)
                bbox_pred = gfl_reg(reg_feat)
                # [遷移] torch.cat -> jt.concat
                output = jt.concat([cls_score, bbox_pred], dim=1)
            outputs.append(output.flatten(start_dim=2))
        return jt.concat(outputs, dim=2).permute