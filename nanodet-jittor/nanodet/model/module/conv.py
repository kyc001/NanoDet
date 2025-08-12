import warnings

import numpy as np
import jittor as jt
from jittor import nn

from .activation import act_layers
from .init_weights import constant_init, kaiming_init
from .norm import build_norm_layer


class ConvModule(nn.Module):
    """ConvModule (Jittor Version)."""
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, bias="auto",
        conv_cfg=None, norm_cfg=None, activation="ReLU",
        inplace=True, order=("conv", "norm", "act"),
    ):
        super(ConvModule, self).__init__()
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        assert activation is None or isinstance(activation, str)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.activation = activation
        self.inplace = inplace
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == {"conv", "norm", "act"}

        self.with_norm = norm_cfg is not None
        if bias == "auto":
            bias = not self.with_norm
        self.with_bias = bias

        if self.with_norm and self.with_bias:
            warnings.warn("ConvModule has norm and bias at the same time")

        # build convolution layer
        self.conv = nn.Conv(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias)
        
        # build normalization layers
        if self.with_norm:
            if order.index("norm") > order.index("conv"):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            self.norm_name, norm = build_norm_layer(norm_cfg, norm_channels)
            setattr(self, self.norm_name, norm)
        else:
            self.norm_name = None

        # build activation layer
        if self.activation:
            self.act = act_layers(self.activation)

        self.init_weights()

    @property
    def norm(self):
        return getattr(self, self.norm_name) if self.norm_name else None

    def init_weights(self):
        nonlinearity = "leaky_relu" if self.activation == "LeakyReLU" else "relu"
        kaiming_init(self.conv, nonlinearity=nonlinearity)
        if self.with_norm:
            constant_init(self.norm, 1, bias=0)

    def execute(self, x, norm=True):
        for layer in self.order:
            if layer == "conv":
                x = self.conv(x)
            elif layer == "norm" and norm and self.with_norm:
                x = self.norm(x)
            elif layer == "act" and self.activation:
                x = self.act(x)
        return x

class DepthwiseConvModule(nn.Module):
    """DepthwiseConvModule (Jittor Version) - 使用 jittordet ConvModule 实现."""
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, bias="auto", norm_cfg=dict(type="BN"),
        activation="ReLU", inplace=True,
        order=("depthwise", "dwnorm", "act", "pointwise", "pwnorm", "act"),
    ):
        super(DepthwiseConvModule, self).__init__()
        assert activation is None or isinstance(activation, str)
        self.activation = activation
        self.inplace = inplace
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 6
        assert set(order) == {"depthwise", "dwnorm", "act", "pointwise", "pwnorm", "act"}

        self.with_norm = norm_cfg is not None
        if bias == "auto":
            bias = not self.with_norm
        self.with_bias = bias

        if self.with_norm and self.with_bias:
            warnings.warn("ConvModule has norm and bias at the same time")

        # 🔧 正确的 depthwise 卷积实现：groups=in_channels
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=in_channels,  # 🔧 关键：groups=in_channels 实现真正的 depthwise
            bias=False  # depthwise 不使用 bias
        )

        # Pointwise 卷积：1x1 卷积调整通道数
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1,
            stride=1, padding=0, bias=bias
        )

        self._use_custom_depthwise = True

        # 批归一化层 - 精确复现 PyTorch 版本
        if self.with_norm:
            _, self.dwnorm = build_norm_layer(norm_cfg, in_channels)   # depthwise 后的 norm
            _, self.pwnorm = build_norm_layer(norm_cfg, out_channels)  # pointwise 后的 norm

        # 激活函数 - 精确复现 PyTorch 版本
        if self.activation:
            self.act = act_layers(self.activation)

        # 导出属性 - 精确复现 PyTorch 版本
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.init_weights()

    def init_weights(self):
        nonlinearity = "leaky_relu" if self.activation == "LeakyReLU" else "relu"

        # 标准卷积初始化
        kaiming_init(self.depthwise, nonlinearity=nonlinearity)

        # Pointwise 卷积初始化
        kaiming_init(self.pointwise, nonlinearity=nonlinearity)

        # 批归一化层初始化
        if self.with_norm:
            constant_init(self.dwnorm, 1, bias=0)
            constant_init(self.pwnorm, 1, bias=0)

    def execute(self, x, norm=True):
        # 🔧 与PyTorch版本完全一致的实现
        for layer_name in self.order:
            if layer_name != "act":
                # 🔧 对于norm层，需要检查norm参数
                if ("norm" in layer_name and not norm):
                    continue
                layer = getattr(self, layer_name)
                x = layer(x)
            elif layer_name == "act" and self.activation:
                x = self.act(x)
        return x

class RepVGGConvModule(nn.Module):
    """RepVGG Conv Block (Jittor Version)."""
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1,
        padding=1, dilation=1, groups=1, activation="ReLU",
        padding_mode="zeros", deploy=False, **kwargs
    ):
        super(RepVGGConvModule, self).__init__()
        assert activation is None or isinstance(activation, str)
        self.activation = activation
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        assert kernel_size == 3 and padding == 1
        padding_11 = padding - kernel_size // 2

        if self.activation:
            self.act = act_layers(self.activation)

        if deploy:
            self.rbr_reparam = nn.Conv(
                in_channels, out_channels, kernel_size, stride,
                padding, dilation, groups, bias=True, padding_mode=padding_mode)
        else:
            self.rbr_identity = (
                nn.BatchNorm(in_channels)
                if out_channels == in_channels and stride == 1
                else None)
            self.rbr_dense = nn.Sequential(
                nn.Conv(in_channels, out_channels, kernel_size, stride,
                        padding, groups=groups, bias=False),
                nn.BatchNorm(out_channels))
            self.rbr_1x1 = nn.Sequential(
                nn.Conv(in_channels, out_channels, 1, stride,
                        padding_11, groups=groups, bias=False),
                nn.BatchNorm(out_channels))
            print("RepVGG Block, identity = ", self.rbr_identity)

    def execute(self, inputs):
        if hasattr(self, "rbr_reparam"):
            return self.act(self.rbr_reparam(inputs))
        id_out = self.rbr_identity(inputs) if self.rbr_identity is not None else 0
        return self.act(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return (kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
                bias3x3 + bias1x1 + biasid)

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            # Jittor's pad for 4D NCHW format is (left, right, top, bottom) for the last two dims
            return nn.pad(kernel1x1, (1, 1, 1, 1))

    def _fuse_bn_tensor(self, branch):
        if branch is None: return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch[0].weight
            running_mean = branch[1].running_mean
            running_var = branch[1].running_var
            gamma = branch[1].weight
            beta = branch[1].bias
            eps = branch[1].eps
        else:
            assert isinstance(branch, nn.BatchNorm)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = jt.array(kernel_value)
            kernel = self.id_tensor
            running_mean, running_var, gamma, beta, eps = \
                branch.running_mean, branch.running_var, branch.weight, branch.bias, branch.eps
        
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def repvgg_convert(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        return (kernel.numpy(), bias.numpy())
