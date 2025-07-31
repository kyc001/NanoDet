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

import jittor as jt
from jittor import nn


class ConvModule(nn.Module):
    """A conv block that bundles conv/norm/activation layers.
    
    Args:
        in_channels (int): Number of channels in the input feature map.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or tuple, optional): Stride of the convolution. Default: 1.
        padding (int or tuple, optional): Zero-padding added to both sides of
            the input. Default: 0.
        dilation (int or tuple, optional): Spacing between kernel elements.
            Default: 1.
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1.
        bias (bool or str, optional): If specified as `auto`, it will be
            decided by the norm_cfg. Bias will be set as True if `norm_cfg`
            is None, otherwise False. Default: "auto".
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None.
        activation (str or None, optional): Activation layer type.
            Default: 'ReLU'.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias="auto",
        conv_cfg=None,
        norm_cfg=None,
        activation="ReLU",
    ):
        super(ConvModule, self).__init__()
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.activation = activation
        self.with_norm = norm_cfg is not None
        self.with_activation = activation is not None
        
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == "auto":
            bias = not self.with_norm
        self.with_bias = bias

        # build convolution layer
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        
        # build normalization layers - match PyTorch naming
        if self.with_norm:
            norm_type = norm_cfg.get("type", "BN")
            if norm_type == "BN":
                self.bn = nn.BatchNorm2d(out_channels)  # Use 'bn' to match PyTorch
            elif norm_type == "GN":
                num_groups = norm_cfg.get("num_groups", 32)
                self.gn = nn.GroupNorm(num_groups, out_channels)  # Use 'gn' to match PyTorch
            else:
                raise NotImplementedError(f"Normalization {norm_type} not implemented")
        
        # build activation layer
        if self.with_activation:
            if activation == "ReLU":
                self.activate = nn.ReLU()
            elif activation == "LeakyReLU":
                self.activate = nn.LeakyReLU(0.1)
            elif activation == "Swish":
                self.activate = nn.Swish()
            else:
                raise NotImplementedError(f"Activation {activation} not implemented")

    def execute(self, x):
        x = self.conv(x)
        if self.with_norm:
            norm_type = self.norm_cfg.get("type", "BN")
            if norm_type == "BN":
                x = self.bn(x)
            elif norm_type == "GN":
                x = self.gn(x)
        if self.with_activation:
            x = self.activate(x)
        return x


class DepthwiseConvModule(nn.Module):
    """Depthwise separable convolution module.

    Args:
        in_channels (int): Number of channels in the input feature map.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or tuple, optional): Stride of the convolution. Default: 1.
        padding (int or tuple, optional): Zero-padding added to both sides of
            the input. Default: 0.
        dilation (int or tuple, optional): Spacing between kernel elements.
            Default: 1.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None.
        activation (str or None, optional): Activation layer type.
            Default: 'ReLU'.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        norm_cfg=None,
        activation="ReLU",
    ):
        super(DepthwiseConvModule, self).__init__()

        # Depthwise convolution - match PyTorch naming exactly
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False
        )

        # Pointwise convolution - match PyTorch naming exactly
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            stride=1,
            padding=0,
            bias=False  # Match PyTorch: no bias when using norm
        )

        # Normalization layers - match PyTorch naming exactly
        if norm_cfg is not None and norm_cfg.get('type') == 'BN':
            self.dwnorm = nn.BatchNorm2d(in_channels)
            self.pwnorm = nn.BatchNorm2d(out_channels)
        else:
            self.dwnorm = None
            self.pwnorm = None

        # Activation
        if activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU(0.1)
        elif activation == 'ReLU':
            self.activation = nn.ReLU()
        else:
            self.activation = None

    def execute(self, x):
        # Depthwise
        x = self.depthwise(x)
        if self.dwnorm is not None:
            x = self.dwnorm(x)
        if self.activation is not None:
            x = self.activation(x)

        # Pointwise
        x = self.pointwise(x)
        if self.pwnorm is not None:
            x = self.pwnorm(x)
        if self.activation is not None:
            x = self.activation(x)

        return x
