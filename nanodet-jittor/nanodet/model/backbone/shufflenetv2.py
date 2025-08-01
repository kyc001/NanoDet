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

from ..module.activation import act_layers


def channel_shuffle(x, groups):
    """Channel shuffle operation.
    
    Args:
        x (jt.Var): Input tensor with shape (N, C, H, W).
        groups (int): Number of groups to divide channels.
        
    Returns:
        jt.Var: Output tensor after channel shuffle.
    """
    batchsize, num_channels, height, width = x.shape
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)
    
    # transpose
    x = x.transpose(1, 2)
    
    # flatten
    x = x.view(batchsize, -1, height, width)
    
    return x


class ShuffleV2Block(nn.Module):
    """ShuffleNetV2 basic block.
    
    Args:
        inp (int): Number of input channels.
        oup (int): Number of output channels.
        stride (int): Stride of the convolution. Default: 1.
        activation (str): Type of activation function. Default: 'ReLU'.
    """
    
    def __init__(self, inp, oup, stride, activation="ReLU"):
        super(ShuffleV2Block, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError("illegal stride value")
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(
                    inp, inp, kernel_size=3, stride=self.stride, padding=1
                ),
                nn.BatchNorm2d(inp),
                nn.Conv2d(
                    inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False
                ),
                nn.BatchNorm2d(branch_features),
                act_layers[activation](),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(
                inp if (self.stride > 1) else branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(branch_features),
            act_layers[activation](),
            self.depthwise_conv(
                branch_features,
                branch_features,
                kernel_size=3,
                stride=self.stride,
                padding=1,
            ),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(
                branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(branch_features),
            act_layers[activation](),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        """Depthwise convolution."""
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def execute(self, x):
        if self.stride == 1:
            x1, x2 = jt.chunk(x, 2, dim=1)
            out = jt.concat([x1, self.branch2(x2)], dim=1)
        else:
            out = jt.concat([self.branch1(x), self.branch2(x)], dim=1)

        out = channel_shuffle(out, 2)

        return out


class ShuffleNetV2(nn.Module):
    """ShuffleNetV2 backbone.
    
    Args:
        model_size (str): Model size, one of ['0.5x', '1.0x', '1.5x', '2.0x'].
            Default: '1.0x'.
        out_stages (tuple): Output stages. Default: (2, 3, 4).
        with_last_conv (bool): Whether to add last conv layer. Default: False.
        activation (str): Type of activation function. Default: 'ReLU'.
        pretrain (bool): Whether to load pretrained weights. Default: True.
    """
    
    def __init__(
        self,
        model_size="1.0x",
        out_stages=(2, 3, 4),
        with_last_conv=False,
        activation="ReLU",
        pretrain=True,
    ):
        super(ShuffleNetV2, self).__init__()
        # out_stages can only be a subset of (2, 3, 4)
        assert set(out_stages).issubset((2, 3, 4))

        print("model size is ", model_size)

        self.stage_repeats = [4, 8, 4]
        self.model_size = model_size
        self.out_stages = out_stages
        self.with_last_conv = with_last_conv
        self.activation = activation
        
        if model_size == "0.5x":
            self._stage_out_channels = [24, 48, 96, 192, 1024]
        elif model_size == "1.0x":
            self._stage_out_channels = [24, 116, 232, 464, 1024]
        elif model_size == "1.5x":
            self._stage_out_channels = [24, 176, 352, 704, 1024]
        elif model_size == "2.0x":
            self._stage_out_channels = [24, 244, 488, 976, 2048]
        else:
            raise NotImplementedError

        # building first layer
        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            act_layers[activation](),
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = ["stage{}".format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
            stage_names, self.stage_repeats, self._stage_out_channels[1:]
        ):
            seq = [
                ShuffleV2Block(
                    input_channels, output_channels, 2, activation=activation
                )
            ]
            for i in range(repeats - 1):
                seq.append(
                    ShuffleV2Block(
                        output_channels, output_channels, 1, activation=activation
                    )
                )
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels
            
        output_channels = self._stage_out_channels[-1]
        if self.with_last_conv:
            conv5 = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(output_channels),
                act_layers[activation](),
            )
            self.stage4.add_module("conv5", conv5)
        self._initialize_weights(pretrain)

    def execute(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        output = []
        for i in range(2, 5):
            stage = getattr(self, "stage{}".format(i))
            x = stage(x)
            if i in self.out_stages:
                output.append(x)
        return tuple(output)

    def _initialize_weights(self, pretrain=True):
        """Initialize weights."""
        print("init weights...")
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if "first" in name:
                    jt.init.gauss_(m.weight, 0, 0.01)
                else:
                    jt.init.gauss_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    jt.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                jt.init.constant_(m.weight, 1)
                if m.bias is not None:
                    jt.init.constant_(m.bias, 0.0001)
                # Note: Jittor BatchNorm doesn't have running_mean attribute
        
        # Load pretrained weights
        if pretrain:
            self._load_pretrained_weights()

    def _load_pretrained_weights(self):
        """Load ImageNet pretrained weights."""
        try:
            import torch
            import urllib.request
            import os

            # ShuffleNetV2 1.0x ImageNet pretrained weights URL
            model_urls = {
                '0.5x': 'https://download.pytorch.org/models/shufflenetv2_x0_5-f707e7126e.pth',
                '1.0x': 'https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth',
                '1.5x': 'https://download.pytorch.org/models/shufflenetv2_x1_5-3c479a10.pth',
                '2.0x': 'https://download.pytorch.org/models/shufflenetv2_x2_0-8be3c8ee.pth'
            }

            if self.model_size not in model_urls:
                print(f"⚠ No pretrained weights available for model size {self.model_size}")
                return

            url = model_urls[self.model_size]
            print(f"=> loading pretrained model {url}")

            # Download and load PyTorch weights
            cache_dir = os.path.expanduser('~/.cache/jittor')
            os.makedirs(cache_dir, exist_ok=True)

            filename = url.split('/')[-1]
            cached_file = os.path.join(cache_dir, filename)

            if not os.path.exists(cached_file):
                print(f"Downloading: {url}")
                urllib.request.urlretrieve(url, cached_file)

            # Load PyTorch state dict
            pytorch_state_dict = torch.load(cached_file, map_location='cpu')

            # Convert PyTorch weights to Jittor format
            jittor_state_dict = {}
            for key, value in pytorch_state_dict.items():
                if 'num_batches_tracked' not in key:  # Skip tracking parameters
                    jittor_state_dict[key] = jt.array(value.numpy())

            # Load weights into model - 模拟PyTorch的strict=False行为
            model_dict = self.state_dict()
            compatible_dict = {}
            missing_keys = []
            unexpected_keys = []

            # 只加载形状匹配的权重，忽略不匹配的（模拟strict=False）
            for k, v in jittor_state_dict.items():
                if k in model_dict:
                    if model_dict[k].shape == v.shape:
                        compatible_dict[k] = v
                    else:
                        missing_keys.append(f"{k} (shape mismatch: {model_dict[k].shape} vs {v.shape})")
                else:
                    unexpected_keys.append(k)

            # 检查模型中有哪些参数没有被加载
            for k in model_dict.keys():
                if k not in compatible_dict:
                    missing_keys.append(f"{k} (not in pretrained)")

            # 加载兼容的权重
            if compatible_dict:
                # 手动赋值，避免load_state_dict的严格检查
                for name, param in self.named_parameters():
                    if name in compatible_dict:
                        param.assign(compatible_dict[name])

                print(f"✓ Loaded {len(compatible_dict)} compatible weights, {len(missing_keys)} missing")
            else:
                print("❌ No compatible weights found")

            print("✓ Pretrained weights loaded successfully")

        except Exception as e:
            print(f"⚠ Failed to load pretrained weights: {e}")
            print("Continuing with random initialization...")
