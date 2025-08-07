
import jittor as jt
from jittor import init
import logging
import math
import warnings
from jittor import nn
from ..module.activation import act_layers
import urllib.request
import os


def get_url(width_mult=1.0):
    if (width_mult == 1.0):
        return 'https://raw.githubusercontent.com/huawei-noah/CV-Backbones/master/ghostnet_pytorch/models/state_dict_73.98.pth'
    else:
        logging.info('GhostNet only has 1.0 pretrain model. ')
        return None

def _make_divisible(v, divisor, min_value=None):
    if (min_value is None):
        min_value = divisor
    new_v = max(min_value, ((int((v + (divisor / 2))) // divisor) * divisor))
    if (new_v < (0.9 * v)):
        new_v += divisor
    return new_v

def hard_sigmoid(x, inplace: bool=False):
    if inplace:
        return x.add_(3.0).clamp_(0.0, 6.0).div_(6.0)
    else:
        return (jt.nn.relu6((x + 3.0)) / 6.0)

class SqueezeExcite(nn.Module):

    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None, activation='ReLU', gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible(((reduced_base_chs or in_chs) * se_ratio), divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layers(activation)
        self.conv_expand = nn.Conv(reduced_chs, in_chs, 1, bias=True)

    def execute(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = (x * self.gate_fn(x_se))
        return x

class ConvBnAct(nn.Module):

    def __init__(self, in_chs, out_chs, kernel_size, stride=1, activation='ReLU'):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv(in_chs, out_chs, kernel_size, stride=stride, padding=(kernel_size // 2), bias=False)
        self.bn1 = nn.BatchNorm(out_chs)
        self.act1 = act_layers(activation)

    def execute(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x

class GhostModule(nn.Module):

    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, activation='ReLU'):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil((oup / ratio))
        new_channels = (init_channels * (ratio - 1))
        self.primary_conv = nn.Sequential(nn.Conv(inp, init_channels, kernel_size, stride=stride, padding=(kernel_size // 2), bias=False), nn.BatchNorm(init_channels), (act_layers(activation) if activation else nn.Sequential()))
        # 🔧 紧急修复：避开 Jittor depthwise_conv.py bug
        self.cheap_operation = nn.Sequential(nn.Conv(init_channels, new_channels, dw_size, stride=1, padding=(dw_size // 2), bias=False), nn.BatchNorm(new_channels), (act_layers(activation) if activation else nn.Sequential()))

    def execute(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = jt.contrib.concat([x1, x2], dim=1)
        return out

class GhostBottleneck(nn.Module):

    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3, stride=1, activation='ReLU', se_ratio=0.0):
        super(GhostBottleneck, self).__init__()
        has_se = ((se_ratio is not None) and (se_ratio > 0.0))
        self.stride = stride
        self.ghost1 = GhostModule(in_chs, mid_chs, activation=activation)
        if (self.stride > 1):
            # 🔧 紧急修复：避开 Jittor depthwise_conv.py bug
            self.conv_dw = nn.Conv(mid_chs, mid_chs, dw_kernel_size, stride=stride, padding=((dw_kernel_size - 1) // 2), bias=False)
            self.bn_dw = nn.BatchNorm(mid_chs)
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None
        self.ghost2 = GhostModule(mid_chs, out_chs, activation=None)
        if ((in_chs == out_chs) and (self.stride == 1)):
            self.shortcut = nn.Sequential()
        else:
            # 🔧 紧急修复：避开 Jittor depthwise_conv.py bug
            self.shortcut = nn.Sequential(nn.Conv(in_chs, in_chs, dw_kernel_size, stride=stride, padding=((dw_kernel_size - 1) // 2), bias=False), nn.BatchNorm(in_chs), nn.Conv(in_chs, out_chs, 1, stride=1, padding=0, bias=False), nn.BatchNorm(out_chs))

    def execute(self, x):
        residual = x
        x = self.ghost1(x)
        if (self.stride > 1):
            x = self.conv_dw(x)
            x = self.bn_dw(x)
        if (self.se is not None):
            x = self.se(x)
        x = self.ghost2(x)
        x += self.shortcut(residual)
        return x

class GhostNet(nn.Module):

    def __init__(self, width_mult=1.0, out_stages=(4, 6, 9), activation='ReLU', pretrain=True, act=None):
        super(GhostNet, self).__init__()
        assert set(out_stages).issubset((i for i in range(10)))
        self.width_mult = width_mult
        self.out_stages = out_stages
        self.cfgs = [[[3, 16, 16, 0, 1]], [[3, 48, 24, 0, 2]], [[3, 72, 24, 0, 1]], [[5, 72, 40, 0.25, 2]], [[5, 120, 40, 0.25, 1]], [[3, 240, 80, 0, 2]], [[3, 200, 80, 0, 1], [3, 184, 80, 0, 1], [3, 184, 80, 0, 1], [3, 480, 112, 0.25, 1], [3, 672, 112, 0.25, 1]], [[5, 672, 160, 0.25, 2]], [[5, 960, 160, 0, 1], [5, 960, 160, 0.25, 1], [5, 960, 160, 0, 1], [5, 960, 160, 0.25, 1]]]
        self.activation = activation
        if (act is not None):
            warnings.warn('Warning! act argument has been deprecated, use activation instead!')
            self.activation = act
        output_channel = _make_divisible((16 * width_mult), 4)
        self.conv_stem = nn.Conv(3, output_channel, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm(output_channel)
        self.act1 = act_layers(self.activation)
        input_channel = output_channel
        stages = []
        block = GhostBottleneck
        for cfg in self.cfgs:
            layers = []
            for (k, exp_size, c, se_ratio, s) in cfg:
                output_channel = _make_divisible((c * width_mult), 4)
                hidden_channel = _make_divisible((exp_size * width_mult), 4)
                layers.append(block(input_channel, hidden_channel, output_channel, k, s, activation=self.activation, se_ratio=se_ratio))
                input_channel = output_channel
            stages.append(nn.Sequential(*layers))
        output_channel = _make_divisible((exp_size * width_mult), 4)
        stages.append(nn.Sequential(ConvBnAct(input_channel, output_channel, 1, activation=self.activation)))
        self.blocks = nn.Sequential(*stages)
        self._initialize_weights(pretrain)

    def execute(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        output = []
        for i in range(10):
            x = self.blocks[i](x)
            if (i in self.out_stages):
                output.append(x)
        return tuple(output)

    def _initialize_weights(self, pretrain=True):
        print('init weights...')
        for (name, m) in self.named_modules():
            if isinstance(m, nn.Conv):
                if ('conv_stem' in name):
                    init.gauss_(m.weight, mean=0, std=0.01)
                else:
                    init.gauss_(m.weight, mean=0, std=(1.0 / m.weight.shape[1]))
                if (m.bias is not None):
                    init.constant_(m.bias, value=0)
            elif isinstance(m, nn.BatchNorm):
                init.constant_(m.weight, value=1)
                if (m.bias is not None):
                    init.constant_(m.bias, value=0.0001)
                init.constant_(m.running_mean, value=0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, value=1)
                if (m.bias is not None):
                    init.constant_(m.bias, value=0.0001)
                init.constant_(m.running_mean, value=0)
            elif isinstance(m, nn.Linear):
                init.gauss_(m.weight, mean=0, std=0.01)
                if (m.bias is not None):
                    init.constant_(m.bias, value=0)
        if pretrain:
            url = get_url(self.width_mult)
            if (url is not None):
                # <--- 调整/MODIFIED: 使用正确的方式从 URL 加载权重 --->
                
                # 步骤 1: 使用 urllib 将文件下载到本地
                # 定义缓存目录和本地文件路径
                cache_dir = "./pretrained_models"
                os.makedirs(cache_dir, exist_ok=True)
                filename = os.path.basename(url)
                local_path = os.path.join(cache_dir, filename)
                
                # 如果本地不存在该文件，则从 URL 下载
                if not os.path.exists(local_path):
                    print(f'Downloading pretrained model from "{url}" to "{local_path}"')
                    urllib.request.urlretrieve(url, local_path)
                
                # 步骤 2: 使用 jt.load() 从本地路径加载权重
                print(f"=> loading pretrained model from {local_path}")
                pretrained_dict = jt.load(local_path)

                # 后续的非严格加载逻辑（筛选、更新、加载）保持不变
                model_dict = self.state_dict()
                pretrained_dict_filtered = {
                    k: v for k, v in pretrained_dict.items() 
                    if k in model_dict and v.shape == model_dict[k].shape
                }
                model_dict.update(pretrained_dict_filtered)
                self.load_state_dict(model_dict)
                
                print(f"=> loaded {len(pretrained_dict_filtered)} weights from pretrained model")