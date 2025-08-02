import jittor as jt
from jittor import init
from jittor import nn
import math
from ..module.activation import act_layers

import os
import urllib.request

efficientnet_lite_params = {'efficientnet_lite0': [1.0, 1.0, 224, 0.2], 'efficientnet_lite1': [1.0, 1.1, 240, 0.2], 'efficientnet_lite2': [1.1, 1.2, 260, 0.3], 'efficientnet_lite3': [1.2, 1.4, 280, 0.3], 'efficientnet_lite4': [1.4, 1.8, 300, 0.3]}
model_urls = {'efficientnet_lite0': 'https://github.com/RangiLyu/EfficientNet-Lite/releases/download/v1.0/efficientnet_lite0.pth', 'efficientnet_lite1': 'https://github.com/RangiLyu/EfficientNet-Lite/releases/download/v1.0/efficientnet_lite1.pth', 'efficientnet_lite2': 'https://github.com/RangiLyu/EfficientNet-Lite/releases/download/v1.0/efficientnet_lite2.pth', 'efficientnet_lite3': 'https://github.com/RangiLyu/EfficientNet-Lite/releases/download/v1.0/efficientnet_lite3.pth', 'efficientnet_lite4': 'https://github.com/RangiLyu/EfficientNet-Lite/releases/download/v1.0/efficientnet_lite4.pth'}

def round_filters(filters, multiplier, divisor=8, min_width=None):
    if (not multiplier):
        return filters
    filters *= multiplier
    min_width = (min_width or divisor)
    new_filters = max(min_width, ((int((filters + (divisor / 2))) // divisor) * divisor))
    if (new_filters < (0.9 * filters)):
        new_filters += divisor
    return int(new_filters)

def round_repeats(repeats, multiplier):
    if (not multiplier):
        return repeats
    return int(math.ceil((multiplier * repeats)))

def drop_connect(x, drop_connect_rate, training):
    if (not training):
        return x
    keep_prob = (1.0 - drop_connect_rate)
    batch_size = x.shape[0]
    random_tensor = keep_prob
    random_tensor += jt.rand([batch_size, 1, 1, 1], dtype=x.dtype, device=x.device)
    binary_mask = jt.floor(random_tensor)
    x = ((x / keep_prob) * binary_mask)
    return x

class MBConvBlock(nn.Module):

    def __init__(self, inp, final_oup, k, s, expand_ratio, se_ratio, has_se=False, activation='ReLU6'):
        super(MBConvBlock, self).__init__()
        self._momentum = 0.01
        self._epsilon = 0.001
        self.input_filters = inp
        self.output_filters = final_oup
        self.stride = s
        self.expand_ratio = expand_ratio
        self.has_se = has_se
        self.id_skip = True
        oup = (inp * expand_ratio)
        if (expand_ratio != 1):
            self._expand_conv = nn.Conv(inp, oup, 1, bias=False)
            self._bn0 = nn.BatchNorm(oup, momentum=self._momentum, eps=self._epsilon)
        self._depthwise_conv = nn.Conv(oup, oup, k, groups=oup, padding=((k - 1) // 2), stride=s, bias=False)
        self._bn1 = nn.BatchNorm(oup, momentum=self._momentum, eps=self._epsilon)
        if self.has_se:
            num_squeezed_channels = max(1, int((inp * se_ratio)))
            self._se_reduce = nn.Conv(oup, num_squeezed_channels, 1)
            self._se_expand = nn.Conv(num_squeezed_channels, oup, 1)
        self._project_conv = nn.Conv(oup, final_oup, 1, bias=False)
        self._bn2 = nn.BatchNorm(final_oup, momentum=self._momentum, eps=self._epsilon)
        self._relu = act_layers(activation)

    def execute(self, x, drop_connect_rate=None):
        identity = x
        if (self.expand_ratio != 1):
            x = self._relu(self._bn0(self._expand_conv(x)))
        x = self._relu(self._bn1(self._depthwise_conv(x)))
        if self.has_se:
            x_squeezed = jt.nn.AdaptiveAvgPool2d(x, 1) # F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(self._relu(self._se_reduce(x_squeezed)))
            x = (jt.nn.sigmoid(x_squeezed) * x)
        x = self._bn2(self._project_conv(x))
        if (self.id_skip and (self.stride == 1) and (self.input_filters == self.output_filters)):
            if drop_connect_rate:
                x = drop_connect(x, drop_connect_rate, training=self.training)
            x += identity
        return x

class EfficientNetLite(nn.Module):

    def __init__(self, model_name, out_stages=(2, 4, 6), activation='ReLU6', pretrain=True):
        super(EfficientNetLite, self).__init__()
        assert set(out_stages).issubset((i for i in range(0, 7)))
        assert (model_name in efficientnet_lite_params)
        self.model_name = model_name
        momentum = 0.01
        epsilon = 0.001
        (width_multiplier, depth_multiplier, _, dropout_rate) = efficientnet_lite_params[model_name]
        self.drop_connect_rate = 0.2
        self.out_stages = out_stages
        mb_block_settings = [[1, 3, 1, 1, 32, 16, 0.25], [2, 3, 2, 6, 16, 24, 0.25], [2, 5, 2, 6, 24, 40, 0.25], [3, 3, 2, 6, 40, 80, 0.25], [3, 5, 1, 6, 80, 112, 0.25], [4, 5, 2, 6, 112, 192, 0.25], [1, 3, 1, 6, 192, 320, 0.25]]
        out_channels = 32
        self.stem = nn.Sequential(nn.Conv(3, out_channels, 3, stride=2, padding=1, bias=False), nn.BatchNorm(out_channels, momentum=momentum, eps=epsilon), act_layers(activation))
        self.blocks = nn.ModuleList([])
        for (i, stage_setting) in enumerate(mb_block_settings):
            stage = nn.ModuleList([])
            (num_repeat, kernal_size, stride, expand_ratio, input_filters, output_filters, se_ratio) = stage_setting
            input_filters = (input_filters if (i == 0) else round_filters(input_filters, width_multiplier))
            output_filters = round_filters(output_filters, width_multiplier)
            num_repeat = (num_repeat if ((i == 0) or (i == (len(mb_block_settings) - 1))) else round_repeats(num_repeat, depth_multiplier))
            stage.append(MBConvBlock(input_filters, output_filters, kernal_size, stride, expand_ratio, se_ratio, has_se=False))
            if (num_repeat > 1):
                input_filters = output_filters
                stride = 1
            for _ in range((num_repeat - 1)):
                stage.append(MBConvBlock(input_filters, output_filters, kernal_size, stride, expand_ratio, se_ratio, has_se=False))
            self.blocks.append(stage)
        self._initialize_weights(pretrain)

    def execute(self, x):
        x = self.stem(x)
        output = []
        idx = 0
        for (j, stage) in enumerate(self.blocks):
            for block in stage:
                drop_connect_rate = self.drop_connect_rate
                if drop_connect_rate:
                    drop_connect_rate *= (float(idx) / len(self.blocks))
                x = block(x, drop_connect_rate)
                idx += 1
            if (j in self.out_stages):
                output.append(x)
        return output

    def _initialize_weights(self, pretrain=True):
        for m in self.modules():
            if isinstance(m, nn.Conv):
                n = ((m.kernel_size[0] * m.kernel_size[1]) * m.out_channels)
                init.gauss_(0, mean=math.sqrt((2.0 / n)))
                if (m.bias is not None):
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if pretrain:
            url = model_urls[self.model_name]
            if (url is not None):
                # <--- 调整/MODIFIED: 替换为从 URL 下载并以非严格模式加载的完整逻辑 --->
                cache_dir = "./pretrained_models"
                os.makedirs(cache_dir, exist_ok=True)
                filename = os.path.basename(url)
                local_path = os.path.join(cache_dir, filename)
                
                if not os.path.exists(local_path):
                    print(f'Downloading pretrained model from "{url}" to "{local_path}"')
                    urllib.request.urlretrieve(url, local_path)
                    
                print(f"=> loading pretrained model from {local_path}")
                pretrained_dict = jt.load(local_path)
                
                model_dict = self.state_dict()
                pretrained_dict_filtered = {
                    k: v for k, v in pretrained_dict.items() 
                    if k in model_dict and v.shape == model_dict[k].shape
                }
                model_dict.update(pretrained_dict_filtered)
                self.load_state_dict(model_dict)
                print(f"=> loaded {len(pretrained_dict_filtered)} weights from pretrained model")
                
    
    def load_pretrain(self, path):
        state_dict = jt.load(path)
        self.load_parameters(state_dict, strict=True)
