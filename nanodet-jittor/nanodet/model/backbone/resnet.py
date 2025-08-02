import jittor as jt
from jittor import nn,init
from __future__ import absolute_import, division, print_function
from ..module.activation import act_layers


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, activation='ReLU'):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm(planes)
        self.act = act_layers(activation)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm(planes)
        self.downsample = downsample
        self.stride = stride

    def execute(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if (self.downsample is not None):
            residual = self.downsample(x)
        out += residual
        out = self.act(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, activation='ReLU'):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm(planes)
        self.conv2 = nn.Conv(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm(planes)
        self.conv3 = nn.Conv(planes, (planes * self.expansion), kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm((planes * self.expansion))
        self.act = act_layers(activation)
        self.downsample = downsample
        self.stride = stride

    def execute(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if (self.downsample is not None):
            residual = self.downsample(x)
        out += residual
        out = self.act(out)
        return out

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            init.gauss_(m.weight, std=0.001)
            if (m.bias is not None):
                init.constant_(m.bias, value=0)

class ResNet(nn.Module):
    resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]), 34: (BasicBlock, [3, 4, 6, 3]), 50: (Bottleneck, [3, 4, 6, 3]), 101: (Bottleneck, [3, 4, 23, 3]), 152: (Bottleneck, [3, 8, 36, 3])}

    def __init__(self, depth, out_stages=(1, 2, 3, 4), activation='ReLU', pretrain=True):
        super(ResNet, self).__init__()
        if (depth not in self.resnet_spec):
            raise KeyError('invalid resnet depth {}'.format(depth))
        assert set(out_stages).issubset((1, 2, 3, 4))
        self.activation = activation
        (block, layers) = self.resnet_spec[depth]
        self.depth = depth
        self.inplanes = 64
        self.out_stages = out_stages
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm(64)
        self.act = act_layers(self.activation)
        # [必要修正] Jittor 中最大池化的参数是 op='max'
        self.maxpool = nn.Pool(3, stride=2, padding=1, op='max')
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.init_weights(pretrain=pretrain)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if ((stride != 1) or (self.inplanes != (planes * block.expansion))):
            downsample = nn.Sequential(nn.Conv(self.inplanes, (planes * block.expansion), 1, stride=stride, bias=False), nn.BatchNorm((planes * block.expansion)))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, activation=self.activation))
        self.inplanes = (planes * block.expansion)
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, activation=self.activation))
        return nn.Sequential(*layers)

    def execute(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.maxpool(x)
        output = []
        for i in range(1, 5):
            res_layer = getattr(self, 'layer{}'.format(i))
            x = res_layer(x)
            if (i in self.out_stages):
                output.append(x)
        return tuple(output)

    # =======================================================================
    # vv 以下是核心修改部分 vv
    # =======================================================================
    def init_weights(self, pretrain=True):
        if pretrain:
            # 使用 jittor.models 加载预训练权重
            print(f'=> loading pretrained model resnet{self.depth} from jittor models...')
            
            # 1. 创建一个带预训练权重的 Jittor 内置 ResNet 模型
            model_factory = {
                18: jt.models.resnet18, 34: jt.models.resnet34,
                50: jt.models.resnet50, 101: jt.models.resnet101,
                152: jt.models.resnet152
            }
            if self.depth not in model_factory:
                raise ValueError(f"Unsupported ResNet depth for jittor.models: {self.depth}")
            
            pretrained_model = model_factory[self.depth](pretrained=True)
            
            # 2. 将预训练模型的参数加载到当前自定义模型中
            self.load_parameters(pretrained_model.state_dict())

        else:
            # [必要修正] 将随机初始化改为正确的 Jittor 写法
            print('=> initializing weights from scratch...')
            for m in self.modules():
                nonlinearity = 'leaky_relu' if self.activation == 'LeakyReLU' else 'relu'
                if isinstance(m, nn.Conv2d):
                    init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=nonlinearity)
                elif isinstance(m, nn.BatchNorm):
                    init.constant_(m.weight, 1.0)
                    init.constant_(m.bias, 0.0)
