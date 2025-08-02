import jittor as jt
from jittor import nn, init, models
from ..module.activation import act_layers

def channel_shuffle(x, groups):
    # type: (jt.Var, int) -> jt.Var
    batchsize, num_channels, height, width = x.shape
    channels_per_group = num_channels // groups

    # reshape
    # PyTorch: x.view(batchsize, groups, channels_per_group, height, width)
    x = x.reshape(batchsize, groups, channels_per_group, height, width)
    
    # transpose
    # PyTorch: torch.transpose(x, 1, 2).contiguous()
    x = x.permute(0, 2, 1, 3, 4)

    # flatten
    x = x.reshape(batchsize, -1, height, width)
    return x

class ShuffleV2Block(nn.Module):
    def __init__(self, inp, oup, stride, activation="ReLU"):
        super(ShuffleV2Block, self).__init__()
        if not (1 <= stride <= 3):
            raise ValueError("illegal stride value")
        self.stride = stride
        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm(inp),
                nn.Conv(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm(branch_features),
                act_layers(activation),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv(
                inp if (self.stride > 1) else branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm(branch_features),
            act_layers(activation),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm(branch_features),
            nn.Conv(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm(branch_features),
            act_layers(activation),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        # nn.Conv2d -> nn.Conv
        return nn.Conv(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    # forward -> execute
    def execute(self, x):
        if self.stride == 1:
            # x.chunk -> jt.chunk
            x1, x2 = jt.chunk(x, 2, dim=1)
            # torch.cat -> jt.concat
            out = jt.concat((x1, self.branch2(x2)), dim=1)
        else:
            out = jt.concat((self.branch1(x), self.branch2(x)), dim=1)
        out = channel_shuffle(out, 2)
        return out

class ShuffleNetV2(nn.Module):
    def __init__(
        self,
        model_size="1.0x",
        out_stages=(2, 3, 4),
        with_last_conv=False,
        kernal_size=3, # 单词拼写错误，但保持与您代码一致
        activation="ReLU",
        pretrain=True,
    ):
        super(ShuffleNetV2, self).__init__()
        assert set(out_stages).issubset((2, 3, 4))
        print("model size is ", model_size)

        self.stage_repeats = [4, 8, 4]
        self.model_size = model_size
        self.out_stages = out_stages
        self.with_last_conv = with_last_conv
        self.kernal_size = kernal_size
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

        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm(output_channels),
            act_layers(activation),
        )
        input_channels = output_channels
        self.maxpool = nn.Pool(kernel_size=3, stride=2, padding=1, op='max')

        stage_names = ["stage{}".format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
            stage_names, self.stage_repeats, self._stage_out_channels[1:]
        ):
            seq = [ShuffleV2Block(input_channels, output_channels, 2, activation=activation)]
            for i in range(repeats - 1):
                seq.append(ShuffleV2Block(output_channels, output_channels, 1, activation=activation))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels
        
        output_channels = self._stage_out_channels[-1]
        if self.with_last_conv:
            conv5 = nn.Sequential(
                nn.Conv(input_channels, output_channels, 1, 1, 0, bias=False),
                nn.BatchNorm(output_channels),
                act_layers(activation),
            )
            self.stage4.add_module("conv5", conv5)
        
        self._initialize_weights(pretrain)

    # forward -> execute
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
        print("init weights...")
        if pretrain:
            # 采纳您的建议，使用 Jittor 内置模型加载预训练权重
            print(f"=> loading pretrained model shufflenetv2_{self.model_size} from jittor models...")
            model_factory = {
                "0.5x": models.shufflenet_v2_x0_5,
                "1.0x": models.shufflenet_v2_x1_0,
                # Jittor models 库目前主要提供 0.5x 和 1.0x 的预训练权重
            }
            if self.model_size not in model_factory:
                 print(f"Warning: Jittor models zoo doesn't have pretrained shufflenetv2_{self.model_size}. Training from scratch.")
                 self._initialize_weights(pretrain=False) # 调用随机初始化
                 return
            
            pretrained_model = model_factory[self.model_size](pretrained=True)
            self.load_parameters(pretrained_model.state_dict())

        else:
            # Jittor 版本的随机初始化
            for name, m in self.named_modules():
                if isinstance(m, nn.Conv):
                    if "first" in name:
                        init.gauss_(m.weight, mean=0, std=0.01)
                    else:
                        init.gauss_(m.weight, mean=0, std=1.0 / m.weight.shape[1])
                    if m.bias is not None:
                        init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm):
                    init.constant_(m.weight, 1)
                    if m.bias is not None:
                        init.constant_(m.bias, 0.0001)
                    # Jittor 的 BatchNorm 会自动处理 running_mean，无需手动初始化