import jittor as jt
import jittor.nn as nn

class SELU(nn.Module):
    def __init__(self, scale=1.0507009873554804934193349852946, alpha=1.6732632423543772848170429916717):
        super().__init__()
        self.alpha = alpha
        self.scale = scale

    def execute(self, x):
        return self.scale * jt.ternary(x > 0, x, self.alpha * (jt.exp(x) - 1))

class Hardswish(nn.Module):
    def execute(self, x):
        return x * nn.relu6(x + 3) / 6

activations = {
    "ReLU": nn.ReLU,
    "LeakyReLU": nn.LeakyReLU,
    "ReLU6": nn.ReLU6,
    "SELU": SELU,
    "ELU": nn.ELU,
    "GELU": nn.GELU,
    "PReLU": nn.PReLU,
    "SiLU": nn.SiLU,
    "HardSwish": Hardswish,
    "Hardswish": Hardswish,
    None: nn.Identity,
}


def act_layers(name):
    assert name in activations.keys()
    if name == "LeakyReLU":
        return nn.LeakyReLU(scale=0.1)
    elif name == "GELU":
        return nn.GELU()
    elif name == "PReLU":
        return nn.PReLU()
    elif name == "ReLU":
        return nn.ReLU()  # 🔧 Jittor 的 ReLU 不支持 inplace 参数
    elif name == "ReLU6":
        return nn.ReLU6()  # 🔧 Jittor 的 ReLU6 不支持 inplace 参数
    else:
        return activations[name]()
