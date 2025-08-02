import jittor as jt
from jittor import nn


class Scale(nn.Module):
    """
    一個可學習的縮放參數 (Jittor 版本)。
    """
    def __init__(self, scale=1.0):
        super(Scale, self).__init__()
        # [遷移] nn.Parameter(torch.tensor(...)) -> nn.Parameter(jt.float32(...))
        # 使用 nn.Parameter 使其成為可訓練的參數
        self.scale = nn.Parameter(jt.float32(scale))

    # [遷移] forward -> execute
    def execute(self, x):
        return x * self.scale


        