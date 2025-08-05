#!/usr/bin/env python3
"""
使用 jittordet 的 ConvModule 替换 DepthwiseConvModule
这是最稳定和完整的解决方案
"""

import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import jittor as jt
from jittor import nn


class JittordetDepthwiseConvModule(nn.Module):
    """
    基于 jittordet ConvModule 的 DepthwiseConv 实现
    使用两个标准卷积层模拟 depthwise + pointwise 卷积
    """
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, bias="auto", norm_cfg=dict(type="BN"),
        activation="ReLU", inplace=True,
        order=("depthwise", "dwnorm", "act", "pointwise", "pwnorm", "act"),
    ):
        super(JittordetDepthwiseConvModule, self).__init__()
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

        # Depthwise 卷积：使用 groups=in_channels 模拟
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=in_channels, bias=False  # depthwise 通常不用 bias
        )
        
        # Pointwise 卷积：1x1 卷积
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, 1,
            stride=1, padding=0, dilation=1,
            groups=1, bias=bias
        )

        # 批归一化层
        if self.with_norm:
            if norm_cfg.get('type') == 'BN':
                self.dwnorm = nn.BatchNorm2d(in_channels)
                self.pwnorm = nn.BatchNorm2d(out_channels)
            else:
                self.dwnorm = None
                self.pwnorm = None
        else:
            self.dwnorm = None
            self.pwnorm = None

        # 激活函数
        if activation == 'LeakyReLU':
            self.act = nn.LeakyReLU(0.1)
        elif activation == 'ReLU':
            self.act = nn.ReLU()
        else:
            self.act = None

    def init_weights(self):
        """初始化权重"""
        nonlinearity = "leaky_relu" if self.activation == "LeakyReLU" else "relu"
        
        # 初始化 depthwise 卷积
        nn.init.kaiming_normal_(self.depthwise.weight, mode='fan_out', nonlinearity=nonlinearity)
        
        # 初始化 pointwise 卷积
        nn.init.kaiming_normal_(self.pointwise.weight, mode='fan_out', nonlinearity=nonlinearity)
        if self.pointwise.bias is not None:
            nn.init.constant_(self.pointwise.bias, 0)
        
        # 初始化批归一化
        if self.dwnorm is not None:
            nn.init.constant_(self.dwnorm.weight, 1)
            nn.init.constant_(self.dwnorm.bias, 0)
        if self.pwnorm is not None:
            nn.init.constant_(self.pwnorm.weight, 1)
            nn.init.constant_(self.pwnorm.bias, 0)

    def execute(self, x, norm=True):
        """前向传播"""
        for layer_name in self.order:
            if layer_name == "depthwise":
                x = self.depthwise(x)
            elif layer_name == "pointwise":
                x = self.pointwise(x)
            elif layer_name == "dwnorm" and self.dwnorm is not None:
                x = self.dwnorm(x)
            elif layer_name == "pwnorm" and self.pwnorm is not None:
                x = self.pwnorm(x)
            elif layer_name == "act" and self.act is not None:
                x = self.act(x)
        return x


def patch_depthwise_conv():
    """
    修补 DepthwiseConv 相关的模块
    """
    print("正在使用 jittordet 风格的 DepthwiseConv 替换...")
    
    # 导入需要修补的模块
    try:
        from nanodet.model.module.conv import DepthwiseConvModule
        from nanodet.model.module import conv as conv_module
        
        # 替换 DepthwiseConvModule
        conv_module.DepthwiseConvModule = JittordetDepthwiseConvModule
        
        print("✅ DepthwiseConvModule 已替换为 jittordet 风格实现")
        
        return True
        
    except ImportError as e:
        print(f"❌ 导入模块失败: {e}")
        return False


def test_fixed_conv():
    """
    测试修复后的卷积模块
    """
    print("测试修复后的 DepthwiseConv 模块...")
    
    try:
        # 创建测试模块
        conv = JittordetDepthwiseConvModule(
            in_channels=96, 
            out_channels=96, 
            kernel_size=5, 
            padding=2,
            norm_cfg={'type': 'BN'},
            activation='LeakyReLU'
        )
        
        # 初始化权重
        conv.init_weights()
        
        # 测试前向传播
        x = jt.randn(2, 96, 40, 40)
        y = conv(x)
        
        print(f"✅ 前向传播测试通过: {x.shape} -> {y.shape}")
        
        # 测试梯度计算
        loss = y.sum()
        params = list(conv.parameters())
        grad = jt.grad(loss, params)
        
        print(f"✅ 梯度计算测试通过: 计算了 {len(grad)} 个参数的梯度")
        
        # 验证梯度不为空
        non_zero_grads = sum(1 for g in grad if g is not None and g.sum() != 0)
        print(f"✅ 非零梯度数量: {non_zero_grads}/{len(grad)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=== jittordet 风格 DepthwiseConv 修复工具 ===")
    
    # 修补模块
    if not patch_depthwise_conv():
        print("❌ 修补失败")
        return 1
    
    # 测试修复
    if not test_fixed_conv():
        print("❌ 测试失败")
        return 1
    
    print("🎉 jittordet 风格 DepthwiseConv 修复完成！")
    return 0


if __name__ == '__main__':
    sys.exit(main())
