#!/usr/bin/env python3
"""
修复 DepthwiseConv 梯度问题的脚本
将 DepthwiseConvModule 替换为标准 ConvModule
"""

import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import jittor as jt
from jittor import nn


class FixedDepthwiseConvModule(nn.Module):
    """
    修复的 DepthwiseConv 模块，使用标准卷积替代
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, dilation=1, groups=1, bias=True, 
                 norm_cfg=None, activation=None):
        super().__init__()
        
        # 使用标准卷积替代 depthwise 卷积
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, dilation=dilation, 
            groups=1, bias=bias  # 强制使用 groups=1
        )
        
        # 批归一化
        if norm_cfg is not None:
            if norm_cfg.get('type') == 'BN':
                self.bn = nn.BatchNorm2d(out_channels)
            else:
                self.bn = None
        else:
            self.bn = None
            
        # 激活函数
        if activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU(0.1)
        elif activation == 'ReLU':
            self.activation = nn.ReLU()
        else:
            self.activation = None
    
    def execute(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


def patch_depthwise_conv():
    """
    修补 DepthwiseConv 相关的模块
    """
    print("正在修补 DepthwiseConv 模块...")
    
    # 导入需要修补的模块
    try:
        from nanodet.model.module.conv import DepthwiseConvModule
        from nanodet.model.module import conv as conv_module
        
        # 替换 DepthwiseConvModule
        conv_module.DepthwiseConvModule = FixedDepthwiseConvModule
        
        print("✅ DepthwiseConvModule 已替换为标准卷积实现")
        
    except ImportError as e:
        print(f"❌ 导入模块失败: {e}")
        return False
    
    return True


def test_fixed_conv():
    """
    测试修复后的卷积模块
    """
    print("测试修复后的卷积模块...")
    
    try:
        # 创建测试模块
        conv = FixedDepthwiseConvModule(
            in_channels=96, 
            out_channels=96, 
            kernel_size=5, 
            padding=2,
            norm_cfg={'type': 'BN'},
            activation='LeakyReLU'
        )
        
        # 测试前向传播
        x = jt.randn(1, 96, 40, 40)
        y = conv(x)
        
        print(f"✅ 前向传播测试通过: {x.shape} -> {y.shape}")
        
        # 测试梯度计算
        loss = y.sum()
        grad = jt.grad(loss, conv.parameters())
        
        print(f"✅ 梯度计算测试通过: 计算了 {len(grad)} 个参数的梯度")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=== DepthwiseConv 修复工具 ===")
    
    # 修补模块
    if not patch_depthwise_conv():
        print("❌ 修补失败")
        return 1
    
    # 测试修复
    if not test_fixed_conv():
        print("❌ 测试失败")
        return 1
    
    print("🎉 DepthwiseConv 修复完成！")
    return 0


if __name__ == '__main__':
    sys.exit(main())
