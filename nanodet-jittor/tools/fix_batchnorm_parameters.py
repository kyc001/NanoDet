#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
修复BatchNorm参数问题
将running_mean和running_var从参数中排除
"""

import os
import sys
import re

# 添加路径
sys.path.append('/home/kyc/project/nanodet/nanodet-jittor')


def find_batchnorm_files():
    """找到所有包含BatchNorm的文件"""
    batchnorm_files = []
    
    # 搜索目录
    search_dirs = [
        'nanodet/model/backbone',
        'nanodet/model/fpn', 
        'nanodet/model/head',
        'nanodet/model/module'
    ]
    
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            for root, dirs, files in os.walk(search_dir):
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        
                        # 检查文件是否包含BatchNorm
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                if 'BatchNorm' in content or 'nn.BatchNorm' in content:
                                    batchnorm_files.append(file_path)
                        except:
                            pass
    
    return batchnorm_files


def create_custom_batchnorm():
    """创建自定义BatchNorm，排除统计参数"""
    custom_bn_code = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
自定义BatchNorm，与PyTorch参数机制对齐
排除running_mean和running_var从named_parameters()
"""

import jittor as jt
from jittor import nn


class BatchNormAligned(nn.Module):
    """
    与PyTorch对齐的BatchNorm
    running_mean和running_var不被计入named_parameters()
    """
    
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(BatchNormAligned, self).__init__()
        
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        if self.affine:
            # 这些是可训练参数，会被计入named_parameters()
            self.weight = jt.ones(num_features)
            self.bias = jt.zeros(num_features)
        else:
            self.weight = None
            self.bias = None
        
        if self.track_running_stats:
            # 使用下划线前缀，不被计入named_parameters()
            self._running_mean = jt.zeros(num_features)
            self._running_var = jt.ones(num_features)
            self._running_mean.requires_grad = False
            self._running_var.requires_grad = False
            # 兼容性属性
            self._num_batches_tracked = 0
        else:
            self._running_mean = None
            self._running_var = None
            self._num_batches_tracked = None
    
    @property
    def running_mean(self):
        """兼容性属性访问"""
        return self._running_mean
    
    @property 
    def running_var(self):
        """兼容性属性访问"""
        return self._running_var
    
    @property
    def num_batches_tracked(self):
        """兼容性属性访问"""
        return self._num_batches_tracked
    
    def execute(self, x):
        """前向传播"""
        if self.training and self.track_running_stats:
            # 训练模式：更新统计信息
            if self._running_mean is not None:
                # 计算当前批次统计
                mean = x.mean(dim=[0, 2, 3], keepdims=False)
                var = x.var(dim=[0, 2, 3], keepdims=False)
                
                # 更新移动平均
                self._running_mean = (1 - self.momentum) * self._running_mean + self.momentum * mean
                self._running_var = (1 - self.momentum) * self._running_var + self.momentum * var
                self._num_batches_tracked += 1
        
        # 使用BatchNorm
        if self.track_running_stats and not self.training:
            # 推理模式：使用统计信息
            mean = self._running_mean
            var = self._running_var
        else:
            # 训练模式：使用当前批次统计
            mean = x.mean(dim=[0, 2, 3], keepdims=True)
            var = x.var(dim=[0, 2, 3], keepdims=True)
        
        # 标准化
        x_norm = (x - mean) / jt.sqrt(var + self.eps)
        
        # 仿射变换
        if self.affine:
            if len(x.shape) == 4:  # [N, C, H, W]
                weight = self.weight.view(1, -1, 1, 1)
                bias = self.bias.view(1, -1, 1, 1)
            else:  # [N, C]
                weight = self.weight
                bias = self.bias
            
            x_norm = x_norm * weight + bias
        
        return x_norm


# 兼容性别名
BatchNorm = BatchNormAligned
BatchNorm1d = BatchNormAligned
BatchNorm2d = BatchNormAligned
BatchNorm3d = BatchNormAligned
'''
    
    # 保存自定义BatchNorm
    with open('nanodet/model/module/batchnorm_aligned.py', 'w', encoding='utf-8') as f:
        f.write(custom_bn_code)
    
    print("✓ 创建了自定义BatchNorm: nanodet/model/module/batchnorm_aligned.py")


def create_custom_scale():
    """创建自定义Scale，使用标量形状"""
    custom_scale_code = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
自定义Scale模块，与PyTorch标量参数对齐
"""

import jittor as jt
from jittor import nn
import numpy as np


class ScaleAligned(nn.Module):
    """
    与PyTorch对齐的Scale模块
    尝试创建标量参数
    """
    
    def __init__(self, scale=1.0):
        super(ScaleAligned, self).__init__()
        
        # 尝试多种方式创建标量参数
        # 方法1: 使用下划线前缀存储数据，用属性访问
        self._scale_value = float(scale)
    
    @property
    def scale(self):
        """动态创建标量张量"""
        # 每次访问时创建新的标量张量
        # 这样可以避免被计入named_parameters()
        return jt.array(self._scale_value)
    
    def execute(self, x):
        """前向传播"""
        return x * self.scale


# 如果上面的方法不行，尝试这个
class ScaleParameter(nn.Module):
    """
    Scale参数模块 - 备选方案
    """
    
    def __init__(self, scale=1.0):
        super(ScaleParameter, self).__init__()
        
        # 创建1维张量，在权重加载时特殊处理
        self.scale = jt.array([scale])
    
    def execute(self, x):
        """前向传播"""
        return x * self.scale


# 默认使用对齐版本
Scale = ScaleAligned
'''
    
    # 保存自定义Scale
    with open('nanodet/model/module/scale_aligned.py', 'w', encoding='utf-8') as f:
        f.write(custom_scale_code)
    
    print("✓ 创建了自定义Scale: nanodet/model/module/scale_aligned.py")


def main():
    """主函数"""
    print("🚀 开始修复BatchNorm和Scale参数问题")
    
    # 找到BatchNorm文件
    batchnorm_files = find_batchnorm_files()
    print(f"找到 {len(batchnorm_files)} 个包含BatchNorm的文件:")
    for file in batchnorm_files:
        print(f"  - {file}")
    
    # 创建自定义模块
    create_custom_batchnorm()
    create_custom_scale()
    
    print(f"\\n✅ 修复完成！")
    print(f"下一步：")
    print(f"1. 将现有代码中的 nn.BatchNorm 替换为 BatchNormAligned")
    print(f"2. 将现有代码中的 Scale 替换为 ScaleAligned")
    print(f"3. 重新测试参数对齐")


if __name__ == '__main__':
    main()
