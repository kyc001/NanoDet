#!/usr/bin/env python3
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
