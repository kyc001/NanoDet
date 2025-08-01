#!/usr/bin/env python3
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
