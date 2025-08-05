#!/usr/bin/env python3
"""
PyTorch 到 Jittor 权重转换脚本
用于将 PyTorch 预训练权重转换为 Jittor 格式
"""

import os
import sys
import argparse
import warnings
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import jittor as jt
import torch
import numpy as np
from collections import OrderedDict


def convert_pytorch_to_jittor_weights(pytorch_path, jittor_path, model_type="nanodet"):
    """
    将 PyTorch 权重转换为 Jittor 格式
    
    Args:
        pytorch_path: PyTorch 权重文件路径
        jittor_path: 输出的 Jittor 权重文件路径
        model_type: 模型类型 ("nanodet", "shufflenet", "resnet")
    """
    print(f"正在转换权重: {pytorch_path} -> {jittor_path}")
    
    # 加载 PyTorch 权重
    if pytorch_path.endswith('.pth') or pytorch_path.endswith('.pt'):
        pytorch_weights = torch.load(pytorch_path, map_location='cpu')
    else:
        raise ValueError(f"不支持的文件格式: {pytorch_path}")
    
    # 处理不同的权重格式
    if isinstance(pytorch_weights, dict):
        if 'state_dict' in pytorch_weights:
            state_dict = pytorch_weights['state_dict']
        elif 'model' in pytorch_weights:
            state_dict = pytorch_weights['model']
        else:
            state_dict = pytorch_weights
    else:
        state_dict = pytorch_weights
    
    # 转换权重格式
    jittor_weights = OrderedDict()
    
    for key, value in state_dict.items():
        # 跳过不需要的键
        if any(skip in key for skip in ['num_batches_tracked', 'total_ops', 'total_params']):
            continue
            
        # 转换 tensor 到 numpy
        if isinstance(value, torch.Tensor):
            numpy_value = value.detach().cpu().numpy()
        else:
            numpy_value = value
            
        # 处理键名映射
        jittor_key = convert_key_name(key, model_type)
        jittor_weights[jittor_key] = numpy_value
        
    print(f"转换完成，共 {len(jittor_weights)} 个参数")
    
    # 保存 Jittor 权重
    os.makedirs(os.path.dirname(jittor_path), exist_ok=True)
    jt.save(jittor_weights, jittor_path)
    print(f"权重已保存到: {jittor_path}")
    
    return jittor_weights


def convert_key_name(pytorch_key, model_type):
    """
    转换 PyTorch 键名到 Jittor 格式
    """
    jittor_key = pytorch_key
    
    # 通用转换规则
    replacements = [
        ('running_mean', 'running_mean'),
        ('running_var', 'running_var'),
        ('weight', 'weight'),
        ('bias', 'bias'),
    ]
    
    # 模型特定的转换规则
    if model_type == "nanodet":
        # NanoDet 特定的键名转换
        if 'head.' in jittor_key:
            # 处理 head 相关的键名
            pass
        elif 'backbone.' in jittor_key:
            # 处理 backbone 相关的键名
            pass
    
    return jittor_key


def load_and_verify_weights(jittor_path, model=None):
    """
    加载并验证转换后的权重
    """
    print(f"验证权重文件: {jittor_path}")
    
    weights = jt.load(jittor_path)
    print(f"权重文件包含 {len(weights)} 个参数")
    
    # 显示前几个参数的信息
    for i, (key, value) in enumerate(weights.items()):
        if i < 5:
            print(f"  {key}: {value.shape if hasattr(value, 'shape') else type(value)}")
        elif i == 5:
            print("  ...")
            break
    
    # 如果提供了模型，尝试加载权重
    if model is not None:
        try:
            model.load_parameters(weights)
            print("✅ 权重加载到模型成功")
        except Exception as e:
            print(f"❌ 权重加载到模型失败: {e}")
    
    return weights


def main():
    parser = argparse.ArgumentParser(description='PyTorch 到 Jittor 权重转换工具')
    parser.add_argument('--pytorch_path', type=str, required=True,
                        help='PyTorch 权重文件路径')
    parser.add_argument('--jittor_path', type=str, required=True,
                        help='输出的 Jittor 权重文件路径')
    parser.add_argument('--model_type', type=str, default='nanodet',
                        choices=['nanodet', 'shufflenet', 'resnet'],
                        help='模型类型')
    parser.add_argument('--verify', action='store_true',
                        help='验证转换后的权重')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.pytorch_path):
        print(f"错误：输入文件不存在: {args.pytorch_path}")
        return 1
    
    try:
        # 转换权重
        jittor_weights = convert_pytorch_to_jittor_weights(
            args.pytorch_path, 
            args.jittor_path, 
            args.model_type
        )
        
        # 验证权重
        if args.verify:
            load_and_verify_weights(args.jittor_path)
        
        print("🎉 权重转换完成！")
        return 0
        
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
