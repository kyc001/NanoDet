#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PyTorch权重转换工具
将PyTorch训练的权重转换为Jittor格式
"""

import os
import sys
import argparse
import jittor as jt
import numpy as np


def load_pytorch_checkpoint(ckpt_path):
    """加载PyTorch检查点文件"""
    print(f"Loading PyTorch checkpoint: {ckpt_path}")
    
    try:
        import torch
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        print(f"✓ Checkpoint loaded successfully")
        print(f"  Keys: {list(checkpoint.keys())}")
        
        # 获取模型权重
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # 过滤模型权重（移除'model.'前缀）
        model_weights = {}
        for key, value in state_dict.items():
            if key.startswith('model.'):
                new_key = key[6:]  # 移除'model.'前缀
                model_weights[new_key] = value.numpy()  # 转换为numpy
            elif not key.startswith('avg_model.') and not key.startswith('optimizer'):
                model_weights[key] = value.numpy()
        
        print(f"✓ Extracted model weights: {len(model_weights)} parameters")
        return model_weights
        
    except Exception as e:
        print(f"✗ Failed to load PyTorch checkpoint: {e}")
        return None


def convert_weights_to_jittor(pytorch_weights):
    """将PyTorch权重转换为Jittor格式"""
    print("Converting weights to Jittor format...")
    
    jittor_weights = {}
    
    for key, value in pytorch_weights.items():
        # 跳过不需要的参数
        if 'num_batches_tracked' in key:
            continue
            
        # 转换为Jittor数组
        jittor_weights[key] = jt.array(value)
    
    print(f"✓ Converted {len(jittor_weights)} parameters to Jittor format")
    return jittor_weights


def save_jittor_weights(weights, output_path):
    """保存Jittor权重"""
    print(f"Saving Jittor weights to: {output_path}")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存权重
    jt.save(weights, output_path)
    print(f"✓ Weights saved successfully")


def load_jittor_model():
    """创建Jittor模型用于验证"""
    try:
        from nanodet.model import build_model
        
        # 模型配置
        model_cfg = {
            'name': 'NanoDetPlus',
            'backbone': {
                'name': 'ShuffleNetV2',
                'model_size': '1.0x',
                'out_stages': [2, 3, 4],
                'activation': 'LeakyReLU'
            },
            'fpn': {
                'name': 'GhostPAN',
                'in_channels': [116, 232, 464],
                'out_channels': 96,
                'kernel_size': 5,
                'num_extra_level': 1,
                'use_depthwise': True,
                'activation': 'LeakyReLU'
            },
            'aux_head': {
                'name': 'SimpleConvHead',
                'num_classes': 20,
                'input_channel': 192,  # 96 * 2
                'feat_channels': 192,
                'stacked_convs': 4,
                'strides': [8, 16, 32, 64],
                'activation': 'LeakyReLU',
                'reg_max': 7
            },
            'head': {
                'name': 'NanoDetPlusHead',
                'num_classes': 20,
                'input_channel': 96,
                'feat_channels': 96,
                'stacked_convs': 2,
                'kernel_size': 5,
                'strides': [8, 16, 32, 64],
                'conv_type': 'DWConv',
                'norm_cfg': dict(type='BN'),
                'reg_max': 7,
                'activation': 'LeakyReLU',
                'loss': {
                    'loss_qfl': {'beta': 2.0, 'loss_weight': 1.0},
                    'loss_dfl': {'loss_weight': 0.25},
                    'loss_bbox': {'loss_weight': 2.0}
                }
            },
            'detach_epoch': 10
        }
        
        model = build_model(model_cfg)
        return model
        
    except Exception as e:
        print(f"✗ Failed to create Jittor model: {e}")
        return None


def verify_weight_loading(model, weights):
    """验证权重加载"""
    print("Verifying weight loading...")
    
    try:
        # 加载权重
        model.load_state_dict(weights)
        print("✓ Weights loaded successfully")
        
        # 测试前向传播
        x = jt.randn(1, 3, 320, 320)
        with jt.no_grad():
            output = model(x)
        
        print(f"✓ Forward pass successful, output shape: {output.shape}")
        return True
        
    except Exception as e:
        print(f"✗ Weight loading verification failed: {e}")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Convert PyTorch weights to Jittor format')
    parser.add_argument('--pytorch_ckpt', required=True, help='Path to PyTorch checkpoint')
    parser.add_argument('--output', required=True, help='Output path for Jittor weights')
    parser.add_argument('--verify', action='store_true', help='Verify weight loading')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PyTorch to Jittor Weight Conversion")
    print("=" * 60)
    
    # 检查输入文件
    if not os.path.exists(args.pytorch_ckpt):
        print(f"✗ PyTorch checkpoint not found: {args.pytorch_ckpt}")
        return False
    
    # 加载PyTorch权重
    pytorch_weights = load_pytorch_checkpoint(args.pytorch_ckpt)
    if pytorch_weights is None:
        return False
    
    # 转换为Jittor格式
    jittor_weights = convert_weights_to_jittor(pytorch_weights)
    
    # 保存Jittor权重
    save_jittor_weights(jittor_weights, args.output)
    
    # 验证权重加载（可选）
    if args.verify:
        model = load_jittor_model()
        if model is not None:
            verify_weight_loading(model, jittor_weights)
    
    print("\n" + "=" * 60)
    print("Conversion completed successfully!")
    print("=" * 60)
    print(f"Input: {args.pytorch_ckpt}")
    print(f"Output: {args.output}")
    print(f"Converted parameters: {len(jittor_weights)}")
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
