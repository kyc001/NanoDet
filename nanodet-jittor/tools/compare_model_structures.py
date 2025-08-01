#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
对比PyTorch和Jittor模型结构
找出参数数量差异的原因
"""

import os
import sys
import torch
import jittor as jt
from collections import defaultdict

# 添加路径
sys.path.insert(0, '/home/kyc/project/nanodet/nanodet-pytorch')
sys.path.append('/home/kyc/project/nanodet/nanodet-jittor')

# PyTorch版本
from nanodet.model.arch import build_model as build_pytorch_model
from nanodet.util import cfg as pytorch_cfg, load_config

# Jittor版本
from nanodet.model.arch.nanodet_plus import NanoDetPlus as JittorNanoDetPlus


def create_pytorch_model():
    """创建PyTorch模型"""
    print("创建PyTorch模型...")
    
    config_path = "/home/kyc/project/nanodet/nanodet-pytorch/config/nanodet-plus-m_320_voc.yml"
    load_config(pytorch_cfg, config_path)
    
    model = build_pytorch_model(pytorch_cfg.model)
    
    return model


def create_jittor_model():
    """创建Jittor模型"""
    print("创建Jittor模型...")
    
    # 创建配置字典
    backbone_cfg = {
        'name': 'ShuffleNetV2',
        'model_size': '1.0x',
        'out_stages': [2, 3, 4],
        'activation': 'LeakyReLU',
        'pretrain': True
    }
    
    fpn_cfg = {
        'name': 'GhostPAN',
        'in_channels': [116, 232, 464],
        'out_channels': 96,
        'kernel_size': 5,
        'num_extra_level': 1,
        'use_depthwise': True,
        'activation': 'LeakyReLU'
    }
    
    head_cfg = {
        'name': 'NanoDetPlusHead',
        'num_classes': 20,
        'input_channel': 96,
        'feat_channels': 96,
        'stacked_convs': 2,
        'kernel_size': 5,
        'strides': [8, 16, 32, 64],
        'activation': 'LeakyReLU',
        'reg_max': 7,
        'norm_cfg': {'type': 'BN'},
        'loss': {
            'loss_qfl': {
                'name': 'QualityFocalLoss',
                'use_sigmoid': True,
                'beta': 2.0,
                'loss_weight': 1.0
            },
            'loss_dfl': {
                'name': 'DistributionFocalLoss',
                'loss_weight': 0.25
            },
            'loss_bbox': {
                'name': 'GIoULoss',
                'loss_weight': 2.0
            }
        }
    }
    
    # 创建aux_head配置 - 使用正确的SimpleConvHead
    aux_head_cfg = {
        'name': 'SimpleConvHead',
        'num_classes': 20,
        'input_channel': 192,
        'feat_channels': 192,
        'stacked_convs': 4,
        'strides': [8, 16, 32, 64],
        'activation': 'LeakyReLU',
        'reg_max': 7
    }
    
    # 创建完整模型
    model = JittorNanoDetPlus(backbone_cfg, fpn_cfg, aux_head_cfg, head_cfg)
    
    return model


def analyze_model_structure(model, model_name):
    """分析模型结构"""
    print(f"\n📊 分析{model_name}模型结构:")
    
    total_params = 0
    module_stats = defaultdict(int)
    param_details = {}
    
    for name, param in model.named_parameters():
        param_count = param.numel() if hasattr(param, 'numel') else param.size
        total_params += param_count
        
        # 按模块分组
        parts = name.split('.')
        if len(parts) >= 2:
            module_name = f"{parts[0]}.{parts[1]}" if len(parts) > 2 else parts[0]
        else:
            module_name = parts[0]
        
        module_stats[module_name] += param_count
        
        # 记录参数详情
        param_details[name] = {
            'shape': list(param.shape) if hasattr(param, 'shape') else list(param.size()),
            'count': param_count
        }
    
    print(f"  总参数数量: {total_params:,}")
    print(f"  参数项数量: {len(param_details)}")
    
    print(f"\n📊 按模块统计:")
    for module, count in sorted(module_stats.items(), key=lambda x: x[1], reverse=True):
        percentage = count / total_params * 100
        print(f"  {module:<20}: {count:>8,} 参数 ({percentage:5.1f}%)")
    
    return param_details, module_stats


def compare_specific_modules(pytorch_details, jittor_details):
    """对比特定模块的差异"""
    print(f"\n🔍 详细对比关键模块:")
    
    # 对比aux_head
    print(f"\n1️⃣ aux_head模块对比:")
    pytorch_aux = {k: v for k, v in pytorch_details.items() if k.startswith('aux_head')}
    jittor_aux = {k: v for k, v in jittor_details.items() if k.startswith('aux_head')}
    
    print(f"  PyTorch aux_head参数数: {len(pytorch_aux)}")
    print(f"  Jittor aux_head参数数: {len(jittor_aux)}")
    
    # 显示前10个参数
    print(f"\n  PyTorch aux_head参数示例:")
    for i, (name, details) in enumerate(list(pytorch_aux.items())[:10]):
        print(f"    {name}: {details['shape']} ({details['count']} 参数)")
    
    print(f"\n  Jittor aux_head参数示例:")
    for i, (name, details) in enumerate(list(jittor_aux.items())[:10]):
        print(f"    {name}: {details['shape']} ({details['count']} 参数)")
    
    # 对比head
    print(f"\n2️⃣ head模块对比:")
    pytorch_head = {k: v for k, v in pytorch_details.items() if k.startswith('head') and not k.startswith('aux_head')}
    jittor_head = {k: v for k, v in jittor_details.items() if k.startswith('head') and not k.startswith('aux_head')}
    
    print(f"  PyTorch head参数数: {len(pytorch_head)}")
    print(f"  Jittor head参数数: {len(jittor_head)}")
    
    # 对比backbone
    print(f"\n3️⃣ backbone模块对比:")
    pytorch_backbone = {k: v for k, v in pytorch_details.items() if k.startswith('backbone')}
    jittor_backbone = {k: v for k, v in jittor_details.items() if k.startswith('backbone')}
    
    print(f"  PyTorch backbone参数数: {len(pytorch_backbone)}")
    print(f"  Jittor backbone参数数: {len(jittor_backbone)}")


def find_missing_modules(pytorch_details, jittor_details):
    """找出缺失的模块"""
    print(f"\n🔍 查找缺失的模块:")
    
    pytorch_modules = set()
    jittor_modules = set()
    
    for name in pytorch_details.keys():
        parts = name.split('.')
        if len(parts) >= 2:
            module = f"{parts[0]}.{parts[1]}"
        else:
            module = parts[0]
        pytorch_modules.add(module)
    
    for name in jittor_details.keys():
        parts = name.split('.')
        if len(parts) >= 2:
            module = f"{parts[0]}.{parts[1]}"
        else:
            module = parts[0]
        jittor_modules.add(module)
    
    only_in_pytorch = pytorch_modules - jittor_modules
    only_in_jittor = jittor_modules - pytorch_modules
    common_modules = pytorch_modules & jittor_modules
    
    print(f"  共同模块: {len(common_modules)} 个")
    print(f"  只在PyTorch中: {len(only_in_pytorch)} 个")
    print(f"  只在Jittor中: {len(only_in_jittor)} 个")
    
    if only_in_pytorch:
        print(f"\n  只在PyTorch中的模块:")
        for module in sorted(only_in_pytorch):
            print(f"    - {module}")
    
    if only_in_jittor:
        print(f"\n  只在Jittor中的模块:")
        for module in sorted(only_in_jittor):
            print(f"    - {module}")


def main():
    """主函数"""
    print("🚀 开始对比PyTorch和Jittor模型结构")
    print("=" * 80)
    
    # 创建模型
    pytorch_model = create_pytorch_model()
    jittor_model = create_jittor_model()
    
    # 分析结构
    pytorch_details, pytorch_stats = analyze_model_structure(pytorch_model, "PyTorch")
    jittor_details, jittor_stats = analyze_model_structure(jittor_model, "Jittor")
    
    # 对比差异
    print(f"\n" + "=" * 80)
    print("🔍 模型结构对比分析")
    print("=" * 80)
    
    # 总体对比
    pytorch_total = sum(pytorch_stats.values())
    jittor_total = sum(jittor_stats.values())
    
    print(f"📊 总体参数对比:")
    print(f"  PyTorch总参数: {pytorch_total:,}")
    print(f"  Jittor总参数: {jittor_total:,}")
    print(f"  差异: {pytorch_total - jittor_total:,} ({(pytorch_total - jittor_total) / pytorch_total * 100:.1f}%)")
    
    # 详细对比
    compare_specific_modules(pytorch_details, jittor_details)
    
    # 查找缺失模块
    find_missing_modules(pytorch_details, jittor_details)
    
    print(f"\n✅ 模型结构对比完成!")
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
