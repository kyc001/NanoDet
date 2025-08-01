#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
精确找出29,116参数差异的根源
必须做到100%对齐，不能有任何妥协
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
    
    # 创建配置字典 - 完全对齐PyTorch版本
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
    
    # 创建aux_head配置 - 完全对齐PyTorch版本
    aux_head_cfg = {
        'name': 'SimpleConvHead',
        'num_classes': 20,
        'input_channel': 192,
        'feat_channels': 192,  # 与PyTorch版本一致
        'stacked_convs': 4,    # 与PyTorch版本一致
        'strides': [8, 16, 32, 64],
        'activation': 'LeakyReLU',
        'reg_max': 7
    }
    
    # 创建完整模型
    model = JittorNanoDetPlus(backbone_cfg, fpn_cfg, aux_head_cfg, head_cfg)
    
    return model


def analyze_parameter_differences():
    """精确分析参数差异"""
    print("🔍 精确分析参数差异")
    print("=" * 80)
    
    # 创建模型
    try:
        pytorch_model = create_pytorch_model()
        print("✓ PyTorch模型创建成功")
    except Exception as e:
        print(f"❌ PyTorch模型创建失败: {e}")
        return False
    
    jittor_model = create_jittor_model()
    print("✓ Jittor模型创建成功")
    
    # 统计PyTorch参数
    pytorch_params = {}
    pytorch_total = 0
    
    for name, param in pytorch_model.named_parameters():
        param_count = param.numel()
        pytorch_params[name] = {
            'shape': list(param.shape),
            'count': param_count
        }
        pytorch_total += param_count
    
    print(f"✓ PyTorch模型: {pytorch_total:,} 参数, {len(pytorch_params)} 项")
    
    # 统计Jittor参数
    jittor_params = {}
    jittor_total = 0
    
    for name, param in jittor_model.named_parameters():
        param_count = param.numel() if hasattr(param, 'numel') else param.size
        jittor_params[name] = {
            'shape': list(param.shape),
            'count': param_count
        }
        jittor_total += param_count
    
    print(f"✓ Jittor模型: {jittor_total:,} 参数, {len(jittor_params)} 项")
    
    # 计算差异
    difference = abs(pytorch_total - jittor_total)
    print(f"\n📊 参数差异: {difference:,} ({difference/pytorch_total*100:.3f}%)")
    
    if difference == 0:
        print("🎉 参数数量100%对齐！")
        return True
    
    # 详细分析差异
    print(f"\n🔍 详细分析差异来源:")
    
    # 按模块分组统计
    pytorch_modules = defaultdict(int)
    jittor_modules = defaultdict(int)
    
    for name, details in pytorch_params.items():
        module = name.split('.')[0]
        pytorch_modules[module] += details['count']
    
    for name, details in jittor_params.items():
        module = name.split('.')[0]
        jittor_modules[module] += details['count']
    
    # 对比每个模块
    all_modules = set(pytorch_modules.keys()) | set(jittor_modules.keys())
    
    print(f"\n📊 按模块对比:")
    print(f"{'模块':<20} {'PyTorch':<12} {'Jittor':<12} {'差异':<12}")
    print("-" * 60)
    
    total_diff = 0
    for module in sorted(all_modules):
        pytorch_count = pytorch_modules.get(module, 0)
        jittor_count = jittor_modules.get(module, 0)
        diff = pytorch_count - jittor_count
        total_diff += abs(diff)
        
        status = "✅" if diff == 0 else "❌"
        print(f"{module:<20} {pytorch_count:<12,} {jittor_count:<12,} {diff:<12,} {status}")
    
    print("-" * 60)
    print(f"{'总计':<20} {pytorch_total:<12,} {jittor_total:<12,} {pytorch_total-jittor_total:<12,}")
    
    # 找出差异最大的模块
    max_diff_module = None
    max_diff = 0
    
    for module in all_modules:
        pytorch_count = pytorch_modules.get(module, 0)
        jittor_count = jittor_modules.get(module, 0)
        diff = abs(pytorch_count - jittor_count)
        
        if diff > max_diff:
            max_diff = diff
            max_diff_module = module
    
    if max_diff_module:
        print(f"\n🎯 差异最大的模块: {max_diff_module} (差异: {max_diff:,} 参数)")
        
        # 详细分析该模块
        print(f"\n🔍 详细分析 {max_diff_module} 模块:")
        
        pytorch_module_params = {k: v for k, v in pytorch_params.items() if k.startswith(max_diff_module)}
        jittor_module_params = {k: v for k, v in jittor_params.items() if k.startswith(max_diff_module)}
        
        print(f"  PyTorch {max_diff_module} 参数项: {len(pytorch_module_params)}")
        print(f"  Jittor {max_diff_module} 参数项: {len(jittor_module_params)}")
        
        # 找出只在一边存在的参数
        pytorch_names = set(pytorch_module_params.keys())
        jittor_names = set(jittor_module_params.keys())
        
        only_pytorch = pytorch_names - jittor_names
        only_jittor = jittor_names - pytorch_names
        common = pytorch_names & jittor_names
        
        print(f"  共同参数: {len(common)}")
        print(f"  只在PyTorch中: {len(only_pytorch)}")
        print(f"  只在Jittor中: {len(only_jittor)}")
        
        if only_pytorch:
            print(f"\n  只在PyTorch中的参数:")
            for name in sorted(only_pytorch)[:10]:
                details = pytorch_module_params[name]
                print(f"    {name}: {details['shape']} ({details['count']} 参数)")
            if len(only_pytorch) > 10:
                print(f"    ... 还有 {len(only_pytorch) - 10} 个")
        
        if only_jittor:
            print(f"\n  只在Jittor中的参数:")
            for name in sorted(only_jittor)[:10]:
                details = jittor_module_params[name]
                print(f"    {name}: {details['shape']} ({details['count']} 参数)")
            if len(only_jittor) > 10:
                print(f"    ... 还有 {len(only_jittor) - 10} 个")
        
        # 检查形状不匹配的参数
        shape_mismatches = []
        for name in common:
            pytorch_shape = pytorch_module_params[name]['shape']
            jittor_shape = jittor_module_params[name]['shape']
            if pytorch_shape != jittor_shape:
                shape_mismatches.append({
                    'name': name,
                    'pytorch_shape': pytorch_shape,
                    'jittor_shape': jittor_shape,
                    'pytorch_count': pytorch_module_params[name]['count'],
                    'jittor_count': jittor_module_params[name]['count']
                })
        
        if shape_mismatches:
            print(f"\n  形状不匹配的参数:")
            for mismatch in shape_mismatches[:10]:
                print(f"    {mismatch['name']}:")
                print(f"      PyTorch: {mismatch['pytorch_shape']} ({mismatch['pytorch_count']} 参数)")
                print(f"      Jittor: {mismatch['jittor_shape']} ({mismatch['jittor_count']} 参数)")
    
    return False


def main():
    """主函数"""
    print("🚀 开始精确查找29,116参数差异")
    print("目标: 100%参数对齐，不允许任何差异")
    
    success = analyze_parameter_differences()
    
    if success:
        print("\n✅ 参数100%对齐成功！")
    else:
        print("\n❌ 仍有参数差异，需要继续修复")
    
    return success


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
