#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
精确找出最后8个参数的差异
"""

import json
import sys
from collections import defaultdict

# 添加路径
sys.path.append('/home/kyc/project/nanodet/nanodet-jittor')
from nanodet.model.arch.nanodet_plus import NanoDetPlus


def load_pytorch_record():
    """加载PyTorch参考记录"""
    try:
        with open('pytorch_reference_record.json', 'r') as f:
            record = json.load(f)
        return record
    except FileNotFoundError:
        print("❌ PyTorch参考记录文件不存在")
        return None


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
    model = NanoDetPlus(backbone_cfg, fpn_cfg, aux_head_cfg, head_cfg)
    
    return model


def filter_trainable_parameters(params_dict):
    """过滤出可训练参数，排除BatchNorm统计参数"""
    filtered_params = {}
    
    for name, details in params_dict.items():
        # 排除BatchNorm的统计参数
        if 'running_mean' in name or 'running_var' in name or 'num_batches_tracked' in name:
            continue
        
        # 只保留可训练参数
        if 'requires_grad' in details:
            if details['requires_grad']:
                filtered_params[name] = details
        else:
            # 对于Jittor，默认都是可训练的
            filtered_params[name] = details
    
    return filtered_params


def find_8_parameter_difference():
    """找出最后8个参数的差异"""
    print("🔍 精确找出最后8个参数的差异")
    print("=" * 80)
    
    # 加载PyTorch记录
    pytorch_record = load_pytorch_record()
    if not pytorch_record:
        return False
    
    pytorch_params = pytorch_record['model_params']
    
    # 过滤PyTorch可训练参数
    pytorch_trainable = filter_trainable_parameters(pytorch_params)
    
    # 创建Jittor模型
    jittor_model = create_jittor_model()
    
    # 统计Jittor可训练参数
    jittor_params = {}
    
    for name, param in jittor_model.named_parameters():
        # 排除BatchNorm统计参数
        if 'running_mean' in name or 'running_var' in name or 'num_batches_tracked' in name:
            continue
        
        count = param.numel() if hasattr(param, 'numel') else param.size
        jittor_params[name] = {
            'shape': list(param.shape),
            'count': count
        }
    
    # 重点分析head模块
    print(f"🔍 重点分析head模块:")
    
    # 提取head模块参数
    pytorch_head = {k: v for k, v in pytorch_trainable.items() if k.startswith('head')}
    jittor_head = {k: v for k, v in jittor_params.items() if k.startswith('head')}
    
    pytorch_head_total = sum(v['count'] for v in pytorch_head.values())
    jittor_head_total = sum(v['count'] for v in jittor_head.values())
    
    print(f"  PyTorch head参数: {pytorch_head_total:,} ({len(pytorch_head)} 项)")
    print(f"  Jittor head参数: {jittor_head_total:,} ({len(jittor_head)} 项)")
    print(f"  差异: {jittor_head_total - pytorch_head_total} 参数")
    
    # 找出参数名差异
    pytorch_head_names = set(pytorch_head.keys())
    jittor_head_names = set(jittor_head.keys())
    
    common = pytorch_head_names & jittor_head_names
    only_pytorch = pytorch_head_names - jittor_head_names
    only_jittor = jittor_head_names - pytorch_head_names
    
    print(f"\n📊 head模块参数对比:")
    print(f"  共同参数: {len(common)}")
    print(f"  只在PyTorch中: {len(only_pytorch)}")
    print(f"  只在Jittor中: {len(only_jittor)}")
    
    if only_pytorch:
        print(f"\n  只在PyTorch中的参数:")
        only_pytorch_total = 0
        for name in sorted(only_pytorch):
            details = pytorch_head[name]
            only_pytorch_total += details['count']
            print(f"    {name}: {details['shape']} ({details['count']} 参数)")
        print(f"  只在PyTorch中的参数总计: {only_pytorch_total}")
    
    if only_jittor:
        print(f"\n  只在Jittor中的参数:")
        only_jittor_total = 0
        for name in sorted(only_jittor):
            details = jittor_head[name]
            only_jittor_total += details['count']
            print(f"    {name}: {details['shape']} ({details['count']} 参数)")
        print(f"  只在Jittor中的参数总计: {only_jittor_total}")
    
    # 检查形状不匹配
    shape_mismatches = []
    for name in common:
        pytorch_shape = pytorch_head[name]['shape']
        jittor_shape = jittor_head[name]['shape']
        if pytorch_shape != jittor_shape:
            shape_mismatches.append({
                'name': name,
                'pytorch_shape': pytorch_shape,
                'jittor_shape': jittor_shape,
                'pytorch_count': pytorch_head[name]['count'],
                'jittor_count': jittor_head[name]['count']
            })
    
    if shape_mismatches:
        print(f"\n  形状不匹配的参数:")
        for mismatch in shape_mismatches:
            print(f"    {mismatch['name']}:")
            print(f"      PyTorch: {mismatch['pytorch_shape']} ({mismatch['pytorch_count']} 参数)")
            print(f"      Jittor: {mismatch['jittor_shape']} ({mismatch['jittor_count']} 参数)")
            print(f"      差异: {mismatch['jittor_count'] - mismatch['pytorch_count']} 参数")
    
    # 详细列出所有head参数
    print(f"\n📋 所有head参数详细对比:")
    print(f"{'参数名':<50} {'PyTorch形状':<20} {'Jittor形状':<20} {'差异'}")
    print("-" * 100)
    
    all_head_names = sorted(pytorch_head_names | jittor_head_names)
    total_diff = 0
    
    for name in all_head_names:
        pytorch_info = pytorch_head.get(name, {'shape': 'N/A', 'count': 0})
        jittor_info = jittor_head.get(name, {'shape': 'N/A', 'count': 0})
        
        pytorch_shape = str(pytorch_info['shape'])
        jittor_shape = str(jittor_info['shape'])
        diff = jittor_info['count'] - pytorch_info['count']
        total_diff += diff
        
        status = "✅" if diff == 0 else "❌"
        print(f"{name:<50} {pytorch_shape:<20} {jittor_shape:<20} {diff:+4d} {status}")
    
    print("-" * 100)
    print(f"{'总计':<50} {'':<20} {'':<20} {total_diff:+4d}")
    
    if total_diff == 0:
        print("\n🎉 head模块参数100%对齐！")
        return True
    else:
        print(f"\n❌ head模块仍有 {total_diff} 个参数差异")
        return False


def main():
    """主函数"""
    print("🚀 开始精确查找最后8个参数差异")
    
    success = find_8_parameter_difference()
    
    if success:
        print("\n✅ 找到并修复了所有参数差异！")
    else:
        print("\n❌ 仍需要继续修复参数差异")
    
    return success


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
