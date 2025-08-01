#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
验证100%参数对齐
排除BatchNorm统计参数，只统计可训练参数
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


def verify_100_percent_alignment():
    """验证100%参数对齐"""
    print("🔍 验证100%参数对齐")
    print("=" * 80)
    
    # 加载PyTorch记录
    pytorch_record = load_pytorch_record()
    if not pytorch_record:
        return False
    
    pytorch_params = pytorch_record['model_params']
    
    # 过滤PyTorch可训练参数
    pytorch_trainable = filter_trainable_parameters(pytorch_params)
    
    # 统计PyTorch可训练参数
    pytorch_total = sum(details['count'] for details in pytorch_trainable.values())
    pytorch_modules = defaultdict(int)
    
    for name, details in pytorch_trainable.items():
        count = details['count']
        module = name.split('.')[0]
        pytorch_modules[module] += count
    
    print(f"✓ PyTorch可训练参数: {pytorch_total:,} 参数, {len(pytorch_trainable)} 项")
    
    # 创建Jittor模型
    jittor_model = create_jittor_model()
    
    # 统计Jittor可训练参数
    jittor_params = {}
    jittor_total = 0
    jittor_modules = defaultdict(int)
    
    for name, param in jittor_model.named_parameters():
        # 排除BatchNorm统计参数
        if 'running_mean' in name or 'running_var' in name or 'num_batches_tracked' in name:
            continue
        
        count = param.numel() if hasattr(param, 'numel') else param.size
        jittor_params[name] = {
            'shape': list(param.shape),
            'count': count
        }
        jittor_total += count
        
        module = name.split('.')[0]
        jittor_modules[module] += count
    
    print(f"✓ Jittor可训练参数: {jittor_total:,} 参数, {len(jittor_params)} 项")
    
    # 计算差异
    difference = abs(pytorch_total - jittor_total)
    print(f"\n📊 可训练参数差异: {difference:,} ({difference/pytorch_total*100:.6f}%)")
    
    if difference == 0:
        print("🎉 可训练参数100%对齐！")
        
        # 进一步验证参数名和形状
        print(f"\n🔍 验证参数名和形状对齐:")
        
        pytorch_names = set(pytorch_trainable.keys())
        jittor_names = set(jittor_params.keys())
        
        common = pytorch_names & jittor_names
        only_pytorch = pytorch_names - jittor_names
        only_jittor = jittor_names - pytorch_names
        
        print(f"  共同参数: {len(common)}")
        print(f"  只在PyTorch中: {len(only_pytorch)}")
        print(f"  只在Jittor中: {len(only_jittor)}")
        
        if len(only_pytorch) == 0 and len(only_jittor) == 0:
            print("✅ 参数名100%对齐！")
            
            # 检查形状对齐
            shape_mismatches = 0
            for name in common:
                pytorch_shape = pytorch_trainable[name]['shape']
                jittor_shape = jittor_params[name]['shape']
                if pytorch_shape != jittor_shape:
                    shape_mismatches += 1
                    print(f"❌ 形状不匹配: {name}")
                    print(f"   PyTorch: {pytorch_shape}")
                    print(f"   Jittor: {jittor_shape}")
            
            if shape_mismatches == 0:
                print("✅ 参数形状100%对齐！")
                print("\n🎉 恭喜！实现了100%完美对齐！")
                return True
            else:
                print(f"❌ 有 {shape_mismatches} 个参数形状不匹配")
        else:
            if only_pytorch:
                print(f"\n  只在PyTorch中的参数:")
                for name in sorted(only_pytorch)[:10]:
                    details = pytorch_trainable[name]
                    print(f"    {name}: {details['shape']} ({details['count']} 参数)")
            
            if only_jittor:
                print(f"\n  只在Jittor中的参数:")
                for name in sorted(only_jittor)[:10]:
                    details = jittor_params[name]
                    print(f"    {name}: {details['shape']} ({details['count']} 参数)")
        
        return False
    
    # 如果还有差异，继续分析
    print(f"\n📊 按模块对比可训练参数:")
    print(f"{'模块':<20} {'PyTorch':<12} {'Jittor':<12} {'差异':<12}")
    print("-" * 60)
    
    all_modules = set(pytorch_modules.keys()) | set(jittor_modules.keys())
    
    for module in sorted(all_modules):
        pytorch_count = pytorch_modules.get(module, 0)
        jittor_count = jittor_modules.get(module, 0)
        diff = pytorch_count - jittor_count
        
        status = "✅" if diff == 0 else "❌"
        print(f"{module:<20} {pytorch_count:<12,} {jittor_count:<12,} {diff:<12,} {status}")
    
    print("-" * 60)
    print(f"{'总计':<20} {pytorch_total:<12,} {jittor_total:<12,} {pytorch_total-jittor_total:<12,}")
    
    return False


def main():
    """主函数"""
    print("🚀 开始验证100%参数对齐")
    print("目标: 可训练参数100%对齐，不允许任何差异")
    
    success = verify_100_percent_alignment()
    
    if success:
        print("\n✅ 100%参数对齐验证成功！")
        print("🎉 模型结构完全正确，可以进行最终mAP测试！")
    else:
        print("\n❌ 仍有参数差异，需要继续修复")
    
    return success


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
