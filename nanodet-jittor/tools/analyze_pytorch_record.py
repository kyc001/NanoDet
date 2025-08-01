#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
分析PyTorch参考记录，找出29,116参数差异
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


def analyze_parameter_differences():
    """分析参数差异"""
    print("🔍 分析参数差异")
    print("=" * 80)
    
    # 加载PyTorch记录
    pytorch_record = load_pytorch_record()
    if not pytorch_record:
        return False
    
    pytorch_params = pytorch_record['model_params']
    
    # 统计PyTorch参数
    pytorch_total = 0
    pytorch_modules = defaultdict(int)
    
    for name, details in pytorch_params.items():
        count = details['count']
        pytorch_total += count
        
        module = name.split('.')[0]
        pytorch_modules[module] += count
    
    print(f"✓ PyTorch模型: {pytorch_total:,} 参数, {len(pytorch_params)} 项")
    
    # 创建Jittor模型
    jittor_model = create_jittor_model()
    
    # 统计Jittor参数
    jittor_params = {}
    jittor_total = 0
    jittor_modules = defaultdict(int)
    
    for name, param in jittor_model.named_parameters():
        count = param.numel() if hasattr(param, 'numel') else param.size
        jittor_params[name] = {
            'shape': list(param.shape),
            'count': count
        }
        jittor_total += count
        
        module = name.split('.')[0]
        jittor_modules[module] += count
    
    print(f"✓ Jittor模型: {jittor_total:,} 参数, {len(jittor_params)} 项")
    
    # 计算差异
    difference = abs(pytorch_total - jittor_total)
    print(f"\n📊 参数差异: {difference:,} ({difference/pytorch_total*100:.3f}%)")
    
    if difference == 0:
        print("🎉 参数数量100%对齐！")
        return True
    
    # 按模块对比
    all_modules = set(pytorch_modules.keys()) | set(jittor_modules.keys())
    
    print(f"\n📊 按模块对比:")
    print(f"{'模块':<20} {'PyTorch':<12} {'Jittor':<12} {'差异':<12}")
    print("-" * 60)
    
    module_differences = {}
    for module in sorted(all_modules):
        pytorch_count = pytorch_modules.get(module, 0)
        jittor_count = jittor_modules.get(module, 0)
        diff = pytorch_count - jittor_count
        module_differences[module] = diff
        
        status = "✅" if diff == 0 else "❌"
        print(f"{module:<20} {pytorch_count:<12,} {jittor_count:<12,} {diff:<12,} {status}")
    
    print("-" * 60)
    print(f"{'总计':<20} {pytorch_total:<12,} {jittor_total:<12,} {pytorch_total-jittor_total:<12,}")
    
    # 找出差异最大的模块
    max_diff_module = max(module_differences.keys(), key=lambda x: abs(module_differences[x]))
    max_diff = abs(module_differences[max_diff_module])
    
    print(f"\n🎯 差异最大的模块: {max_diff_module} (差异: {module_differences[max_diff_module]:,} 参数)")
    
    # 详细分析差异最大的模块
    if max_diff > 0:
        print(f"\n🔍 详细分析 {max_diff_module} 模块:")
        
        # PyTorch该模块的参数
        pytorch_module_params = {k: v for k, v in pytorch_params.items() if k.startswith(max_diff_module)}
        jittor_module_params = {k: v for k, v in jittor_params.items() if k.startswith(max_diff_module)}
        
        print(f"  PyTorch {max_diff_module} 参数项: {len(pytorch_module_params)}")
        print(f"  Jittor {max_diff_module} 参数项: {len(jittor_module_params)}")
        
        # 找出差异
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
            only_pytorch_total = 0
            for name in sorted(only_pytorch)[:20]:
                details = pytorch_module_params[name]
                only_pytorch_total += details['count']
                print(f"    {name}: {details['shape']} ({details['count']} 参数)")
            if len(only_pytorch) > 20:
                remaining = len(only_pytorch) - 20
                remaining_total = sum(pytorch_module_params[name]['count'] for name in list(only_pytorch)[20:])
                print(f"    ... 还有 {remaining} 个参数 ({remaining_total:,} 参数)")
                only_pytorch_total += remaining_total
            print(f"  只在PyTorch中的参数总计: {only_pytorch_total:,}")
        
        if only_jittor:
            print(f"\n  只在Jittor中的参数:")
            only_jittor_total = 0
            for name in sorted(only_jittor)[:20]:
                details = jittor_module_params[name]
                only_jittor_total += details['count']
                print(f"    {name}: {details['shape']} ({details['count']} 参数)")
            if len(only_jittor) > 20:
                remaining = len(only_jittor) - 20
                remaining_total = sum(jittor_module_params[name]['count'] for name in list(only_jittor)[20:])
                print(f"    ... 还有 {remaining} 个参数 ({remaining_total:,} 参数)")
                only_jittor_total += remaining_total
            print(f"  只在Jittor中的参数总计: {only_jittor_total:,}")
        
        # 检查形状不匹配
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
            for mismatch in shape_mismatches:
                print(f"    {mismatch['name']}:")
                print(f"      PyTorch: {mismatch['pytorch_shape']} ({mismatch['pytorch_count']} 参数)")
                print(f"      Jittor: {mismatch['jittor_shape']} ({mismatch['jittor_count']} 参数)")
    
    # 分析所有模块的差异
    print(f"\n📊 所有模块差异汇总:")
    for module in sorted(module_differences.keys(), key=lambda x: abs(module_differences[x]), reverse=True):
        diff = module_differences[module]
        if diff != 0:
            print(f"  {module}: {diff:+,} 参数")
    
    return False


def main():
    """主函数"""
    print("🚀 开始分析29,116参数差异")
    print("目标: 100%参数对齐，不允许任何差异")
    
    success = analyze_parameter_differences()
    
    if success:
        print("\n✅ 参数100%对齐成功！")
    else:
        print("\n❌ 仍有参数差异，需要继续修复")
        print("\n💡 修复建议:")
        print("1. 检查模块结构是否完全一致")
        print("2. 检查配置参数是否完全对齐")
        print("3. 检查是否有遗漏的层或参数")
    
    return success


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
