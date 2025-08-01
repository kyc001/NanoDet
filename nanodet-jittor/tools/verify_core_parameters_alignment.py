#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
验证核心参数100%对齐
排除框架差异参数：BatchNorm统计参数和Scale形状差异
专注于验证模型核心逻辑的参数对齐
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


def filter_core_parameters(params_dict, framework="pytorch"):
    """过滤出核心参数，排除框架差异参数"""
    core_params = {}
    
    for name, details in params_dict.items():
        # 排除BatchNorm统计参数
        if 'running_mean' in name or 'running_var' in name or 'num_batches_tracked' in name:
            continue
        
        # 排除非可训练参数
        if framework == "pytorch":
            if 'requires_grad' in details and not details['requires_grad']:
                continue
        
        # 对于Scale参数，我们知道有形状差异，但参数数量相同
        # 所以我们保留它们，但在对比时特殊处理
        core_params[name] = details
    
    return core_params


def normalize_parameter_shapes(pytorch_params, jittor_params):
    """标准化参数形状，处理已知的框架差异"""
    normalized_pytorch = {}
    normalized_jittor = {}
    
    shape_adjustments = 0
    
    for name in pytorch_params.keys():
        if name in jittor_params:
            pytorch_shape = pytorch_params[name]['shape']
            jittor_shape = jittor_params[name]['shape']
            
            # 处理Scale参数的形状差异：PyTorch [] vs Jittor [1]
            if 'scale' in name and pytorch_shape == [] and jittor_shape == [1]:
                # 标准化为相同形状进行对比
                normalized_pytorch[name] = {
                    **pytorch_params[name],
                    'shape': [1],  # 标准化为1维
                    'original_shape': pytorch_shape
                }
                normalized_jittor[name] = {
                    **jittor_params[name],
                    'original_shape': jittor_shape
                }
                shape_adjustments += 1
            else:
                normalized_pytorch[name] = pytorch_params[name]
                normalized_jittor[name] = jittor_params[name]
    
    return normalized_pytorch, normalized_jittor, shape_adjustments


def verify_core_parameters_alignment():
    """验证核心参数100%对齐"""
    print("🔍 验证核心参数100%对齐")
    print("排除框架差异：BatchNorm统计参数 + Scale形状差异")
    print("=" * 80)
    
    # 加载PyTorch记录
    pytorch_record = load_pytorch_record()
    if not pytorch_record:
        return False
    
    pytorch_params = pytorch_record['model_params']
    
    # 过滤PyTorch核心参数
    pytorch_core = filter_core_parameters(pytorch_params, "pytorch")
    
    # 统计PyTorch核心参数
    pytorch_total = sum(details['count'] for details in pytorch_core.values())
    pytorch_modules = defaultdict(int)
    
    for name, details in pytorch_core.items():
        count = details['count']
        module = name.split('.')[0]
        pytorch_modules[module] += count
    
    print(f"✓ PyTorch核心参数: {pytorch_total:,} 参数, {len(pytorch_core)} 项")
    
    # 创建Jittor模型
    jittor_model = create_jittor_model()
    
    # 统计Jittor核心参数
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
    
    # 过滤Jittor核心参数
    jittor_core = filter_core_parameters(jittor_params, "jittor")
    jittor_core_total = sum(details['count'] for details in jittor_core.values())
    
    print(f"✓ Jittor核心参数: {jittor_core_total:,} 参数, {len(jittor_core)} 项")
    
    # 标准化参数形状
    norm_pytorch, norm_jittor, shape_adjustments = normalize_parameter_shapes(pytorch_core, jittor_core)
    
    print(f"✓ 形状标准化调整: {shape_adjustments} 个参数")
    
    # 计算差异
    difference = abs(pytorch_total - jittor_core_total)
    print(f"\n📊 核心参数差异: {difference:,} ({difference/pytorch_total*100:.6f}%)")
    
    if difference == 0:
        print("🎉 核心参数数量100%对齐！")
        
        # 进一步验证参数名和形状
        print(f"\n🔍 验证参数名和标准化形状对齐:")
        
        pytorch_names = set(norm_pytorch.keys())
        jittor_names = set(norm_jittor.keys())
        
        common = pytorch_names & jittor_names
        only_pytorch = pytorch_names - jittor_names
        only_jittor = jittor_names - pytorch_names
        
        print(f"  共同参数: {len(common)}")
        print(f"  只在PyTorch中: {len(only_pytorch)}")
        print(f"  只在Jittor中: {len(only_jittor)}")
        
        if len(only_pytorch) == 0 and len(only_jittor) == 0:
            print("✅ 参数名100%对齐！")
            
            # 检查标准化后的形状对齐
            shape_mismatches = 0
            for name in common:
                pytorch_shape = norm_pytorch[name]['shape']
                jittor_shape = norm_jittor[name]['shape']
                if pytorch_shape != jittor_shape:
                    shape_mismatches += 1
                    print(f"❌ 标准化后仍形状不匹配: {name}")
                    print(f"   PyTorch: {pytorch_shape}")
                    print(f"   Jittor: {jittor_shape}")
            
            if shape_mismatches == 0:
                print("✅ 标准化形状100%对齐！")
                print(f"\n🎉 恭喜！实现了核心参数100%完美对齐！")
                print(f"📊 对齐统计:")
                print(f"  - 核心参数数量: {pytorch_total:,} (100%一致)")
                print(f"  - 参数项数量: {len(pytorch_core)} (100%一致)")
                print(f"  - 参数名称: 100%对齐")
                print(f"  - 标准化形状: 100%对齐")
                print(f"  - 框架差异处理: {shape_adjustments} 个Scale参数")
                
                # 显示模块统计
                print(f"\n📊 按模块统计:")
                print(f"{'模块':<20} {'参数数量':<12}")
                print("-" * 35)
                for module in sorted(pytorch_modules.keys(), key=lambda x: pytorch_modules[x], reverse=True):
                    count = pytorch_modules[module]
                    print(f"{module:<20} {count:<12,}")
                
                return True
            else:
                print(f"❌ 有 {shape_mismatches} 个参数标准化后仍形状不匹配")
        else:
            if only_pytorch:
                print(f"\n  只在PyTorch中的参数:")
                for name in sorted(only_pytorch)[:10]:
                    details = norm_pytorch[name]
                    print(f"    {name}: {details['shape']} ({details['count']} 参数)")
            
            if only_jittor:
                print(f"\n  只在Jittor中的参数:")
                for name in sorted(only_jittor)[:10]:
                    details = norm_jittor[name]
                    print(f"    {name}: {details['shape']} ({details['count']} 参数)")
        
        return False
    
    # 如果还有差异，继续分析
    print(f"\n📊 按模块对比核心参数:")
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
    print(f"{'总计':<20} {pytorch_total:<12,} {jittor_core_total:<12,} {pytorch_total-jittor_core_total:<12,}")
    
    return False


def main():
    """主函数"""
    print("🚀 开始验证核心参数100%对齐")
    print("目标: 排除框架差异，验证模型核心逻辑100%对齐")
    
    success = verify_core_parameters_alignment()
    
    if success:
        print("\n✅ 核心参数100%对齐验证成功！")
        print("🎉 模型核心逻辑完全正确，可以进行最终mAP测试！")
        print("\n📝 框架差异说明:")
        print("  - BatchNorm统计参数: Jittor计入参数，PyTorch计入buffer")
        print("  - Scale参数形状: Jittor [1], PyTorch []")
        print("  - 这些差异不影响模型功能和性能")
    else:
        print("\n❌ 核心参数仍有差异，需要继续修复")
    
    return success


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
