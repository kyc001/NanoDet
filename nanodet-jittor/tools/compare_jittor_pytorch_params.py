#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
逐个对比Jittor和PyTorch模型参数
使用转换脚本进行精确检查
"""

import os
import sys
import json
import jittor as jt
import torch

# 添加路径
sys.path.append('/home/kyc/project/nanodet/nanodet-jittor')
from nanodet.model.arch.nanodet_plus import NanoDetPlus


def load_jittor_model():
    """加载Jittor模型"""
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

    # 创建aux_head配置
    aux_head_cfg = {
        'name': 'NanoDetPlusHead',
        'num_classes': 20,
        'input_channel': 192,  # 96 * 2
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

    # 创建完整模型
    model = NanoDetPlus(backbone_cfg, fpn_cfg, aux_head_cfg, head_cfg)

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Jittor模型参数数量: {total_params:,}")

    return model, None


def analyze_jittor_model_structure(model):
    """分析Jittor模型结构"""
    print("\n" + "=" * 80)
    print("🔍 分析Jittor模型结构")
    print("=" * 80)
    
    jittor_params = {}
    
    # 获取所有参数
    for name, param in model.named_parameters():
        jittor_params[name] = {
            "shape": list(param.shape),
            "numel": param.numel(),
            "dtype": str(param.dtype),
        }
        
        # 计算统计信息
        if param.dtype in [jt.float32, jt.float64, jt.float16]:
            jittor_params[name].update({
                "mean": float(param.mean()),
                "std": float(param.std()),
                "min": float(param.min()),
                "max": float(param.max())
            })
    
    print(f"✓ Jittor模型总参数数: {len(jittor_params)}")
    
    # 按模块分类
    modules = {"backbone": {}, "fpn": {}, "head": {}, "other": {}}
    
    for name, info in jittor_params.items():
        if "backbone" in name:
            modules["backbone"][name] = info
        elif "fpn" in name:
            modules["fpn"][name] = info
        elif "head" in name:
            modules["head"][name] = info
        else:
            modules["other"][name] = info
    
    # 打印模块统计
    for module_name, module_params in modules.items():
        if module_params:
            total_params = sum(p["numel"] for p in module_params.values())
            print(f"\n🔹 {module_name.upper()}:")
            print(f"   参数数量: {len(module_params)}")
            print(f"   总参数量: {total_params:,}")
            
            # 显示关键参数
            for param_name, param_info in list(module_params.items())[:3]:
                print(f"   {param_name}: {param_info['shape']}")
    
    return jittor_params


def compare_key_parameters():
    """对比关键参数"""
    print("\n" + "=" * 80)
    print("🔍 对比关键参数")
    print("=" * 80)
    
    # 加载PyTorch分析结果
    pytorch_analysis_file = "pytorch_model_analysis.json"
    if not os.path.exists(pytorch_analysis_file):
        print("❌ 请先运行 analyze_pytorch_model_detailed.py")
        return False
    
    with open(pytorch_analysis_file, 'r') as f:
        pytorch_analysis = json.load(f)
    
    # 加载Jittor模型
    jittor_model, cfg = load_jittor_model()
    jittor_params = analyze_jittor_model_structure(jittor_model)
    
    print("\n" + "=" * 80)
    print("🔍 关键参数对比")
    print("=" * 80)
    
    # 对比输出层
    print("\n🔹 输出层对比:")
    
    # PyTorch输出层
    pytorch_head = pytorch_analysis.get("head", {})
    pytorch_output_layers = {}
    for name, info in pytorch_head.items():
        if "gfl_cls" in name and "weight" in name:
            pytorch_output_layers[name] = info["shape"]
    
    print("PyTorch输出层:")
    for name, shape in pytorch_output_layers.items():
        print(f"   {name}: {shape}")
    
    # Jittor输出层
    jittor_output_layers = {}
    for name, info in jittor_params.items():
        if "gfl_cls" in name and "weight" in name:
            jittor_output_layers[name] = info["shape"]
    
    print("\nJittor输出层:")
    for name, shape in jittor_output_layers.items():
        print(f"   {name}: {shape}")
    
    # 对比backbone第一层
    print("\n🔹 Backbone第一层对比:")
    
    # PyTorch backbone
    pytorch_backbone = pytorch_analysis.get("backbone", {})
    pytorch_first_conv = None
    for name, info in pytorch_backbone.items():
        if "conv1.0.weight" in name:
            pytorch_first_conv = (name, info["shape"])
            break
    
    if pytorch_first_conv:
        print(f"PyTorch: {pytorch_first_conv[0]} -> {pytorch_first_conv[1]}")
    
    # Jittor backbone
    jittor_first_conv = None
    for name, info in jittor_params.items():
        if "backbone" in name and "conv" in name and "weight" in name:
            jittor_first_conv = (name, info["shape"])
            break
    
    if jittor_first_conv:
        print(f"Jittor: {jittor_first_conv[0]} -> {jittor_first_conv[1]}")
    
    # 检查匹配情况
    print("\n🔹 匹配检查:")
    
    if pytorch_first_conv and jittor_first_conv:
        if pytorch_first_conv[1] == jittor_first_conv[1]:
            print("✅ Backbone第一层形状匹配")
        else:
            print("❌ Backbone第一层形状不匹配")
            print(f"   PyTorch: {pytorch_first_conv[1]}")
            print(f"   Jittor: {jittor_first_conv[1]}")
    
    # 检查输出层匹配
    if len(pytorch_output_layers) > 0 and len(jittor_output_layers) > 0:
        pytorch_shapes = list(pytorch_output_layers.values())
        jittor_shapes = list(jittor_output_layers.values())
        
        if len(pytorch_shapes) == len(jittor_shapes):
            all_match = all(p == j for p, j in zip(pytorch_shapes, jittor_shapes))
            if all_match:
                print("✅ 输出层形状完全匹配")
            else:
                print("❌ 输出层形状不匹配")
                for i, (p, j) in enumerate(zip(pytorch_shapes, jittor_shapes)):
                    if p != j:
                        print(f"   层{i}: PyTorch{p} vs Jittor{j}")
        else:
            print("❌ 输出层数量不匹配")
            print(f"   PyTorch: {len(pytorch_shapes)} 层")
            print(f"   Jittor: {len(jittor_shapes)} 层")
    
    return True


def test_model_output():
    """测试模型输出形状"""
    print("\n" + "=" * 80)
    print("🔍 测试模型输出形状")
    print("=" * 80)
    
    # 加载Jittor模型
    jittor_model, cfg = load_jittor_model()
    
    # 创建测试输入
    test_input = jt.randn(1, 3, 320, 320)
    
    print("进行前向推理...")
    with jt.no_grad():
        output = jittor_model(test_input)
    
    print(f"✓ 输出形状: {output.shape}")
    print(f"✓ 输出数据类型: {output.dtype}")
    
    # 分析输出通道
    if len(output.shape) == 3:  # [B, N, C]
        batch_size, num_anchors, num_channels = output.shape
        print(f"✓ 批次大小: {batch_size}")
        print(f"✓ 锚点数量: {num_anchors}")
        print(f"✓ 输出通道: {num_channels}")
        
        # 分析通道分配
        expected_cls_channels = 20  # VOC 20类
        expected_reg_channels = 32  # 4 * (7+1) = 32
        expected_total = expected_cls_channels + expected_reg_channels
        
        print(f"\n🔹 通道分析:")
        print(f"   期望分类通道: {expected_cls_channels}")
        print(f"   期望回归通道: {expected_reg_channels}")
        print(f"   期望总通道: {expected_total}")
        print(f"   实际总通道: {num_channels}")
        
        if num_channels == expected_total:
            print("✅ 输出通道数正确")
        else:
            print("❌ 输出通道数不正确")
            print(f"   差异: {num_channels - expected_total}")
    
    return True


def main():
    """主函数"""
    print("🚀 开始对比Jittor和PyTorch模型参数")
    
    # 对比关键参数
    success = compare_key_parameters()
    
    if success:
        # 测试模型输出
        test_model_output()
        print("\n🎉 参数对比完成!")
    else:
        print("\n❌ 参数对比失败")
        return False
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
