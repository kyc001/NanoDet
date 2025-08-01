#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
深度分析权重加载失败的原因
找出每个失败参数的具体问题
"""

import os
import sys
import torch
import jittor as jt
from collections import defaultdict

# 添加路径
sys.path.append('/home/kyc/project/nanodet/nanodet-jittor')
from nanodet.model.arch.nanodet_plus import NanoDetPlus


def create_nanodet_model():
    """创建NanoDet模型"""
    print("创建NanoDet模型...")
    
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
    model = NanoDetPlus(backbone_cfg, fpn_cfg, aux_head_cfg, head_cfg)
    
    return model


def analyze_parameter_mismatch():
    """深度分析参数不匹配的原因"""
    print("🔍 深度分析参数不匹配原因")
    print("=" * 80)
    
    # 加载PyTorch checkpoint
    checkpoint_path = "/home/kyc/project/nanodet/nanodet-pytorch/workspace/nanodet-plus-m_320_voc_bs64/model_best/model_best.ckpt"
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    pytorch_state_dict = checkpoint.get('state_dict', checkpoint)
    
    print(f"✓ PyTorch checkpoint包含 {len(pytorch_state_dict)} 个参数")
    
    # 创建Jittor模型
    jittor_model = create_nanodet_model()
    
    # 获取Jittor模型的参数字典
    jittor_state_dict = {}
    for name, param in jittor_model.named_parameters():
        jittor_state_dict[name] = param
    
    print(f"✓ Jittor模型包含 {len(jittor_state_dict)} 个参数")
    
    # 分类分析失败原因
    analysis_result = {
        "成功加载": [],
        "跳过_BatchNorm统计": [],
        "跳过_权重平均": [],
        "跳过_aux_head": [],
        "形状不匹配": [],
        "参数名不存在": [],
        "其他错误": []
    }
    
    for pytorch_name, pytorch_param in pytorch_state_dict.items():
        # 移除PyTorch特有的前缀
        jittor_name = pytorch_name
        if jittor_name.startswith("model."):
            jittor_name = jittor_name[6:]  # 移除"model."前缀
        
        # 分类处理
        if "num_batches_tracked" in jittor_name:
            analysis_result["跳过_BatchNorm统计"].append({
                "pytorch_name": pytorch_name,
                "jittor_name": jittor_name,
                "shape": list(pytorch_param.shape)
            })
            continue
        
        if jittor_name.startswith("avg_"):
            analysis_result["跳过_权重平均"].append({
                "pytorch_name": pytorch_name,
                "jittor_name": jittor_name,
                "shape": list(pytorch_param.shape)
            })
            continue
        
        # 不再跳过aux_head，现在应该能正确匹配
        # if "aux_head" in jittor_name:
        #     analysis_result["跳过_aux_head"].append({
        #         "pytorch_name": pytorch_name,
        #         "jittor_name": jittor_name,
        #         "shape": list(pytorch_param.shape)
        #     })
        #     continue
        
        if jittor_name in jittor_state_dict:
            jittor_param = jittor_state_dict[jittor_name]
            
            # 检查形状匹配
            if list(pytorch_param.shape) == list(jittor_param.shape):
                analysis_result["成功加载"].append({
                    "pytorch_name": pytorch_name,
                    "jittor_name": jittor_name,
                    "shape": list(pytorch_param.shape)
                })
            else:
                analysis_result["形状不匹配"].append({
                    "pytorch_name": pytorch_name,
                    "jittor_name": jittor_name,
                    "pytorch_shape": list(pytorch_param.shape),
                    "jittor_shape": list(jittor_param.shape)
                })
        else:
            analysis_result["参数名不存在"].append({
                "pytorch_name": pytorch_name,
                "jittor_name": jittor_name,
                "shape": list(pytorch_param.shape)
            })
    
    # 打印详细分析结果
    print("\n" + "=" * 80)
    print("📊 详细分析结果")
    print("=" * 80)
    
    for category, items in analysis_result.items():
        print(f"\n🔹 {category}: {len(items)} 个")
        
        if len(items) > 0:
            print(f"   前5个示例:")
            for i, item in enumerate(items[:5]):
                if category == "形状不匹配":
                    print(f"   {i+1}. {item['jittor_name']}")
                    print(f"      PyTorch: {item['pytorch_shape']}")
                    print(f"      Jittor: {item['jittor_shape']}")
                else:
                    print(f"   {i+1}. {item['jittor_name']} -> {item['shape']}")
            
            if len(items) > 5:
                print(f"   ... 还有 {len(items) - 5} 个")
    
    # 重点分析：参数名不存在的问题
    print("\n" + "=" * 80)
    print("🚨 重点分析：参数名不存在的问题")
    print("=" * 80)
    
    missing_params = analysis_result["参数名不存在"]
    if len(missing_params) > 0:
        # 按模块分组
        module_groups = defaultdict(list)
        for param in missing_params:
            module_name = param['jittor_name'].split('.')[0]
            module_groups[module_name].append(param)
        
        for module_name, params in module_groups.items():
            print(f"\n🔸 {module_name} 模块缺失: {len(params)} 个参数")
            for param in params[:3]:  # 只显示前3个
                print(f"   - {param['jittor_name']}")
            if len(params) > 3:
                print(f"   ... 还有 {len(params) - 3} 个")
    
    # 重点分析：形状不匹配的问题
    print("\n" + "=" * 80)
    print("🚨 重点分析：形状不匹配的问题")
    print("=" * 80)
    
    shape_mismatch = analysis_result["形状不匹配"]
    if len(shape_mismatch) > 0:
        for item in shape_mismatch:
            print(f"\n❌ {item['jittor_name']}")
            print(f"   PyTorch: {item['pytorch_shape']}")
            print(f"   Jittor: {item['jittor_shape']}")
            print(f"   差异: {[p-j for p, j in zip(item['pytorch_shape'], item['jittor_shape']) if len(item['pytorch_shape']) == len(item['jittor_shape'])]}")
    
    return analysis_result


def compare_jittor_pytorch_parameters():
    """对比Jittor和PyTorch的参数名"""
    print("\n" + "=" * 80)
    print("🔍 对比Jittor和PyTorch的参数名")
    print("=" * 80)
    
    # 加载PyTorch checkpoint
    checkpoint_path = "/home/kyc/project/nanodet/nanodet-pytorch/workspace/nanodet-plus-m_320_voc_bs64/model_best/model_best.ckpt"
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    pytorch_state_dict = checkpoint.get('state_dict', checkpoint)
    
    # 创建Jittor模型
    jittor_model = create_nanodet_model()
    
    # 获取参数名列表
    pytorch_names = set()
    for name in pytorch_state_dict.keys():
        clean_name = name
        if clean_name.startswith("model."):
            clean_name = clean_name[6:]
        if not clean_name.startswith("avg_") and "num_batches_tracked" not in clean_name:
            pytorch_names.add(clean_name)
    
    jittor_names = set()
    for name, param in jittor_model.named_parameters():
        jittor_names.add(name)
    
    print(f"PyTorch有效参数名: {len(pytorch_names)} 个")
    print(f"Jittor参数名: {len(jittor_names)} 个")
    
    # 找出差异
    only_in_pytorch = pytorch_names - jittor_names
    only_in_jittor = jittor_names - pytorch_names
    common = pytorch_names & jittor_names
    
    print(f"\n✅ 共同参数: {len(common)} 个")
    print(f"❌ 只在PyTorch中: {len(only_in_pytorch)} 个")
    print(f"❌ 只在Jittor中: {len(only_in_jittor)} 个")
    
    if len(only_in_pytorch) > 0:
        print(f"\n🔸 只在PyTorch中的参数 (前10个):")
        for i, name in enumerate(sorted(only_in_pytorch)[:10]):
            print(f"   {i+1}. {name}")
    
    if len(only_in_jittor) > 0:
        print(f"\n🔸 只在Jittor中的参数 (前10个):")
        for i, name in enumerate(sorted(only_in_jittor)[:10]):
            print(f"   {i+1}. {name}")
    
    return {
        "common": common,
        "only_pytorch": only_in_pytorch,
        "only_jittor": only_in_jittor
    }


def main():
    """主函数"""
    print("🚀 开始深度分析参数加载失败原因")
    
    # 详细分析参数不匹配
    analysis_result = analyze_parameter_mismatch()
    
    # 对比参数名差异
    name_comparison = compare_jittor_pytorch_parameters()
    
    # 总结
    print("\n" + "=" * 80)
    print("📋 问题总结")
    print("=" * 80)
    
    total_failed = (len(analysis_result["跳过_BatchNorm统计"]) + 
                   len(analysis_result["跳过_权重平均"]) + 
                   len(analysis_result["跳过_aux_head"]) + 
                   len(analysis_result["形状不匹配"]) + 
                   len(analysis_result["参数名不存在"]))
    
    print(f"✅ 成功加载: {len(analysis_result['成功加载'])} 个")
    print(f"❌ 总失败数: {total_failed} 个")
    print(f"   - BatchNorm统计: {len(analysis_result['跳过_BatchNorm统计'])} 个 (可忽略)")
    print(f"   - 权重平均: {len(analysis_result['跳过_权重平均'])} 个 (可忽略)")
    print(f"   - aux_head: {len(analysis_result['跳过_aux_head'])} 个 (需修复)")
    print(f"   - 形状不匹配: {len(analysis_result['形状不匹配'])} 个 (需修复)")
    print(f"   - 参数名不存在: {len(analysis_result['参数名不存在'])} 个 (需修复)")
    
    critical_failures = (len(analysis_result["形状不匹配"]) + 
                        len(analysis_result["参数名不存在"]))
    
    print(f"\n🚨 关键失败数: {critical_failures} 个 (必须修复)")
    
    if critical_failures > 0:
        print("❌ 权重加载存在严重问题，需要立即修复！")
        return False
    else:
        print("✅ 权重加载基本正常")
        return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
