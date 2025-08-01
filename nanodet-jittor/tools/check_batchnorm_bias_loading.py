#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
检查BatchNorm bias的权重加载情况
确认bias是否被正确覆盖
"""

import os
import sys
import torch
import jittor as jt
import numpy as np

# 添加路径
sys.path.append('/home/kyc/project/nanodet/nanodet-jittor')
from nanodet.model.arch.nanodet_plus import NanoDetPlus


def check_batchnorm_bias_loading():
    """检查BatchNorm bias的权重加载"""
    print("🔍 检查BatchNorm bias的权重加载")
    print("=" * 60)
    
    # 创建模型配置
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
    
    print("1️⃣ 创建模型...")
    model = NanoDetPlus(backbone_cfg, fpn_cfg, aux_head_cfg, head_cfg)
    
    # 检查初始化后的BatchNorm bias
    print("\n2️⃣ 检查初始化后的BatchNorm bias...")
    initial_bias_values = {}
    for name, module in model.named_modules():
        if isinstance(module, jt.nn.BatchNorm2d) and hasattr(module, 'bias') and module.bias is not None:
            bias_value = module.bias.numpy().copy()
            initial_bias_values[name] = bias_value
            if len(initial_bias_values) <= 5:  # 只显示前5个
                print(f"  {name}: bias范围[{bias_value.min():.6f}, {bias_value.max():.6f}]")
    
    print(f"✓ 记录了 {len(initial_bias_values)} 个BatchNorm层的初始bias")
    
    # 加载PyTorch权重
    print("\n3️⃣ 加载PyTorch权重...")
    checkpoint_path = "/home/kyc/project/nanodet/nanodet-pytorch/workspace/nanodet-plus-m_320_voc_bs64/model_best/model_best.ckpt"
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)
    
    # 检查PyTorch权重中的BatchNorm bias
    print("\n4️⃣ 检查PyTorch权重中的BatchNorm bias...")
    pytorch_bias_values = {}
    for pytorch_name, pytorch_param in state_dict.items():
        if 'bias' in pytorch_name and ('bn' in pytorch_name.lower() or 'norm' in pytorch_name.lower() or '.1.' in pytorch_name):
            jittor_name = pytorch_name
            if jittor_name.startswith("model."):
                jittor_name = jittor_name[6:]
            
            pytorch_bias_values[jittor_name] = pytorch_param.detach().numpy()
            if len(pytorch_bias_values) <= 5:  # 只显示前5个
                bias_data = pytorch_param.detach().numpy()
                print(f"  {pytorch_name} -> {jittor_name}: 范围[{bias_data.min():.6f}, {bias_data.max():.6f}]")
    
    print(f"✓ 找到 {len(pytorch_bias_values)} 个PyTorch BatchNorm bias")
    
    # 获取Jittor模型的参数字典
    jittor_state_dict = {}
    for name, param in model.named_parameters():
        jittor_state_dict[name] = param
    
    print("\n5️⃣ 执行权重加载...")
    
    # 记录bias加载过程
    bias_loaded_count = 0
    bias_changes = {}
    
    for pytorch_name, pytorch_param in state_dict.items():
        jittor_name = pytorch_name
        if jittor_name.startswith("model."):
            jittor_name = jittor_name[6:]
        
        if "num_batches_tracked" in jittor_name or jittor_name.startswith("avg_"):
            continue
        
        if "distribution_project.project" in jittor_name:
            continue
        
        if jittor_name in jittor_state_dict:
            jittor_param = jittor_state_dict[jittor_name]
            
            # 记录bias参数的加载
            if 'bias' in jittor_name and ('bn' in jittor_name.lower() or 'norm' in jittor_name.lower() or '.1.' in jittor_name):
                before_data = jittor_param.numpy().copy()
                
                if list(pytorch_param.shape) == list(jittor_param.shape):
                    jittor_param.assign(jt.array(pytorch_param.detach().numpy()))
                    bias_loaded_count += 1
                    
                    after_data = jittor_param.numpy().copy()
                    pytorch_data = pytorch_param.detach().numpy()
                    
                    bias_changes[jittor_name] = {
                        'before': before_data,
                        'after': after_data,
                        'pytorch': pytorch_data,
                        'diff': np.abs(after_data - pytorch_data).max()
                    }
                    
                    if bias_loaded_count <= 5:  # 只显示前5个
                        print(f"\n  加载 {jittor_name}:")
                        print(f"    加载前: 范围[{before_data.min():.6f}, {before_data.max():.6f}]")
                        print(f"    PyTorch: 范围[{pytorch_data.min():.6f}, {pytorch_data.max():.6f}]")
                        print(f"    加载后: 范围[{after_data.min():.6f}, {after_data.max():.6f}]")
                        print(f"    差异: {np.abs(after_data - pytorch_data).max():.10f}")
            
            elif list(pytorch_param.shape) == list(jittor_param.shape):
                jittor_param.assign(jt.array(pytorch_param.detach().numpy()))
            elif "scale" in jittor_name and len(pytorch_param.shape) == 0 and list(jittor_param.shape) == [1]:
                jittor_param.assign(jt.array([pytorch_param.detach().numpy()]))
    
    print(f"\n✓ 加载了 {bias_loaded_count} 个BatchNorm bias参数")
    
    # 验证最终的bias状态
    print("\n6️⃣ 验证最终的bias状态...")
    
    final_bias_check = {}
    for name, module in model.named_modules():
        if isinstance(module, jt.nn.BatchNorm2d) and hasattr(module, 'bias') and module.bias is not None:
            final_bias = module.bias.numpy().copy()
            final_bias_check[name] = final_bias
            
            if len(final_bias_check) <= 5:  # 只显示前5个
                print(f"  {name}: 最终bias范围[{final_bias.min():.6f}, {final_bias.max():.6f}]")
                
                # 检查是否还是初始值0.0001
                if np.allclose(final_bias, 0.0001):
                    print(f"    ❌ 仍然是初始值0.0001，权重加载失败")
                else:
                    print(f"    ✅ 已被正确覆盖")
    
    # 统计分析
    print(f"\n📊 统计分析:")
    
    still_initial = 0
    properly_loaded = 0
    
    for name, final_bias in final_bias_check.items():
        if np.allclose(final_bias, 0.0001):
            still_initial += 1
        else:
            properly_loaded += 1
    
    print(f"  仍为初始值(0.0001): {still_initial}")
    print(f"  正确加载: {properly_loaded}")
    print(f"  总BatchNorm层: {len(final_bias_check)}")
    
    if still_initial > 0:
        print(f"  ❌ 有 {still_initial} 个BatchNorm bias未被正确加载")
        return False
    else:
        print(f"  ✅ 所有BatchNorm bias都被正确加载")
        return True


def check_specific_bias_parameters():
    """检查特定的bias参数"""
    print(f"\n🔍 检查特定的bias参数")
    print("=" * 60)
    
    # 加载PyTorch权重
    checkpoint_path = "/home/kyc/project/nanodet/nanodet-pytorch/workspace/nanodet-plus-m_320_voc_bs64/model_best/model_best.ckpt"
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)
    
    # 查找所有bias参数
    bias_params = {}
    for name, param in state_dict.items():
        if 'bias' in name:
            bias_params[name] = param.detach().numpy()
    
    print(f"PyTorch权重中包含 {len(bias_params)} 个bias参数")
    
    # 分析bias参数的值分布
    all_bias_values = []
    for name, bias_data in bias_params.items():
        all_bias_values.extend(bias_data.flatten())
    
    all_bias_values = np.array(all_bias_values)
    
    print(f"\n所有bias参数统计:")
    print(f"  总数: {len(all_bias_values)}")
    print(f"  范围: [{all_bias_values.min():.6f}, {all_bias_values.max():.6f}]")
    print(f"  均值: {all_bias_values.mean():.6f}")
    print(f"  标准差: {all_bias_values.std():.6f}")
    
    # 检查有多少bias是0.0001
    close_to_0001 = np.isclose(all_bias_values, 0.0001, atol=1e-6)
    print(f"  接近0.0001的bias数量: {close_to_0001.sum()}")
    print(f"  接近0.0001的比例: {close_to_0001.sum() / len(all_bias_values) * 100:.2f}%")
    
    # 显示一些具体的bias参数
    print(f"\n具体的bias参数示例:")
    count = 0
    for name, bias_data in bias_params.items():
        if count < 10:
            print(f"  {name}: 范围[{bias_data.min():.6f}, {bias_data.max():.6f}], 形状{bias_data.shape}")
            if np.allclose(bias_data, 0.0001):
                print(f"    ⚠️ 全部为0.0001")
            count += 1


def main():
    """主函数"""
    print("🚀 开始检查BatchNorm bias加载")
    
    # 检查bias加载过程
    bias_ok = check_batchnorm_bias_loading()
    
    # 检查特定bias参数
    check_specific_bias_parameters()
    
    print(f"\n✅ 检查完成")
    
    if bias_ok:
        print(f"BatchNorm bias加载正确")
    else:
        print(f"BatchNorm bias加载有问题")


if __name__ == '__main__':
    main()
