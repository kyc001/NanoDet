#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
跟踪权重加载过程
找出权重差异的具体原因
"""

import os
import sys
import torch
import jittor as jt
import numpy as np

# 添加路径
sys.path.append('/home/kyc/project/nanodet/nanodet-jittor')
from nanodet.model.arch.nanodet_plus import NanoDetPlus


def trace_weight_loading():
    """跟踪权重加载的详细过程"""
    print("🔍 跟踪权重加载详细过程")
    print("=" * 60)
    
    # 创建模型配置
    backbone_cfg = {
        'name': 'ShuffleNetV2',
        'model_size': '1.0x',
        'out_stages': [2, 3, 4],
        'activation': 'LeakyReLU',
        'pretrain': True  # 这里会加载预训练权重
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
    
    print("1️⃣ 创建模型（包含预训练权重加载）...")
    model = NanoDetPlus(backbone_cfg, fpn_cfg, aux_head_cfg, head_cfg)
    
    # 记录预训练权重加载后的状态
    print("\n2️⃣ 记录预训练权重加载后的状态...")
    pretrained_weights = {}
    for name, param in model.named_parameters():
        if name.startswith('backbone.'):
            pretrained_weights[name] = param.numpy().copy()
    
    print(f"记录了 {len(pretrained_weights)} 个backbone参数")
    
    # 选择几个关键参数进行跟踪
    trace_params = [
        'backbone.conv1.0.weight',
        'backbone.conv1.1.weight',
        'backbone.conv1.1.bias',
        'backbone.conv1.1.running_mean',
        'backbone.conv1.1.running_var',
    ]
    
    print("\n预训练权重加载后的关键参数:")
    for param_name in trace_params:
        if param_name in pretrained_weights:
            param_data = pretrained_weights[param_name]
            print(f"  {param_name}: 范围[{param_data.min():.6f}, {param_data.max():.6f}], 均值{param_data.mean():.6f}")
    
    print("\n3️⃣ 加载NanoDet训练权重...")
    
    # 加载PyTorch权重
    checkpoint_path = "/home/kyc/project/nanodet/nanodet-pytorch/workspace/nanodet-plus-m_320_voc_bs64/model_best/model_best.ckpt"
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)
    
    print(f"PyTorch checkpoint包含 {len(state_dict)} 个参数")
    
    # 检查要覆盖的backbone参数
    print("\nPyTorch权重中的关键backbone参数:")
    for param_name in trace_params:
        pytorch_name = f"model.{param_name}"
        if pytorch_name in state_dict:
            pytorch_param = state_dict[pytorch_name]
            pytorch_data = pytorch_param.detach().numpy()
            print(f"  {pytorch_name}: 范围[{pytorch_data.min():.6f}, {pytorch_data.max():.6f}], 均值{pytorch_data.mean():.6f}")
            
            # 对比预训练权重
            if param_name in pretrained_weights:
                pretrained_data = pretrained_weights[param_name]
                diff = np.abs(pytorch_data - pretrained_data).max()
                print(f"    与预训练权重差异: {diff:.6f}")
    
    # 获取Jittor模型的参数字典
    jittor_state_dict = {}
    for name, param in model.named_parameters():
        jittor_state_dict[name] = param
    
    print("\n4️⃣ 执行权重覆盖...")
    
    # 记录覆盖过程
    overwritten_count = 0
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
            
            # 记录覆盖前的值
            if jittor_name in trace_params:
                before_data = jittor_param.numpy().copy()
                print(f"\n覆盖 {jittor_name}:")
                print(f"  覆盖前: 范围[{before_data.min():.6f}, {before_data.max():.6f}], 均值{before_data.mean():.6f}")
            
            if list(pytorch_param.shape) == list(jittor_param.shape):
                jittor_param.assign(jt.array(pytorch_param.detach().numpy()))
                overwritten_count += 1
                
                # 记录覆盖后的值
                if jittor_name in trace_params:
                    after_data = jittor_param.numpy().copy()
                    pytorch_data = pytorch_param.detach().numpy()
                    print(f"  覆盖后: 范围[{after_data.min():.6f}, {after_data.max():.6f}], 均值{after_data.mean():.6f}")
                    print(f"  PyTorch: 范围[{pytorch_data.min():.6f}, {pytorch_data.max():.6f}], 均值{pytorch_data.mean():.6f}")
                    
                    # 验证覆盖是否成功
                    diff = np.abs(after_data - pytorch_data).max()
                    print(f"  覆盖差异: {diff:.10f}")
                    
                    if diff < 1e-6:
                        print(f"  ✅ 覆盖成功")
                    else:
                        print(f"  ❌ 覆盖失败")
                        
            elif "scale" in jittor_name and len(pytorch_param.shape) == 0 and list(jittor_param.shape) == [1]:
                jittor_param.assign(jt.array([pytorch_param.detach().numpy()]))
                overwritten_count += 1
    
    print(f"\n✓ 覆盖了 {overwritten_count} 个参数")
    
    print("\n5️⃣ 验证最终权重状态...")
    
    # 验证最终状态
    for param_name in trace_params:
        if param_name in jittor_state_dict:
            final_data = jittor_state_dict[param_name].numpy()
            print(f"\n最终 {param_name}:")
            print(f"  最终值: 范围[{final_data.min():.6f}, {final_data.max():.6f}], 均值{final_data.mean():.6f}")
            
            # 与PyTorch权重对比
            pytorch_name = f"model.{param_name}"
            if pytorch_name in state_dict:
                pytorch_data = state_dict[pytorch_name].detach().numpy()
                diff = np.abs(final_data - pytorch_data).max()
                print(f"  与PyTorch差异: {diff:.10f}")
                
                if diff < 1e-6:
                    print(f"  ✅ 与PyTorch一致")
                else:
                    print(f"  ❌ 与PyTorch不一致")
                    
                    # 详细分析不一致的原因
                    print(f"    最终统计: 均值={final_data.mean():.6f}, 标准差={final_data.std():.6f}")
                    print(f"    PyTorch统计: 均值={pytorch_data.mean():.6f}, 标准差={pytorch_data.std():.6f}")
    
    print(f"\n✅ 权重加载跟踪完成")
    return model


def compare_with_fresh_model():
    """与新创建的模型对比"""
    print("\n🔍 与新创建的模型对比")
    print("=" * 60)
    
    # 创建一个新的模型（不加载任何权重）
    backbone_cfg = {
        'name': 'ShuffleNetV2',
        'model_size': '1.0x',
        'out_stages': [2, 3, 4],
        'activation': 'LeakyReLU',
        'pretrain': False  # 不加载预训练权重
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
    
    print("创建新模型（无预训练权重）...")
    fresh_model = NanoDetPlus(backbone_cfg, fpn_cfg, aux_head_cfg, head_cfg)
    
    # 手动加载PyTorch权重
    print("手动加载PyTorch权重...")
    checkpoint_path = "/home/kyc/project/nanodet/nanodet-pytorch/workspace/nanodet-plus-m_320_voc_bs64/model_best/model_best.ckpt"
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)
    
    # 获取新模型的参数字典
    fresh_state_dict = {}
    for name, param in fresh_model.named_parameters():
        fresh_state_dict[name] = param
    
    # 手动加载权重
    loaded_count = 0
    for pytorch_name, pytorch_param in state_dict.items():
        jittor_name = pytorch_name
        if jittor_name.startswith("model."):
            jittor_name = jittor_name[6:]
        
        if "num_batches_tracked" in jittor_name or jittor_name.startswith("avg_"):
            continue
        
        if "distribution_project.project" in jittor_name:
            continue
        
        if jittor_name in fresh_state_dict:
            jittor_param = fresh_state_dict[jittor_name]
            
            if list(pytorch_param.shape) == list(jittor_param.shape):
                jittor_param.assign(jt.array(pytorch_param.detach().numpy()))
                loaded_count += 1
            elif "scale" in jittor_name and len(pytorch_param.shape) == 0 and list(jittor_param.shape) == [1]:
                jittor_param.assign(jt.array([pytorch_param.detach().numpy()]))
                loaded_count += 1
    
    print(f"✓ 手动加载了 {loaded_count} 个参数")
    
    # 对比两个模型的权重
    print("\n对比两个模型的权重:")
    
    trace_params = [
        'backbone.conv1.0.weight',
        'backbone.conv1.1.weight',
        'backbone.conv1.1.bias',
        'backbone.conv1.1.running_mean',
        'backbone.conv1.1.running_var',
    ]
    
    # 先跟踪加载权重的模型
    traced_model = trace_weight_loading()
    
    for param_name in trace_params:
        if param_name in fresh_state_dict and param_name in traced_model.state_dict():
            fresh_data = fresh_state_dict[param_name].numpy()
            traced_data = traced_model.state_dict()[param_name].numpy()
            
            diff = np.abs(fresh_data - traced_data).max()
            
            print(f"\n{param_name}:")
            print(f"  新模型: 范围[{fresh_data.min():.6f}, {fresh_data.max():.6f}]")
            print(f"  跟踪模型: 范围[{traced_data.min():.6f}, {traced_data.max():.6f}]")
            print(f"  差异: {diff:.10f}")
            
            if diff < 1e-6:
                print(f"  ✅ 两个模型一致")
            else:
                print(f"  ❌ 两个模型不一致")


def main():
    """主函数"""
    print("🚀 开始跟踪权重加载过程")
    
    # 跟踪权重加载
    traced_model = trace_weight_loading()
    
    # 与新模型对比
    # compare_with_fresh_model()
    
    print(f"\n✅ 跟踪完成")


if __name__ == '__main__':
    main()
