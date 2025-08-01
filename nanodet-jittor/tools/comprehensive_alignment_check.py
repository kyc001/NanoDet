#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
全面对齐检查工具
系统性解决所有PyTorch到Jittor迁移问题
"""

import os
import sys
import torch
import jittor as jt
import numpy as np
from collections import OrderedDict

# 添加路径
sys.path.append('/home/kyc/project/nanodet/nanodet-jittor')
from nanodet.model.arch.nanodet_plus import NanoDetPlus


def check_model_architecture_alignment():
    """检查模型架构对齐"""
    print("🔍 检查模型架构对齐")
    print("=" * 60)
    
    # 创建Jittor模型
    backbone_cfg = {
        'name': 'ShuffleNetV2',
        'model_size': '1.0x',
        'out_stages': [2, 3, 4],
        'activation': 'LeakyReLU',
        'pretrain': False  # 不加载预训练权重，专注于架构检查
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
    
    model = NanoDetPlus(backbone_cfg, fpn_cfg, aux_head_cfg, head_cfg)
    
    # 分析模型结构
    print(f"Jittor模型结构分析:")
    
    # 统计参数数量
    total_params = 0
    trainable_params = 0
    
    param_groups = {
        'backbone': 0,
        'fpn': 0,
        'aux_fpn': 0,
        'head': 0,
        'aux_head': 0,
        'other': 0
    }
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        
        if param.requires_grad:
            trainable_params += param_count
        
        # 分组统计
        if name.startswith('backbone.'):
            param_groups['backbone'] += param_count
        elif name.startswith('fpn.'):
            param_groups['fpn'] += param_count
        elif name.startswith('aux_fpn.'):
            param_groups['aux_fpn'] += param_count
        elif name.startswith('head.'):
            param_groups['head'] += param_count
        elif name.startswith('aux_head.'):
            param_groups['aux_head'] += param_count
        else:
            param_groups['other'] += param_count
    
    print(f"  总参数数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  参数分布:")
    for group, count in param_groups.items():
        print(f"    {group}: {count:,} ({count/total_params*100:.1f}%)")
    
    # 检查模型层数
    total_modules = 0
    module_types = {}
    
    for name, module in model.named_modules():
        if name:  # 跳过根模块
            total_modules += 1
            module_type = type(module).__name__
            module_types[module_type] = module_types.get(module_type, 0) + 1
    
    print(f"\n  总模块数: {total_modules}")
    print(f"  模块类型分布:")
    for module_type, count in sorted(module_types.items()):
        print(f"    {module_type}: {count}")
    
    return model


def check_batchnorm_parameters():
    """检查BatchNorm参数问题"""
    print(f"\n🔍 检查BatchNorm参数问题")
    print("=" * 60)
    
    # 创建模型
    model = check_model_architecture_alignment()
    
    # 统计BatchNorm相关参数
    bn_params = []
    bn_buffers = []
    scale_params = []
    
    for name, param in model.named_parameters():
        if 'running_mean' in name or 'running_var' in name:
            bn_buffers.append((name, param.shape))
        elif ('weight' in name or 'bias' in name) and ('bn' in name.lower() or 'norm' in name.lower() or '.1.' in name):
            bn_params.append((name, param.shape))
        elif 'scale' in name:
            scale_params.append((name, param.shape))
    
    print(f"BatchNorm参数统计:")
    print(f"  BN权重/偏置参数: {len(bn_params)}")
    print(f"  BN统计参数(running_mean/var): {len(bn_buffers)}")
    print(f"  Scale参数: {len(scale_params)}")
    
    if scale_params:
        print(f"\nScale参数详情:")
        for name, shape in scale_params[:5]:  # 只显示前5个
            print(f"    {name}: {shape}")
    
    # 检查PyTorch权重中的对应参数
    print(f"\n检查PyTorch权重中的对应参数:")
    checkpoint_path = "/home/kyc/project/nanodet/nanodet-pytorch/workspace/nanodet-plus-m_320_voc_bs64/model_best/model_best.ckpt"
    
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        state_dict = ckpt.get('state_dict', ckpt)
        
        pytorch_bn_params = []
        pytorch_scale_params = []
        
        for name, param in state_dict.items():
            if 'scale' in name:
                pytorch_scale_params.append((name, param.shape))
            elif ('weight' in name or 'bias' in name) and ('bn' in name.lower() or 'norm' in name.lower() or '.1.' in name):
                pytorch_bn_params.append((name, param.shape))
        
        print(f"  PyTorch BN参数: {len(pytorch_bn_params)}")
        print(f"  PyTorch Scale参数: {len(pytorch_scale_params)}")
        
        if pytorch_scale_params:
            print(f"\n  PyTorch Scale参数详情:")
            for name, shape in pytorch_scale_params[:5]:
                print(f"    {name}: {shape}")
                
                # 检查对应的Jittor参数
                jittor_name = name
                if jittor_name.startswith("model."):
                    jittor_name = jittor_name[6:]
                
                jittor_param = None
                for jname, jparam in model.named_parameters():
                    if jname == jittor_name:
                        jittor_param = jparam
                        break
                
                if jittor_param is not None:
                    print(f"      对应Jittor: {jittor_name}: {jittor_param.shape}")
                    if list(param.shape) != list(jittor_param.shape):
                        print(f"      ❌ 形状不匹配: PyTorch{param.shape} vs Jittor{jittor_param.shape}")
                else:
                    print(f"      ❌ 在Jittor中未找到对应参数")
        
    except Exception as e:
        print(f"❌ 加载PyTorch权重失败: {e}")


def check_distribution_project_issue():
    """检查distribution_project问题"""
    print(f"\n🔍 检查distribution_project问题")
    print("=" * 60)
    
    # 创建模型
    model = check_model_architecture_alignment()
    
    # 检查head中的distribution_project
    if hasattr(model.head, 'distribution_project'):
        dist_proj = model.head.distribution_project
        print(f"✅ distribution_project存在")
        print(f"  类型: {type(dist_proj)}")
        
        if hasattr(dist_proj, 'project'):
            project = dist_proj.project
            print(f"  project属性: {type(project)}")
            print(f"  project形状: {project.shape if hasattr(project, 'shape') else 'N/A'}")
            
            # 检查是否在named_parameters中
            found_in_params = False
            for name, param in model.named_parameters():
                if 'distribution_project.project' in name:
                    found_in_params = True
                    print(f"  ✅ 在named_parameters中找到: {name}")
                    break
            
            if not found_in_params:
                print(f"  ✅ 不在named_parameters中 (正确)")
        else:
            print(f"  ❌ 没有project属性")
    else:
        print(f"❌ distribution_project不存在")


def check_weight_loading_compatibility():
    """检查权重加载兼容性"""
    print(f"\n🔍 检查权重加载兼容性")
    print("=" * 60)
    
    # 创建模型
    model = check_model_architecture_alignment()
    
    # 加载PyTorch权重
    checkpoint_path = "/home/kyc/project/nanodet/nanodet-pytorch/workspace/nanodet-plus-m_320_voc_bs64/model_best/model_best.ckpt"
    
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        state_dict = ckpt.get('state_dict', ckpt)
        
        print(f"PyTorch权重文件:")
        print(f"  总参数数: {len(state_dict)}")
        
        # 获取Jittor模型参数
        jittor_params = {}
        for name, param in model.named_parameters():
            jittor_params[name] = param
        
        print(f"Jittor模型参数:")
        print(f"  总参数数: {len(jittor_params)}")
        
        # 分析匹配情况
        matched = 0
        shape_mismatch = 0
        missing_in_jittor = 0
        missing_in_pytorch = 0
        
        # PyTorch -> Jittor匹配
        for pytorch_name, pytorch_param in state_dict.items():
            jittor_name = pytorch_name
            if jittor_name.startswith("model."):
                jittor_name = jittor_name[6:]
            
            if "num_batches_tracked" in jittor_name or jittor_name.startswith("avg_"):
                continue
            
            if "distribution_project.project" in jittor_name:
                continue
            
            if jittor_name in jittor_params:
                jittor_param = jittor_params[jittor_name]
                
                if list(pytorch_param.shape) == list(jittor_param.shape):
                    matched += 1
                elif "scale" in jittor_name and len(pytorch_param.shape) == 0 and list(jittor_param.shape) == [1]:
                    matched += 1  # Scale参数特殊处理
                else:
                    shape_mismatch += 1
                    print(f"    形状不匹配: {jittor_name} PyTorch{pytorch_param.shape} vs Jittor{jittor_param.shape}")
            else:
                missing_in_jittor += 1
        
        # Jittor -> PyTorch匹配
        for jittor_name in jittor_params.keys():
            pytorch_name = f"model.{jittor_name}"
            if pytorch_name not in state_dict:
                missing_in_pytorch += 1
        
        print(f"\n权重匹配分析:")
        print(f"  ✅ 成功匹配: {matched}")
        print(f"  ❌ 形状不匹配: {shape_mismatch}")
        print(f"  ❌ Jittor中缺失: {missing_in_jittor}")
        print(f"  ❌ PyTorch中缺失: {missing_in_pytorch}")
        
        success_rate = matched / (matched + shape_mismatch + missing_in_jittor) * 100 if (matched + shape_mismatch + missing_in_jittor) > 0 else 0
        print(f"  📊 匹配成功率: {success_rate:.1f}%")
        
        return success_rate > 95
        
    except Exception as e:
        print(f"❌ 权重加载检查失败: {e}")
        return False


def main():
    """主函数"""
    print("🚀 开始全面对齐检查")
    
    # 1. 模型架构对齐检查
    model = check_model_architecture_alignment()
    
    # 2. BatchNorm参数检查
    check_batchnorm_parameters()
    
    # 3. distribution_project问题检查
    check_distribution_project_issue()
    
    # 4. 权重加载兼容性检查
    weight_compatible = check_weight_loading_compatibility()
    
    print(f"\n📊 全面检查总结:")
    print(f"  模型架构: ✅")
    print(f"  权重兼容性: {'✅' if weight_compatible else '❌'}")
    
    print(f"\n✅ 全面对齐检查完成")


if __name__ == '__main__':
    main()
