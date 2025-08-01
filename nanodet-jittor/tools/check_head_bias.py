#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
检查Head bias的问题
找出为什么分类预测全是负值
"""

import os
import sys
import torch
import jittor as jt
import numpy as np

# 添加路径
sys.path.append('/home/kyc/project/nanodet/nanodet-jittor')
from nanodet.model.arch.nanodet_plus import NanoDetPlus


def check_head_bias():
    """检查Head bias的详细情况"""
    print("🔍 检查Head bias详细情况")
    print("=" * 60)
    
    # 创建模型
    print("1️⃣ 创建Jittor模型...")
    
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
    
    model = NanoDetPlus(backbone_cfg, fpn_cfg, aux_head_cfg, head_cfg)
    
    print("2️⃣ 检查初始化后的Head bias...")
    
    # 检查初始化后的bias
    for i, gfl_cls in enumerate(model.head.gfl_cls):
        if hasattr(gfl_cls, 'bias') and gfl_cls.bias is not None:
            bias_value = gfl_cls.bias.numpy()
            print(f"   gfl_cls[{i}] bias: {bias_value[:5]}... (前5个值)")
            print(f"   gfl_cls[{i}] bias范围: [{bias_value.min():.6f}, {bias_value.max():.6f}]")
        else:
            print(f"   gfl_cls[{i}] 没有bias")
    
    print("\n3️⃣ 加载PyTorch权重...")
    
    # 加载PyTorch权重
    checkpoint_path = "/home/kyc/project/nanodet/nanodet-pytorch/workspace/nanodet-plus-m_320_voc_bs64/model_best/model_best.ckpt"
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)
    
    # 检查PyTorch权重中的Head bias
    print("\n4️⃣ 检查PyTorch权重中的Head bias...")
    
    head_bias_params = {}
    for name, param in state_dict.items():
        if 'head.gfl_cls' in name and 'bias' in name:
            head_bias_params[name] = param
            print(f"   {name}: {param.shape}, 范围[{param.min():.6f}, {param.max():.6f}]")
            print(f"     前5个值: {param[:5].tolist()}")
    
    print(f"\n5️⃣ 加载权重到Jittor模型...")
    
    # 获取Jittor模型的参数字典
    jittor_state_dict = {}
    for name, param in model.named_parameters():
        jittor_state_dict[name] = param
    
    # 权重加载
    loaded_head_bias = []
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
            
            if list(pytorch_param.shape) == list(jittor_param.shape):
                jittor_param.assign(jt.array(pytorch_param.detach().numpy()))
                
                # 记录Head bias的加载
                if 'head.gfl_cls' in jittor_name and 'bias' in jittor_name:
                    loaded_head_bias.append((jittor_name, pytorch_param.detach().numpy()))
            elif "scale" in jittor_name and len(pytorch_param.shape) == 0 and list(jittor_param.shape) == [1]:
                jittor_param.assign(jt.array([pytorch_param.detach().numpy()]))
    
    print(f"\n6️⃣ 检查加载后的Head bias...")
    
    # 检查加载后的bias
    for i, gfl_cls in enumerate(model.head.gfl_cls):
        if hasattr(gfl_cls, 'bias') and gfl_cls.bias is not None:
            bias_value = gfl_cls.bias.numpy()
            print(f"   gfl_cls[{i}] bias: {bias_value[:5]}... (前5个值)")
            print(f"   gfl_cls[{i}] bias范围: [{bias_value.min():.6f}, {bias_value.max():.6f}]")
    
    print(f"\n7️⃣ 测试模型输出...")
    
    # 测试模型输出
    model.eval()
    test_input = jt.randn(1, 3, 320, 320)
    
    with jt.no_grad():
        output = model(test_input)
    
    cls_preds = output[:, :, :20]
    cls_scores = jt.sigmoid(cls_preds)
    
    print(f"   模型输出形状: {output.shape}")
    print(f"   分类预测范围: [{cls_preds.min():.6f}, {cls_preds.max():.6f}]")
    print(f"   最高置信度: {cls_scores.max():.6f}")
    
    # 分析每个level的输出
    print(f"\n8️⃣ 分析每个level的输出...")
    
    # 重新运行Head，获取每个level的输出
    backbone_features = model.backbone(test_input)
    fpn_features = model.fpn(backbone_features)
    
    level_outputs = []
    for i, (feat, cls_convs, gfl_cls) in enumerate(zip(
        fpn_features,
        model.head.cls_convs,
        model.head.gfl_cls,
    )):
        for conv in cls_convs:
            feat = conv(feat)
        level_output = gfl_cls(feat)
        level_outputs.append(level_output)
        
        # 分析这个level的输出
        level_flat = level_output.flatten(start_dim=2)
        level_cls = level_flat[:, :20, :]
        level_cls_scores = jt.sigmoid(level_cls)
        
        print(f"   Level {i}: 输出形状{level_output.shape}")
        print(f"   Level {i}: 分类范围[{level_cls.min():.6f}, {level_cls.max():.6f}]")
        print(f"   Level {i}: 最高置信度{level_cls_scores.max():.6f}")
    
    print(f"\n✅ Head bias检查完成")


def main():
    """主函数"""
    print("🚀 开始检查Head bias问题")
    
    check_head_bias()
    
    print("\n✅ 检查完成")


if __name__ == '__main__':
    main()
