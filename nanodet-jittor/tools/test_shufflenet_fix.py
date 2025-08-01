#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试修复后的ShuffleNetV2预训练权重加载
"""

import os
import sys
import torch
import jittor as jt
import numpy as np

# 添加路径
sys.path.append('/home/kyc/project/nanodet/nanodet-jittor')
from nanodet.model.arch.nanodet_plus import NanoDetPlus


def test_shufflenet_fix():
    """测试修复后的ShuffleNetV2"""
    print("🔍 测试修复后的ShuffleNetV2预训练权重加载")
    print("=" * 60)
    
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    jt.set_global_seed(42)
    
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
    
    print("1️⃣ 创建修复后的Jittor模型...")
    model = NanoDetPlus(backbone_cfg, fpn_cfg, aux_head_cfg, head_cfg)
    
    # 加载NanoDet权重
    print("\n2️⃣ 加载NanoDet训练权重...")
    checkpoint_path = "/home/kyc/project/nanodet/nanodet-pytorch/workspace/nanodet-plus-m_320_voc_bs64/model_best/model_best.ckpt"
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)
    
    # 获取Jittor模型的参数字典
    jittor_state_dict = {}
    for name, param in model.named_parameters():
        jittor_state_dict[name] = param
    
    # 权重加载
    loaded_count = 0
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
                loaded_count += 1
            elif "scale" in jittor_name and len(pytorch_param.shape) == 0 and list(jittor_param.shape) == [1]:
                jittor_param.assign(jt.array([pytorch_param.detach().numpy()]))
                loaded_count += 1
    
    print(f"✓ 加载了 {loaded_count} 个NanoDet权重")
    
    model.eval()
    
    print("\n3️⃣ 测试修复后的模型输出...")
    
    # 使用固定输入
    if os.path.exists("fixed_input_data.npy"):
        input_data = np.load("fixed_input_data.npy")
        print("✓ 使用固定输入")
    else:
        input_data = np.random.randn(1, 3, 320, 320).astype(np.float32)
        print("✓ 创建新的固定输入")
    
    jittor_input = jt.array(input_data)
    
    # 逐层测试
    with jt.no_grad():
        # Backbone
        backbone_features = model.backbone(jittor_input)
        print(f"\n🔍 修复后的Backbone输出:")
        for i, feat in enumerate(backbone_features):
            print(f"   特征{i}: {feat.shape}, 范围[{feat.min():.6f}, {feat.max():.6f}]")
            # 保存修复后的backbone特征
            np.save(f"jittor_fixed_backbone_feat_{i}.npy", feat.numpy())
        
        # FPN
        fpn_features = model.fpn(backbone_features)
        print(f"\n🔍 修复后的FPN输出:")
        for i, feat in enumerate(fpn_features):
            print(f"   FPN特征{i}: {feat.shape}, 范围[{feat.min():.6f}, {feat.max():.6f}]")
            # 保存修复后的FPN特征
            np.save(f"jittor_fixed_fpn_feat_{i}.npy", feat.numpy())
        
        # Head
        head_output = model.head(fpn_features)
        print(f"\n🔍 修复后的Head输出:")
        print(f"   Head输出: {head_output.shape}, 范围[{head_output.min():.6f}, {head_output.max():.6f}]")
        
        # 分析Head输出
        cls_preds = head_output[:, :, :20]
        reg_preds = head_output[:, :, 20:]
        cls_scores = jt.sigmoid(cls_preds)
        
        print(f"   分类预测: 范围[{cls_preds.min():.6f}, {cls_preds.max():.6f}]")
        print(f"   回归预测: 范围[{reg_preds.min():.6f}, {reg_preds.max():.6f}]")
        print(f"   最高置信度: {cls_scores.max():.6f}")
        
        # 保存修复后的Head输出
        np.save("jittor_fixed_head_output.npy", head_output.numpy())
        
        # 完整模型
        full_output = model(jittor_input)
        print(f"\n🔍 修复后的完整模型输出:")
        print(f"   完整输出: {full_output.shape}, 范围[{full_output.min():.6f}, {full_output.max():.6f}]")
        
        # 保存修复后的完整输出
        np.save("jittor_fixed_full_output.npy", full_output.numpy())
    
    print("\n4️⃣ 与之前的输出对比...")
    
    # 对比修复前后的差异
    if os.path.exists("jittor_backbone_feat_0.npy"):
        for i in range(3):
            old_feat = np.load(f"jittor_backbone_feat_{i}.npy")
            new_feat = np.load(f"jittor_fixed_backbone_feat_{i}.npy")
            
            diff = np.abs(old_feat - new_feat)
            max_diff = diff.max()
            mean_diff = diff.mean()
            
            print(f"   Backbone特征{i}修复前后差异: 最大{max_diff:.6f}, 平均{mean_diff:.6f}")
    
    # 对比与PyTorch的差异
    print("\n5️⃣ 与PyTorch输出对比...")
    
    if os.path.exists("pytorch_backbone_feat_0.npy"):
        for i in range(3):
            pytorch_feat = np.load(f"pytorch_backbone_feat_{i}.npy")
            jittor_feat = np.load(f"jittor_fixed_backbone_feat_{i}.npy")
            
            diff = np.abs(pytorch_feat - jittor_feat)
            max_diff = diff.max()
            mean_diff = diff.mean()
            
            print(f"   修复后Backbone特征{i}与PyTorch差异: 最大{max_diff:.6f}, 平均{mean_diff:.6f}")
    
    if os.path.exists("pytorch_fixed_output.npy"):
        pytorch_output = np.load("pytorch_fixed_output.npy")
        jittor_output = np.load("jittor_fixed_full_output.npy")
        
        diff = np.abs(pytorch_output - jittor_output)
        max_diff = diff.max()
        mean_diff = diff.mean()
        
        print(f"\n📊 修复后完整输出与PyTorch差异:")
        print(f"   最大差异: {max_diff:.6f}")
        print(f"   平均差异: {mean_diff:.6f}")
        
        if max_diff < 1e-3:
            print(f"   ✅ 修复成功！差异显著减小")
        elif max_diff < 1e-1:
            print(f"   ⚠️ 有改善，但仍有差异")
        else:
            print(f"   ❌ 修复效果不明显")
    
    print(f"\n✅ 修复测试完成")


def main():
    """主函数"""
    print("🚀 开始测试ShuffleNetV2修复")
    
    test_shufflenet_fix()
    
    print("\n✅ 测试完成")


if __name__ == '__main__':
    main()
