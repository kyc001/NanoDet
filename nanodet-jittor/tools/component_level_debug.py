#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
组件级别调试工具
逐个检查模型组件，找出问题根源
"""

import os
import sys
import torch
import jittor as jt
import numpy as np

# 添加路径
sys.path.append('/home/kyc/project/nanodet/nanodet-jittor')
from nanodet.model.arch.nanodet_plus import NanoDetPlus


def create_test_input():
    """创建固定的测试输入"""
    np.random.seed(42)
    torch.manual_seed(42)
    jt.set_global_seed(42)
    
    # 使用固定的测试数据
    if os.path.exists("fixed_input_data.npy"):
        input_data = np.load("fixed_input_data.npy")
    else:
        input_data = np.random.randn(1, 3, 320, 320).astype(np.float32)
        np.save("fixed_input_data.npy", input_data)
    
    return input_data


def create_jittor_model():
    """创建Jittor模型"""
    print("🔍 创建Jittor模型...")
    
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
    
    # 加载权重
    print("加载PyTorch权重...")
    checkpoint_path = "/home/kyc/project/nanodet/nanodet-pytorch/workspace/nanodet-plus-m_320_voc_bs64/model_best/model_best.ckpt"
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)
    
    jittor_state_dict = {}
    for name, param in model.named_parameters():
        jittor_state_dict[name] = param
    
    # 权重加载
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
            elif "scale" in jittor_name and len(pytorch_param.shape) == 0 and list(jittor_param.shape) == [1]:
                jittor_param.assign(jt.array([pytorch_param.detach().numpy()]))
    
    model.eval()
    return model


def test_backbone_only():
    """测试仅Backbone"""
    print("🔍 测试仅Backbone")
    print("=" * 60)
    
    input_data = create_test_input()
    jittor_input = jt.array(input_data)
    
    model = create_jittor_model()
    
    with jt.no_grad():
        backbone_features = model.backbone(jittor_input)
        
        print(f"Backbone输出:")
        for i, feat in enumerate(backbone_features):
            print(f"  特征{i}: {feat.shape}, 范围[{feat.min():.6f}, {feat.max():.6f}]")
            
            # 检查是否有异常值
            if feat.max() > 100 or feat.min() < -100:
                print(f"    ⚠️ 特征{i}数值范围异常")
            elif feat.max() > 10 or feat.min() < -10:
                print(f"    ⚠️ 特征{i}数值范围偏大")
            else:
                print(f"    ✅ 特征{i}数值范围正常")
    
    return backbone_features


def test_fpn_only(backbone_features):
    """测试仅FPN"""
    print(f"\n🔍 测试仅FPN")
    print("=" * 60)
    
    model = create_jittor_model()
    
    with jt.no_grad():
        fpn_features = model.fpn(backbone_features)
        
        print(f"FPN输出:")
        for i, feat in enumerate(fpn_features):
            print(f"  FPN特征{i}: {feat.shape}, 范围[{feat.min():.6f}, {feat.max():.6f}]")
            
            # 检查是否有异常值
            if feat.max() > 100 or feat.min() < -100:
                print(f"    ⚠️ FPN特征{i}数值范围异常")
            elif feat.max() > 10 or feat.min() < -10:
                print(f"    ⚠️ FPN特征{i}数值范围偏大")
            else:
                print(f"    ✅ FPN特征{i}数值范围正常")
    
    return fpn_features


def test_head_only(fpn_features):
    """测试仅Head"""
    print(f"\n🔍 测试仅Head")
    print("=" * 60)
    
    model = create_jittor_model()
    
    with jt.no_grad():
        head_output = model.head(fpn_features)
        
        print(f"Head输出:")
        print(f"  输出形状: {head_output.shape}")
        print(f"  输出范围: [{head_output.min():.6f}, {head_output.max():.6f}]")
        
        # 分析Head输出
        cls_preds = head_output[:, :, :20]
        reg_preds = head_output[:, :, 20:]
        cls_scores = jt.sigmoid(cls_preds)
        
        print(f"  分类预测: 范围[{cls_preds.min():.6f}, {cls_preds.max():.6f}]")
        print(f"  回归预测: 范围[{reg_preds.min():.6f}, {reg_preds.max():.6f}]")
        print(f"  分类置信度: 范围[{cls_scores.min():.6f}, {cls_scores.max():.6f}]")
        print(f"  最高置信度: {cls_scores.max():.6f}")
        
        # 检查Head输出是否正常
        if cls_preds.max() > 0:
            print(f"    ⚠️ 分类预测有正值，可能有问题")
        else:
            print(f"    ✅ 分类预测全为负值，符合预期")
        
        if cls_scores.max() < 0.1:
            print(f"    ❌ 最高置信度过低，Head有问题")
        elif cls_scores.max() < 0.5:
            print(f"    ⚠️ 最高置信度偏低")
        else:
            print(f"    ✅ 最高置信度正常")
    
    return head_output


def test_individual_head_layers(fpn_features):
    """测试Head的各个层"""
    print(f"\n🔍 测试Head的各个层")
    print("=" * 60)
    
    model = create_jittor_model()
    head = model.head
    
    with jt.no_grad():
        # 测试cls_convs
        print(f"测试分类卷积层:")
        cls_feat = fpn_features[0]  # 使用第一个特征
        for i, conv in enumerate(head.cls_convs):
            cls_feat = conv(cls_feat)
            print(f"  cls_conv{i}: {cls_feat.shape}, 范围[{cls_feat.min():.6f}, {cls_feat.max():.6f}]")
        
        # 测试reg_convs
        print(f"\n测试回归卷积层:")
        reg_feat = fpn_features[0]  # 使用第一个特征
        for i, conv in enumerate(head.reg_convs):
            reg_feat = conv(reg_feat)
            print(f"  reg_conv{i}: {reg_feat.shape}, 范围[{reg_feat.min():.6f}, {reg_feat.max():.6f}]")
        
        # 测试最终预测层
        print(f"\n测试最终预测层:")
        cls_pred = head.gfl_cls(cls_feat)
        reg_pred = head.gfl_reg(reg_feat)
        
        print(f"  cls_pred: {cls_pred.shape}, 范围[{cls_pred.min():.6f}, {cls_pred.max():.6f}]")
        print(f"  reg_pred: {reg_pred.shape}, 范围[{reg_pred.min():.6f}, {reg_pred.max():.6f}]")
        
        # 检查预测层输出
        if cls_pred.max() > 10 or cls_pred.min() < -10:
            print(f"    ❌ 分类预测数值范围异常")
        else:
            print(f"    ✅ 分类预测数值范围正常")
        
        if reg_pred.max() > 100 or reg_pred.min() < -100:
            print(f"    ❌ 回归预测数值范围异常")
        else:
            print(f"    ✅ 回归预测数值范围正常")


def main():
    """主函数"""
    print("🚀 开始组件级别调试")
    
    # 1. 测试Backbone
    backbone_features = test_backbone_only()
    
    # 2. 测试FPN
    fpn_features = test_fpn_only(backbone_features)
    
    # 3. 测试Head
    head_output = test_head_only(fpn_features)
    
    # 4. 测试Head各个层
    test_individual_head_layers(fpn_features)
    
    print(f"\n✅ 组件级别调试完成")
    
    # 总结
    print(f"\n📊 调试总结:")
    print(f"  如果Backbone正常，FPN异常 -> FPN实现有问题")
    print(f"  如果FPN正常，Head异常 -> Head实现有问题")
    print(f"  如果所有组件都正常，但整体异常 -> 组合或权重加载有问题")


if __name__ == '__main__':
    main()
