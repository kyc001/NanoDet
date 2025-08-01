#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
精确的权重调试工具
逐步验证权重加载过程，确保微调权重正确加载
"""

import os
import sys
import torch
import jittor as jt
import numpy as np

# 添加路径
sys.path.append('/home/kyc/project/nanodet/nanodet-jittor')
from nanodet.model.arch.nanodet_plus import NanoDetPlus


def create_model_step_by_step():
    """逐步创建模型，监控每个步骤"""
    print("🔍 逐步创建模型并监控权重变化")
    print("=" * 60)
    
    # 1. 创建模型配置
    backbone_cfg = {
        'name': 'ShuffleNetV2',
        'model_size': '1.0x',
        'out_stages': [2, 3, 4],
        'activation': 'LeakyReLU',
        'pretrain': False  # 明确不加载ImageNet预训练
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
    
    print("1. 创建模型...")
    model = NanoDetPlus(backbone_cfg, fpn_cfg, aux_head_cfg, head_cfg)
    
    # 2. 检查初始权重
    print("2. 检查初始权重...")
    initial_weights = {}
    for name, param in model.named_parameters():
        if 'head.gfl_cls.0.bias' in name:
            initial_weights[name] = param.numpy().copy()
            print(f"  初始 {name}: 范围[{param.min():.6f}, {param.max():.6f}]")
            print(f"    前5个值: {param.numpy()[:5]}")
    
    # 3. 加载PyTorch权重
    print("3. 加载PyTorch微调权重...")
    checkpoint_path = "/home/kyc/project/nanodet/nanodet-pytorch/workspace/nanodet-plus-m_320_voc_bs64/model_best/model_best.ckpt"
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)
    
    # 检查PyTorch权重
    print("4. 检查PyTorch权重...")
    for pytorch_name, pytorch_param in state_dict.items():
        if 'head.gfl_cls.0.bias' in pytorch_name:
            print(f"  PyTorch {pytorch_name}: 范围[{pytorch_param.min():.6f}, {pytorch_param.max():.6f}]")
            print(f"    前5个值: {pytorch_param.numpy()[:5]}")
    
    # 5. 手动加载权重
    print("5. 手动加载权重...")
    jittor_state_dict = {}
    for name, param in model.named_parameters():
        jittor_state_dict[name] = param
    
    loaded_count = 0
    total_count = 0
    
    for pytorch_name, pytorch_param in state_dict.items():
        jittor_name = pytorch_name
        if jittor_name.startswith("model."):
            jittor_name = jittor_name[6:]
        
        if "num_batches_tracked" in jittor_name or jittor_name.startswith("avg_"):
            continue
        
        if "distribution_project.project" in jittor_name:
            continue
        
        total_count += 1
        
        if jittor_name in jittor_state_dict:
            jittor_param = jittor_state_dict[jittor_name]
            
            if list(pytorch_param.shape) == list(jittor_param.shape):
                # 记录加载前的值
                if 'head.gfl_cls.0.bias' in jittor_name:
                    print(f"  加载前 {jittor_name}: {jittor_param.numpy()[:5]}")
                
                jittor_param.assign(jt.array(pytorch_param.detach().numpy()))
                loaded_count += 1
                
                # 记录加载后的值
                if 'head.gfl_cls.0.bias' in jittor_name:
                    print(f"  加载后 {jittor_name}: {jittor_param.numpy()[:5]}")
                    
            elif "scale" in jittor_name and len(pytorch_param.shape) == 0 and list(jittor_param.shape) == [1]:
                jittor_param.assign(jt.array([pytorch_param.detach().numpy()]))
                loaded_count += 1
    
    print(f"✅ 权重加载: {loaded_count}/{total_count} ({loaded_count/total_count*100:.1f}%)")
    
    # 6. 验证权重加载结果
    print("6. 验证权重加载结果...")
    for name, param in model.named_parameters():
        if 'head.gfl_cls.0.bias' in name:
            final_weights = param.numpy()
            print(f"  最终 {name}: 范围[{param.min():.6f}, {param.max():.6f}]")
            print(f"    前5个值: {final_weights[:5]}")
            
            # 检查是否真的改变了
            if name in initial_weights:
                diff = np.abs(final_weights - initial_weights[name]).max()
                print(f"    与初始值最大差异: {diff:.6f}")
                if diff > 1e-6:
                    print(f"    ✅ 权重已成功更新")
                else:
                    print(f"    ❌ 权重未更新！")
    
    model.eval()
    return model


def test_model_with_debug():
    """测试模型并调试输出"""
    print("\n🔍 测试模型并调试输出")
    print("=" * 60)
    
    model = create_model_step_by_step()
    
    # 创建测试输入
    np.random.seed(42)
    input_data = np.random.randn(1, 3, 320, 320).astype(np.float32)
    jittor_input = jt.array(input_data)
    
    print(f"输入数据: {input_data.shape}, 范围[{input_data.min():.6f}, {input_data.max():.6f}]")
    
    with jt.no_grad():
        # 推理
        output = model(jittor_input)
        
        print(f"\n模型输出:")
        print(f"  输出形状: {output.shape}")
        print(f"  输出范围: [{output.min():.6f}, {output.max():.6f}]")
        
        # 分析分类预测
        cls_preds = output[:, :, :20]
        cls_scores = jt.sigmoid(cls_preds)
        
        print(f"  分类预测: 范围[{cls_preds.min():.6f}, {cls_preds.max():.6f}]")
        print(f"  置信度: 范围[{cls_scores.min():.6f}, {cls_scores.max():.6f}]")
        
        # 详细分析置信度
        cls_scores_np = cls_scores.numpy()
        max_conf = cls_scores_np.max()
        mean_conf = cls_scores_np.mean()
        
        print(f"  置信度统计:")
        print(f"    最高: {max_conf:.6f}")
        print(f"    均值: {mean_conf:.6f}")
        print(f"    >0.01: {(cls_scores_np > 0.01).sum()}")
        print(f"    >0.05: {(cls_scores_np > 0.05).sum()}")
        print(f"    >0.1: {(cls_scores_np > 0.1).sum()}")
        
        # 分析分类预测的分布
        print(f"  分类预测分析:")
        cls_preds_np = cls_preds.numpy()
        print(f"    最大值: {cls_preds_np.max():.6f}")
        print(f"    最小值: {cls_preds_np.min():.6f}")
        print(f"    均值: {cls_preds_np.mean():.6f}")
        print(f"    标准差: {cls_preds_np.std():.6f}")
        
        # 检查是否所有预测都是负值（这是正常的）
        positive_preds = (cls_preds_np > 0).sum()
        print(f"    正值预测数: {positive_preds}")
        
        if positive_preds == 0:
            print(f"    ✅ 所有分类预测都是负值（符合预期）")
        else:
            print(f"    ⚠️ 有正值预测，可能有问题")
        
        return max_conf, mean_conf


def compare_with_expected():
    """与预期结果对比"""
    print(f"\n🔍 与预期结果对比")
    print("=" * 60)
    
    max_conf, mean_conf = test_model_with_debug()
    
    # 预期的结果（基于之前的测试）
    expected_max_conf = 0.082834  # 之前测试的结果
    
    print(f"结果对比:")
    print(f"  当前最高置信度: {max_conf:.6f}")
    print(f"  预期最高置信度: {expected_max_conf:.6f}")
    
    diff = abs(max_conf - expected_max_conf)
    print(f"  差异: {diff:.6f}")
    
    if diff < 0.001:
        print(f"  ✅ 结果一致，权重加载正确")
    elif diff < 0.01:
        print(f"  ⚠️ 结果基本一致，可能有小的差异")
    else:
        print(f"  ❌ 结果差异较大，权重加载可能有问题")
    
    # 分析可能的问题
    if max_conf < 0.01:
        print(f"\n问题分析:")
        print(f"  最高置信度过低，可能的原因:")
        print(f"  1. Head的bias没有正确加载")
        print(f"  2. 某些关键权重缺失")
        print(f"  3. 模型架构与PyTorch不完全一致")
        print(f"  4. 预处理方式不同")


def main():
    """主函数"""
    print("🚀 开始精确的权重调试")
    print("目标: 确保微调权重正确加载，找出mAP为0的原因")
    
    try:
        compare_with_expected()
        
        print(f"\n✅ 精确权重调试完成")
        
    except Exception as e:
        print(f"❌ 调试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
