#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Head深度调试工具
专门检查Head的权重加载和bias设置
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


def check_head_weights():
    """检查Head权重"""
    print("🔍 检查Head权重")
    print("=" * 60)
    
    model = create_jittor_model()
    head = model.head
    
    # 加载PyTorch权重
    checkpoint_path = "/home/kyc/project/nanodet/nanodet-pytorch/workspace/nanodet-plus-m_320_voc_bs64/model_best/model_best.ckpt"
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)
    
    print(f"Head权重检查:")
    
    # 检查gfl_cls权重
    for i in range(len(head.gfl_cls)):
        jittor_layer = head.gfl_cls[i]
        
        # 检查权重
        weight_name = f"head.gfl_cls.{i}.weight"
        bias_name = f"head.gfl_cls.{i}.bias"
        
        pytorch_weight_name = f"model.{weight_name}"
        pytorch_bias_name = f"model.{bias_name}"
        
        if pytorch_weight_name in state_dict:
            pytorch_weight = state_dict[pytorch_weight_name].detach().numpy()
            jittor_weight = jittor_layer.weight.numpy()
            
            weight_diff = np.abs(pytorch_weight - jittor_weight).max()
            print(f"  gfl_cls.{i}.weight: 差异{weight_diff:.10f}")
            
            if weight_diff < 1e-6:
                print(f"    ✅ 权重一致")
            else:
                print(f"    ❌ 权重不一致")
                print(f"      PyTorch: 范围[{pytorch_weight.min():.6f}, {pytorch_weight.max():.6f}]")
                print(f"      Jittor: 范围[{jittor_weight.min():.6f}, {jittor_weight.max():.6f}]")
        
        if pytorch_bias_name in state_dict:
            pytorch_bias = state_dict[pytorch_bias_name].detach().numpy()
            jittor_bias = jittor_layer.bias.numpy()
            
            bias_diff = np.abs(pytorch_bias - jittor_bias).max()
            print(f"  gfl_cls.{i}.bias: 差异{bias_diff:.10f}")
            
            if bias_diff < 1e-6:
                print(f"    ✅ bias一致")
            else:
                print(f"    ❌ bias不一致")
                print(f"      PyTorch: 范围[{pytorch_bias.min():.6f}, {pytorch_bias.max():.6f}]")
                print(f"      Jittor: 范围[{jittor_bias.min():.6f}, {jittor_bias.max():.6f}]")
            
            # 特别检查分类bias
            cls_bias = pytorch_bias[:20]  # 前20个是分类bias
            print(f"      分类bias: 范围[{cls_bias.min():.6f}, {cls_bias.max():.6f}]")
            print(f"      分类bias均值: {cls_bias.mean():.6f}")


def check_head_initialization():
    """检查Head初始化"""
    print(f"\n🔍 检查Head初始化")
    print("=" * 60)
    
    model = create_jittor_model()
    head = model.head
    
    print(f"Head配置:")
    print(f"  num_classes: {head.num_classes}")
    print(f"  feat_channels: {head.feat_channels}")
    print(f"  stacked_convs: {head.stacked_convs}")
    print(f"  strides: {head.strides}")
    print(f"  reg_max: {head.reg_max}")
    
    # 检查gfl_cls层的配置
    for i, layer in enumerate(head.gfl_cls):
        print(f"  gfl_cls.{i}: in_channels={layer.in_channels}, out_channels={layer.out_channels}")
        
        # 检查bias初始化
        bias = layer.bias.numpy()
        cls_bias = bias[:20]  # 分类bias
        reg_bias = bias[20:]  # 回归bias
        
        print(f"    分类bias: 范围[{cls_bias.min():.6f}, {cls_bias.max():.6f}], 均值{cls_bias.mean():.6f}")
        print(f"    回归bias: 范围[{reg_bias.min():.6f}, {reg_bias.max():.6f}], 均值{reg_bias.mean():.6f}")
        
        # 检查是否符合预期的初始化
        expected_cls_bias = -4.595  # 来自init_weights
        if abs(cls_bias.mean() - expected_cls_bias) < 0.1:
            print(f"    ✅ 分类bias初始化正确")
        else:
            print(f"    ❌ 分类bias初始化可能有问题，预期约{expected_cls_bias}")


def test_head_forward_step_by_step():
    """逐步测试Head前向传播"""
    print(f"\n🔍 逐步测试Head前向传播")
    print("=" * 60)
    
    model = create_jittor_model()
    head = model.head
    
    # 创建测试输入
    input_data = create_test_input()
    jittor_input = jt.array(input_data)
    
    with jt.no_grad():
        # 获取FPN特征
        backbone_features = model.backbone(jittor_input)
        fpn_features = model.fpn(backbone_features)
        
        print(f"FPN特征:")
        for i, feat in enumerate(fpn_features):
            print(f"  特征{i}: {feat.shape}, 范围[{feat.min():.6f}, {feat.max():.6f}]")
        
        # 手动执行Head的前向传播
        outputs = []
        for level, (feat, cls_convs, gfl_cls) in enumerate(zip(fpn_features, head.cls_convs, head.gfl_cls)):
            print(f"\n  处理level {level}:")
            print(f"    输入特征: {feat.shape}, 范围[{feat.min():.6f}, {feat.max():.6f}]")
            
            # 通过cls_convs
            current_feat = feat
            for conv_idx, conv in enumerate(cls_convs):
                current_feat = conv(current_feat)
                print(f"    cls_conv{conv_idx}: {current_feat.shape}, 范围[{current_feat.min():.6f}, {current_feat.max():.6f}]")
            
            # 通过gfl_cls
            output = gfl_cls(current_feat)
            print(f"    gfl_cls输出: {output.shape}, 范围[{output.min():.6f}, {output.max():.6f}]")
            
            # 分析输出
            cls_pred = output[:, :20, :, :]
            reg_pred = output[:, 20:, :, :]
            
            print(f"    分类预测: 范围[{cls_pred.min():.6f}, {cls_pred.max():.6f}]")
            print(f"    回归预测: 范围[{reg_pred.min():.6f}, {reg_pred.max():.6f}]")
            
            # 计算置信度
            cls_scores = jt.sigmoid(cls_pred)
            max_conf = cls_scores.max()
            print(f"    最高置信度: {max_conf:.6f}")
            
            # reshape并添加到输出
            output = output.permute(0, 2, 3, 1).reshape(output.shape[0], -1, output.shape[1])
            outputs.append(output)
        
        # 合并所有输出
        final_output = jt.concat(outputs, dim=1)
        print(f"\n最终输出: {final_output.shape}, 范围[{final_output.min():.6f}, {final_output.max():.6f}]")
        
        # 分析最终输出
        final_cls_pred = final_output[:, :, :20]
        final_cls_scores = jt.sigmoid(final_cls_pred)
        final_max_conf = final_cls_scores.max()
        
        print(f"最终最高置信度: {final_max_conf:.6f}")
        
        # 对比完整Head输出
        complete_output = head(fpn_features)
        diff = jt.abs(final_output - complete_output).max()
        print(f"手动vs完整Head差异: {diff:.10f}")


def main():
    """主函数"""
    print("🚀 开始Head深度调试")
    
    # 检查Head权重
    check_head_weights()
    
    # 检查Head初始化
    check_head_initialization()
    
    # 逐步测试Head前向传播
    test_head_forward_step_by_step()
    
    print(f"\n✅ Head深度调试完成")


if __name__ == '__main__':
    main()
