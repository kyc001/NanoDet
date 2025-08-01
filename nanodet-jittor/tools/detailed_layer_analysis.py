#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
详细的逐层分析
找出模型差异的具体位置
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


def analyze_first_few_layers():
    """分析前几层的详细行为"""
    print(f"🔍 分析前几层的详细行为")
    print("=" * 60)
    
    # 创建模型
    model = create_jittor_model()
    
    # 创建测试输入
    input_data = create_test_input()
    jittor_input = jt.array(input_data)
    
    print(f"输入数据: {input_data.shape}, 范围[{input_data.min():.6f}, {input_data.max():.6f}]")
    
    with jt.no_grad():
        print(f"\n🔍 逐层前向传播分析:")
        
        # 1. 输入层
        x = jittor_input
        print(f"  输入: {x.shape}, 范围[{x.min():.6f}, {x.max():.6f}]")
        
        # 2. Conv1层详细分析
        print(f"\n  🔍 Conv1层详细分析:")
        
        # Conv1的各个子层
        conv1_conv = model.backbone.conv1[0]  # Conv2d
        conv1_bn = model.backbone.conv1[1]    # BatchNorm2d
        conv1_act = model.backbone.conv1[2]   # LeakyReLU
        
        # Conv输出
        conv_out = conv1_conv(x)
        print(f"    Conv2d输出: {conv_out.shape}, 范围[{conv_out.min():.6f}, {conv_out.max():.6f}]")
        
        # BatchNorm输出
        bn_out = conv1_bn(conv_out)
        print(f"    BatchNorm输出: {bn_out.shape}, 范围[{bn_out.min():.6f}, {bn_out.max():.6f}]")
        
        # 激活函数输出
        act_out = conv1_act(bn_out)
        print(f"    LeakyReLU输出: {act_out.shape}, 范围[{act_out.min():.6f}, {act_out.max():.6f}]")
        
        # 完整conv1输出
        conv1_full_out = model.backbone.conv1(x)
        print(f"    Conv1完整输出: {conv1_full_out.shape}, 范围[{conv1_full_out.min():.6f}, {conv1_full_out.max():.6f}]")
        
        # 验证一致性
        diff = jt.abs(act_out - conv1_full_out).max()
        print(f"    一致性检查差异: {diff:.10f}")
        
        # 3. MaxPool层
        print(f"\n  🔍 MaxPool层:")
        x = conv1_full_out
        maxpool_out = model.backbone.maxpool(x)
        print(f"    MaxPool输出: {maxpool_out.shape}, 范围[{maxpool_out.min():.6f}, {maxpool_out.max():.6f}]")
        
        # 4. Stage2第一个block详细分析
        print(f"\n  🔍 Stage2第一个block详细分析:")
        x = maxpool_out
        
        # Stage2的第一个block
        first_block = model.backbone.stage2[0]
        
        # 分析branch1和branch2
        if hasattr(first_block, 'branch1') and len(first_block.branch1) > 0:
            branch1_out = first_block.branch1(x)
            print(f"    Branch1输出: {branch1_out.shape}, 范围[{branch1_out.min():.6f}, {branch1_out.max():.6f}]")
        
        if hasattr(first_block, 'branch2'):
            branch2_out = first_block.branch2(x)
            print(f"    Branch2输出: {branch2_out.shape}, 范围[{branch2_out.min():.6f}, {branch2_out.max():.6f}]")
        
        # 完整block输出
        block_out = first_block(x)
        print(f"    Block完整输出: {block_out.shape}, 范围[{block_out.min():.6f}, {block_out.max():.6f}]")
        
        # 5. 继续分析更多层
        print(f"\n  🔍 继续分析Stage2:")
        x = block_out
        
        for i, block in enumerate(model.backbone.stage2[1:], 1):
            x = block(x)
            print(f"    Stage2 Block{i}输出: {x.shape}, 范围[{x.min():.6f}, {x.max():.6f}]")
        
        stage2_out = x
        print(f"    Stage2完整输出: {stage2_out.shape}, 范围[{stage2_out.min():.6f}, {stage2_out.max():.6f}]")

        # 6. Stage3分析
        print(f"\n  🔍 Stage3分析:")
        x = stage2_out

        for i, block in enumerate(model.backbone.stage3):
            x = block(x)
            print(f"    Stage3 Block{i}输出: {x.shape}, 范围[{x.min():.6f}, {x.max():.6f}]")

        stage3_out = x
        print(f"    Stage3完整输出: {stage3_out.shape}, 范围[{stage3_out.min():.6f}, {stage3_out.max():.6f}]")

        # 7. Stage4分析
        print(f"\n  🔍 Stage4分析:")
        x = stage3_out

        for i, block in enumerate(model.backbone.stage4):
            x = block(x)
            print(f"    Stage4 Block{i}输出: {x.shape}, 范围[{x.min():.6f}, {x.max():.6f}]")

        stage4_out = x
        print(f"    Stage4完整输出: {stage4_out.shape}, 范围[{stage4_out.min():.6f}, {stage4_out.max():.6f}]")

        # 8. 完整Backbone输出
        print(f"\n  🔍 完整Backbone输出:")
        backbone_features = model.backbone(jittor_input)
        for i, feat in enumerate(backbone_features):
            print(f"    Backbone特征{i}: {feat.shape}, 范围[{feat.min():.6f}, {feat.max():.6f}]")

        # 9. FPN详细分析
        print(f"\n  🔍 FPN详细分析:")
        fpn_features = model.fpn(backbone_features)
        for i, feat in enumerate(fpn_features):
            print(f"    FPN特征{i}: {feat.shape}, 范围[{feat.min():.6f}, {feat.max():.6f}]")

        # 10. Head详细分析
        print(f"\n  🔍 Head详细分析:")
        head_output = model.head(fpn_features)
        print(f"    Head原始输出: {head_output.shape}, 范围[{head_output.min():.6f}, {head_output.max():.6f}]")

        # 分析Head输出的组成
        cls_preds = head_output[:, :, :20]
        reg_preds = head_output[:, :, 20:]
        cls_scores = jt.sigmoid(cls_preds)

        print(f"    分类预测: {cls_preds.shape}, 范围[{cls_preds.min():.6f}, {cls_preds.max():.6f}]")
        print(f"    回归预测: {reg_preds.shape}, 范围[{reg_preds.min():.6f}, {reg_preds.max():.6f}]")
        print(f"    分类置信度: 范围[{cls_scores.min():.6f}, {cls_scores.max():.6f}]")
        print(f"    最高置信度: {cls_scores.max():.6f}")

        # 11. 对比完整模型输出
        print(f"\n  🔍 完整模型输出对比:")
        full_output = model(jittor_input)
        print(f"    完整模型输出: {full_output.shape}, 范围[{full_output.min():.6f}, {full_output.max():.6f}]")

        # 验证一致性
        head_vs_full_diff = jt.abs(head_output - full_output).max()
        print(f"    Head vs 完整模型差异: {head_vs_full_diff:.10f}")

        if head_vs_full_diff < 1e-6:
            print(f"    ✅ Head输出与完整模型一致")
        else:
            print(f"    ❌ Head输出与完整模型不一致")


def check_batchnorm_detailed():
    """详细检查BatchNorm行为"""
    print(f"\n🔍 详细检查BatchNorm行为")
    print("=" * 60)
    
    # 创建模型
    model = create_jittor_model()
    
    # 创建测试输入
    input_data = create_test_input()
    jittor_input = jt.array(input_data)
    
    with jt.no_grad():
        # 获取第一个BatchNorm层
        conv1_conv = model.backbone.conv1[0]
        conv1_bn = model.backbone.conv1[1]
        
        # 获取conv输出作为BatchNorm输入
        conv_out = conv1_conv(jittor_input)
        
        print(f"BatchNorm输入: {conv_out.shape}, 范围[{conv_out.min():.6f}, {conv_out.max():.6f}]")
        
        # 检查BatchNorm参数
        print(f"\nBatchNorm参数:")
        print(f"  weight: 形状{conv1_bn.weight.shape}, 范围[{conv1_bn.weight.min():.6f}, {conv1_bn.weight.max():.6f}]")
        print(f"  bias: 形状{conv1_bn.bias.shape}, 范围[{conv1_bn.bias.min():.6f}, {conv1_bn.bias.max():.6f}]")
        print(f"  running_mean: 形状{conv1_bn.running_mean.shape}, 范围[{conv1_bn.running_mean.min():.6f}, {conv1_bn.running_mean.max():.6f}]")
        print(f"  running_var: 形状{conv1_bn.running_var.shape}, 范围[{conv1_bn.running_var.min():.6f}, {conv1_bn.running_var.max():.6f}]")
        print(f"  eps: {conv1_bn.eps}")
        print(f"  momentum: {conv1_bn.momentum}")
        print(f"  is_train: {conv1_bn.is_train}")
        
        # BatchNorm输出
        bn_out = conv1_bn(conv_out)
        print(f"\nBatchNorm输出: {bn_out.shape}, 范围[{bn_out.min():.6f}, {bn_out.max():.6f}]")
        
        # 手动计算BatchNorm（评估模式）
        print(f"\n手动计算BatchNorm:")
        
        # 重塑参数以便广播
        mean = conv1_bn.running_mean.view(1, -1, 1, 1)
        var = conv1_bn.running_var.view(1, -1, 1, 1)
        weight = conv1_bn.weight.view(1, -1, 1, 1)
        bias = conv1_bn.bias.view(1, -1, 1, 1)
        
        print(f"  mean形状: {mean.shape}, 范围[{mean.min():.6f}, {mean.max():.6f}]")
        print(f"  var形状: {var.shape}, 范围[{var.min():.6f}, {var.max():.6f}]")
        print(f"  weight形状: {weight.shape}, 范围[{weight.min():.6f}, {weight.max():.6f}]")
        print(f"  bias形状: {bias.shape}, 范围[{bias.min():.6f}, {bias.max():.6f}]")
        
        # 标准化
        normalized = (conv_out - mean) / jt.sqrt(var + conv1_bn.eps)
        print(f"  标准化后: 范围[{normalized.min():.6f}, {normalized.max():.6f}]")
        
        # 仿射变换
        manual_bn_out = normalized * weight + bias
        print(f"  手动BatchNorm输出: 范围[{manual_bn_out.min():.6f}, {manual_bn_out.max():.6f}]")
        
        # 对比差异
        bn_diff = jt.abs(bn_out - manual_bn_out).max()
        print(f"  BatchNorm差异: {bn_diff:.10f}")
        
        if bn_diff < 1e-5:
            print(f"  ✅ BatchNorm计算正确")
        else:
            print(f"  ❌ BatchNorm计算有误")
            
            # 详细分析差异
            print(f"    详细差异分析:")
            print(f"    最大绝对差异: {bn_diff:.10f}")
            print(f"    平均绝对差异: {jt.abs(bn_out - manual_bn_out).mean():.10f}")
            print(f"    相对差异: {(bn_diff / jt.abs(bn_out).max()):.10f}")


def analyze_fpn_internal():
    """分析FPN内部结构"""
    print(f"\n🔍 分析FPN内部结构")
    print("=" * 60)

    # 创建模型
    model = create_jittor_model()

    # 创建测试输入
    input_data = create_test_input()
    jittor_input = jt.array(input_data)

    with jt.no_grad():
        # 获取backbone特征
        backbone_features = model.backbone(jittor_input)
        print(f"Backbone特征输入到FPN:")
        for i, feat in enumerate(backbone_features):
            print(f"  特征{i}: {feat.shape}, 范围[{feat.min():.6f}, {feat.max():.6f}]")

        # 分析FPN的内部处理
        fpn = model.fpn

        # 检查FPN的各个组件
        print(f"\nFPN组件分析:")

        # 如果FPN有特定的处理步骤，我们需要手动执行
        # 这里我们先获取FPN的输出，然后分析
        fpn_features = fpn(backbone_features)
        print(f"FPN输出特征:")
        for i, feat in enumerate(fpn_features):
            print(f"  FPN特征{i}: {feat.shape}, 范围[{feat.min():.6f}, {feat.max():.6f}]")

        # 检查特征变化
        print(f"\n特征变化分析:")
        for i, (backbone_feat, fpn_feat) in enumerate(zip(backbone_features, fpn_features[:len(backbone_features)])):
            if backbone_feat.shape == fpn_feat.shape:
                diff = jt.abs(backbone_feat - fpn_feat).max()
                print(f"  特征{i}变化: {diff:.6f}")
            else:
                print(f"  特征{i}形状变化: {backbone_feat.shape} -> {fpn_feat.shape}")


def analyze_head_internal():
    """分析Head内部结构"""
    print(f"\n🔍 分析Head内部结构")
    print("=" * 60)

    # 创建模型
    model = create_jittor_model()

    # 创建测试输入
    input_data = create_test_input()
    jittor_input = jt.array(input_data)

    with jt.no_grad():
        # 获取FPN特征
        backbone_features = model.backbone(jittor_input)
        fpn_features = model.fpn(backbone_features)

        print(f"FPN特征输入到Head:")
        for i, feat in enumerate(fpn_features):
            print(f"  特征{i}: {feat.shape}, 范围[{feat.min():.6f}, {feat.max():.6f}]")

        # 分析Head的处理
        head = model.head

        # Head输出
        head_output = head(fpn_features)
        print(f"\nHead输出:")
        print(f"  Head输出: {head_output.shape}, 范围[{head_output.min():.6f}, {head_output.max():.6f}]")

        # 分析输出组成
        cls_preds = head_output[:, :, :20]
        reg_preds = head_output[:, :, 20:]

        print(f"  分类预测: {cls_preds.shape}, 范围[{cls_preds.min():.6f}, {cls_preds.max():.6f}]")
        print(f"  回归预测: {reg_preds.shape}, 范围[{reg_preds.min():.6f}, {reg_preds.max():.6f}]")

        # 分析置信度
        cls_scores = jt.sigmoid(cls_preds)
        print(f"  置信度: 范围[{cls_scores.min():.6f}, {cls_scores.max():.6f}]")
        print(f"  最高置信度: {cls_scores.max():.6f}")

        # 分析每个尺度的输出
        print(f"\n按尺度分析:")
        # 假设有4个尺度，每个尺度的anchor数量不同
        # 这需要根据具体的Head实现来调整
        total_anchors = head_output.shape[1]
        print(f"  总anchor数: {total_anchors}")


def main():
    """主函数"""
    print("🚀 开始详细层分析")

    # 分析前几层
    analyze_first_few_layers()

    # 详细检查BatchNorm
    check_batchnorm_detailed()

    # 分析FPN内部
    analyze_fpn_internal()

    # 分析Head内部
    analyze_head_internal()

    print(f"\n✅ 详细分析完成")


if __name__ == '__main__':
    main()
