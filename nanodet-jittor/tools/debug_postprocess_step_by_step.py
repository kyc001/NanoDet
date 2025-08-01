#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
逐步调试后处理算法
找出为什么模型输出高置信度但后处理后没有检测结果
"""

import os
import sys
import cv2
import torch
import jittor as jt
import numpy as np

# 添加路径
sys.path.append('/home/kyc/project/nanodet/nanodet-jittor')
from nanodet.model.arch.nanodet_plus import NanoDetPlus
from nanodet.util.postprocess_pytorch_aligned import nanodet_postprocess


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
    
    # 创建aux_head配置
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


def load_pytorch_weights_100_percent(model, checkpoint_path):
    """100%修复的权重加载函数"""
    print(f"加载PyTorch checkpoint: {checkpoint_path}")
    
    # 使用PyTorch加载checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)
    
    # 获取Jittor模型的参数字典
    jittor_state_dict = {}
    for name, param in model.named_parameters():
        jittor_state_dict[name] = param
    
    # 100%修复的权重加载
    loaded_count = 0
    skipped_count = 0
    scale_fixed_count = 0
    
    for pytorch_name, pytorch_param in state_dict.items():
        # 移除PyTorch特有的前缀
        jittor_name = pytorch_name
        if jittor_name.startswith("model."):
            jittor_name = jittor_name[6:]  # 移除"model."前缀
        
        # 跳过Jittor中不存在的BatchNorm统计参数
        if "num_batches_tracked" in jittor_name:
            skipped_count += 1
            continue
        
        # 跳过avg_model参数（权重平均相关）
        if jittor_name.startswith("avg_"):
            skipped_count += 1
            continue
        
        # 特殊处理：distribution_project.project参数在Jittor中不存在（已改为非参数）
        if "distribution_project.project" in jittor_name:
            skipped_count += 1
            continue
        
        if jittor_name in jittor_state_dict:
            jittor_param = jittor_state_dict[jittor_name]
            
            # 检查形状匹配
            if list(pytorch_param.shape) == list(jittor_param.shape):
                # 转换并加载参数
                jittor_param.assign(jt.array(pytorch_param.detach().numpy()))
                loaded_count += 1
            elif "scale" in jittor_name and len(pytorch_param.shape) == 0 and list(jittor_param.shape) == [1]:
                # 特殊处理Scale参数：PyTorch标量 -> Jittor 1维张量
                jittor_param.assign(jt.array([pytorch_param.detach().numpy()]))
                loaded_count += 1
                scale_fixed_count += 1
    
    print(f"✅ 成功加载: {loaded_count} 个参数")
    print(f"✅ Scale参数修复: {scale_fixed_count} 个")
    print(f"⏭️ 跳过无关: {skipped_count} 个参数")
    
    return True


def debug_postprocess_step_by_step():
    """逐步调试后处理算法"""
    print("🔍 开始逐步调试后处理算法")
    print("=" * 80)
    
    # 创建模型
    model = create_nanodet_model()
    
    # 加载权重
    checkpoint_path = "/home/kyc/project/nanodet/nanodet-pytorch/workspace/nanodet-plus-m_320_voc_bs64/model_best/model_best.ckpt"
    load_pytorch_weights_100_percent(model, checkpoint_path)
    
    # 设置为评估模式
    model.eval()
    
    # 创建测试图像
    test_img_path = "data/VOCdevkit/VOC2007/JPEGImages/000001.jpg"
    
    if not os.path.exists(test_img_path):
        print(f"❌ 测试图像不存在: {test_img_path}")
        # 创建一个随机图像进行测试
        test_img = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
        print("✓ 使用随机图像进行测试")
    else:
        test_img = cv2.imread(test_img_path)
        test_img = cv2.resize(test_img, (320, 320))
        print(f"✓ 使用真实图像进行测试: {test_img_path}")
    
    # 预处理
    img_tensor = jt.array(test_img.transpose(2, 0, 1)).unsqueeze(0).float()
    
    # 使用ImageNet归一化（之前测试显示这是最佳方式）
    mean = jt.array([123.675, 116.28, 103.53]).reshape(1, 3, 1, 1)
    std = jt.array([58.395, 57.12, 57.375]).reshape(1, 3, 1, 1)
    img_normalized = (img_tensor - mean) / std
    
    print(f"\n📊 输入分析:")
    print(f"   输入形状: {img_normalized.shape}")
    print(f"   输入数值范围: [{img_normalized.min():.6f}, {img_normalized.max():.6f}]")
    
    # 模型推理
    with jt.no_grad():
        output = model(img_normalized)
    
    print(f"\n📊 模型输出分析:")
    print(f"   输出形状: {output.shape}")
    print(f"   输出数值范围: [{output.min():.6f}, {output.max():.6f}]")
    
    # 分离分类和回归预测
    cls_preds = output[:, :, :20]  # [B, N, 20]
    reg_preds = output[:, :, 20:]  # [B, N, 32]
    
    print(f"\n📊 分离后的预测:")
    print(f"   分类预测形状: {cls_preds.shape}")
    print(f"   分类预测范围: [{cls_preds.min():.6f}, {cls_preds.max():.6f}]")
    print(f"   回归预测形状: {reg_preds.shape}")
    print(f"   回归预测范围: [{reg_preds.min():.6f}, {reg_preds.max():.6f}]")
    
    # 计算sigmoid后的分类分数
    cls_scores = jt.sigmoid(cls_preds)
    print(f"   Sigmoid后分类分数范围: [{cls_scores.min():.6f}, {cls_scores.max():.6f}]")
    print(f"   最高分类分数: {cls_scores.max():.6f}")
    
    # 统计高置信度预测
    high_conf_mask = cls_scores > 0.5
    high_conf_count = high_conf_mask.sum()
    print(f"   置信度>0.5的预测数: {high_conf_count}")
    
    if high_conf_count > 0:
        high_conf_indices = jt.where(high_conf_mask)
        print(f"   高置信度预测位置: 批次{high_conf_indices[0][:5]}, 锚点{high_conf_indices[1][:5]}, 类别{high_conf_indices[2][:5]}")
    
    # 逐步调试后处理
    print(f"\n🔍 逐步调试后处理:")
    
    # 步骤1: 调用后处理函数
    print(f"\n1️⃣ 调用nanodet_postprocess函数:")
    try:
        # 测试不同的置信度阈值
        test_thresholds = [0.001, 0.01, 0.05, 0.1]

        for threshold in test_thresholds:
            print(f"\n   测试阈值 {threshold}:")
            results = nanodet_postprocess(cls_preds, reg_preds, (320, 320), score_thr=threshold)
            print(f"     ✓ 后处理函数调用成功")
            print(f"     结果数量: {len(results)}")

            for i, (dets, labels) in enumerate(results):
                print(f"     批次{i}: {len(dets)}个检测, {len(labels)}个标签")
                if len(dets) > 0:
                    print(f"       检测框形状: {dets.shape}")
                    print(f"       标签形状: {labels.shape}")
                    print(f"       置信度范围: [{dets[:, 4].min():.6f}, {dets[:, 4].max():.6f}]")
                    # 显示前几个检测结果
                    for j in range(min(3, len(dets))):
                        bbox = dets[j][:4]
                        score = dets[j][4]
                        label = labels[j]
                        print(f"       检测{j+1}: bbox=[{bbox[0]:.1f},{bbox[1]:.1f},{bbox[2]:.1f},{bbox[3]:.1f}], score={score:.4f}, label={label}")
                else:
                    print(f"       ❌ 没有检测结果")

    except Exception as e:
        print(f"   ❌ 后处理函数调用失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 步骤2: 测试不同的置信度阈值
    print(f"\n2️⃣ 测试不同置信度阈值:")
    thresholds = [0.001, 0.01, 0.05, 0.1, 0.3, 0.5]
    
    for threshold in thresholds:
        # 简单的阈值过滤
        valid_mask = cls_scores.max(dim=-1)[0] > threshold
        valid_count = valid_mask.sum()
        print(f"   阈值{threshold:5.3f}: {valid_count}个有效预测")
    
    # 步骤3: 手动检查anchor生成
    print(f"\n3️⃣ 检查anchor生成:")
    
    # 这里需要实现anchor生成的检查
    # 暂时跳过，专注于后处理逻辑
    
    print(f"\n✅ 后处理调试完成")
    
    return True


def main():
    """主函数"""
    print("🚀 开始逐步调试后处理算法")
    
    success = debug_postprocess_step_by_step()
    
    if success:
        print("\n✅ 后处理调试完成")
    else:
        print("\n❌ 后处理调试失败")
    
    return success


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
