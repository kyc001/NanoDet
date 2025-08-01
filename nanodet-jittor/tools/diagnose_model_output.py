#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
诊断模型输出问题
深入分析为什么mAP为0
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
    
    # 创建aux_head配置 - 使用正确的SimpleConvHead
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


def diagnose_model_output():
    """诊断模型输出"""
    print("🔍 开始诊断模型输出问题")
    
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
    
    print(f"\n📊 输入分析:")
    print(f"   原始图像形状: {test_img.shape}")
    print(f"   输入张量形状: {img_tensor.shape}")
    print(f"   输入数值范围: [{img_tensor.min():.2f}, {img_tensor.max():.2f}]")
    
    # 测试不同的归一化方式
    print(f"\n🔍 测试不同的预处理方式:")
    
    # 方式1: 无归一化
    print(f"\n1️⃣ 无归一化:")
    with jt.no_grad():
        output1 = model(img_tensor)
    
    print(f"   输出形状: {output1.shape}")
    print(f"   输出数值范围: [{output1.min():.6f}, {output1.max():.6f}]")
    
    # 分析分类和回归输出
    cls_preds1 = output1[:, :, :20]  # [B, N, 20]
    reg_preds1 = output1[:, :, 20:]  # [B, N, 32]
    
    print(f"   分类预测形状: {cls_preds1.shape}")
    print(f"   分类预测范围: [{cls_preds1.min():.6f}, {cls_preds1.max():.6f}]")
    print(f"   回归预测形状: {reg_preds1.shape}")
    print(f"   回归预测范围: [{reg_preds1.min():.6f}, {reg_preds1.max():.6f}]")
    
    # 计算sigmoid后的分类分数
    cls_scores1 = jt.sigmoid(cls_preds1)
    print(f"   Sigmoid后分类分数范围: [{cls_scores1.min():.6f}, {cls_scores1.max():.6f}]")
    print(f"   最高分类分数: {cls_scores1.max():.6f}")
    
    # 方式2: ImageNet归一化
    print(f"\n2️⃣ ImageNet归一化:")
    mean = jt.array([123.675, 116.28, 103.53]).reshape(1, 3, 1, 1)
    std = jt.array([58.395, 57.12, 57.375]).reshape(1, 3, 1, 1)
    img_normalized = (img_tensor - mean) / std
    
    print(f"   归一化后数值范围: [{img_normalized.min():.6f}, {img_normalized.max():.6f}]")
    
    with jt.no_grad():
        output2 = model(img_normalized)
    
    print(f"   输出形状: {output2.shape}")
    print(f"   输出数值范围: [{output2.min():.6f}, {output2.max():.6f}]")
    
    cls_preds2 = output2[:, :, :20]
    cls_scores2 = jt.sigmoid(cls_preds2)
    print(f"   Sigmoid后分类分数范围: [{cls_scores2.min():.6f}, {cls_scores2.max():.6f}]")
    print(f"   最高分类分数: {cls_scores2.max():.6f}")
    
    # 方式3: 0-1归一化
    print(f"\n3️⃣ 0-1归一化:")
    img_01 = img_tensor / 255.0
    
    print(f"   归一化后数值范围: [{img_01.min():.6f}, {img_01.max():.6f}]")
    
    with jt.no_grad():
        output3 = model(img_01)
    
    print(f"   输出形状: {output3.shape}")
    print(f"   输出数值范围: [{output3.min():.6f}, {output3.max():.6f}]")
    
    cls_preds3 = output3[:, :, :20]
    cls_scores3 = jt.sigmoid(cls_preds3)
    print(f"   Sigmoid后分类分数范围: [{cls_scores3.min():.6f}, {cls_scores3.max():.6f}]")
    print(f"   最高分类分数: {cls_scores3.max():.6f}")
    
    # 分析哪种方式产生最高的置信度
    max_scores = [
        float(cls_scores1.max()),
        float(cls_scores2.max()),
        float(cls_scores3.max())
    ]
    
    print(f"\n📊 不同预处理方式的最高置信度对比:")
    print(f"   无归一化: {max_scores[0]:.6f}")
    print(f"   ImageNet归一化: {max_scores[1]:.6f}")
    print(f"   0-1归一化: {max_scores[2]:.6f}")
    
    best_method = np.argmax(max_scores)
    method_names = ["无归一化", "ImageNet归一化", "0-1归一化"]
    print(f"   🏆 最佳方式: {method_names[best_method]} (置信度: {max_scores[best_method]:.6f})")
    
    # 使用最佳方式进行后处理测试
    print(f"\n🔍 使用最佳方式进行后处理测试:")
    
    if best_method == 0:
        best_output = output1
    elif best_method == 1:
        best_output = output2
    else:
        best_output = output3
    
    # 测试不同的置信度阈值
    thresholds = [0.001, 0.01, 0.05, 0.1, 0.3, 0.5]
    
    for threshold in thresholds:
        cls_preds = best_output[:, :, :20]
        reg_preds = best_output[:, :, 20:]
        
        # 使用真正的NanoDet后处理
        results = nanodet_postprocess(cls_preds, reg_preds, (320, 320))
        
        # 统计检测数量
        total_detections = 0
        for dets, labels in results:
            valid_dets = dets[dets[:, 4] > threshold]
            total_detections += len(valid_dets)
        
        print(f"   阈值 {threshold:5.3f}: {total_detections} 个检测")
    
    print(f"\n✅ 诊断完成!")
    
    return True


if __name__ == '__main__':
    success = diagnose_model_output()
    sys.exit(0 if success else 1)
