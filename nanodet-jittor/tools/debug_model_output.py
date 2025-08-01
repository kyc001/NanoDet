#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
调试模型输出
深入分析为什么模型没有产生检测结果
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


def create_jittor_model():
    """创建Jittor模型并加载微调权重"""
    print("🔍 创建Jittor模型...")
    
    backbone_cfg = {
        'name': 'ShuffleNetV2',
        'model_size': '1.0x',
        'out_stages': [2, 3, 4],
        'activation': 'LeakyReLU',
        'pretrain': False  # 不加载ImageNet预训练
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
    
    # 加载微调后的权重
    print("加载微调后的PyTorch权重...")
    checkpoint_path = "/home/kyc/project/nanodet/nanodet-pytorch/workspace/nanodet-plus-m_320_voc_bs64/model_best/model_best.ckpt"
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)
    
    print(f"PyTorch checkpoint包含 {len(state_dict)} 个参数")
    
    # 检查一些关键权重
    print("检查关键权重:")
    for name in ['model.backbone.stage2.0.branch1.0.weight', 'model.head.gfl_cls.0.weight', 'model.head.gfl_cls.0.bias']:
        if name in state_dict:
            param = state_dict[name]
            print(f"  {name}: {param.shape}, 范围[{param.min():.6f}, {param.max():.6f}]")
    
    jittor_state_dict = {}
    for name, param in model.named_parameters():
        jittor_state_dict[name] = param
    
    print(f"Jittor模型包含 {len(jittor_state_dict)} 个参数")
    
    # 权重加载
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
                jittor_param.assign(jt.array(pytorch_param.detach().numpy()))
                loaded_count += 1
            elif "scale" in jittor_name and len(pytorch_param.shape) == 0 and list(jittor_param.shape) == [1]:
                jittor_param.assign(jt.array([pytorch_param.detach().numpy()]))
                loaded_count += 1
    
    print(f"✅ 权重加载: {loaded_count}/{total_count} ({loaded_count/total_count*100:.1f}%)")
    
    # 验证关键权重是否正确加载
    print("验证关键权重加载:")
    for jittor_name in ['head.gfl_cls.0.weight', 'head.gfl_cls.0.bias']:
        if jittor_name in jittor_state_dict:
            param = jittor_state_dict[jittor_name].numpy()
            print(f"  {jittor_name}: {param.shape}, 范围[{param.min():.6f}, {param.max():.6f}]")
    
    model.eval()
    return model


def debug_model_inference():
    """调试模型推理过程"""
    print("\n🔍 调试模型推理过程")
    print("=" * 60)
    
    model = create_jittor_model()
    
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
        
        # 分析分类和回归预测
        cls_preds = output[:, :, :20]  # [1, num_anchors, 20]
        reg_preds = output[:, :, 20:]  # [1, num_anchors, 32]
        
        print(f"  分类预测: {cls_preds.shape}, 范围[{cls_preds.min():.6f}, {cls_preds.max():.6f}]")
        print(f"  回归预测: {reg_preds.shape}, 范围[{reg_preds.min():.6f}, {reg_preds.max():.6f}]")
        
        # 计算置信度
        cls_scores = jt.sigmoid(cls_preds)
        print(f"  置信度: 范围[{cls_scores.min():.6f}, {cls_scores.max():.6f}]")
        
        # 统计置信度分布
        cls_scores_np = cls_scores.numpy()
        print(f"  置信度统计:")
        print(f"    均值: {cls_scores_np.mean():.6f}")
        print(f"    标准差: {cls_scores_np.std():.6f}")
        print(f"    >0.01的数量: {(cls_scores_np > 0.01).sum()}")
        print(f"    >0.05的数量: {(cls_scores_np > 0.05).sum()}")
        print(f"    >0.1的数量: {(cls_scores_np > 0.1).sum()}")
        print(f"    >0.5的数量: {(cls_scores_np > 0.5).sum()}")
        
        # 找出最高置信度的预测
        max_conf_idx = np.unravel_index(np.argmax(cls_scores_np), cls_scores_np.shape)
        max_conf = cls_scores_np[max_conf_idx]
        
        print(f"  最高置信度预测:")
        print(f"    位置: {max_conf_idx}")
        print(f"    置信度: {max_conf:.6f}")
        print(f"    类别: {max_conf_idx[2]}")
        
        # 检查不同置信度阈值下的检测数量
        print(f"\n不同置信度阈值下的检测数量:")
        for threshold in [0.01, 0.05, 0.1, 0.3, 0.5]:
            max_scores = jt.max(cls_scores, dim=2)[0]  # [1, num_anchors]
            valid_detections = (max_scores > threshold).sum()
            print(f"    阈值 {threshold}: {valid_detections} 个检测")
        
        return output, cls_scores


def test_with_real_image():
    """使用真实图像测试"""
    print(f"\n🔍 使用真实图像测试")
    print("=" * 60)
    
    model = create_jittor_model()
    
    # 加载一张真实的VOC图像
    voc_root = "/home/kyc/project/nanodet/data/VOCdevkit/VOC2007"
    val_file = os.path.join(voc_root, "ImageSets/Main/val.txt")
    
    with open(val_file, 'r') as f:
        image_ids = [line.strip() for line in f.readlines()]
    
    # 使用第一张图像
    image_id = image_ids[0]
    image_path = os.path.join(voc_root, f"JPEGImages/{image_id}.jpg")
    
    print(f"测试图像: {image_path}")
    
    # 预处理图像
    image = cv2.imread(image_path)
    original_height, original_width = image.shape[:2]
    
    # 调整大小
    input_size = 320
    scale = min(input_size / original_width, input_size / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    image = cv2.resize(image, (new_width, new_height))
    
    # 填充
    padded_image = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    padded_image[:new_height, :new_width] = image
    
    # 归一化
    image = cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    
    # 使用与PyTorch训练时相同的归一化参数
    mean = np.array([103.53, 116.28, 123.675])
    std = np.array([57.375, 57.12, 58.395])
    image = (image - mean) / std
    
    # 转换为CHW格式
    image = image.transpose(2, 0, 1)
    image = image[np.newaxis, ...]
    
    print(f"预处理后: {image.shape}, 范围[{image.min():.6f}, {image.max():.6f}]")
    
    jittor_input = jt.array(image)
    
    with jt.no_grad():
        output = model(jittor_input)
        
        # 分析输出
        cls_preds = output[:, :, :20]
        cls_scores = jt.sigmoid(cls_preds)
        
        max_conf = float(cls_scores.max().numpy())
        mean_conf = float(cls_scores.mean().numpy())
        
        print(f"真实图像推理结果:")
        print(f"  最高置信度: {max_conf:.6f}")
        print(f"  平均置信度: {mean_conf:.6f}")
        
        # 检查不同阈值下的检测数量
        for threshold in [0.01, 0.05, 0.1, 0.3, 0.5]:
            max_scores = jt.max(cls_scores, dim=2)[0]
            valid_detections = (max_scores > threshold).sum()
            print(f"  阈值 {threshold}: {valid_detections} 个检测")


def main():
    """主函数"""
    print("🚀 开始调试模型输出")
    print("目标: 找出为什么模型没有产生检测结果")
    
    try:
        # 1. 调试模型推理
        output, cls_scores = debug_model_inference()
        
        # 2. 使用真实图像测试
        test_with_real_image()
        
        print(f"\n🎯 调试总结:")
        print("=" * 60)
        
        max_conf = float(cls_scores.max().numpy())
        
        if max_conf < 0.01:
            print(f"  ❌ 最高置信度过低 ({max_conf:.6f})")
            print(f"  可能原因:")
            print(f"    1. 权重加载有问题")
            print(f"    2. 模型架构不匹配")
            print(f"    3. 预处理不一致")
            print(f"    4. Head的bias初始化问题")
        elif max_conf < 0.05:
            print(f"  ⚠️ 最高置信度偏低 ({max_conf:.6f})")
            print(f"  需要调整置信度阈值或优化模型")
        else:
            print(f"  ✅ 最高置信度正常 ({max_conf:.6f})")
            print(f"  问题可能在后处理流程")
        
        print(f"\n✅ 调试完成")
        
    except Exception as e:
        print(f"❌ 调试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
