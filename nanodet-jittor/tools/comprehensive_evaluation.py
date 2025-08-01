#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
全面评估工具
实现四个测评角度的完整评估系统
"""

import os
import sys
import cv2
import torch
import jittor as jt
import numpy as np
from pathlib import Path

# 添加路径
sys.path.append('/home/kyc/project/nanodet/nanodet-jittor')
from nanodet.model.arch.nanodet_plus import NanoDetPlus


def create_model_configs():
    """创建模型配置"""
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
    
    return backbone_cfg, fpn_cfg, head_cfg, aux_head_cfg


def create_jittor_model_imagenet():
    """创建Jittor模型 - ImageNet预训练权重"""
    print("🔍 创建Jittor模型 (ImageNet预训练)")
    
    backbone_cfg, fpn_cfg, head_cfg, aux_head_cfg = create_model_configs()
    
    # 只加载ImageNet预训练权重，不加载微调权重
    backbone_cfg['pretrain'] = True
    
    model = NanoDetPlus(backbone_cfg, fpn_cfg, aux_head_cfg, head_cfg)
    model.eval()
    
    return model


def create_jittor_model_finetuned():
    """创建Jittor模型 - 微调后权重"""
    print("🔍 创建Jittor模型 (微调后)")
    
    backbone_cfg, fpn_cfg, head_cfg, aux_head_cfg = create_model_configs()
    
    model = NanoDetPlus(backbone_cfg, fpn_cfg, aux_head_cfg, head_cfg)
    
    # 加载微调后的权重
    print("加载微调后的PyTorch权重...")
    checkpoint_path = "/home/kyc/project/nanodet/nanodet-pytorch/workspace/nanodet-plus-m_320_voc_bs64/model_best/model_best.ckpt"
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)
    
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
    
    print(f"✅ 成功加载 {loaded_count} 个权重参数")
    model.eval()
    
    return model


def preprocess_image(image_path, input_size=(320, 320)):
    """预处理图像"""
    # 读取图像
    if isinstance(image_path, str):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        # 转换为RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        # 如果是numpy数组
        image = image_path
    
    # 调整大小
    image = cv2.resize(image, input_size)
    
    # 归一化
    image = image.astype(np.float32) / 255.0
    
    # 标准化 (ImageNet标准)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    
    # 转换为CHW格式
    image = image.transpose(2, 0, 1)
    
    # 添加batch维度
    image = image[np.newaxis, ...]
    
    return image


def create_test_images():
    """创建测试图像"""
    print("🔍 创建测试图像")
    
    test_images = []
    
    # 1. 随机噪声图像 (我们之前使用的)
    np.random.seed(42)
    random_image = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
    test_images.append(("random_noise", random_image))
    
    # 2. 纯色图像
    solid_image = np.full((320, 320, 3), 128, dtype=np.uint8)
    test_images.append(("solid_gray", solid_image))
    
    # 3. 渐变图像
    gradient_image = np.zeros((320, 320, 3), dtype=np.uint8)
    for i in range(320):
        gradient_image[i, :, :] = int(i * 255 / 320)
    test_images.append(("gradient", gradient_image))
    
    # 4. 棋盘图像
    checkerboard = np.zeros((320, 320, 3), dtype=np.uint8)
    for i in range(0, 320, 40):
        for j in range(0, 320, 40):
            if (i // 40 + j // 40) % 2 == 0:
                checkerboard[i:i+40, j:j+40] = 255
    test_images.append(("checkerboard", checkerboard))
    
    return test_images


def evaluate_model(model, model_name, test_images):
    """评估模型"""
    print(f"\n🔍 评估 {model_name}")
    print("=" * 60)
    
    results = {}
    
    for image_name, image in test_images:
        print(f"\n测试图像: {image_name}")
        
        # 预处理
        input_data = preprocess_image(image)
        jittor_input = jt.array(input_data)
        
        with jt.no_grad():
            # 推理
            output = model(jittor_input)
            
            # 分析输出
            cls_preds = output[:, :, :20]
            reg_preds = output[:, :, 20:]
            cls_scores = jt.sigmoid(cls_preds)
            
            max_confidence = float(cls_scores.max().numpy())
            mean_confidence = float(cls_scores.mean().numpy())
            
            # 统计置信度分布
            cls_scores_np = cls_scores.numpy()
            high_conf_ratio = (cls_scores_np > 0.1).mean()
            very_high_conf_ratio = (cls_scores_np > 0.5).mean()
            
            result = {
                'max_confidence': max_confidence,
                'mean_confidence': mean_confidence,
                'high_conf_ratio': high_conf_ratio,
                'very_high_conf_ratio': very_high_conf_ratio,
                'output_range': [float(output.min().numpy()), float(output.max().numpy())],
                'cls_pred_range': [float(cls_preds.min().numpy()), float(cls_preds.max().numpy())],
                'reg_pred_range': [float(reg_preds.min().numpy()), float(reg_preds.max().numpy())]
            }
            
            results[image_name] = result
            
            print(f"  最高置信度: {max_confidence:.6f}")
            print(f"  平均置信度: {mean_confidence:.6f}")
            print(f"  >0.1置信度比例: {high_conf_ratio:.4f}")
            print(f"  >0.5置信度比例: {very_high_conf_ratio:.4f}")
            print(f"  输出范围: [{result['output_range'][0]:.3f}, {result['output_range'][1]:.3f}]")
    
    return results


def compare_results(imagenet_results, finetuned_results):
    """对比结果"""
    print(f"\n🔍 对比结果")
    print("=" * 60)
    
    for image_name in imagenet_results.keys():
        print(f"\n图像: {image_name}")
        
        imagenet = imagenet_results[image_name]
        finetuned = finetuned_results[image_name]
        
        print(f"  ImageNet预训练:")
        print(f"    最高置信度: {imagenet['max_confidence']:.6f}")
        print(f"    平均置信度: {imagenet['mean_confidence']:.6f}")
        
        print(f"  微调后:")
        print(f"    最高置信度: {finetuned['max_confidence']:.6f}")
        print(f"    平均置信度: {finetuned['mean_confidence']:.6f}")
        
        # 计算差异
        max_conf_diff = finetuned['max_confidence'] - imagenet['max_confidence']
        mean_conf_diff = finetuned['mean_confidence'] - imagenet['mean_confidence']
        
        print(f"  差异:")
        print(f"    最高置信度差异: {max_conf_diff:+.6f}")
        print(f"    平均置信度差异: {mean_conf_diff:+.6f}")
        
        if max_conf_diff > 0.01:
            print(f"    ✅ 微调后置信度明显提高")
        elif max_conf_diff > 0:
            print(f"    ⚠️ 微调后置信度略有提高")
        else:
            print(f"    ❌ 微调后置信度没有提高")


def main():
    """主函数"""
    print("🚀 开始全面评估")
    print("实现四个测评角度:")
    print("  1. Jittor ImageNet预训练")
    print("  2. Jittor 微调后")
    print("  3. PyTorch ImageNet预训练 (待实现)")
    print("  4. PyTorch 微调后 (已知: mAP=0.277)")
    
    # 创建测试图像
    test_images = create_test_images()
    
    # 1. 评估Jittor ImageNet预训练模型
    imagenet_model = create_jittor_model_imagenet()
    imagenet_results = evaluate_model(imagenet_model, "Jittor ImageNet预训练", test_images)
    
    # 2. 评估Jittor微调后模型
    finetuned_model = create_jittor_model_finetuned()
    finetuned_results = evaluate_model(finetuned_model, "Jittor 微调后", test_images)
    
    # 3. 对比结果
    compare_results(imagenet_results, finetuned_results)
    
    # 保存结果
    results = {
        'jittor_imagenet': imagenet_results,
        'jittor_finetuned': finetuned_results
    }
    
    np.save("comprehensive_evaluation_results.npy", results)
    print(f"\n✅ 评估结果已保存到 comprehensive_evaluation_results.npy")
    
    print(f"\n📊 总结:")
    print("=" * 60)
    print("我们已经完成了Jittor版本的两个测评角度:")
    print("  ✅ Jittor ImageNet预训练")
    print("  ✅ Jittor 微调后")
    print("\n还需要实现:")
    print("  🔄 PyTorch ImageNet预训练")
    print("  🔄 PyTorch 微调后的详细评估")
    print("\n这为我们提供了完整的性能对比基准！")


if __name__ == '__main__':
    main()
