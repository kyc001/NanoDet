#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
交叉验证测试
验证Jittor模型迁移的正确性
"""

import os
import sys
import cv2
import torch
import jittor as jt
import numpy as np

# 添加路径
sys.path.insert(0, '/home/kyc/project/nanodet/nanodet-pytorch')
sys.path.append('/home/kyc/project/nanodet/nanodet-jittor')

# PyTorch版本导入
from nanodet.model.arch import build_model as build_pytorch_model
from nanodet.util import cfg as pytorch_cfg, load_config
from nanodet.util.postprocess import postprocess as pytorch_postprocess

# Jittor版本导入
from nanodet.model.arch.nanodet_plus import NanoDetPlus as JittorNanoDetPlus
from nanodet.util.postprocess_pytorch_aligned import nanodet_postprocess as jittor_postprocess


def create_pytorch_model():
    """创建PyTorch模型"""
    print("创建PyTorch模型...")
    
    config_path = "/home/kyc/project/nanodet/nanodet-pytorch/config/nanodet-plus-m_320_voc.yml"
    load_config(pytorch_cfg, config_path)
    
    model = build_pytorch_model(pytorch_cfg.model)
    
    # 加载权重
    checkpoint_path = "/home/kyc/project/nanodet/nanodet-pytorch/workspace/nanodet-plus-m_320_voc_bs64/model_best/model_best.ckpt"
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    # 移除前缀
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('model.', '') if key.startswith('model.') else key
        new_state_dict[new_key] = value
    
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    
    print("✓ PyTorch模型创建并加载权重成功")
    return model


def create_jittor_model():
    """创建Jittor模型"""
    print("创建Jittor模型...")
    
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
    
    model = JittorNanoDetPlus(backbone_cfg, fpn_cfg, aux_head_cfg, head_cfg)
    
    # 加载权重
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
    
    model.eval()
    print(f"✓ Jittor模型创建并加载权重成功 ({loaded_count}个参数)")
    return model


def prepare_test_image():
    """准备测试图像"""
    test_img_path = "data/VOCdevkit/VOC2007/JPEGImages/000001.jpg"
    
    if not os.path.exists(test_img_path):
        # 创建随机图像
        test_img = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
        print("✓ 使用随机图像进行测试")
    else:
        test_img = cv2.imread(test_img_path)
        test_img = cv2.resize(test_img, (320, 320))
        print(f"✓ 使用真实图像: {test_img_path}")
    
    return test_img


def cross_validation_test():
    """交叉验证测试"""
    print("🔍 开始交叉验证测试")
    print("=" * 80)
    
    # 准备测试图像
    test_img = prepare_test_image()
    
    # 预处理
    img_tensor_torch = torch.from_numpy(test_img.transpose(2, 0, 1)).unsqueeze(0).float()
    img_tensor_jittor = jt.array(test_img.transpose(2, 0, 1)).unsqueeze(0).float()
    
    # ImageNet归一化
    mean_torch = torch.tensor([123.675, 116.28, 103.53]).reshape(1, 3, 1, 1)
    std_torch = torch.tensor([58.395, 57.12, 57.375]).reshape(1, 3, 1, 1)
    img_normalized_torch = (img_tensor_torch - mean_torch) / std_torch
    
    mean_jittor = jt.array([123.675, 116.28, 103.53]).reshape(1, 3, 1, 1)
    std_jittor = jt.array([58.395, 57.12, 57.375]).reshape(1, 3, 1, 1)
    img_normalized_jittor = (img_tensor_jittor - mean_jittor) / std_jittor
    
    print(f"✓ 图像预处理完成")
    
    # 创建模型
    try:
        pytorch_model = create_pytorch_model()
    except Exception as e:
        print(f"❌ PyTorch模型创建失败: {e}")
        pytorch_model = None
    
    jittor_model = create_jittor_model()
    
    # 测试1: PyTorch模型推理
    if pytorch_model is not None:
        print(f"\n1️⃣ PyTorch模型推理:")
        with torch.no_grad():
            pytorch_output = pytorch_model(img_normalized_torch)
        
        print(f"   输出形状: {pytorch_output.shape}")
        print(f"   输出范围: [{pytorch_output.min():.6f}, {pytorch_output.max():.6f}]")
        
        # 分离分类和回归
        pytorch_cls = pytorch_output[:, :, :20]
        pytorch_reg = pytorch_output[:, :, 20:]
        pytorch_cls_scores = torch.sigmoid(pytorch_cls)
        print(f"   最高置信度: {pytorch_cls_scores.max():.6f}")
    else:
        print(f"\n1️⃣ PyTorch模型推理: 跳过（模型创建失败）")
        pytorch_output = None
    
    # 测试2: Jittor模型推理
    print(f"\n2️⃣ Jittor模型推理:")
    with jt.no_grad():
        jittor_output = jittor_model(img_normalized_jittor)
    
    print(f"   输出形状: {jittor_output.shape}")
    print(f"   输出范围: [{jittor_output.min():.6f}, {jittor_output.max():.6f}]")
    
    # 分离分类和回归
    jittor_cls = jittor_output[:, :, :20]
    jittor_reg = jittor_output[:, :, 20:]
    jittor_cls_scores = jt.sigmoid(jittor_cls)
    print(f"   最高置信度: {jittor_cls_scores.max():.6f}")
    
    # 测试3: 输出对比
    if pytorch_output is not None:
        print(f"\n3️⃣ 输出对比:")
        
        # 转换为numpy进行对比
        pytorch_np = pytorch_output.detach().numpy()
        jittor_np = jittor_output.numpy()
        
        # 计算差异
        diff = np.abs(pytorch_np - jittor_np)
        max_diff = diff.max()
        mean_diff = diff.mean()
        
        print(f"   最大差异: {max_diff:.6f}")
        print(f"   平均差异: {mean_diff:.6f}")
        
        if max_diff < 1e-4:
            print(f"   ✅ 输出高度一致 (差异 < 1e-4)")
        elif max_diff < 1e-2:
            print(f"   ⚠️ 输出基本一致 (差异 < 1e-2)")
        else:
            print(f"   ❌ 输出差异较大")
    
    # 测试4: Jittor后处理
    print(f"\n4️⃣ Jittor后处理测试:")
    try:
        jittor_results = jittor_postprocess(jittor_cls, jittor_reg, (320, 320), score_thr=0.001)
        
        total_detections = 0
        for dets, labels in jittor_results:
            total_detections += len(dets)
            if len(dets) > 0:
                print(f"   检测数量: {len(dets)}")
                print(f"   置信度范围: [{dets[:, 4].min():.6f}, {dets[:, 4].max():.6f}]")
        
        if total_detections > 0:
            print(f"   ✅ Jittor后处理成功 ({total_detections}个检测)")
        else:
            print(f"   ❌ Jittor后处理无检测结果")
    
    except Exception as e:
        print(f"   ❌ Jittor后处理失败: {e}")
    
    # 测试5: PyTorch后处理（如果可用）
    if pytorch_output is not None:
        print(f"\n5️⃣ PyTorch后处理测试:")
        try:
            # 这里需要实现PyTorch版本的后处理调用
            print(f"   ⚠️ PyTorch后处理接口需要进一步实现")
        except Exception as e:
            print(f"   ❌ PyTorch后处理失败: {e}")
    
    print(f"\n✅ 交叉验证测试完成")


def main():
    """主函数"""
    print("🚀 开始交叉验证测试")
    
    cross_validation_test()
    
    print("\n✅ 交叉验证完成")


if __name__ == '__main__':
    main()
