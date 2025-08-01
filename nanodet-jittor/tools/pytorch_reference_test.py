#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PyTorch版本参考测试
记录所有细节：参数数量、函数调用、mAP结果
作为Jittor版本对齐的标准参考
"""

import os
import sys
import json
import cv2
import torch
import numpy as np
from collections import defaultdict

# 添加PyTorch版本路径
sys.path.insert(0, '/home/kyc/project/nanodet/nanodet-pytorch')

# 导入PyTorch版本的模块
from nanodet.model.arch import build_model
from nanodet.util import cfg, load_config
from nanodet.data.transform import Pipeline


def load_pytorch_config():
    """加载PyTorch配置"""
    config_path = "/home/kyc/project/nanodet/nanodet-pytorch/config/nanodet-plus-m_320_voc.yml"
    
    print(f"加载PyTorch配置: {config_path}")
    
    # 加载配置
    load_config(cfg, config_path)
    
    print("✓ PyTorch配置加载成功")
    print(f"  模型类型: {cfg.model.arch.name}")
    print(f"  输入尺寸: {cfg.data.train.input_size}")
    print(f"  类别数量: {cfg.model.arch.head.num_classes}")
    
    return cfg


def create_pytorch_model(cfg):
    """创建PyTorch模型"""
    print("创建PyTorch模型...")
    
    # 创建模型
    model = build_model(cfg.model)
    
    print("✓ PyTorch模型创建成功")
    
    # 统计参数数量
    total_params = 0
    trainable_params = 0
    
    param_details = {}
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        
        if param.requires_grad:
            trainable_params += param_count
        
        # 记录参数详情
        param_details[name] = {
            'shape': list(param.shape),
            'count': param_count,
            'requires_grad': param.requires_grad,
            'dtype': str(param.dtype)
        }
    
    print(f"📊 PyTorch模型参数统计:")
    print(f"  总参数数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  参数项数量: {len(param_details)}")
    
    # 按模块分组统计
    module_stats = defaultdict(int)
    for name, details in param_details.items():
        module_name = name.split('.')[0]
        module_stats[module_name] += details['count']
    
    print(f"\n📊 按模块统计:")
    for module, count in sorted(module_stats.items()):
        print(f"  {module}: {count:,} 参数")
    
    return model, param_details


def load_pytorch_checkpoint(model, checkpoint_path):
    """加载PyTorch checkpoint"""
    print(f"加载PyTorch checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ checkpoint文件不存在: {checkpoint_path}")
        return False
    
    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    print(f"✓ checkpoint包含 {len(state_dict)} 个参数")
    
    # 分析checkpoint参数
    checkpoint_details = {}
    for name, param in state_dict.items():
        checkpoint_details[name] = {
            'shape': list(param.shape),
            'count': param.numel(),
            'dtype': str(param.dtype)
        }
    
    # 加载到模型
    try:
        # 移除可能的前缀
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('model.', '') if key.startswith('model.') else key
            new_state_dict[new_key] = value
        
        # 加载参数
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        
        print(f"✓ PyTorch模型权重加载完成")
        print(f"  缺失参数: {len(missing_keys)}")
        print(f"  多余参数: {len(unexpected_keys)}")
        
        if len(missing_keys) > 0:
            print(f"  缺失参数示例: {missing_keys[:5]}")
        if len(unexpected_keys) > 0:
            print(f"  多余参数示例: {unexpected_keys[:5]}")
        
        return True, checkpoint_details
        
    except Exception as e:
        print(f"❌ 权重加载失败: {e}")
        return False, {}


def test_pytorch_inference(model, cfg):
    """测试PyTorch推理"""
    print("\n🔍 测试PyTorch推理...")
    
    # 设置为评估模式
    model.eval()
    
    # 创建测试输入
    input_size = cfg.data.train.input_size  # [w, h]
    test_input = torch.randn(1, 3, input_size[1], input_size[0])  # [B, C, H, W]
    
    print(f"  输入形状: {test_input.shape}")
    print(f"  输入数值范围: [{test_input.min():.6f}, {test_input.max():.6f}]")
    
    # 推理
    with torch.no_grad():
        output = model(test_input)
    
    print(f"✅ PyTorch推理成功!")
    print(f"  输出形状: {output.shape}")
    print(f"  输出数值范围: [{output.min():.6f}, {output.max():.6f}]")
    
    # 分析输出通道
    if len(output.shape) == 3:  # [B, N, C]
        batch_size, num_anchors, num_channels = output.shape
        print(f"  批次大小: {batch_size}")
        print(f"  锚点数量: {num_anchors}")
        print(f"  输出通道: {num_channels}")
        
        # 分析通道分配
        num_classes = cfg.model.arch.head.num_classes
        reg_max = cfg.model.arch.head.reg_max
        expected_cls_channels = num_classes
        expected_reg_channels = 4 * (reg_max + 1)
        expected_total = expected_cls_channels + expected_reg_channels
        
        print(f"\n🔹 通道分析:")
        print(f"  类别数: {num_classes}")
        print(f"  reg_max: {reg_max}")
        print(f"  期望分类通道: {expected_cls_channels}")
        print(f"  期望回归通道: {expected_reg_channels}")
        print(f"  期望总通道: {expected_total}")
        print(f"  实际总通道: {num_channels}")
        
        if num_channels == expected_total:
            print("✅ 输出通道数正确")
        else:
            print("❌ 输出通道数不正确")
    
    return output


def test_pytorch_postprocess(model, cfg):
    """测试PyTorch后处理"""
    print("\n🔍 测试PyTorch后处理...")
    
    # 加载一张真实图像
    test_img_path = "data/VOCdevkit/VOC2007/JPEGImages/000001.jpg"
    
    if not os.path.exists(test_img_path):
        print(f"❌ 测试图像不存在: {test_img_path}")
        return None
    
    # 读取图像
    img = cv2.imread(test_img_path)
    original_shape = img.shape[:2]  # (H, W)
    
    print(f"  原始图像形状: {original_shape}")
    
    # 预处理
    input_size = cfg.data.train.input_size  # [w, h]
    img_resized = cv2.resize(img, tuple(input_size))
    
    # 转换为tensor
    img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).unsqueeze(0).float()
    
    # 测试不同的归一化方式
    print(f"\n📊 测试不同预处理方式:")
    
    # 方式1: 无归一化
    print(f"\n1️⃣ 无归一化:")
    with torch.no_grad():
        output1 = model(img_tensor)
    
    cls_preds1 = output1[:, :, :cfg.model.arch.head.num_classes]
    cls_scores1 = torch.sigmoid(cls_preds1)
    max_score1 = cls_scores1.max().item()
    print(f"  最高置信度: {max_score1:.6f}")
    
    # 方式2: ImageNet归一化
    print(f"\n2️⃣ ImageNet归一化:")
    mean = torch.tensor([123.675, 116.28, 103.53]).reshape(1, 3, 1, 1)
    std = torch.tensor([58.395, 57.12, 57.375]).reshape(1, 3, 1, 1)
    img_normalized = (img_tensor - mean) / std
    
    with torch.no_grad():
        output2 = model(img_normalized)
    
    cls_preds2 = output2[:, :, :cfg.model.arch.head.num_classes]
    cls_scores2 = torch.sigmoid(cls_preds2)
    max_score2 = cls_scores2.max().item()
    print(f"  最高置信度: {max_score2:.6f}")
    
    # 方式3: 0-1归一化
    print(f"\n3️⃣ 0-1归一化:")
    img_01 = img_tensor / 255.0
    
    with torch.no_grad():
        output3 = model(img_01)
    
    cls_preds3 = output3[:, :, :cfg.model.arch.head.num_classes]
    cls_scores3 = torch.sigmoid(cls_preds3)
    max_score3 = cls_scores3.max().item()
    print(f"  最高置信度: {max_score3:.6f}")
    
    # 选择最佳方式
    max_scores = [max_score1, max_score2, max_score3]
    best_method = np.argmax(max_scores)
    method_names = ["无归一化", "ImageNet归一化", "0-1归一化"]
    
    print(f"\n🏆 最佳预处理方式: {method_names[best_method]} (置信度: {max_scores[best_method]:.6f})")
    
    # 使用最佳方式的输出
    if best_method == 0:
        best_output = output1
    elif best_method == 1:
        best_output = output2
    else:
        best_output = output3
    
    return best_output, method_names[best_method], max_scores[best_method]


def analyze_pytorch_postprocess_functions():
    """分析PyTorch版本使用的后处理函数"""
    print("\n🔍 分析PyTorch后处理函数...")
    
    # 检查PyTorch版本的后处理模块
    try:
        from nanodet.util.postprocess import postprocess
        print("✓ 找到PyTorch后处理函数: nanodet.util.postprocess.postprocess")
    except ImportError:
        print("❌ 未找到标准后处理函数")
    
    # 检查其他可能的后处理模块
    postprocess_modules = [
        'nanodet.util.postprocess',
        'nanodet.model.head.nanodet_plus_head',
        'nanodet.util.nms',
        'nanodet.util.bbox_util'
    ]
    
    for module_name in postprocess_modules:
        try:
            module = __import__(module_name, fromlist=[''])
            functions = [name for name in dir(module) if not name.startswith('_')]
            print(f"✓ 模块 {module_name} 包含函数: {functions[:10]}")
        except ImportError:
            print(f"❌ 模块 {module_name} 不存在")


def main():
    """主函数"""
    print("🚀 开始PyTorch版本参考测试")
    print("=" * 80)
    
    # 1. 加载配置
    cfg = load_pytorch_config()
    
    # 2. 创建模型
    model, param_details = create_pytorch_model(cfg)
    
    # 3. 加载权重
    checkpoint_path = "/home/kyc/project/nanodet/nanodet-pytorch/workspace/nanodet-plus-m_320_voc_bs64/model_best/model_best.ckpt"
    success, checkpoint_details = load_pytorch_checkpoint(model, checkpoint_path)
    
    if not success:
        print("❌ PyTorch模型加载失败")
        return False
    
    # 4. 测试推理
    output = test_pytorch_inference(model, cfg)
    
    # 5. 测试后处理
    postprocess_result = test_pytorch_postprocess(model, cfg)
    
    # 6. 分析后处理函数
    analyze_pytorch_postprocess_functions()
    
    # 7. 保存详细记录
    record = {
        'config': {
            'model_name': cfg.model.arch.name,
            'input_size': cfg.data.train.input_size,
            'num_classes': cfg.model.arch.head.num_classes,
            'reg_max': cfg.model.arch.head.reg_max
        },
        'model_params': param_details,
        'checkpoint_params': checkpoint_details,
        'inference_result': {
            'output_shape': list(output.shape),
            'output_range': [float(output.min()), float(output.max())]
        }
    }
    
    if postprocess_result:
        best_output, best_method, best_score = postprocess_result
        record['postprocess_result'] = {
            'best_method': best_method,
            'best_score': best_score,
            'output_shape': list(best_output.shape)
        }
    
    # 保存记录
    with open('pytorch_reference_record.json', 'w') as f:
        json.dump(record, f, indent=2)
    
    print("\n✅ PyTorch参考测试完成!")
    print("  详细记录已保存到: pytorch_reference_record.json")
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
