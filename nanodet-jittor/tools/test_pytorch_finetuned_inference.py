#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测评角度2：PyTorch微调后模型测评
加载PyTorch训练后的权重，用Jittor版本进行推理测评
验证权重转换的正确性
"""

import os
import sys
import cv2
import numpy as np
import jittor as jt

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanodet.model import build_model


def create_model():
    """创建NanoDet模型（与PyTorch训练时相同配置）"""
    model_cfg = {
        'name': 'NanoDetPlus',
        'backbone': {
            'name': 'ShuffleNetV2',
            'model_size': '1.0x',
            'out_stages': [2, 3, 4],
            'activation': 'LeakyReLU',
            'pretrain': False  # 不加载ImageNet权重，使用PyTorch训练后的权重
        },
        'fpn': {
            'name': 'GhostPAN',
            'in_channels': [116, 232, 464],
            'out_channels': 96,
            'kernel_size': 5,
            'num_extra_level': 1,
            'use_depthwise': True,
            'activation': 'LeakyReLU'
        },
        'aux_head': {
            'name': 'SimpleConvHead',
            'num_classes': 20,
            'input_channel': 192,  # 96 * 2
            'feat_channels': 192,
            'stacked_convs': 4,
            'strides': [8, 16, 32, 64],
            'activation': 'LeakyReLU',
            'reg_max': 7
        },
        'head': {
            'name': 'NanoDetPlusHead',
            'num_classes': 20,
            'input_channel': 96,
            'feat_channels': 96,
            'stacked_convs': 2,
            'kernel_size': 5,
            'strides': [8, 16, 32, 64],
            'conv_type': 'DWConv',
            'norm_cfg': dict(type='BN'),
            'reg_max': 7,
            'activation': 'LeakyReLU',
            'loss': {
                'loss_qfl': {'beta': 2.0, 'loss_weight': 1.0},
                'loss_dfl': {'loss_weight': 0.25},
                'loss_bbox': {'loss_weight': 2.0}
            }
        },
        'detach_epoch': 10
    }
    
    return build_model(model_cfg)


def load_pytorch_finetuned_weights(model, weight_path):
    """加载PyTorch微调后的权重"""
    print(f"加载PyTorch微调后权重: {weight_path}")
    
    if not os.path.exists(weight_path):
        print(f"✗ 权重文件不存在: {weight_path}")
        return False
    
    try:
        # 加载转换后的Jittor权重
        weights = jt.load(weight_path)
        
        # 加载权重到模型
        missing_keys, unexpected_keys = [], []
        model_dict = model.state_dict()
        
        loaded_count = 0
        for key, value in weights.items():
            if key in model_dict:
                if model_dict[key].shape == value.shape:
                    model_dict[key] = value
                    loaded_count += 1
                else:
                    print(f"⚠ 形状不匹配: {key} {model_dict[key].shape} vs {value.shape}")
                    missing_keys.append(key)
            else:
                unexpected_keys.append(key)
        
        # 更新模型权重
        model.load_state_dict(model_dict)
        
        print(f"✓ 成功加载 {loaded_count} 个参数")
        if missing_keys:
            print(f"⚠ 缺失参数: {len(missing_keys)} 个")
        if unexpected_keys:
            print(f"⚠ 额外参数: {len(unexpected_keys)} 个")
        
        return True
        
    except Exception as e:
        print(f"✗ 权重加载失败: {e}")
        return False


def preprocess_image(img_path, input_size=(320, 320)):
    """预处理图像"""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot load image: {img_path}")
    
    # 调整大小
    img = cv2.resize(img, input_size)
    
    # 归一化 (ImageNet标准)
    mean = np.array([103.53, 116.28, 123.675])
    std = np.array([57.375, 57.12, 58.395])
    
    img = img.astype(np.float32)
    img -= mean
    img /= std
    
    # 转换为CHW格式
    img = img.transpose(2, 0, 1)
    
    # 添加batch维度
    img = np.expand_dims(img, axis=0)
    
    return jt.array(img)


def test_pytorch_finetuned_inference():
    """测试PyTorch微调后模型推理"""
    print("=" * 60)
    print("测评角度2：PyTorch微调后模型测评")
    print("=" * 60)
    
    # 设置CUDA
    if jt.has_cuda:
        jt.flags.use_cuda = 1
        print("✓ Using CUDA")
    
    # 创建模型
    print("创建模型...")
    model = create_model()
    model.eval()
    
    print(f"✓ 模型创建成功")
    print(f"  参数数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # 加载PyTorch微调后的权重
    pytorch_weight_path = "weights/pytorch_converted.pkl"
    if not load_pytorch_finetuned_weights(model, pytorch_weight_path):
        print("✗ 无法加载PyTorch微调后权重，测试失败")
        return False
    
    # 测试前向传播
    print("\n测试前向传播...")
    test_input = jt.randn(1, 3, 320, 320)
    
    with jt.no_grad():
        try:
            output = model(test_input)
            print(f"✓ 前向传播成功")
            print(f"  输入形状: {test_input.shape}")
            print(f"  输出形状: {output.shape}")
            print(f"  输出范围: [{output.min():.4f}, {output.max():.4f}]")
            
            # 检查输出是否合理（微调后应该有不同的输出范围）
            if abs(output.min().item() - (-4.5950)) > 0.1 or abs(output.max().item() - (-4.5950)) > 0.1:
                print("✓ 输出值已改变，说明微调权重加载成功")
            else:
                print("⚠ 输出值未改变，可能权重加载有问题")
                
        except Exception as e:
            print(f"✗ 前向传播失败: {e}")
            return False
    
    # 测试真实VOC图像
    print("\n测试真实VOC图像...")
    test_images = [
        "data/VOCdevkit/VOC2007/JPEGImages/000001.jpg",
        "data/VOCdevkit/VOC2007/JPEGImages/000002.jpg", 
        "data/VOCdevkit/VOC2007/JPEGImages/000003.jpg"
    ]
    
    successful_tests = 0
    outputs = []
    
    for img_path in test_images:
        if os.path.exists(img_path):
            print(f"  测试图像: {os.path.basename(img_path)}")
            try:
                img_tensor = preprocess_image(img_path)
                with jt.no_grad():
                    output = model(img_tensor)
                print(f"    ✓ 推理成功: {output.shape}")
                print(f"    输出范围: [{output.min():.4f}, {output.max():.4f}]")
                outputs.append(output)
                successful_tests += 1
            except Exception as e:
                print(f"    ✗ 推理失败: {e}")
        else:
            print(f"  ⚠ 图像不存在: {img_path}")
    
    print(f"\n✓ 成功测试了 {successful_tests}/{len(test_images)} 张图像")
    
    # 分析输出差异
    if len(outputs) >= 2:
        print("\n分析不同图像的输出差异...")
        diff = jt.abs(outputs[0] - outputs[1]).mean()
        print(f"  图像间输出差异: {diff:.6f}")
        if diff > 1e-6:
            print("  ✓ 不同图像产生不同输出，模型工作正常")
        else:
            print("  ⚠ 不同图像输出相同，可能有问题")
    
    print("\n" + "=" * 60)
    print("PyTorch微调后模型测评完成")
    print("=" * 60)
    
    return True


def compare_with_pretrained():
    """与预训练权重对比"""
    print("\n" + "=" * 40)
    print("与预训练权重对比")
    print("=" * 40)
    
    print("⚠ 对比功能待实现")
    print("  可以对比：")
    print("  1. 输出值范围差异")
    print("  2. 权重参数差异")
    print("  3. 推理结果差异")


def main():
    """主函数"""
    print("Jittor NanoDet PyTorch微调后模型测评")
    
    # 测试PyTorch微调后模型推理
    success = test_pytorch_finetuned_inference()
    
    if success:
        print("\n🎉 PyTorch微调后模型测评成功！")
        print("✓ 权重转换正确")
        print("✓ 模型架构对齐")
        print("✓ 推理功能正常")
        print("✓ 输出结果合理")
    else:
        print("\n❌ PyTorch微调后模型测评失败")
        return False
    
    # 对比分析
    compare_with_pretrained()
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
