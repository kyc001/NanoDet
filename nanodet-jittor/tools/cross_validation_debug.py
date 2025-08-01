#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
交叉验证调试工具
逐个替换组件，精确定位问题根源
"""

import os
import sys
import torch
import jittor as jt
import numpy as np

# 添加路径
sys.path.append('/home/kyc/project/nanodet/nanodet-jittor')
sys.path.append('/home/kyc/project/nanodet/nanodet-pytorch')

# Jittor版本
from nanodet.model.arch.nanodet_plus import NanoDetPlus as JittorNanoDetPlus

# PyTorch版本 (需要重命名避免冲突)
import importlib.util
pytorch_spec = importlib.util.spec_from_file_location(
    "pytorch_nanodet", 
    "/home/kyc/project/nanodet/nanodet-pytorch/nanodet/model/arch/nanodet_plus.py"
)
pytorch_nanodet = importlib.util.module_from_spec(pytorch_spec)
pytorch_spec.loader.exec_module(pytorch_nanodet)
PyTorchNanoDetPlus = pytorch_nanodet.NanoDetPlus


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


def load_pytorch_weights():
    """加载PyTorch权重"""
    checkpoint_path = "/home/kyc/project/nanodet/nanodet-pytorch/workspace/nanodet-plus-m_320_voc_bs64/model_best/model_best.ckpt"
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)
    return state_dict


def create_pytorch_model():
    """创建PyTorch模型"""
    print("🔍 创建PyTorch模型...")
    
    backbone_cfg, fpn_cfg, head_cfg, aux_head_cfg = create_model_configs()
    
    # 创建PyTorch模型
    pytorch_model = PyTorchNanoDetPlus(backbone_cfg, fpn_cfg, aux_head_cfg, head_cfg)
    
    # 加载权重
    state_dict = load_pytorch_weights()
    pytorch_model.load_state_dict(state_dict, strict=False)
    pytorch_model.eval()
    
    return pytorch_model


def create_jittor_model():
    """创建Jittor模型"""
    print("🔍 创建Jittor模型...")
    
    backbone_cfg, fpn_cfg, head_cfg, aux_head_cfg = create_model_configs()
    
    # 创建Jittor模型
    jittor_model = JittorNanoDetPlus(backbone_cfg, fpn_cfg, aux_head_cfg, head_cfg)
    
    # 加载权重
    state_dict = load_pytorch_weights()
    
    jittor_state_dict = {}
    for name, param in jittor_model.named_parameters():
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
    
    jittor_model.eval()
    return jittor_model


def test_model_output(model, input_tensor, model_name):
    """测试模型输出"""
    print(f"\n🔍 测试{model_name}模型输出:")
    
    if model_name == "PyTorch":
        with torch.no_grad():
            output = model(input_tensor)
            
            # 分析输出
            cls_preds = output[:, :, :20]
            reg_preds = output[:, :, 20:]
            cls_scores = torch.sigmoid(cls_preds)
            
            print(f"  输出形状: {output.shape}")
            print(f"  输出范围: [{output.min():.6f}, {output.max():.6f}]")
            print(f"  分类预测: 范围[{cls_preds.min():.6f}, {cls_preds.max():.6f}]")
            print(f"  回归预测: 范围[{reg_preds.min():.6f}, {reg_preds.max():.6f}]")
            print(f"  分类置信度: 范围[{cls_scores.min():.6f}, {cls_scores.max():.6f}]")
            print(f"  最高置信度: {cls_scores.max():.6f}")
            
            return output.detach().numpy(), cls_scores.max().item()
    
    else:  # Jittor
        with jt.no_grad():
            output = model(input_tensor)
            
            # 分析输出
            cls_preds = output[:, :, :20]
            reg_preds = output[:, :, 20:]
            cls_scores = jt.sigmoid(cls_preds)
            
            print(f"  输出形状: {output.shape}")
            print(f"  输出范围: [{output.min():.6f}, {output.max():.6f}]")
            print(f"  分类预测: 范围[{cls_preds.min():.6f}, {cls_preds.max():.6f}]")
            print(f"  回归预测: 范围[{reg_preds.min():.6f}, {reg_preds.max():.6f}]")
            print(f"  分类置信度: 范围[{cls_scores.min():.6f}, {cls_scores.max():.6f}]")
            print(f"  最高置信度: {cls_scores.max():.6f}")
            
            return output.numpy(), float(cls_scores.max().numpy())


def cross_validation_test():
    """交叉验证测试"""
    print("🚀 开始交叉验证测试")
    print("=" * 60)
    
    # 创建测试输入
    input_data = create_test_input()
    pytorch_input = torch.from_numpy(input_data)
    jittor_input = jt.array(input_data)
    
    print(f"测试输入: {input_data.shape}, 范围[{input_data.min():.6f}, {input_data.max():.6f}]")
    
    # 1. 测试纯PyTorch模型
    print(f"\n1️⃣ 测试纯PyTorch模型")
    try:
        pytorch_model = create_pytorch_model()
        pytorch_output, pytorch_confidence = test_model_output(pytorch_model, pytorch_input, "PyTorch")
        print(f"  ✅ PyTorch模型测试成功")
    except Exception as e:
        print(f"  ❌ PyTorch模型测试失败: {e}")
        pytorch_output, pytorch_confidence = None, 0
    
    # 2. 测试纯Jittor模型
    print(f"\n2️⃣ 测试纯Jittor模型")
    try:
        jittor_model = create_jittor_model()
        jittor_output, jittor_confidence = test_model_output(jittor_model, jittor_input, "Jittor")
        print(f"  ✅ Jittor模型测试成功")
    except Exception as e:
        print(f"  ❌ Jittor模型测试失败: {e}")
        jittor_output, jittor_confidence = None, 0
    
    # 3. 对比结果
    print(f"\n3️⃣ 对比结果")
    if pytorch_output is not None and jittor_output is not None:
        output_diff = np.abs(pytorch_output - jittor_output).max()
        confidence_diff = abs(pytorch_confidence - jittor_confidence)
        
        print(f"  输出最大差异: {output_diff:.6f}")
        print(f"  置信度差异: {confidence_diff:.6f}")
        print(f"  PyTorch最高置信度: {pytorch_confidence:.6f}")
        print(f"  Jittor最高置信度: {jittor_confidence:.6f}")
        
        if output_diff < 0.01:
            print(f"  ✅ 输出基本一致")
        else:
            print(f"  ❌ 输出差异较大")
        
        if confidence_diff < 0.1:
            print(f"  ✅ 置信度基本一致")
        else:
            print(f"  ❌ 置信度差异较大")
    
    # 4. 保存结果
    results = {
        'pytorch_output': pytorch_output,
        'jittor_output': jittor_output,
        'pytorch_confidence': pytorch_confidence,
        'jittor_confidence': jittor_confidence,
        'input_data': input_data
    }
    
    np.save("cross_validation_results.npy", results)
    print(f"\n📊 结果已保存到 cross_validation_results.npy")
    
    return results


def main():
    """主函数"""
    print("🚀 开始交叉验证调试")
    
    # 交叉验证测试
    results = cross_validation_test()
    
    print(f"\n✅ 交叉验证调试完成")


if __name__ == '__main__':
    main()
