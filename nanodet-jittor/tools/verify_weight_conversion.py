#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
验证权重转换过程
检查PyTorch到Jittor的权重转换是否正确
"""

import torch
import jittor as jt
import numpy as np


def test_basic_conversion():
    """测试基础的PyTorch到Jittor转换"""
    print("🔍 测试基础PyTorch到Jittor转换")
    print("=" * 50)
    
    # 创建测试数据
    test_cases = [
        ("标量", torch.tensor(3.14159)),
        ("1D张量", torch.randn(10)),
        ("2D张量", torch.randn(3, 4)),
        ("3D张量", torch.randn(2, 3, 4)),
        ("4D张量", torch.randn(1, 3, 32, 32)),
    ]
    
    for name, pytorch_tensor in test_cases:
        print(f"\n测试 {name}:")
        print(f"  PyTorch: {pytorch_tensor.shape}, dtype={pytorch_tensor.dtype}")
        
        # 方法1: 直接转换
        jittor_array1 = jt.array(pytorch_tensor.detach().numpy())
        print(f"  Jittor方法1: {jittor_array1.shape}, dtype={jittor_array1.dtype}")
        
        # 方法2: 显式numpy转换
        numpy_array = pytorch_tensor.detach().numpy()
        jittor_array2 = jt.array(numpy_array)
        print(f"  Jittor方法2: {jittor_array2.shape}, dtype={jittor_array2.dtype}")
        
        # 检查数值一致性
        diff1 = np.abs(pytorch_tensor.detach().numpy() - jittor_array1.numpy()).max()
        diff2 = np.abs(pytorch_tensor.detach().numpy() - jittor_array2.numpy()).max()
        
        print(f"  差异方法1: {diff1:.10f}")
        print(f"  差异方法2: {diff2:.10f}")
        
        if diff1 < 1e-6 and diff2 < 1e-6:
            print(f"  ✅ 转换正确")
        else:
            print(f"  ❌ 转换有误")


def test_parameter_assignment():
    """测试参数赋值过程"""
    print("\n🔍 测试参数赋值过程")
    print("=" * 50)
    
    # 创建简单的Jittor模型
    class SimpleModel(jt.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = jt.nn.Conv2d(3, 16, 3)
            self.bn = jt.nn.BatchNorm2d(16)
    
    model = SimpleModel()
    
    # 创建PyTorch权重
    pytorch_conv_weight = torch.randn(16, 3, 3, 3)
    pytorch_conv_bias = torch.randn(16)
    pytorch_bn_weight = torch.randn(16)
    pytorch_bn_bias = torch.randn(16)
    pytorch_bn_mean = torch.randn(16)
    pytorch_bn_var = torch.randn(16)
    
    print("原始Jittor参数:")
    for name, param in model.named_parameters():
        print(f"  {name}: {param.shape}, 范围[{param.min():.6f}, {param.max():.6f}]")
    
    # 测试不同的赋值方法
    print("\n测试赋值方法:")
    
    # 方法1: 直接assign
    print("方法1: 直接assign")
    try:
        model.conv.weight.assign(jt.array(pytorch_conv_weight.detach().numpy()))
        model.conv.bias.assign(jt.array(pytorch_conv_bias.detach().numpy()))
        model.bn.weight.assign(jt.array(pytorch_bn_weight.detach().numpy()))
        model.bn.bias.assign(jt.array(pytorch_bn_bias.detach().numpy()))
        model.bn.running_mean.assign(jt.array(pytorch_bn_mean.detach().numpy()))
        model.bn.running_var.assign(jt.array(pytorch_bn_var.detach().numpy()))
        
        print("  ✅ assign成功")
        
        # 验证赋值结果
        conv_weight_diff = np.abs(pytorch_conv_weight.detach().numpy() - model.conv.weight.numpy()).max()
        conv_bias_diff = np.abs(pytorch_conv_bias.detach().numpy() - model.conv.bias.numpy()).max()
        bn_weight_diff = np.abs(pytorch_bn_weight.detach().numpy() - model.bn.weight.numpy()).max()
        bn_bias_diff = np.abs(pytorch_bn_bias.detach().numpy() - model.bn.bias.numpy()).max()
        bn_mean_diff = np.abs(pytorch_bn_mean.detach().numpy() - model.bn.running_mean.numpy()).max()
        bn_var_diff = np.abs(pytorch_bn_var.detach().numpy() - model.bn.running_var.numpy()).max()
        
        print(f"  conv.weight差异: {conv_weight_diff:.10f}")
        print(f"  conv.bias差异: {conv_bias_diff:.10f}")
        print(f"  bn.weight差异: {bn_weight_diff:.10f}")
        print(f"  bn.bias差异: {bn_bias_diff:.10f}")
        print(f"  bn.running_mean差异: {bn_mean_diff:.10f}")
        print(f"  bn.running_var差异: {bn_var_diff:.10f}")
        
        max_diff = max(conv_weight_diff, conv_bias_diff, bn_weight_diff, bn_bias_diff, bn_mean_diff, bn_var_diff)
        if max_diff < 1e-6:
            print(f"  ✅ 赋值数值正确")
        else:
            print(f"  ❌ 赋值数值有误，最大差异: {max_diff:.10f}")
            
    except Exception as e:
        print(f"  ❌ assign失败: {e}")


def test_real_weight_loading():
    """测试真实权重加载过程"""
    print("\n🔍 测试真实权重加载过程")
    print("=" * 50)
    
    # 加载真实的PyTorch权重
    checkpoint_path = "/home/kyc/project/nanodet/nanodet-pytorch/workspace/nanodet-plus-m_320_voc_bs64/model_best/model_best.ckpt"
    
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        state_dict = ckpt.get('state_dict', ckpt)
        
        print(f"✓ 成功加载PyTorch权重，包含 {len(state_dict)} 个参数")
        
        # 选择几个典型参数进行测试
        test_params = [
            "model.backbone.conv1.0.weight",
            "model.backbone.conv1.1.weight", 
            "model.backbone.conv1.1.bias",
            "model.backbone.conv1.1.running_mean",
            "model.backbone.conv1.1.running_var",
        ]
        
        for param_name in test_params:
            if param_name in state_dict:
                pytorch_param = state_dict[param_name]
                print(f"\n测试参数: {param_name}")
                print(f"  PyTorch: {pytorch_param.shape}, dtype={pytorch_param.dtype}")
                print(f"  范围: [{pytorch_param.min():.6f}, {pytorch_param.max():.6f}]")
                
                # 转换到Jittor
                jittor_param = jt.array(pytorch_param.detach().numpy())
                print(f"  Jittor: {jittor_param.shape}, dtype={jittor_param.dtype}")
                print(f"  范围: [{jittor_param.min():.6f}, {jittor_param.max():.6f}]")
                
                # 检查差异
                diff = np.abs(pytorch_param.detach().numpy() - jittor_param.numpy()).max()
                print(f"  转换差异: {diff:.10f}")
                
                if diff < 1e-6:
                    print(f"  ✅ 转换正确")
                else:
                    print(f"  ❌ 转换有误")
                    
                    # 详细分析差异
                    pytorch_np = pytorch_param.detach().numpy()
                    jittor_np = jittor_param.numpy()
                    
                    print(f"    PyTorch统计: 均值={pytorch_np.mean():.6f}, 标准差={pytorch_np.std():.6f}")
                    print(f"    Jittor统计: 均值={jittor_np.mean():.6f}, 标准差={jittor_np.std():.6f}")
                    
                    # 检查是否是数据类型问题
                    print(f"    PyTorch numpy dtype: {pytorch_np.dtype}")
                    print(f"    Jittor numpy dtype: {jittor_np.dtype}")
            else:
                print(f"⚠️ 参数 {param_name} 不存在")
                
    except Exception as e:
        print(f"❌ 加载PyTorch权重失败: {e}")


def test_scale_parameter_issue():
    """测试Scale参数问题"""
    print("\n🔍 测试Scale参数问题")
    print("=" * 50)
    
    # 模拟PyTorch的标量参数
    pytorch_scale = torch.tensor(1.0)  # 标量
    print(f"PyTorch scale: {pytorch_scale.shape}, 值={pytorch_scale.item()}")
    
    # 模拟Jittor的Scale参数
    jittor_scale = jt.array([1.0])  # 1维数组
    print(f"Jittor scale: {jittor_scale.shape}, 值={jittor_scale.numpy()}")
    
    # 测试转换
    print("\n转换测试:")
    
    # 方法1: 直接转换（会失败）
    try:
        converted1 = jt.array(pytorch_scale.detach().numpy())
        print(f"方法1成功: {converted1.shape}, 值={converted1.numpy()}")
    except Exception as e:
        print(f"方法1失败: {e}")
    
    # 方法2: 包装成数组
    try:
        if len(pytorch_scale.shape) == 0:  # 标量
            converted2 = jt.array([pytorch_scale.detach().numpy()])
        else:
            converted2 = jt.array(pytorch_scale.detach().numpy())
        print(f"方法2成功: {converted2.shape}, 值={converted2.numpy()}")
        
        # 赋值测试
        jittor_scale.assign(converted2)
        print(f"赋值后: {jittor_scale.shape}, 值={jittor_scale.numpy()}")
        
    except Exception as e:
        print(f"方法2失败: {e}")


def main():
    """主函数"""
    print("🚀 开始验证权重转换过程")
    
    # 基础转换测试
    test_basic_conversion()
    
    # 参数赋值测试
    test_parameter_assignment()
    
    # 真实权重加载测试
    test_real_weight_loading()
    
    # Scale参数问题测试
    test_scale_parameter_issue()
    
    print(f"\n✅ 验证完成")


if __name__ == '__main__':
    main()
