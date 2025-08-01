#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
检查激活函数的实现差异
特别是LeakyReLU的参数和行为
"""

import torch
import jittor as jt
import numpy as np


def test_leaky_relu():
    """测试LeakyReLU的行为差异"""
    print("🔍 测试LeakyReLU行为差异")
    print("=" * 50)
    
    # 创建测试数据
    test_data = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0], dtype=np.float32)
    
    print(f"测试数据: {test_data}")
    
    # PyTorch LeakyReLU
    torch_data = torch.from_numpy(test_data)
    torch_leaky_relu = torch.nn.LeakyReLU()
    torch_result = torch_leaky_relu(torch_data)
    
    print(f"PyTorch LeakyReLU结果: {torch_result.numpy()}")
    print(f"PyTorch LeakyReLU负斜率: {torch_leaky_relu.negative_slope}")
    
    # Jittor LeakyReLU
    jittor_data = jt.array(test_data)
    jittor_leaky_relu = jt.nn.LeakyReLU()
    jittor_result = jittor_leaky_relu(jittor_data)
    
    print(f"Jittor LeakyReLU结果: {jittor_result.numpy()}")
    
    # 检查Jittor LeakyReLU的参数
    if hasattr(jittor_leaky_relu, 'negative_slope'):
        print(f"Jittor LeakyReLU负斜率: {jittor_leaky_relu.negative_slope}")
    else:
        print("Jittor LeakyReLU没有negative_slope属性")
    
    # 计算差异
    diff = np.abs(torch_result.numpy() - jittor_result.numpy())
    print(f"差异: {diff}")
    print(f"最大差异: {diff.max()}")
    
    if diff.max() < 1e-6:
        print("✅ LeakyReLU行为一致")
    else:
        print("❌ LeakyReLU行为不一致")
    
    return diff.max() < 1e-6


def test_batch_norm():
    """测试BatchNorm的行为差异"""
    print("\n🔍 测试BatchNorm行为差异")
    print("=" * 50)
    
    # 创建测试数据
    test_data = np.random.randn(2, 64, 8, 8).astype(np.float32)
    
    print(f"测试数据形状: {test_data.shape}")
    print(f"测试数据范围: [{test_data.min():.6f}, {test_data.max():.6f}]")
    
    # PyTorch BatchNorm
    torch_data = torch.from_numpy(test_data)
    torch_bn = torch.nn.BatchNorm2d(64)
    torch_bn.eval()  # 设置为评估模式
    torch_result = torch_bn(torch_data)
    
    print(f"PyTorch BatchNorm结果范围: [{torch_result.min():.6f}, {torch_result.max():.6f}]")
    
    # Jittor BatchNorm
    jittor_data = jt.array(test_data)
    jittor_bn = jt.nn.BatchNorm2d(64)
    jittor_bn.eval()  # 设置为评估模式
    jittor_result = jittor_bn(jittor_data)
    
    print(f"Jittor BatchNorm结果范围: [{jittor_result.min():.6f}, {jittor_result.max():.6f}]")
    
    # 计算差异
    diff = np.abs(torch_result.detach().numpy() - jittor_result.numpy())
    print(f"最大差异: {diff.max():.6f}")
    print(f"平均差异: {diff.mean():.6f}")
    
    if diff.max() < 1e-4:
        print("✅ BatchNorm行为基本一致")
        return True
    else:
        print("❌ BatchNorm行为差异较大")
        return False


def test_conv2d():
    """测试Conv2d的行为差异"""
    print("\n🔍 测试Conv2d行为差异")
    print("=" * 50)
    
    # 创建测试数据
    test_data = np.random.randn(1, 3, 32, 32).astype(np.float32)
    
    print(f"测试数据形状: {test_data.shape}")
    print(f"测试数据范围: [{test_data.min():.6f}, {test_data.max():.6f}]")
    
    # 创建相同的权重
    weight = np.random.randn(64, 3, 3, 3).astype(np.float32)
    bias = np.random.randn(64).astype(np.float32)
    
    # PyTorch Conv2d
    torch_data = torch.from_numpy(test_data)
    torch_conv = torch.nn.Conv2d(3, 64, 3, padding=1)
    torch_conv.weight.data = torch.from_numpy(weight)
    torch_conv.bias.data = torch.from_numpy(bias)
    torch_result = torch_conv(torch_data)
    
    print(f"PyTorch Conv2d结果形状: {torch_result.shape}")
    print(f"PyTorch Conv2d结果范围: [{torch_result.min():.6f}, {torch_result.max():.6f}]")
    
    # Jittor Conv2d
    jittor_data = jt.array(test_data)
    jittor_conv = jt.nn.Conv2d(3, 64, 3, padding=1)
    jittor_conv.weight.assign(jt.array(weight))
    jittor_conv.bias.assign(jt.array(bias))
    jittor_result = jittor_conv(jittor_data)
    
    print(f"Jittor Conv2d结果形状: {jittor_result.shape}")
    print(f"Jittor Conv2d结果范围: [{jittor_result.min():.6f}, {jittor_result.max():.6f}]")
    
    # 计算差异
    diff = np.abs(torch_result.detach().numpy() - jittor_result.numpy())
    print(f"最大差异: {diff.max():.6f}")
    print(f"平均差异: {diff.mean():.6f}")
    
    if diff.max() < 1e-4:
        print("✅ Conv2d行为基本一致")
        return True
    else:
        print("❌ Conv2d行为差异较大")
        return False


def test_combined_operations():
    """测试组合操作的行为差异"""
    print("\n🔍 测试组合操作行为差异")
    print("=" * 50)
    
    # 创建测试数据
    test_data = np.random.randn(1, 64, 16, 16).astype(np.float32)
    
    print(f"测试数据形状: {test_data.shape}")
    print(f"测试数据范围: [{test_data.min():.6f}, {test_data.max():.6f}]")
    
    # 创建相同的权重
    weight = np.random.randn(64, 64, 3, 3).astype(np.float32)
    bias = np.random.randn(64).astype(np.float32)
    
    # PyTorch组合操作: Conv2d + BatchNorm + LeakyReLU
    torch_data = torch.from_numpy(test_data)
    
    torch_conv = torch.nn.Conv2d(64, 64, 3, padding=1)
    torch_conv.weight.data = torch.from_numpy(weight)
    torch_conv.bias.data = torch.from_numpy(bias)
    
    torch_bn = torch.nn.BatchNorm2d(64)
    torch_bn.eval()
    
    torch_leaky = torch.nn.LeakyReLU()
    
    # PyTorch前向传播
    torch_x = torch_conv(torch_data)
    torch_x = torch_bn(torch_x)
    torch_result = torch_leaky(torch_x)
    
    print(f"PyTorch组合操作结果范围: [{torch_result.min():.6f}, {torch_result.max():.6f}]")
    
    # Jittor组合操作
    jittor_data = jt.array(test_data)
    
    jittor_conv = jt.nn.Conv2d(64, 64, 3, padding=1)
    jittor_conv.weight.assign(jt.array(weight))
    jittor_conv.bias.assign(jt.array(bias))
    
    jittor_bn = jt.nn.BatchNorm2d(64)
    jittor_bn.eval()
    
    jittor_leaky = jt.nn.LeakyReLU()
    
    # Jittor前向传播
    jittor_x = jittor_conv(jittor_data)
    jittor_x = jittor_bn(jittor_x)
    jittor_result = jittor_leaky(jittor_x)
    
    print(f"Jittor组合操作结果范围: [{jittor_result.min():.6f}, {jittor_result.max():.6f}]")
    
    # 计算差异
    diff = np.abs(torch_result.detach().numpy() - jittor_result.numpy())
    print(f"最大差异: {diff.max():.6f}")
    print(f"平均差异: {diff.mean():.6f}")
    
    if diff.max() < 1e-3:
        print("✅ 组合操作行为基本一致")
        return True
    else:
        print("❌ 组合操作行为差异较大")
        return False


def main():
    """主函数"""
    print("🚀 开始检查激活函数和基础操作差异")
    
    # 测试各个组件
    leaky_relu_ok = test_leaky_relu()
    batch_norm_ok = test_batch_norm()
    conv2d_ok = test_conv2d()
    combined_ok = test_combined_operations()
    
    print(f"\n📊 测试总结:")
    print(f"  LeakyReLU: {'✅' if leaky_relu_ok else '❌'}")
    print(f"  BatchNorm: {'✅' if batch_norm_ok else '❌'}")
    print(f"  Conv2d: {'✅' if conv2d_ok else '❌'}")
    print(f"  组合操作: {'✅' if combined_ok else '❌'}")
    
    if all([leaky_relu_ok, batch_norm_ok, conv2d_ok, combined_ok]):
        print(f"\n✅ 所有基础操作都一致，问题可能在更高层的实现")
    else:
        print(f"\n❌ 发现基础操作差异，这可能是问题根源")
    
    print(f"\n✅ 检查完成")


if __name__ == '__main__':
    main()
