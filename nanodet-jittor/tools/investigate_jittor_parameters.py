#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
深入研究Jittor参数机制
找出BatchNorm统计参数和Scale参数的正确处理方式
"""

import jittor as jt
from jittor import nn
import torch
import torch.nn as torch_nn


def investigate_jittor_batchnorm():
    """研究Jittor BatchNorm的参数机制"""
    print("🔍 研究Jittor BatchNorm参数机制")
    print("=" * 60)
    
    # 创建Jittor BatchNorm
    jittor_bn = nn.BatchNorm(64)
    
    print("Jittor BatchNorm属性:")
    for name in dir(jittor_bn):
        if not name.startswith('_'):
            attr = getattr(jittor_bn, name)
            if hasattr(attr, 'shape'):
                print(f"  {name}: {attr.shape} - {type(attr)}")
    
    print(f"\nJittor BatchNorm named_parameters():")
    for name, param in jittor_bn.named_parameters():
        print(f"  {name}: {param.shape}")
    
    print(f"\nJittor BatchNorm parameters():")
    params = list(jittor_bn.parameters())
    print(f"  总数: {len(params)}")
    for i, param in enumerate(params):
        print(f"  参数{i}: {param.shape}")
    
    # 对比PyTorch BatchNorm
    print(f"\n" + "=" * 60)
    print("对比PyTorch BatchNorm参数机制")
    
    torch_bn = torch_nn.BatchNorm2d(64)
    
    print(f"\nPyTorch BatchNorm named_parameters():")
    for name, param in torch_bn.named_parameters():
        print(f"  {name}: {param.shape}")
    
    print(f"\nPyTorch BatchNorm named_buffers():")
    for name, buffer in torch_bn.named_buffers():
        print(f"  {name}: {buffer.shape}")
    
    print(f"\nPyTorch BatchNorm parameters():")
    params = list(torch_bn.parameters())
    print(f"  总数: {len(params)}")
    for i, param in enumerate(params):
        print(f"  参数{i}: {param.shape}")


def investigate_jittor_scalar():
    """研究Jittor标量参数的创建方式"""
    print("\n🔍 研究Jittor标量参数创建方式")
    print("=" * 60)
    
    # 测试不同的标量创建方式
    methods = [
        ("jt.array(1.0)", lambda: jt.array(1.0)),
        ("jt.float32(1.0)", lambda: jt.float32(1.0)),
        ("jt.Var(1.0)", lambda: jt.Var(1.0)),
        ("jt.array([1.0])", lambda: jt.array([1.0])),
        ("jt.ones([])", lambda: jt.ones([])),
        ("jt.tensor(1.0)", lambda: jt.array(1.0)),
    ]
    
    for method_name, method_func in methods:
        try:
            result = method_func()
            print(f"{method_name:20}: shape={result.shape}, ndim={result.ndim}, value={result}")
        except Exception as e:
            print(f"{method_name:20}: ERROR - {e}")
    
    # 测试PyTorch标量
    print(f"\n对比PyTorch标量:")
    torch_scalar = torch.tensor(1.0)
    print(f"torch.tensor(1.0):   shape={torch_scalar.shape}, ndim={torch_scalar.ndim}, value={torch_scalar}")
    
    torch_param = torch_nn.Parameter(torch.tensor(1.0))
    print(f"nn.Parameter(1.0):   shape={torch_param.shape}, ndim={torch_param.ndim}, value={torch_param}")


def investigate_jittor_module_system():
    """研究Jittor模块系统"""
    print("\n🔍 研究Jittor模块系统")
    print("=" * 60)
    
    class TestModule(nn.Module):
        def __init__(self):
            super().__init__()
            # 测试不同的参数注册方式
            self.param1 = jt.array([1.0])  # 普通参数
            self.param2 = nn.Parameter(jt.array([2.0]))  # 显式参数
            
            # 测试buffer注册
            try:
                self.register_buffer('buffer1', jt.array([3.0]))
                print("✓ register_buffer 可用")
            except:
                print("❌ register_buffer 不可用")
                # 手动设置buffer
                self.buffer1 = jt.array([3.0])
                self.buffer1.requires_grad = False
            
            # 测试BatchNorm
            self.bn = nn.BatchNorm(10)
    
    module = TestModule()
    
    print(f"\nTestModule named_parameters():")
    for name, param in module.named_parameters():
        print(f"  {name}: {param.shape} - requires_grad={param.requires_grad}")
    
    print(f"\nTestModule所有属性:")
    for name in dir(module):
        if not name.startswith('_'):
            attr = getattr(module, name)
            if hasattr(attr, 'shape'):
                print(f"  {name}: {attr.shape} - {type(attr)} - requires_grad={getattr(attr, 'requires_grad', 'N/A')}")


def test_parameter_exclusion():
    """测试参数排除机制"""
    print("\n🔍 测试参数排除机制")
    print("=" * 60)
    
    class CustomModule(nn.Module):
        def __init__(self):
            super().__init__()
            # 创建不同类型的属性
            self.trainable_param = jt.array([1.0])
            
            # 尝试创建非参数属性
            non_param = jt.array([2.0])
            non_param.requires_grad = False
            object.__setattr__(self, 'non_param', non_param)
            
            # 尝试使用_开头的属性
            self._private_param = jt.array([3.0])
            
            # BatchNorm
            self.bn = nn.BatchNorm(5)
    
    module = CustomModule()
    
    print(f"CustomModule named_parameters():")
    for name, param in module.named_parameters():
        print(f"  {name}: {param.shape}")
    
    print(f"\nCustomModule所有Var属性:")
    for name in dir(module):
        if not name.startswith('__'):
            attr = getattr(module, name)
            if hasattr(attr, 'shape') and hasattr(attr, 'requires_grad'):
                print(f"  {name}: {attr.shape} - requires_grad={attr.requires_grad}")


def main():
    """主函数"""
    print("🚀 开始深入研究Jittor参数机制")
    
    # 研究BatchNorm
    investigate_jittor_batchnorm()
    
    # 研究标量参数
    investigate_jittor_scalar()
    
    # 研究模块系统
    investigate_jittor_module_system()
    
    # 测试参数排除
    test_parameter_exclusion()
    
    print("\n✅ 研究完成")


if __name__ == '__main__':
    main()
