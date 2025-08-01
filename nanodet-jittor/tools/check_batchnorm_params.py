#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
检查BatchNorm参数设置
找出PyTorch和Jittor BatchNorm的参数差异
"""

import torch
import jittor as jt
import numpy as np


def check_batchnorm_parameters():
    """检查BatchNorm参数设置"""
    print("🔍 检查BatchNorm参数设置")
    print("=" * 60)
    
    # PyTorch BatchNorm
    print("PyTorch BatchNorm2d默认参数:")
    pytorch_bn = torch.nn.BatchNorm2d(64)
    print(f"  num_features: {pytorch_bn.num_features}")
    print(f"  eps: {pytorch_bn.eps}")
    print(f"  momentum: {pytorch_bn.momentum}")
    print(f"  affine: {pytorch_bn.affine}")
    print(f"  track_running_stats: {pytorch_bn.track_running_stats}")
    print(f"  training: {pytorch_bn.training}")
    
    # Jittor BatchNorm
    print(f"\nJittor BatchNorm2d默认参数:")
    jittor_bn = jt.nn.BatchNorm2d(64)
    print(f"  num_features: {jittor_bn.num_features}")
    print(f"  eps: {jittor_bn.eps}")
    print(f"  momentum: {jittor_bn.momentum}")
    print(f"  affine: {jittor_bn.affine}")
    print(f"  is_train: {jittor_bn.is_train}")
    
    # 对比参数
    print(f"\n参数对比:")
    params_match = True
    
    if pytorch_bn.eps != jittor_bn.eps:
        print(f"  ❌ eps不匹配: PyTorch={pytorch_bn.eps}, Jittor={jittor_bn.eps}")
        params_match = False
    else:
        print(f"  ✅ eps匹配: {pytorch_bn.eps}")
    
    if pytorch_bn.momentum != jittor_bn.momentum:
        print(f"  ❌ momentum不匹配: PyTorch={pytorch_bn.momentum}, Jittor={jittor_bn.momentum}")
        params_match = False
    else:
        print(f"  ✅ momentum匹配: {pytorch_bn.momentum}")
    
    if pytorch_bn.affine != jittor_bn.affine:
        print(f"  ❌ affine不匹配: PyTorch={pytorch_bn.affine}, Jittor={jittor_bn.affine}")
        params_match = False
    else:
        print(f"  ✅ affine匹配: {pytorch_bn.affine}")
    
    return params_match


def test_batchnorm_behavior():
    """测试BatchNorm行为差异"""
    print(f"\n🔍 测试BatchNorm行为差异")
    print("=" * 60)
    
    # 创建测试数据
    np.random.seed(42)
    torch.manual_seed(42)
    jt.set_global_seed(42)
    
    test_data = np.random.randn(2, 64, 8, 8).astype(np.float32)
    
    # PyTorch BatchNorm
    pytorch_bn = torch.nn.BatchNorm2d(64)
    pytorch_input = torch.from_numpy(test_data)
    
    # Jittor BatchNorm
    jittor_bn = jt.nn.BatchNorm2d(64)
    jittor_input = jt.array(test_data)
    
    # 设置相同的权重
    weight = np.random.randn(64).astype(np.float32)
    bias = np.random.randn(64).astype(np.float32)
    running_mean = np.random.randn(64).astype(np.float32)
    running_var = np.random.randn(64).astype(np.float32)
    
    pytorch_bn.weight.data = torch.from_numpy(weight)
    pytorch_bn.bias.data = torch.from_numpy(bias)
    pytorch_bn.running_mean.data = torch.from_numpy(running_mean)
    pytorch_bn.running_var.data = torch.from_numpy(running_var)
    
    jittor_bn.weight.assign(jt.array(weight))
    jittor_bn.bias.assign(jt.array(bias))
    jittor_bn.running_mean.assign(jt.array(running_mean))
    jittor_bn.running_var.assign(jt.array(running_var))
    
    print(f"测试数据: {test_data.shape}, 范围[{test_data.min():.6f}, {test_data.max():.6f}]")
    
    # 测试训练模式
    print(f"\n训练模式测试:")
    pytorch_bn.train()
    jittor_bn.train()
    
    pytorch_output_train = pytorch_bn(pytorch_input)
    jittor_output_train = jittor_bn(jittor_input)
    
    print(f"  PyTorch输出: 范围[{pytorch_output_train.min():.6f}, {pytorch_output_train.max():.6f}]")
    print(f"  Jittor输出: 范围[{jittor_output_train.min():.6f}, {jittor_output_train.max():.6f}]")
    
    train_diff = np.abs(pytorch_output_train.detach().numpy() - jittor_output_train.numpy()).max()
    print(f"  训练模式差异: {train_diff:.8f}")
    
    # 测试评估模式
    print(f"\n评估模式测试:")
    pytorch_bn.eval()
    jittor_bn.eval()
    
    pytorch_output_eval = pytorch_bn(pytorch_input)
    jittor_output_eval = jittor_bn(jittor_input)
    
    print(f"  PyTorch输出: 范围[{pytorch_output_eval.min():.6f}, {pytorch_output_eval.max():.6f}]")
    print(f"  Jittor输出: 范围[{jittor_output_eval.min():.6f}, {jittor_output_eval.max():.6f}]")
    
    eval_diff = np.abs(pytorch_output_eval.detach().numpy() - jittor_output_eval.numpy()).max()
    print(f"  评估模式差异: {eval_diff:.8f}")
    
    # 检查running stats更新
    print(f"\n检查running stats更新:")
    print(f"  PyTorch running_mean: 范围[{pytorch_bn.running_mean.min():.6f}, {pytorch_bn.running_mean.max():.6f}]")
    print(f"  Jittor running_mean: 范围[{jittor_bn.running_mean.min():.6f}, {jittor_bn.running_mean.max():.6f}]")
    
    running_mean_diff = np.abs(pytorch_bn.running_mean.detach().numpy() - jittor_bn.running_mean.numpy()).max()
    print(f"  running_mean差异: {running_mean_diff:.8f}")
    
    print(f"  PyTorch running_var: 范围[{pytorch_bn.running_var.min():.6f}, {pytorch_bn.running_var.max():.6f}]")
    print(f"  Jittor running_var: 范围[{jittor_bn.running_var.min():.6f}, {jittor_bn.running_var.max():.6f}]")
    
    running_var_diff = np.abs(pytorch_bn.running_var.detach().numpy() - jittor_bn.running_var.numpy()).max()
    print(f"  running_var差异: {running_var_diff:.8f}")
    
    return train_diff < 1e-4 and eval_diff < 1e-4


def check_model_batchnorm_settings():
    """检查模型中BatchNorm的设置"""
    print(f"\n🔍 检查模型中BatchNorm的设置")
    print("=" * 60)
    
    import sys
    sys.path.append('/home/kyc/project/nanodet/nanodet-jittor')
    from nanodet.model.arch.nanodet_plus import NanoDetPlus
    
    # 创建模型
    backbone_cfg = {
        'name': 'ShuffleNetV2',
        'model_size': '1.0x',
        'out_stages': [2, 3, 4],
        'activation': 'LeakyReLU',
        'pretrain': False  # 不加载预训练权重，专注于参数检查
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
    model.eval()
    
    # 检查模型中的BatchNorm层
    bn_layers = []
    for name, module in model.named_modules():
        if isinstance(module, jt.nn.BatchNorm2d) or isinstance(module, jt.nn.BatchNorm):
            bn_layers.append((name, module))
    
    print(f"找到 {len(bn_layers)} 个BatchNorm层:")
    
    for i, (name, bn) in enumerate(bn_layers[:10]):  # 只显示前10个
        print(f"  {name}:")
        print(f"    eps: {bn.eps}")
        print(f"    momentum: {bn.momentum}")
        print(f"    affine: {bn.affine}")
        print(f"    is_train: {bn.is_train}")
        
        # 检查权重和bias的范围
        if hasattr(bn, 'weight') and bn.weight is not None:
            print(f"    weight范围: [{bn.weight.min():.6f}, {bn.weight.max():.6f}]")
        if hasattr(bn, 'bias') and bn.bias is not None:
            print(f"    bias范围: [{bn.bias.min():.6f}, {bn.bias.max():.6f}]")
        if hasattr(bn, 'running_mean') and bn.running_mean is not None:
            print(f"    running_mean范围: [{bn.running_mean.min():.6f}, {bn.running_mean.max():.6f}]")
        if hasattr(bn, 'running_var') and bn.running_var is not None:
            print(f"    running_var范围: [{bn.running_var.min():.6f}, {bn.running_var.max():.6f}]")


def test_specific_batchnorm_issue():
    """测试特定的BatchNorm问题"""
    print(f"\n🔍 测试特定的BatchNorm问题")
    print("=" * 60)
    
    # 创建与模型中相同配置的BatchNorm
    pytorch_bn = torch.nn.BatchNorm2d(64, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
    jittor_bn = jt.nn.BatchNorm2d(64, eps=1e-5, momentum=0.1, affine=True)
    
    # 设置相同的初始状态
    weight = np.ones(64, dtype=np.float32)
    bias = np.zeros(64, dtype=np.float32)
    running_mean = np.zeros(64, dtype=np.float32)
    running_var = np.ones(64, dtype=np.float32)
    
    pytorch_bn.weight.data = torch.from_numpy(weight)
    pytorch_bn.bias.data = torch.from_numpy(bias)
    pytorch_bn.running_mean.data = torch.from_numpy(running_mean)
    pytorch_bn.running_var.data = torch.from_numpy(running_var)
    
    jittor_bn.weight.assign(jt.array(weight))
    jittor_bn.bias.assign(jt.array(bias))
    jittor_bn.running_mean.assign(jt.array(running_mean))
    jittor_bn.running_var.assign(jt.array(running_var))
    
    # 设置为评估模式
    pytorch_bn.eval()
    jittor_bn.eval()
    
    # 创建特定的测试输入
    test_input = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)  # [1, 1, 2, 2]
    test_input = np.repeat(test_input, 64, axis=1)  # [1, 64, 2, 2]
    
    pytorch_input = torch.from_numpy(test_input)
    jittor_input = jt.array(test_input)
    
    print(f"测试输入: {test_input.shape}")
    print(f"输入范围: [{test_input.min():.6f}, {test_input.max():.6f}]")
    
    # 前向传播
    pytorch_output = pytorch_bn(pytorch_input)
    jittor_output = jittor_bn(jittor_input)
    
    print(f"\nPyTorch输出: 范围[{pytorch_output.min():.6f}, {pytorch_output.max():.6f}]")
    print(f"Jittor输出: 范围[{jittor_output.min():.6f}, {jittor_output.max():.6f}]")
    
    diff = np.abs(pytorch_output.detach().numpy() - jittor_output.numpy()).max()
    print(f"差异: {diff:.10f}")
    
    if diff < 1e-6:
        print(f"✅ BatchNorm行为一致")
        return True
    else:
        print(f"❌ BatchNorm行为不一致")
        
        # 详细分析
        pytorch_np = pytorch_output.detach().numpy()
        jittor_np = jittor_output.numpy()
        
        print(f"PyTorch详细输出: {pytorch_np.flatten()[:10]}")
        print(f"Jittor详细输出: {jittor_np.flatten()[:10]}")
        
        return False


def main():
    """主函数"""
    print("🚀 开始检查BatchNorm参数设置")
    
    # 检查默认参数
    params_ok = check_batchnorm_parameters()
    
    # 测试行为差异
    behavior_ok = test_batchnorm_behavior()
    
    # 检查模型中的设置
    check_model_batchnorm_settings()
    
    # 测试特定问题
    specific_ok = test_specific_batchnorm_issue()
    
    print(f"\n📊 检查总结:")
    print(f"  默认参数: {'✅' if params_ok else '❌'}")
    print(f"  行为测试: {'✅' if behavior_ok else '❌'}")
    print(f"  特定测试: {'✅' if specific_ok else '❌'}")
    
    if all([params_ok, behavior_ok, specific_ok]):
        print(f"\n✅ BatchNorm没有问题，需要检查其他原因")
    else:
        print(f"\n❌ 发现BatchNorm问题，需要修复")
    
    print(f"\n✅ 检查完成")


if __name__ == '__main__':
    main()
