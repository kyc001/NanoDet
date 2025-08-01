#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
检查激活函数配置差异
对比PyTorch和Jittor版本的激活函数参数
"""

import torch
import jittor as jt
import numpy as np


def check_leaky_relu_config():
    """检查LeakyReLU配置差异"""
    print("🔍 检查LeakyReLU配置差异")
    print("=" * 50)
    
    # PyTorch LeakyReLU默认参数
    pytorch_leaky = torch.nn.LeakyReLU()
    print(f"PyTorch LeakyReLU:")
    print(f"  negative_slope: {pytorch_leaky.negative_slope}")
    print(f"  inplace: {pytorch_leaky.inplace}")
    
    # Jittor LeakyReLU默认参数
    jittor_leaky = jt.nn.LeakyReLU()
    print(f"\nJittor LeakyReLU:")
    if hasattr(jittor_leaky, 'negative_slope'):
        print(f"  negative_slope: {jittor_leaky.negative_slope}")
    else:
        print(f"  negative_slope: 未找到属性")
    
    if hasattr(jittor_leaky, 'inplace'):
        print(f"  inplace: {jittor_leaky.inplace}")
    else:
        print(f"  inplace: 未找到属性")
    
    # 测试数值行为
    test_data = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0], dtype=np.float32)
    
    pytorch_input = torch.from_numpy(test_data)
    pytorch_output = pytorch_leaky(pytorch_input)
    
    jittor_input = jt.array(test_data)
    jittor_output = jittor_leaky(jittor_input)
    
    print(f"\n数值测试:")
    print(f"  输入: {test_data}")
    print(f"  PyTorch输出: {pytorch_output.numpy()}")
    print(f"  Jittor输出: {jittor_output.numpy()}")
    
    diff = np.abs(pytorch_output.numpy() - jittor_output.numpy())
    print(f"  差异: {diff}")
    print(f"  最大差异: {diff.max()}")
    
    return diff.max() < 1e-6


def check_activation_creation():
    """检查激活函数创建方式"""
    print("\n🔍 检查激活函数创建方式")
    print("=" * 50)
    
    # 检查我们的激活函数创建
    import sys
    sys.path.append('/home/kyc/project/nanodet/nanodet-jittor')
    from nanodet.model.module.activation import act_layers
    
    print("Jittor版本激活函数创建:")
    jittor_leaky_class = act_layers['LeakyReLU']
    jittor_leaky = jittor_leaky_class()
    print(f"  类型: {type(jittor_leaky)}")
    if hasattr(jittor_leaky, 'negative_slope'):
        print(f"  negative_slope: {jittor_leaky.negative_slope}")

    # 检查PyTorch版本的激活函数创建
    sys.path.insert(0, '/home/kyc/project/nanodet/nanodet-pytorch')
    try:
        from nanodet.model.module.activation import act_layers as pytorch_act_layers

        print("\nPyTorch版本激活函数创建:")
        pytorch_leaky_class = pytorch_act_layers['LeakyReLU']
        pytorch_leaky = pytorch_leaky_class()
        print(f"  类型: {type(pytorch_leaky)}")
        print(f"  negative_slope: {pytorch_leaky.negative_slope}")

        # 对比参数
        if hasattr(jittor_leaky, 'negative_slope') and hasattr(pytorch_leaky, 'negative_slope'):
            if jittor_leaky.negative_slope == pytorch_leaky.negative_slope:
                print("  ✅ negative_slope参数一致")
            else:
                print(f"  ❌ negative_slope参数不一致: Jittor={jittor_leaky.negative_slope}, PyTorch={pytorch_leaky.negative_slope}")

    except Exception as e:
        print(f"  ❌ 无法导入PyTorch版本: {e}")


def check_shufflenet_activation():
    """检查ShuffleNet中的激活函数配置"""
    print("\n🔍 检查ShuffleNet中的激活函数配置")
    print("=" * 50)
    
    import sys
    sys.path.append('/home/kyc/project/nanodet/nanodet-jittor')
    from nanodet.model.backbone.shufflenetv2 import ShuffleNetV2
    
    # 创建ShuffleNetV2
    print("创建Jittor ShuffleNetV2...")
    jittor_shufflenet = ShuffleNetV2(
        model_size="1.0x",
        out_stages=[2, 3, 4],
        activation='LeakyReLU',
        pretrain=False  # 不加载预训练权重，专注于结构
    )
    
    print("检查ShuffleNetV2中的激活函数:")
    
    # 检查conv1中的激活函数
    conv1_activation = jittor_shufflenet.conv1[2]  # 第三个是激活函数
    print(f"  conv1激活函数: {type(conv1_activation)}")
    if hasattr(conv1_activation, 'negative_slope'):
        print(f"    negative_slope: {conv1_activation.negative_slope}")
    
    # 检查stage2中的激活函数
    stage2_block0 = jittor_shufflenet.stage2[0]
    if hasattr(stage2_block0, 'branch2'):
        branch2_activation = stage2_block0.branch2[2]  # 第三个是激活函数
        print(f"  stage2激活函数: {type(branch2_activation)}")
        if hasattr(branch2_activation, 'negative_slope'):
            print(f"    negative_slope: {branch2_activation.negative_slope}")
    
    # 对比PyTorch版本
    try:
        sys.path.insert(0, '/home/kyc/project/nanodet/nanodet-pytorch')
        from nanodet.model.backbone.shufflenetv2 import ShuffleNetV2 as PyTorchShuffleNetV2
        
        print("\n创建PyTorch ShuffleNetV2...")
        pytorch_shufflenet = PyTorchShuffleNetV2(
            model_size="1.0x",
            out_stages=[2, 3, 4],
            activation='LeakyReLU',
            pretrain=False
        )
        
        print("检查PyTorch ShuffleNetV2中的激活函数:")
        
        # 检查conv1中的激活函数
        pytorch_conv1_activation = pytorch_shufflenet.conv1[2]
        print(f"  conv1激活函数: {type(pytorch_conv1_activation)}")
        print(f"    negative_slope: {pytorch_conv1_activation.negative_slope}")
        
        # 检查stage2中的激活函数
        pytorch_stage2_block0 = pytorch_shufflenet.stage2[0]
        pytorch_branch2_activation = pytorch_stage2_block0.branch2[2]
        print(f"  stage2激活函数: {type(pytorch_branch2_activation)}")
        print(f"    negative_slope: {pytorch_branch2_activation.negative_slope}")
        
        # 对比参数
        if (hasattr(conv1_activation, 'negative_slope') and 
            hasattr(pytorch_conv1_activation, 'negative_slope')):
            if conv1_activation.negative_slope == pytorch_conv1_activation.negative_slope:
                print("  ✅ conv1激活函数参数一致")
            else:
                print(f"  ❌ conv1激活函数参数不一致")
        
    except Exception as e:
        print(f"  ❌ 无法创建PyTorch版本: {e}")


def create_minimal_test():
    """创建最小测试案例"""
    print("\n🔍 创建最小测试案例")
    print("=" * 50)
    
    import sys
    sys.path.append('/home/kyc/project/nanodet/nanodet-jittor')
    from nanodet.model.backbone.shufflenetv2 import ShuffleNetV2
    
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    jt.set_global_seed(42)
    
    # 创建简单的测试输入
    test_input = np.random.randn(1, 3, 64, 64).astype(np.float32)
    
    print("创建Jittor ShuffleNetV2（无预训练）...")
    jittor_model = ShuffleNetV2(
        model_size="1.0x",
        out_stages=[2, 3, 4],
        activation='LeakyReLU',
        pretrain=False
    )
    jittor_model.eval()
    
    jittor_input = jt.array(test_input)
    with jt.no_grad():
        jittor_output = jittor_model(jittor_input)
    
    print(f"Jittor输出:")
    for i, feat in enumerate(jittor_output):
        print(f"  特征{i}: {feat.shape}, 范围[{feat.min():.6f}, {feat.max():.6f}]")
    
    # 保存输出用于对比
    for i, feat in enumerate(jittor_output):
        np.save(f"jittor_minimal_feat_{i}.npy", feat.numpy())
    
    # 尝试创建PyTorch版本进行对比
    try:
        sys.path.insert(0, '/home/kyc/project/nanodet/nanodet-pytorch')
        from nanodet.model.backbone.shufflenetv2 import ShuffleNetV2 as PyTorchShuffleNetV2
        
        print("\n创建PyTorch ShuffleNetV2（无预训练）...")
        pytorch_model = PyTorchShuffleNetV2(
            model_size="1.0x",
            out_stages=[2, 3, 4],
            activation='LeakyReLU',
            pretrain=False
        )
        pytorch_model.eval()
        
        pytorch_input = torch.from_numpy(test_input)
        with torch.no_grad():
            pytorch_output = pytorch_model(pytorch_input)
        
        print(f"PyTorch输出:")
        for i, feat in enumerate(pytorch_output):
            print(f"  特征{i}: {feat.shape}, 范围[{feat.min():.6f}, {feat.max():.6f}]")
        
        # 对比差异
        print(f"\n差异对比:")
        for i, (jittor_feat, pytorch_feat) in enumerate(zip(jittor_output, pytorch_output)):
            diff = np.abs(jittor_feat.numpy() - pytorch_feat.detach().numpy())
            max_diff = diff.max()
            mean_diff = diff.mean()
            print(f"  特征{i}差异: 最大{max_diff:.6f}, 平均{mean_diff:.6f}")
        
    except Exception as e:
        print(f"  ❌ 无法创建PyTorch版本: {e}")


def main():
    """主函数"""
    print("🚀 开始检查激活函数配置差异")
    
    # 检查LeakyReLU配置
    leaky_relu_ok = check_leaky_relu_config()
    
    # 检查激活函数创建方式
    check_activation_creation()
    
    # 检查ShuffleNet中的激活函数
    check_shufflenet_activation()
    
    # 创建最小测试案例
    create_minimal_test()
    
    print(f"\n📊 检查总结:")
    print(f"  LeakyReLU基础行为: {'✅' if leaky_relu_ok else '❌'}")
    
    print(f"\n✅ 检查完成")


if __name__ == '__main__':
    main()
