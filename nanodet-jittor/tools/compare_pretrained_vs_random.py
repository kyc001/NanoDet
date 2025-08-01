#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
对比预训练权重vs随机初始化的差异
找出预训练权重加载的问题
"""

import os
import sys
import torch
import jittor as jt
import numpy as np

# 添加路径
sys.path.append('/home/kyc/project/nanodet/nanodet-jittor')
from nanodet.model.backbone.shufflenetv2 import ShuffleNetV2


def compare_pretrained_vs_random():
    """对比预训练权重vs随机初始化"""
    print("🔍 对比预训练权重vs随机初始化")
    print("=" * 60)
    
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    jt.set_global_seed(42)
    
    # 创建固定输入
    if os.path.exists("fixed_input_data.npy"):
        input_data = np.load("fixed_input_data.npy")
        print("✓ 使用固定输入")
    else:
        input_data = np.random.randn(1, 3, 320, 320).astype(np.float32)
        print("✓ 创建新的固定输入")
    
    jittor_input = jt.array(input_data)
    
    print("\n1️⃣ 测试随机初始化的ShuffleNetV2...")
    
    # 创建随机初始化的模型
    random_model = ShuffleNetV2(
        model_size="1.0x",
        out_stages=[2, 3, 4],
        activation='LeakyReLU',
        pretrain=False  # 不加载预训练权重
    )
    random_model.eval()
    
    with jt.no_grad():
        random_output = random_model(jittor_input)
    
    print(f"随机初始化输出:")
    for i, feat in enumerate(random_output):
        print(f"  特征{i}: {feat.shape}, 范围[{feat.min():.6f}, {feat.max():.6f}]")
        # 保存随机初始化的输出
        np.save(f"random_backbone_feat_{i}.npy", feat.numpy())
    
    print("\n2️⃣ 测试预训练权重的ShuffleNetV2...")
    
    # 创建预训练权重的模型
    pretrained_model = ShuffleNetV2(
        model_size="1.0x",
        out_stages=[2, 3, 4],
        activation='LeakyReLU',
        pretrain=True  # 加载预训练权重
    )
    pretrained_model.eval()
    
    with jt.no_grad():
        pretrained_output = pretrained_model(jittor_input)
    
    print(f"预训练权重输出:")
    for i, feat in enumerate(pretrained_output):
        print(f"  特征{i}: {feat.shape}, 范围[{feat.min():.6f}, {feat.max():.6f}]")
        # 保存预训练权重的输出
        np.save(f"pretrained_backbone_feat_{i}.npy", feat.numpy())
    
    print("\n3️⃣ 对比差异...")
    
    # 对比随机初始化vs预训练权重
    for i in range(len(random_output)):
        random_feat = random_output[i].numpy()
        pretrained_feat = pretrained_output[i].numpy()
        
        diff = np.abs(random_feat - pretrained_feat)
        max_diff = diff.max()
        mean_diff = diff.mean()
        
        print(f"  特征{i}差异: 最大{max_diff:.6f}, 平均{mean_diff:.6f}")
        
        # 分析数值分布
        random_positive = (random_feat > 0).sum()
        random_negative = (random_feat < 0).sum()
        pretrained_positive = (pretrained_feat > 0).sum()
        pretrained_negative = (pretrained_feat < 0).sum()
        
        print(f"    随机初始化: {random_positive}个正值, {random_negative}个负值")
        print(f"    预训练权重: {pretrained_positive}个正值, {pretrained_negative}个负值")
    
    print("\n4️⃣ 检查权重差异...")
    
    # 对比模型权重
    random_params = dict(random_model.named_parameters())
    pretrained_params = dict(pretrained_model.named_parameters())
    
    weight_diffs = []
    for name in random_params.keys():
        if name in pretrained_params:
            random_weight = random_params[name].numpy()
            pretrained_weight = pretrained_params[name].numpy()
            
            diff = np.abs(random_weight - pretrained_weight)
            max_diff = diff.max()
            mean_diff = diff.mean()
            
            weight_diffs.append((name, max_diff, mean_diff))
    
    # 显示权重差异最大的几个层
    weight_diffs.sort(key=lambda x: x[1], reverse=True)
    print(f"权重差异最大的10个层:")
    for name, max_diff, mean_diff in weight_diffs[:10]:
        print(f"  {name}: 最大{max_diff:.6f}, 平均{mean_diff:.6f}")
    
    print("\n5️⃣ 检查预训练权重的数值分布...")
    
    # 分析预训练权重的数值分布
    all_pretrained_weights = []
    for name, param in pretrained_model.named_parameters():
        weights = param.numpy().flatten()
        all_pretrained_weights.extend(weights)
    
    all_pretrained_weights = np.array(all_pretrained_weights)
    
    print(f"预训练权重统计:")
    print(f"  总数: {len(all_pretrained_weights)}")
    print(f"  范围: [{all_pretrained_weights.min():.6f}, {all_pretrained_weights.max():.6f}]")
    print(f"  均值: {all_pretrained_weights.mean():.6f}")
    print(f"  标准差: {all_pretrained_weights.std():.6f}")
    print(f"  正值数量: {(all_pretrained_weights > 0).sum()}")
    print(f"  负值数量: {(all_pretrained_weights < 0).sum()}")
    print(f"  零值数量: {(all_pretrained_weights == 0).sum()}")
    
    # 分析随机权重的数值分布
    all_random_weights = []
    for name, param in random_model.named_parameters():
        weights = param.numpy().flatten()
        all_random_weights.extend(weights)
    
    all_random_weights = np.array(all_random_weights)
    
    print(f"\n随机权重统计:")
    print(f"  总数: {len(all_random_weights)}")
    print(f"  范围: [{all_random_weights.min():.6f}, {all_random_weights.max():.6f}]")
    print(f"  均值: {all_random_weights.mean():.6f}")
    print(f"  标准差: {all_random_weights.std():.6f}")
    print(f"  正值数量: {(all_random_weights > 0).sum()}")
    print(f"  负值数量: {(all_random_weights < 0).sum()}")
    print(f"  零值数量: {(all_random_weights == 0).sum()}")


def check_pytorch_pretrained():
    """检查PyTorch预训练权重"""
    print("\n6️⃣ 检查PyTorch预训练权重...")
    
    try:
        # 直接下载PyTorch预训练权重
        import torch.utils.model_zoo as model_zoo
        
        url = "https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth"
        print(f"下载PyTorch预训练权重: {url}")
        
        pretrained_state_dict = model_zoo.load_url(url)
        
        print(f"PyTorch预训练权重包含 {len(pretrained_state_dict)} 个参数:")
        
        # 分析权重
        all_weights = []
        for name, param in pretrained_state_dict.items():
            weights = param.numpy().flatten()
            all_weights.extend(weights)
            print(f"  {name}: {param.shape}, 范围[{param.min():.6f}, {param.max():.6f}]")
        
        all_weights = np.array(all_weights)
        
        print(f"\nPyTorch预训练权重统计:")
        print(f"  总数: {len(all_weights)}")
        print(f"  范围: [{all_weights.min():.6f}, {all_weights.max():.6f}]")
        print(f"  均值: {all_weights.mean():.6f}")
        print(f"  标准差: {all_weights.std():.6f}")
        print(f"  正值数量: {(all_weights > 0).sum()}")
        print(f"  负值数量: {(all_weights < 0).sum()}")
        print(f"  零值数量: {(all_weights == 0).sum()}")
        
    except Exception as e:
        print(f"❌ 无法下载PyTorch预训练权重: {e}")


def main():
    """主函数"""
    print("🚀 开始对比预训练权重vs随机初始化")
    
    # 对比预训练vs随机
    compare_pretrained_vs_random()
    
    # 检查PyTorch预训练权重
    check_pytorch_pretrained()
    
    print(f"\n✅ 对比完成")


if __name__ == '__main__':
    main()
