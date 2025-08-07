#!/usr/bin/env python3
"""
🔧 预训练权重转换工具
将 depthwise 卷积权重转换为标准卷积权重
"""

import sys
sys.path.insert(0, '.')

import jittor as jt
import pickle
import os

def convert_depthwise_to_standard(depthwise_weight):
    """
    将 depthwise 卷积权重转换为标准卷积权重
    
    Args:
        depthwise_weight: [out_channels, 1, kernel_h, kernel_w]
    
    Returns:
        standard_weight: [out_channels, out_channels, kernel_h, kernel_w]
    """
    out_channels, _, kernel_h, kernel_w = depthwise_weight.shape
    
    # 创建标准卷积权重：对角矩阵形式
    standard_weight = jt.zeros((out_channels, out_channels, kernel_h, kernel_w))
    
    for i in range(out_channels):
        # 将 depthwise 权重放在对角线位置
        standard_weight[i, i, :, :] = depthwise_weight[i, 0, :, :]
    
    return standard_weight

def convert_pretrained_weights(input_path, output_path):
    """转换预训练权重文件"""
    print(f"🔧 开始转换预训练权重...")
    print(f"输入文件: {input_path}")
    print(f"输出文件: {output_path}")
    
    # 加载原始权重
    print("📥 加载原始权重...")
    with open(input_path, 'rb') as f:
        state_dict = pickle.load(f)
    
    print(f"✅ 加载了 {len(state_dict)} 个参数")
    
    # 需要转换的层名称模式
    depthwise_patterns = [
        'branch1.0.weight',  # ShuffleNet depthwise 卷积
        'branch2.3.weight',  # ShuffleNet depthwise 卷积
    ]
    
    converted_count = 0
    
    # 转换权重
    print("\n🔄 开始转换权重...")
    for name, weight in state_dict.items():
        # 检查是否是需要转换的 depthwise 卷积
        is_depthwise = any(pattern in name for pattern in depthwise_patterns)
        
        if is_depthwise and len(weight.shape) == 4 and weight.shape[1] == 1:
            print(f"  转换 {name}: {weight.shape} -> ", end="")
            
            # 转换权重
            converted_weight = convert_depthwise_to_standard(weight)
            state_dict[name] = converted_weight
            
            print(f"{converted_weight.shape}")
            converted_count += 1
    
    print(f"\n✅ 转换了 {converted_count} 个 depthwise 卷积层")
    
    # 保存转换后的权重
    print(f"\n💾 保存转换后的权重到 {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(state_dict, f)
    
    print("🎉 权重转换完成！")

def main():
    # 查找预训练权重文件
    pretrained_paths = [
        'nanodet_plus_m_backbone.ckpt',
        'nanodet-plus-m_320.ckpt',
        'workspace/nanodet_plus_m_backbone.ckpt',
        'workspace/nanodet-plus-m_320.ckpt',
    ]
    
    input_path = None
    for path in pretrained_paths:
        if os.path.exists(path):
            input_path = path
            break
    
    if input_path is None:
        print("❌ 未找到预训练权重文件")
        print("请确保以下文件之一存在：")
        for path in pretrained_paths:
            print(f"  - {path}")
        return
    
    # 生成输出文件名
    base_name = os.path.splitext(input_path)[0]
    output_path = f"{base_name}_converted.ckpt"
    
    # 转换权重
    convert_pretrained_weights(input_path, output_path)
    
    print(f"\n📋 使用说明：")
    print(f"请在配置文件中将预训练权重路径改为：")
    print(f"  load_from: {output_path}")

if __name__ == "__main__":
    main()
