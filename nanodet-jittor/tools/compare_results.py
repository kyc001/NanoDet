#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
结果对比工具
对比Jittor和PyTorch的推理结果
"""

import os
import sys
import numpy as np


def analyze_jittor_results():
    """分析Jittor结果"""
    print("🔍 分析Jittor结果")
    print("=" * 60)
    
    if os.path.exists("jittor_inference_results.npy"):
        jittor_results = np.load("jittor_inference_results.npy", allow_pickle=True).item()
        
        print(f"Jittor推理结果:")
        print(f"  输出形状: {jittor_results['output'].shape}")
        print(f"  输出范围: [{jittor_results['output'].min():.6f}, {jittor_results['output'].max():.6f}]")
        print(f"  最高置信度: {jittor_results['max_confidence']:.6f}")
        
        # 分析置信度分布
        cls_scores = jittor_results['cls_scores']
        print(f"  置信度统计:")
        print(f"    均值: {cls_scores.mean():.6f}")
        print(f"    标准差: {cls_scores.std():.6f}")
        print(f"    >0.1的比例: {(cls_scores > 0.1).mean():.4f}")
        print(f"    >0.5的比例: {(cls_scores > 0.5).mean():.4f}")
        print(f"    >0.9的比例: {(cls_scores > 0.9).mean():.4f}")
        
        return jittor_results
    else:
        print("❌ 未找到Jittor结果文件")
        return None


def analyze_pytorch_results():
    """分析PyTorch结果"""
    print(f"\n🔍 分析PyTorch结果")
    print("=" * 60)
    
    if os.path.exists("pytorch_inference_results.npy"):
        pytorch_results = np.load("pytorch_inference_results.npy", allow_pickle=True).item()
        
        print(f"PyTorch推理结果:")
        print(f"  输出形状: {pytorch_results['output'].shape}")
        print(f"  输出范围: [{pytorch_results['output'].min():.6f}, {pytorch_results['output'].max():.6f}]")
        print(f"  最高置信度: {pytorch_results['max_confidence']:.6f}")
        
        # 分析置信度分布
        cls_scores = pytorch_results['cls_scores']
        print(f"  置信度统计:")
        print(f"    均值: {cls_scores.mean():.6f}")
        print(f"    标准差: {cls_scores.std():.6f}")
        print(f"    >0.1的比例: {(cls_scores > 0.1).mean():.4f}")
        print(f"    >0.5的比例: {(cls_scores > 0.5).mean():.4f}")
        print(f"    >0.9的比例: {(cls_scores > 0.9).mean():.4f}")
        
        return pytorch_results
    else:
        print("❌ 未找到PyTorch结果文件")
        return None


def compare_results(jittor_results, pytorch_results):
    """对比结果"""
    print(f"\n🔍 对比结果")
    print("=" * 60)
    
    if jittor_results is None or pytorch_results is None:
        print("❌ 缺少结果文件，无法对比")
        return
    
    # 输出差异
    output_diff = np.abs(jittor_results['output'] - pytorch_results['output'])
    max_diff = output_diff.max()
    mean_diff = output_diff.mean()
    
    print(f"输出差异:")
    print(f"  最大差异: {max_diff:.6f}")
    print(f"  平均差异: {mean_diff:.6f}")
    print(f"  差异标准差: {output_diff.std():.6f}")
    
    # 置信度差异
    confidence_diff = abs(jittor_results['max_confidence'] - pytorch_results['max_confidence'])
    print(f"\n置信度对比:")
    print(f"  Jittor最高置信度: {jittor_results['max_confidence']:.6f}")
    print(f"  PyTorch最高置信度: {pytorch_results['max_confidence']:.6f}")
    print(f"  置信度差异: {confidence_diff:.6f}")
    
    # 判断一致性
    print(f"\n一致性评估:")
    if max_diff < 0.001:
        print(f"  ✅ 输出高度一致 (差异 < 0.001)")
    elif max_diff < 0.01:
        print(f"  ⚠️ 输出基本一致 (差异 < 0.01)")
    elif max_diff < 0.1:
        print(f"  ⚠️ 输出有一定差异 (差异 < 0.1)")
    else:
        print(f"  ❌ 输出差异较大 (差异 >= 0.1)")
    
    if confidence_diff < 0.01:
        print(f"  ✅ 置信度高度一致")
    elif confidence_diff < 0.1:
        print(f"  ⚠️ 置信度基本一致")
    else:
        print(f"  ❌ 置信度差异较大")
    
    # 分析差异分布
    print(f"\n差异分布分析:")
    print(f"  差异 > 0.001的比例: {(output_diff > 0.001).mean():.4f}")
    print(f"  差异 > 0.01的比例: {(output_diff > 0.01).mean():.4f}")
    print(f"  差异 > 0.1的比例: {(output_diff > 0.1).mean():.4f}")


def analyze_expected_performance():
    """分析预期性能"""
    print(f"\n🔍 分析预期性能")
    print("=" * 60)
    
    print(f"根据PyTorch训练结果，预期性能指标:")
    print(f"  mAP: 0.277")
    print(f"  AP_50: 0.475")
    print(f"  最高置信度应该 > 0.5")
    
    # 分析当前Jittor结果
    jittor_results = analyze_jittor_results()
    if jittor_results:
        current_confidence = jittor_results['max_confidence']
        print(f"\n当前Jittor结果分析:")
        print(f"  当前最高置信度: {current_confidence:.6f}")
        
        if current_confidence > 0.5:
            print(f"  ✅ 置信度正常")
        elif current_confidence > 0.1:
            print(f"  ⚠️ 置信度偏低，可能有问题")
        else:
            print(f"  ❌ 置信度过低，模型有严重问题")
            
            print(f"\n可能的问题:")
            print(f"  1. 权重加载不完整或不正确")
            print(f"  2. 模型架构与PyTorch不一致")
            print(f"  3. 某些操作的实现有差异")
            print(f"  4. 数值精度问题")


def main():
    """主函数"""
    print("🚀 开始结果对比分析")
    
    # 分析Jittor结果
    jittor_results = analyze_jittor_results()
    
    # 分析PyTorch结果
    pytorch_results = analyze_pytorch_results()
    
    # 对比结果
    if jittor_results and pytorch_results:
        compare_results(jittor_results, pytorch_results)
    
    # 分析预期性能
    analyze_expected_performance()
    
    print(f"\n✅ 结果对比分析完成")


if __name__ == '__main__':
    main()
