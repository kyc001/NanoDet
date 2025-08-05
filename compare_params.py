#!/usr/bin/env python3
"""
🔍 详细分析参数分布，找出与 PyTorch 版本的差异
"""

import sys
sys.path.insert(0, '.')

import jittor as jt
from nanodet.util import cfg, load_config
from nanodet.model.arch import build_model

def analyze_model_structure(model, prefix="", level=0):
    """递归分析模型结构和参数"""
    indent = "  " * level
    total_params = 0
    
    for name, child in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        child_params = sum(p.numel() for p in child.parameters())
        total_params += child_params
        
        # 显示有参数的模块
        if child_params > 0:
            print(f"{indent}{name}: {child_params:,} 参数 ({type(child).__name__})")
            
            # 如果参数很多，进一步分析
            if child_params > 50000:
                print(f"{indent}  ⚠️ 大参数模块，进一步分析:")
                analyze_model_structure(child, full_name, level + 2)
        
        # 递归分析子模块（但不重复计算参数）
        elif len(list(child.children())) > 0:
            print(f"{indent}{name}: (容器模块 - {type(child).__name__})")
            sub_params = analyze_model_structure(child, full_name, level + 1)
            total_params += sub_params
    
    return total_params

def compare_with_expected():
    """与期望的参数数量对比"""
    print("🔍 开始详细参数分析...")
    
    # 加载配置
    load_config(cfg, 'config/nanodet-plus-m_320_voc_bs64_50epochs.yml')
    
    # 创建模型
    model = build_model(cfg.model)
    
    # 统计总参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 总体参数统计:")
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"参数大小: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    # 与官方对比
    official_params = 1170000  # 1.17M
    ratio = total_params / official_params
    print(f"\n🔍 与官方 NanoDet-Plus-m 对比:")
    print(f"官方参数数量: {official_params:,}")
    print(f"我们的参数数量: {total_params:,}")
    print(f"差异倍数: {ratio:.2f}x")
    print(f"多出参数: {total_params - official_params:,}")
    
    if ratio > 1.5:
        print("❌ 参数数量严重超标！需要检查实现")
    elif ratio > 1.1:
        print("⚠️ 参数数量偏高，需要优化")
    else:
        print("✅ 参数数量在合理范围内")
    
    print(f"\n🔍 详细模块参数分布:")
    print("=" * 60)
    
    # 分析各个主要模块
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    fpn_params = sum(p.numel() for p in model.fpn.parameters())
    head_params = sum(p.numel() for p in model.head.parameters())
    
    print(f"backbone: {backbone_params:,} 参数")
    print(f"fpn: {fpn_params:,} 参数")
    print(f"head: {head_params:,} 参数")
    
    # 检查是否有 aux_fpn 和 aux_head
    if hasattr(model, 'aux_fpn') and model.aux_fpn is not None:
        aux_fpn_params = sum(p.numel() for p in model.aux_fpn.parameters())
        print(f"aux_fpn: {aux_fpn_params:,} 参数")
    
    if hasattr(model, 'aux_head') and model.aux_head is not None:
        aux_head_params = sum(p.numel() for p in model.aux_head.parameters())
        print(f"aux_head: {aux_head_params:,} 参数")
    
    print(f"\n🔍 FPN 详细分析:")
    print("=" * 40)
    analyze_model_structure(model.fpn, "fpn", 0)
    
    print(f"\n🔍 Head 详细分析:")
    print("=" * 40)
    analyze_model_structure(model.head, "head", 0)
    
    if hasattr(model, 'aux_head') and model.aux_head is not None:
        print(f"\n🔍 Aux Head 详细分析:")
        print("=" * 40)
        analyze_model_structure(model.aux_head, "aux_head", 0)

if __name__ == "__main__":
    compare_with_expected()
