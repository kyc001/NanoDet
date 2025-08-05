#!/usr/bin/env python3
"""
🔍 详细分析参数分布，找出与 PyTorch 版本的差异
"""

import sys
sys.path.insert(0, '.')

import jittor as jt
from nanodet.util import cfg, load_config
from nanodet.model.arch import build_model

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
    
    # 分析各个主要模块
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    fpn_params = sum(p.numel() for p in model.fpn.parameters())
    head_params = sum(p.numel() for p in model.head.parameters())
    
    print(f"\n🔍 主要模块参数分布:")
    print(f"backbone: {backbone_params:,} 参数 ({backbone_params/total_params*100:.1f}%)")
    print(f"fpn: {fpn_params:,} 参数 ({fpn_params/total_params*100:.1f}%)")
    print(f"head: {head_params:,} 参数 ({head_params/total_params*100:.1f}%)")
    
    # 检查是否有 aux_fpn 和 aux_head
    if hasattr(model, 'aux_fpn') and model.aux_fpn is not None:
        aux_fpn_params = sum(p.numel() for p in model.aux_fpn.parameters())
        print(f"aux_fpn: {aux_fpn_params:,} 参数 ({aux_fpn_params/total_params*100:.1f}%)")
    
    if hasattr(model, 'aux_head') and model.aux_head is not None:
        aux_head_params = sum(p.numel() for p in model.aux_head.parameters())
        print(f"aux_head: {aux_head_params:,} 参数 ({aux_head_params/total_params*100:.1f}%)")
    
    # 分析 FPN 中的 DepthwiseConvModule
    print(f"\n🔍 DepthwiseConvModule 分析:")
    dw_count = 0
    dw_total_params = 0
    
    def count_dw_modules(module, prefix=""):
        nonlocal dw_count, dw_total_params
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if "DepthwiseConvModule" in str(type(child)):
                dw_count += 1
                child_params = sum(p.numel() for p in child.parameters())
                dw_total_params += child_params
                print(f"  #{dw_count}: {full_name} - {child_params:,} 参数")
                
                # 分析 DepthwiseConvModule 内部
                if hasattr(child, 'depthwise_weight'):
                    dw_weight_params = child.depthwise_weight.numel()
                    print(f"    depthwise_weight: {dw_weight_params:,} 参数")
                if hasattr(child, 'pointwise'):
                    pw_params = sum(p.numel() for p in child.pointwise.parameters())
                    print(f"    pointwise: {pw_params:,} 参数")
            else:
                count_dw_modules(child, full_name)
    
    count_dw_modules(model)
    print(f"\nDepthwiseConvModule 总计: {dw_count} 个，{dw_total_params:,} 参数 ({dw_total_params/total_params*100:.1f}%)")

if __name__ == "__main__":
    compare_with_expected()
