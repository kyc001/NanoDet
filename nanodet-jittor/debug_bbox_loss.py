#!/usr/bin/env python3
"""
🔍 深度调试 bbox 和 dfl 损失为 0 的问题
严格检查每一个可能的原因
"""

import sys
import numpy as np
import jittor as jt
from nanodet.util import cfg, load_config
from nanodet.model.head.nanodet_plus_head import NanoDetPlusHead
from nanodet.data.dataset import build_dataset
from nanodet.data.collate import naive_collate
from jittordet.datasets.coco import CocoDataset

def print_section(title):
    print(f"\n{'='*80}")
    print(f"🔍 {title}")
    print('='*80)

def debug_label_assignment():
    """调试标签分配过程"""
    print_section("标签分配调试")
    
    # 加载配置
    load_config(cfg, 'config/nanodet-plus-m_320_voc_bs64_50epochs.yml')
    
    # 创建数据集
    train_dataset = build_dataset(cfg.data.train, 'train')
    print(f"✅ 数据集创建成功，样本数: {len(train_dataset)}")
    
    # 获取一个样本
    sample = train_dataset[0]
    print(f"✅ 获取样本成功")
    print(f"   - 图片形状: {sample['img'].shape}")
    print(f"   - 标注数量: {len(sample['gt_bboxes'])}")
    
    # 检查标注内容
    gt_bboxes = sample['gt_bboxes']
    gt_labels = sample['gt_labels']
    
    print(f"✅ 标注内容:")
    print(f"   - bbox 形状: {gt_bboxes.shape}")
    print(f"   - label 形状: {gt_labels.shape}")
    print(f"   - bbox 范围: [{gt_bboxes.min():.2f}, {gt_bboxes.max():.2f}]")
    print(f"   - label 范围: [{gt_labels.min()}, {gt_labels.max()}]")
    print(f"   - 唯一标签: {np.unique(gt_labels)}")
    
    # 检查 bbox 格式
    print(f"✅ 前5个 bbox:")
    for i in range(min(5, len(gt_bboxes))):
        bbox = gt_bboxes[i]
        label = gt_labels[i]
        print(f"   - bbox {i}: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}], label: {label}")
        
        # 检查 bbox 有效性
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            print(f"   ❌ 无效 bbox: x2 <= x1 或 y2 <= y1")
        if bbox[0] < 0 or bbox[1] < 0:
            print(f"   ⚠️ 负坐标 bbox")

def debug_head_forward():
    """调试 Head 前向传播"""
    print_section("Head 前向传播调试")
    
    # 加载配置
    load_config(cfg, 'config/nanodet-plus-m_320_voc_bs64_50epochs.yml')
    
    # 创建 Head
    head_cfg = cfg.model.arch.head
    head = NanoDetPlusHead(
        num_classes=head_cfg.num_classes,
        input_channel=head_cfg.input_channel,
        feat_channels=head_cfg.feat_channels,
        stacked_convs=head_cfg.stacked_convs,
        kernel_size=head_cfg.kernel_size,
        strides=head_cfg.strides,
        loss=head_cfg.loss,
        reg_max=head_cfg.reg_max,
    )
    print(f"✅ Head 创建成功")
    
    # 创建模拟输入
    batch_size = 1  # 使用单个样本便于调试
    strides = [8, 16, 32, 64]
    input_size = 320
    
    feats = []
    for stride in strides:
        feat_h = feat_w = input_size // stride
        feat = jt.randn(batch_size, head_cfg.input_channel, feat_h, feat_w)
        feats.append(feat)
    
    print(f"✅ 模拟特征创建成功:")
    for i, feat in enumerate(feats):
        print(f"   - Level {i}: {feat.shape}")
    
    # 前向传播
    outputs = head(feats)
    cls_scores, bbox_preds = outputs
    
    print(f"✅ 前向传播成功:")
    for i, (cls_score, bbox_pred) in enumerate(zip(cls_scores, bbox_preds)):
        print(f"   - Level {i}:")
        print(f"     cls_score: {cls_score.shape}, 范围: [{cls_score.min().item():.3f}, {cls_score.max().item():.3f}]")
        print(f"     bbox_pred: {bbox_pred.shape}, 范围: [{bbox_pred.min().item():.3f}, {bbox_pred.max().item():.3f}]")
        
        # 检查是否有异常值
        if jt.isnan(cls_score).any():
            print(f"     ❌ cls_score 包含 NaN")
        if jt.isnan(bbox_pred).any():
            print(f"     ❌ bbox_pred 包含 NaN")
        if jt.isinf(cls_score).any():
            print(f"     ❌ cls_score 包含 Inf")
        if jt.isinf(bbox_pred).any():
            print(f"     ❌ bbox_pred 包含 Inf")

def debug_anchor_generation():
    """调试 anchor 生成"""
    print_section("Anchor 生成调试")
    
    # 加载配置
    load_config(cfg, 'config/nanodet-plus-m_320_voc_bs64_50epochs.yml')
    
    # 创建 Head
    head_cfg = cfg.model.arch.head
    head = NanoDetPlusHead(
        num_classes=head_cfg.num_classes,
        input_channel=head_cfg.input_channel,
        feat_channels=head_cfg.feat_channels,
        stacked_convs=head_cfg.stacked_convs,
        kernel_size=head_cfg.kernel_size,
        strides=head_cfg.strides,
        loss=head_cfg.loss,
        reg_max=head_cfg.reg_max,
    )
    
    # 检查 anchor 生成器
    if hasattr(head, 'anchor_generator'):
        print(f"✅ 找到 anchor_generator")
        anchor_gen = head.anchor_generator
        print(f"   - 类型: {type(anchor_gen)}")
        print(f"   - strides: {anchor_gen.strides if hasattr(anchor_gen, 'strides') else 'N/A'}")
    else:
        print(f"⚠️ 没有找到 anchor_generator")
    
    # 检查 prior_generator
    if hasattr(head, 'prior_generator'):
        print(f"✅ 找到 prior_generator")
        prior_gen = head.prior_generator
        print(f"   - 类型: {type(prior_gen)}")
        print(f"   - strides: {prior_gen.strides if hasattr(prior_gen, 'strides') else 'N/A'}")
    else:
        print(f"⚠️ 没有找到 prior_generator")
    
    # 生成 anchor/prior
    input_size = 320
    strides = [8, 16, 32, 64]
    featmap_sizes = [(input_size // s, input_size // s) for s in strides]
    
    print(f"✅ 特征图尺寸: {featmap_sizes}")
    
    # 尝试生成 anchor
    try:
        if hasattr(head, 'anchor_generator'):
            anchors = head.anchor_generator.grid_anchors(featmap_sizes, device='cuda')
            print(f"✅ Anchor 生成成功:")
            for i, anchor in enumerate(anchors):
                print(f"   - Level {i}: {anchor.shape}")
        elif hasattr(head, 'prior_generator'):
            priors = head.prior_generator.grid_priors(featmap_sizes, device='cuda')
            print(f"✅ Prior 生成成功:")
            for i, prior in enumerate(priors):
                print(f"   - Level {i}: {prior.shape}")
    except Exception as e:
        print(f"❌ Anchor/Prior 生成失败: {e}")

def debug_loss_calculation():
    """调试损失计算过程"""
    print_section("损失计算调试")
    
    # 加载配置
    load_config(cfg, 'config/nanodet-plus-m_320_voc_bs64_50epochs.yml')
    
    # 创建数据集和数据加载器
    train_dataset = build_dataset(cfg.data.train, 'train')
    
    # 获取一个批次的数据
    batch_data = []
    for i in range(2):  # 小批次
        sample = train_dataset[i]
        batch_data.append(sample)
    
    # 使用 collate 函数
    batch = naive_collate(batch_data)
    
    print(f"✅ 批次数据准备成功:")
    print(f"   - 图片形状: {batch['img'].shape}")
    print(f"   - gt_bboxes 数量: {len(batch['gt_bboxes'])}")
    print(f"   - gt_labels 数量: {len(batch['gt_labels'])}")
    
    # 检查每个样本的标注
    for i, (bboxes, labels) in enumerate(zip(batch['gt_bboxes'], batch['gt_labels'])):
        print(f"   - 样本 {i}: {len(bboxes)} 个目标")
        if len(bboxes) > 0:
            print(f"     bbox 范围: [{bboxes.min():.2f}, {bboxes.max():.2f}]")
            print(f"     label 范围: [{labels.min()}, {labels.max()}]")
        else:
            print(f"     ❌ 没有目标！")
    
    # 创建模型
    head_cfg = cfg.model.arch.head
    head = NanoDetPlusHead(
        num_classes=head_cfg.num_classes,
        input_channel=head_cfg.input_channel,
        feat_channels=head_cfg.feat_channels,
        stacked_convs=head_cfg.stacked_convs,
        kernel_size=head_cfg.kernel_size,
        strides=head_cfg.strides,
        loss=head_cfg.loss,
        reg_max=head_cfg.reg_max,
    )
    
    # 创建模拟特征
    batch_size = len(batch_data)
    strides = [8, 16, 32, 64]
    input_size = 320
    
    feats = []
    for stride in strides:
        feat_h = feat_w = input_size // stride
        feat = jt.randn(batch_size, head_cfg.input_channel, feat_h, feat_w)
        feats.append(feat)
    
    # 前向传播
    outputs = head(feats)
    
    # 尝试计算损失
    try:
        loss_dict = head.loss(outputs, batch['gt_bboxes'], batch['gt_labels'])
        print(f"✅ 损失计算成功:")
        for key, value in loss_dict.items():
            print(f"   - {key}: {value.item():.6f}")
            
        # 检查是否所有损失都为 0
        all_zero = all(abs(v.item()) < 1e-6 for v in loss_dict.values())
        if all_zero:
            print(f"❌ 所有损失都为 0！这不正常！")
        else:
            print(f"✅ 至少有一些损失不为 0")
            
    except Exception as e:
        print(f"❌ 损失计算失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主调试函数"""
    print("🔍 开始深度调试 bbox 和 dfl 损失为 0 的问题...")
    
    # 设置 Jittor
    jt.flags.use_cuda = 1 if jt.has_cuda else 0
    
    # 运行各项调试
    debug_label_assignment()
    debug_head_forward()
    debug_anchor_generation()
    debug_loss_calculation()
    
    print_section("调试总结")
    print("🎯 深度调试完成！")
    print("如果发现任何异常，需要立即修复！")

if __name__ == "__main__":
    main()
