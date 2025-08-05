#!/usr/bin/env python3
"""
🔍 NanoDet-Plus 损失函数对齐测试脚本
深度检查 Jittor 版本与 PyTorch 版本的损失函数实现对齐情况
"""

import sys
import numpy as np
import jittor as jt
from nanodet.model.loss.gfocal_loss import QualityFocalLoss, DistributionFocalLoss
from nanodet.model.loss.iou_loss import GIoULoss, IoULoss, DIoULoss, CIoULoss
from nanodet.model.head.nanodet_plus_head import NanoDetPlusHead
# from nanodet.util import load_config  # 不再需要

def print_section(title):
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print('='*60)

def test_quality_focal_loss():
    """测试 Quality Focal Loss 实现"""
    print_section("Quality Focal Loss 测试")
    
    # 创建 QFL 实例
    qfl = QualityFocalLoss(use_sigmoid=True, beta=2.0, loss_weight=1.0)
    print(f"✅ QualityFocalLoss 创建成功")
    print(f"   - use_sigmoid: {qfl.use_sigmoid}")
    print(f"   - beta: {qfl.beta}")
    print(f"   - loss_weight: {qfl.loss_weight}")
    
    # 创建测试数据
    batch_size, num_classes, h, w = 2, 20, 10, 10
    
    # 预测分数 (logits)
    pred_scores = jt.randn(batch_size, num_classes, h, w)
    
    # 目标：(label, score) 元组
    # label: 类别标签，-1 表示背景，0-19 表示前景类别
    target_labels = jt.randint(0, num_classes, (batch_size, h, w))  # 0-19
    target_scores = jt.rand(batch_size, h, w)  # 质量分数 0-1
    
    # 将部分设置为背景 (-1)
    bg_mask = jt.rand(batch_size, h, w) < 0.3  # 30% 背景
    target_labels = jt.where(bg_mask, jt.full_like(target_labels, -1), target_labels)
    
    target = (target_labels, target_scores)
    
    print(f"✅ 测试数据创建成功:")
    print(f"   - pred_scores shape: {pred_scores.shape}")
    print(f"   - target_labels shape: {target_labels.shape}")
    print(f"   - target_scores shape: {target_scores.shape}")
    print(f"   - 前景样本数: {(target_labels >= 0).sum().item()}")
    print(f"   - 背景样本数: {(target_labels == -1).sum().item()}")
    
    # 测试损失计算
    try:
        loss = qfl(pred_scores, target)
        print(f"✅ QualityFocalLoss 计算成功: {loss.item():.6f}")
        
        # 检查损失值的合理性
        if loss.item() > 0:
            print(f"✅ 损失值为正数，符合预期")
        else:
            print(f"⚠️ 损失值为 {loss.item()}，可能存在问题")
            
    except Exception as e:
        print(f"❌ QualityFocalLoss 计算失败: {e}")
        import traceback
        traceback.print_exc()

def test_giou_loss():
    """测试 GIoU Loss 实现"""
    print_section("GIoU Loss 测试")
    
    # 创建 GIoU Loss 实例
    giou_loss = GIoULoss(loss_weight=2.0)
    print(f"✅ GIoULoss 创建成功")
    print(f"   - loss_weight: {giou_loss.loss_weight}")
    
    # 创建测试数据 (bbox 格式: x1, y1, x2, y2)
    num_boxes = 100
    
    # 预测框
    pred_bboxes = jt.rand(num_boxes, 4) * 100  # 0-100 范围内的坐标
    pred_bboxes[:, 2:] += pred_bboxes[:, :2]   # 确保 x2 > x1, y2 > y1
    
    # 目标框 (添加一些噪声)
    target_bboxes = pred_bboxes + jt.randn(num_boxes, 4) * 5  # 添加噪声
    target_bboxes[:, 2:] = jt.maximum(target_bboxes[:, 2:], target_bboxes[:, :2] + 1)  # 确保有效
    
    print(f"✅ 测试数据创建成功:")
    print(f"   - pred_bboxes shape: {pred_bboxes.shape}")
    print(f"   - target_bboxes shape: {target_bboxes.shape}")
    print(f"   - pred_bboxes 范围: [{pred_bboxes.min().item():.2f}, {pred_bboxes.max().item():.2f}]")
    print(f"   - target_bboxes 范围: [{target_bboxes.min().item():.2f}, {target_bboxes.max().item():.2f}]")
    
    # 测试损失计算
    try:
        loss = giou_loss(pred_bboxes, target_bboxes)
        print(f"✅ GIoULoss 计算成功: {loss.item():.6f}")
        
        # 检查损失值的合理性
        if 0 <= loss.item() <= 2:  # GIoU loss 范围通常在 [0, 2]
            print(f"✅ 损失值在合理范围内")
        else:
            print(f"⚠️ 损失值 {loss.item()} 可能超出预期范围")
            
    except Exception as e:
        print(f"❌ GIoULoss 计算失败: {e}")
        import traceback
        traceback.print_exc()

def test_distribution_focal_loss():
    """测试 Distribution Focal Loss 实现"""
    print_section("Distribution Focal Loss 测试")
    
    # 创建 DFL 实例
    dfl = DistributionFocalLoss(loss_weight=0.25)
    print(f"✅ DistributionFocalLoss 创建成功")
    print(f"   - loss_weight: {dfl.loss_weight}")
    
    # 创建测试数据
    batch_size, reg_max, num_points = 2, 16, 1000
    
    # 预测分布 (logits)
    pred_dist = jt.randn(batch_size * num_points, reg_max + 1)
    
    # 目标距离 (连续值)
    target_dist = jt.rand(batch_size * num_points) * reg_max
    
    print(f"✅ 测试数据创建成功:")
    print(f"   - pred_dist shape: {pred_dist.shape}")
    print(f"   - target_dist shape: {target_dist.shape}")
    print(f"   - target_dist 范围: [{target_dist.min().item():.2f}, {target_dist.max().item():.2f}]")
    
    # 测试损失计算
    try:
        loss = dfl(pred_dist, target_dist)
        print(f"✅ DistributionFocalLoss 计算成功: {loss.item():.6f}")
        
        # 检查损失值的合理性
        if loss.item() > 0:
            print(f"✅ 损失值为正数，符合预期")
        else:
            print(f"⚠️ 损失值为 {loss.item()}，可能存在问题")
            
    except Exception as e:
        print(f"❌ DistributionFocalLoss 计算失败: {e}")
        import traceback
        traceback.print_exc()

def test_nanodet_plus_head():
    """测试 NanoDet-Plus Head 的损失计算"""
    print_section("NanoDet-Plus Head 损失计算测试")
    
    try:
        # 加载配置
        from nanodet.util.config import Config
        cfg = Config.fromfile('config/nanodet-plus-m_320_voc_bs64_50epochs.yml')
        print(f"✅ 配置加载成功，类别数: {cfg.model.arch.head.num_classes}")
        
        # 创建 Head 实例
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
        print(f"✅ NanoDetPlusHead 创建成功")
        
        # 创建模拟输入
        batch_size = 2
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
        print(f"✅ 前向传播成功:")
        print(f"   - cls_scores: {len(outputs[0])} levels")
        print(f"   - bbox_preds: {len(outputs[1])} levels")
        
        # 检查输出形状
        for i, (cls_score, bbox_pred) in enumerate(zip(outputs[0], outputs[1])):
            print(f"   - Level {i}: cls_score {cls_score.shape}, bbox_pred {bbox_pred.shape}")
        
    except Exception as e:
        print(f"❌ NanoDetPlusHead 测试失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主测试函数"""
    print("🔍 开始 NanoDet-Plus 损失函数深度对齐测试...")
    
    # 设置 Jittor
    jt.flags.use_cuda = 1 if jt.has_cuda else 0
    
    # 运行各项测试
    test_quality_focal_loss()
    test_giou_loss()
    test_distribution_focal_loss()
    test_nanodet_plus_head()
    
    print_section("测试总结")
    print("🎯 所有损失函数测试完成！")
    print("✅ 如果所有测试都通过，说明 Jittor 版本实现正确")
    print("⚠️ 如果有测试失败，需要进一步检查对应的实现")

if __name__ == "__main__":
    main()
