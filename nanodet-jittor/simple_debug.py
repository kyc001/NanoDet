#!/usr/bin/env python3
"""
🔍 简单调试脚本：检查训练中 bbox 和 dfl 损失为 0 的问题
"""

import sys
sys.path.insert(0, '.')

import jittor as jt
import numpy as np
from nanodet.util import cfg, load_config
from nanodet.data.dataset import build_dataset

def main():
    print("🔍 开始简单调试...")
    
    # 设置 Jittor
    jt.flags.use_cuda = 1 if jt.has_cuda else 0
    
    # 加载配置
    load_config(cfg, 'config/nanodet-plus-m_320_voc_bs64_50epochs.yml')
    print(f"✅ 配置加载成功，类别数: {cfg.model.arch.head.num_classes}")
    
    # 创建数据集
    print("🔍 检查数据集...")
    train_dataset = build_dataset(cfg.data.train, 'train')
    print(f"✅ 数据集创建成功，样本数: {len(train_dataset)}")
    
    # 检查前几个样本
    print("🔍 检查前5个样本...")
    for i in range(min(5, len(train_dataset))):
        try:
            sample = train_dataset[i]
            gt_bboxes = sample['gt_bboxes']
            gt_labels = sample['gt_labels']
            
            print(f"样本 {i}:")
            print(f"  - 图片形状: {sample['img'].shape}")
            print(f"  - bbox 数量: {len(gt_bboxes)}")
            print(f"  - label 数量: {len(gt_labels)}")
            
            if len(gt_bboxes) > 0:
                print(f"  - bbox 范围: [{gt_bboxes.min():.2f}, {gt_bboxes.max():.2f}]")
                print(f"  - label 范围: [{gt_labels.min()}, {gt_labels.max()}]")
                print(f"  - 唯一标签: {np.unique(gt_labels)}")
                
                # 检查第一个 bbox
                bbox = gt_bboxes[0]
                label = gt_labels[0]
                print(f"  - 第一个目标: bbox=[{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}], label={label}")
                
                # 检查 bbox 有效性
                if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                    print(f"  ❌ 无效 bbox: x2 <= x1 或 y2 <= y1")
                if bbox[0] < 0 or bbox[1] < 0:
                    print(f"  ⚠️ 负坐标 bbox")
                if label < 0 or label >= 20:
                    print(f"  ❌ 标签超出范围: {label}")
            else:
                print(f"  ❌ 样本 {i} 没有目标！")
                
        except Exception as e:
            print(f"  ❌ 样本 {i} 加载失败: {e}")
    
    print("\n🔍 检查数据集统计...")
    
    # 统计所有样本的目标数量
    total_targets = 0
    empty_samples = 0
    label_counts = {}
    
    for i in range(min(100, len(train_dataset))):  # 只检查前100个样本
        try:
            sample = train_dataset[i]
            gt_labels = sample['gt_labels']
            
            if len(gt_labels) == 0:
                empty_samples += 1
            else:
                total_targets += len(gt_labels)
                for label in gt_labels:
                    label_counts[int(label)] = label_counts.get(int(label), 0) + 1
                    
        except Exception as e:
            print(f"样本 {i} 统计失败: {e}")
    
    print(f"✅ 数据集统计（前100个样本）:")
    print(f"  - 总目标数: {total_targets}")
    print(f"  - 空样本数: {empty_samples}")
    print(f"  - 平均每样本目标数: {total_targets / (100 - empty_samples) if (100 - empty_samples) > 0 else 0:.2f}")
    
    if label_counts:
        print(f"  - 标签分布:")
        for label, count in sorted(label_counts.items()):
            print(f"    标签 {label}: {count} 个")
    else:
        print(f"  ❌ 没有找到任何有效标签！")
    
    # 检查标注文件
    print("\n🔍 检查标注文件...")
    try:
        import json
        with open('data/annotations/voc_train.json', 'r') as f:
            data = json.load(f)
        
        print(f"✅ 标注文件统计:")
        print(f"  - 图片数量: {len(data['images'])}")
        print(f"  - 标注数量: {len(data['annotations'])}")
        print(f"  - 类别数量: {len(data['categories'])}")
        
        # 检查类别定义
        print(f"  - 类别定义:")
        for cat in data['categories'][:5]:
            print(f"    ID: {cat['id']}, Name: {cat['name']}")
        
        # 检查标注中的类别ID分布
        category_ids = [ann['category_id'] for ann in data['annotations']]
        unique_ids = sorted(set(category_ids))
        print(f"  - 标注中的类别ID: {unique_ids}")
        print(f"  - ID范围: [{min(category_ids)}, {max(category_ids)}]")
        
        # 检查是否有不一致
        cat_def_ids = [cat['id'] for cat in data['categories']]
        if set(unique_ids) != set(cat_def_ids):
            print(f"  ❌ 类别定义与标注数据不一致！")
            print(f"    类别定义ID: {sorted(cat_def_ids)}")
            print(f"    标注数据ID: {unique_ids}")
        else:
            print(f"  ✅ 类别定义与标注数据一致")
            
    except Exception as e:
        print(f"❌ 检查标注文件失败: {e}")
    
    print("\n🎯 简单调试完成！")
    
    # 总结可能的问题
    print("\n📋 可能的问题总结:")
    if empty_samples > 50:  # 如果超过50%的样本为空
        print("❌ 大量空样本 - 数据加载可能有问题")
    if total_targets == 0:
        print("❌ 没有任何目标 - 标注文件可能有问题")
    if not label_counts:
        print("❌ 没有有效标签 - 标签映射可能有问题")
    
    print("✅ 如果以上都正常，问题可能在模型的标签分配或损失计算中")

if __name__ == "__main__":
    main()
