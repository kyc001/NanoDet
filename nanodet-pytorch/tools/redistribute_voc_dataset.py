#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
重新分配VOC数据集
按照标准的70%-15%-15%比例重新分配训练集、验证集、测试集
"""

import os
import random
import shutil
from pathlib import Path


def analyze_current_distribution():
    """分析当前数据集分布"""
    print("🔍 分析当前VOC数据集分布...")
    
    voc_root = "/home/kyc/project/nanodet/data/VOCdevkit/VOC2007"
    
    splits = ['train', 'val', 'test']
    current_distribution = {}
    
    for split in splits:
        split_file = os.path.join(voc_root, f"ImageSets/Main/{split}.txt")
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                image_ids = [line.strip() for line in f.readlines()]
            current_distribution[split] = len(image_ids)
            print(f"  {split}: {len(image_ids)} 张图像")
        else:
            current_distribution[split] = 0
            print(f"  {split}: 文件不存在")
    
    total = sum(current_distribution.values())
    print(f"  总计: {total} 张图像")
    
    if total > 0:
        print(f"\n当前分配比例:")
        for split, count in current_distribution.items():
            percentage = count / total * 100
            print(f"  {split}: {percentage:.1f}%")
    
    return current_distribution, total


def get_all_available_images():
    """获取所有可用的图像"""
    print("\n🔍 收集所有可用图像...")
    
    voc_root = "/home/kyc/project/nanodet/data/VOCdevkit/VOC2007"
    images_dir = os.path.join(voc_root, "JPEGImages")
    annotations_dir = os.path.join(voc_root, "Annotations")
    
    # 获取所有图像文件
    image_files = list(Path(images_dir).glob("*.jpg"))
    
    # 检查对应的标注文件是否存在
    valid_images = []
    
    for image_file in image_files:
        image_id = image_file.stem
        annotation_file = os.path.join(annotations_dir, f"{image_id}.xml")
        
        if os.path.exists(annotation_file):
            valid_images.append(image_id)
    
    print(f"  找到 {len(image_files)} 张图像文件")
    print(f"  其中 {len(valid_images)} 张有对应的标注文件")
    
    return valid_images


def redistribute_dataset(all_images, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """重新分配数据集"""
    print(f"\n🔧 重新分配数据集...")
    print(f"  目标比例: 训练集{train_ratio*100:.0f}%, 验证集{val_ratio*100:.0f}%, 测试集{test_ratio*100:.0f}%")
    
    # 确保比例加起来等于1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须等于1"
    
    # 随机打乱
    random.seed(42)  # 固定随机种子，确保可重复
    shuffled_images = all_images.copy()
    random.shuffle(shuffled_images)
    
    total_count = len(shuffled_images)
    
    # 计算各集合的大小
    train_count = int(total_count * train_ratio)
    val_count = int(total_count * val_ratio)
    test_count = total_count - train_count - val_count  # 剩余的都给测试集
    
    # 分配图像
    train_images = shuffled_images[:train_count]
    val_images = shuffled_images[train_count:train_count + val_count]
    test_images = shuffled_images[train_count + val_count:]
    
    print(f"\n新的分配结果:")
    print(f"  训练集: {len(train_images)} 张 ({len(train_images)/total_count*100:.1f}%)")
    print(f"  验证集: {len(val_images)} 张 ({len(val_images)/total_count*100:.1f}%)")
    print(f"  测试集: {len(test_images)} 张 ({len(test_images)/total_count*100:.1f}%)")
    print(f"  总计: {total_count} 张")
    
    return {
        'train': train_images,
        'val': val_images,
        'test': test_images
    }


def backup_original_splits():
    """备份原始的数据集分割"""
    print(f"\n💾 备份原始数据集分割...")
    
    voc_root = "/home/kyc/project/nanodet/data/VOCdevkit/VOC2007"
    imagesets_dir = os.path.join(voc_root, "ImageSets/Main")
    backup_dir = os.path.join(voc_root, "ImageSets/Main_backup")
    
    # 创建备份目录
    os.makedirs(backup_dir, exist_ok=True)
    
    # 备份现有的分割文件
    splits = ['train.txt', 'val.txt', 'test.txt']
    
    for split_file in splits:
        original_path = os.path.join(imagesets_dir, split_file)
        backup_path = os.path.join(backup_dir, split_file)
        
        if os.path.exists(original_path):
            shutil.copy2(original_path, backup_path)
            print(f"  备份: {split_file}")
        else:
            print(f"  跳过: {split_file} (不存在)")
    
    print(f"  备份完成: {backup_dir}")


def save_new_splits(new_distribution):
    """保存新的数据集分割"""
    print(f"\n💾 保存新的数据集分割...")
    
    voc_root = "/home/kyc/project/nanodet/data/VOCdevkit/VOC2007"
    imagesets_dir = os.path.join(voc_root, "ImageSets/Main")
    
    for split_name, image_ids in new_distribution.items():
        split_file = os.path.join(imagesets_dir, f"{split_name}.txt")
        
        with open(split_file, 'w') as f:
            for image_id in sorted(image_ids):  # 排序以保证一致性
                f.write(f"{image_id}\n")
        
        print(f"  保存: {split_name}.txt ({len(image_ids)} 张图像)")
    
    print(f"  保存完成: {imagesets_dir}")


def verify_new_distribution():
    """验证新的数据集分布"""
    print(f"\n✅ 验证新的数据集分布...")
    
    voc_root = "/home/kyc/project/nanodet/data/VOCdevkit/VOC2007"
    
    splits = ['train', 'val', 'test']
    new_distribution = {}
    all_images_check = set()
    
    for split in splits:
        split_file = os.path.join(voc_root, f"ImageSets/Main/{split}.txt")
        with open(split_file, 'r') as f:
            image_ids = [line.strip() for line in f.readlines()]
        
        new_distribution[split] = len(image_ids)
        all_images_check.update(image_ids)
        
        print(f"  {split}: {len(image_ids)} 张图像")
    
    total = sum(new_distribution.values())
    print(f"  总计: {total} 张图像")
    print(f"  去重后: {len(all_images_check)} 张图像")
    
    # 检查是否有重复
    if total == len(all_images_check):
        print(f"  ✅ 无重复图像")
    else:
        print(f"  ❌ 存在重复图像")
    
    # 显示新的比例
    print(f"\n新的分配比例:")
    for split, count in new_distribution.items():
        percentage = count / total * 100
        print(f"  {split}: {percentage:.1f}%")
    
    return new_distribution


def create_class_specific_splits():
    """创建类别特定的分割文件（VOC格式需要）"""
    print(f"\n🔧 创建类别特定的分割文件...")
    
    # VOC 20个类别
    voc_classes = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    
    voc_root = "/home/kyc/project/nanodet/data/VOCdevkit/VOC2007"
    imagesets_dir = os.path.join(voc_root, "ImageSets/Main")
    
    # 读取主要的分割
    splits = ['train', 'val', 'test']
    main_splits = {}
    
    for split in splits:
        split_file = os.path.join(imagesets_dir, f"{split}.txt")
        with open(split_file, 'r') as f:
            main_splits[split] = [line.strip() for line in f.readlines()]
    
    # 为每个类别创建分割文件
    for class_name in voc_classes:
        for split in splits:
            class_split_file = os.path.join(imagesets_dir, f"{class_name}_{split}.txt")
            
            # 简化处理：所有图像都标记为可能包含该类别
            with open(class_split_file, 'w') as f:
                for image_id in main_splits[split]:
                    f.write(f"{image_id}  1\n")  # 1表示可能包含该类别
    
    print(f"  为 {len(voc_classes)} 个类别创建了分割文件")


def main():
    """主函数"""
    print("🚀 开始重新分配VOC数据集")
    print("=" * 60)
    
    try:
        # 1. 分析当前分布
        current_dist, total = analyze_current_distribution()
        
        if total == 0:
            print("❌ 没有找到有效的数据集")
            return
        
        # 2. 获取所有可用图像
        all_images = get_all_available_images()
        
        if len(all_images) == 0:
            print("❌ 没有找到有效的图像")
            return
        
        # 3. 询问用户确认
        print(f"\n📋 重新分配计划:")
        print(f"  当前分布: 训练{current_dist.get('train', 0)}张, 验证{current_dist.get('val', 0)}张, 测试{current_dist.get('test', 0)}张")
        print(f"  目标分布: 训练70%, 验证15%, 测试15%")
        print(f"  总图像数: {len(all_images)}张")
        
        confirm = input(f"\n是否继续重新分配？(y/N): ").strip().lower()
        if confirm not in ['y', 'yes']:
            print("❌ 用户取消操作")
            return
        
        # 4. 备份原始分割
        backup_original_splits()
        
        # 5. 重新分配
        new_distribution = redistribute_dataset(all_images)
        
        # 6. 保存新分割
        save_new_splits(new_distribution)
        
        # 7. 创建类别特定分割
        create_class_specific_splits()
        
        # 8. 验证结果
        verify_new_distribution()
        
        print(f"\n🎯 数据集重新分配完成！")
        print(f"  原始分割已备份到: ImageSets/Main_backup/")
        print(f"  新的分割已保存到: ImageSets/Main/")
        print(f"  现在可以开始训练了！")
        
    except Exception as e:
        print(f"❌ 重新分配失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
