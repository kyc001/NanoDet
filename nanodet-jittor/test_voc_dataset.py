#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VOC数据集测试脚本
验证VOC数据集是否正确准备，并测试数据加载功能
"""

import json
import os
from pathlib import Path
import random
import cv2
import numpy as np


# VOC类别名称和颜色
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
    'bus', 'car', 'cat', 'chair', 'cow', 
    'diningtable', 'dog', 'horse', 'motorbike', 'person', 
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# 为每个类别分配颜色
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
    (128, 0, 128), (0, 128, 128), (255, 128, 0), (255, 0, 128), (128, 255, 0),
    (0, 255, 128), (128, 0, 255), (0, 128, 255), (192, 192, 192), (64, 64, 64)
]


def check_voc_dataset(data_dir):
    """检查VOC数据集结构"""
    data_dir = Path(data_dir)
    
    print("检查VOC数据集结构...")
    
    # 检查目录结构
    required_paths = [
        'images',
        'annotations/voc_train.json',
        'annotations/voc_val.json'
    ]
    
    missing_paths = []
    for path in required_paths:
        full_path = data_dir / path
        if full_path.exists():
            print(f"✓ {path} 存在")
        else:
            print(f"✗ {path} 不存在")
            missing_paths.append(path)
    
    if missing_paths:
        print(f"\n缺少以下文件/目录: {missing_paths}")
        return False
    
    return True


def load_coco_annotation(ann_file):
    """加载COCO格式标注文件"""
    with open(ann_file, 'r') as f:
        data = json.load(f)
    return data


def analyze_dataset(data_dir):
    """分析数据集统计信息"""
    data_dir = Path(data_dir)
    
    print("\n分析数据集统计信息...")
    
    # 加载标注文件
    train_ann = load_coco_annotation(data_dir / 'annotations/voc_train.json')
    val_ann = load_coco_annotation(data_dir / 'annotations/voc_val.json')
    
    print(f"\n训练集:")
    print(f"  图片数量: {len(train_ann['images'])}")
    print(f"  标注数量: {len(train_ann['annotations'])}")
    print(f"  类别数量: {len(train_ann['categories'])}")
    
    print(f"\n验证集:")
    print(f"  图片数量: {len(val_ann['images'])}")
    print(f"  标注数量: {len(val_ann['annotations'])}")
    print(f"  类别数量: {len(val_ann['categories'])}")
    
    # 统计类别分布
    def count_categories(annotations):
        cat_count = {}
        for ann in annotations:
            cat_id = ann['category_id']
            cat_name = VOC_CLASSES[cat_id]
            cat_count[cat_name] = cat_count.get(cat_name, 0) + 1
        return cat_count
    
    train_cat_count = count_categories(train_ann['annotations'])
    val_cat_count = count_categories(val_ann['annotations'])
    
    print(f"\n训练集类别分布:")
    for class_name in VOC_CLASSES:
        count = train_cat_count.get(class_name, 0)
        print(f"  {class_name:12s}: {count:3d}")
    
    print(f"\n验证集类别分布:")
    for class_name in VOC_CLASSES:
        count = val_cat_count.get(class_name, 0)
        print(f"  {class_name:12s}: {count:3d}")
    
    return train_ann, val_ann


def visualize_samples(data_dir, train_ann, val_ann, num_samples=5):
    """可视化数据集样本"""
    data_dir = Path(data_dir)
    
    print(f"\n可视化 {num_samples} 个样本...")
    
    # 创建输出目录
    output_dir = data_dir / 'visualization'
    output_dir.mkdir(exist_ok=True)
    
    # 随机选择样本
    all_images = train_ann['images'] + val_ann['images']
    all_annotations = train_ann['annotations'] + val_ann['annotations']
    
    # 创建图片ID到标注的映射
    img_id_to_anns = {}
    for ann in all_annotations:
        img_id = ann['image_id']
        if img_id not in img_id_to_anns:
            img_id_to_anns[img_id] = []
        img_id_to_anns[img_id].append(ann)
    
    # 随机选择有标注的图片
    images_with_anns = [img for img in all_images if img['id'] in img_id_to_anns]
    selected_images = random.sample(images_with_anns, min(num_samples, len(images_with_anns)))
    
    for i, img_info in enumerate(selected_images):
        img_path = data_dir / 'images' / img_info['file_name']
        
        if not img_path.exists():
            print(f"警告: 图片 {img_path} 不存在")
            continue
        
        # 读取图片
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"警告: 无法读取图片 {img_path}")
            continue
        
        # 绘制标注
        img_id = img_info['id']
        annotations = img_id_to_anns.get(img_id, [])
        
        for ann in annotations:
            # 获取边界框
            x, y, w, h = ann['bbox']
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            
            # 获取类别信息
            cat_id = ann['category_id']
            cat_name = VOC_CLASSES[cat_id]
            color = COLORS[cat_id]
            
            # 绘制边界框
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # 绘制类别标签
            label = f"{cat_name}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(img, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # 保存可视化结果
        output_path = output_dir / f'sample_{i+1}_{img_info["file_name"]}'
        cv2.imwrite(str(output_path), img)
        print(f"  保存可视化结果: {output_path}")
        print(f"    图片: {img_info['file_name']}")
        print(f"    尺寸: {img_info['width']}x{img_info['height']}")
        print(f"    标注数量: {len(annotations)}")
        print(f"    类别: {[VOC_CLASSES[ann['category_id']] for ann in annotations]}")


def test_data_loading():
    """测试数据加载性能"""
    print(f"\n测试数据加载性能...")
    
    # 这里可以添加实际的数据加载器测试
    # 目前只是占位符
    print("  数据加载器测试将在实现数据模块后添加")


def main():
    """主函数"""
    print("=" * 60)
    print("VOC数据集测试工具")
    print("=" * 60)
    
    # 检查数据集路径
    data_dirs = ['data/VOC_mini', 'data/VOCdevkit']
    
    data_dir = None
    for dir_path in data_dirs:
        if Path(dir_path).exists():
            data_dir = dir_path
            break
    
    if data_dir is None:
        print("错误: 未找到VOC数据集")
        print("请运行以下命令之一准备数据集:")
        print("1. python tools/download_voc_dataset.py --download --convert")
        print("2. python tools/create_mini_voc_dataset.py --src-dir /path/to/VOCdevkit")
        return False
    
    print(f"使用数据集: {data_dir}")
    
    # 检查数据集结构
    if not check_voc_dataset(data_dir):
        return False
    
    # 分析数据集
    try:
        train_ann, val_ann = analyze_dataset(data_dir)
    except Exception as e:
        print(f"分析数据集失败: {e}")
        return False
    
    # 可视化样本
    try:
        visualize_samples(data_dir, train_ann, val_ann, num_samples=3)
    except Exception as e:
        print(f"可视化样本失败: {e}")
        print("可能是因为缺少opencv-python，请安装: pip install opencv-python")
    
    # 测试数据加载
    test_data_loading()
    
    print("\n" + "=" * 60)
    print("VOC数据集测试完成！")
    print("=" * 60)
    print("\n数据集已准备就绪，可以开始训练:")
    print("  python tools/train.py config/nanodet-plus-m_320_voc.yml")
    
    return True


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
