#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
创建演示VOC数据集
为训练框架开发创建一个小规模的演示数据集
"""

import os
import json
import numpy as np
from PIL import Image
import random
from pathlib import Path


def create_demo_images(output_dir, num_images=50):
    """创建演示图片"""
    img_dir = Path(output_dir) / 'images'
    img_dir.mkdir(parents=True, exist_ok=True)
    
    image_infos = []
    
    for i in range(num_images):
        # 创建320x320的随机图片，模拟真实图片
        # 使用不同的颜色模式来模拟不同类别的物体
        if i % 4 == 0:  # 蓝色背景 (天空/水)
            base_color = [100, 150, 255]
        elif i % 4 == 1:  # 绿色背景 (草地)
            base_color = [100, 200, 100]
        elif i % 4 == 2:  # 灰色背景 (建筑)
            base_color = [150, 150, 150]
        else:  # 棕色背景 (土地)
            base_color = [180, 140, 100]
        
        # 添加噪声
        img_array = np.random.normal(base_color, 30, (320, 320, 3))
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        
        # 添加一些简单的形状来模拟物体
        for _ in range(random.randint(1, 3)):
            # 随机矩形
            x1 = random.randint(50, 200)
            y1 = random.randint(50, 200)
            x2 = x1 + random.randint(30, 80)
            y2 = y1 + random.randint(30, 80)
            
            # 随机颜色
            color = [random.randint(0, 255) for _ in range(3)]
            img_array[y1:y2, x1:x2] = color
        
        img = Image.fromarray(img_array)
        
        filename = f'demo_{i:04d}.jpg'
        img_path = img_dir / filename
        img.save(img_path)
        
        image_info = {
            'id': i,
            'file_name': filename,
            'width': 320,
            'height': 320
        }
        image_infos.append(image_info)
    
    return image_infos


def create_demo_annotations(image_infos, num_classes=20):
    """创建演示标注"""
    annotations = []
    ann_id = 0
    
    for img_info in image_infos:
        # 每张图片随机生成1-4个标注框
        num_boxes = random.randint(1, 4)
        
        for _ in range(num_boxes):
            # 随机生成边界框
            x = random.randint(10, 200)
            y = random.randint(10, 200)
            w = random.randint(30, 100)
            h = random.randint(30, 100)
            
            # 确保边界框在图片内
            x = min(x, 320 - w - 10)
            y = min(y, 320 - h - 10)
            w = min(w, 320 - x - 10)
            h = min(h, 320 - y - 10)
            
            annotation = {
                'id': ann_id,
                'image_id': img_info['id'],
                'category_id': random.randint(0, num_classes - 1),
                'bbox': [x, y, w, h],
                'area': w * h,
                'iscrowd': 0
            }
            annotations.append(annotation)
            ann_id += 1
    
    return annotations


def create_coco_format_dataset(output_dir, split='train', num_images=50):
    """创建COCO格式的数据集"""
    print(f"创建 {split} 数据集，包含 {num_images} 张图片...")
    
    # VOC类别
    voc_classes = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
        'bus', 'car', 'cat', 'chair', 'cow', 
        'diningtable', 'dog', 'horse', 'motorbike', 'person', 
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    
    categories = []
    for i, class_name in enumerate(voc_classes):
        categories.append({
            'id': i,
            'name': class_name,
            'supercategory': 'object'
        })
    
    # 创建图片
    image_infos = create_demo_images(output_dir, num_images)
    
    # 创建标注
    annotations = create_demo_annotations(image_infos, len(voc_classes))
    
    # 构建COCO格式数据
    coco_data = {
        'info': {
            'description': f'Demo VOC {split} dataset for NanoDet training',
            'version': '1.0',
            'year': 2023,
            'contributor': 'NanoDet-Jittor',
            'date_created': '2023-01-01'
        },
        'licenses': [{'id': 1, 'name': 'Demo License', 'url': ''}],
        'categories': categories,
        'images': image_infos,
        'annotations': annotations
    }
    
    # 保存标注文件
    ann_dir = Path(output_dir) / 'annotations'
    ann_dir.mkdir(exist_ok=True)
    
    ann_file = ann_dir / f'voc_{split}.json'
    with open(ann_file, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"✓ 创建 {split} 数据集完成:")
    print(f"  图片数量: {len(image_infos)}")
    print(f"  标注数量: {len(annotations)}")
    print(f"  保存到: {ann_file}")
    
    return ann_file


def create_symlinks():
    """为PyTorch和Jittor版本创建符号链接"""
    print('\n创建符号链接...')
    
    data_dir = Path('data')
    
    # 为PyTorch版本创建链接
    pytorch_link = Path('nanodet-pytorch/data')
    if pytorch_link.exists():
        pytorch_link.unlink()  # 删除现有链接
    pytorch_link.symlink_to('../data', target_is_directory=True)
    print(f'✓ 创建PyTorch链接: {pytorch_link} -> ../data')
    
    # 为Jittor版本创建链接
    jittor_link = Path('nanodet-jittor/data')
    if jittor_link.exists():
        jittor_link.unlink()  # 删除现有链接
    jittor_link.symlink_to('../data', target_is_directory=True)
    print(f'✓ 创建Jittor链接: {jittor_link} -> ../data')


def main():
    """主函数"""
    print("=" * 60)
    print("创建演示VOC数据集")
    print("=" * 60)
    
    output_dir = 'data'
    
    # 创建训练集 (100张图片)
    train_ann_file = create_coco_format_dataset(output_dir, 'train', num_images=100)
    
    # 创建验证集 (50张图片)
    val_ann_file = create_coco_format_dataset(output_dir, 'val', num_images=50)
    
    # 创建符号链接
    create_symlinks()
    
    print(f"\n✓ 演示数据集创建完成！")
    print(f"数据集结构:")
    print(f"├── data/")
    print(f"│   ├── images/")
    print(f"│   │   ├── demo_0000.jpg")
    print(f"│   │   ├── demo_0001.jpg")
    print(f"│   │   └── ... (150张图片)")
    print(f"│   └── annotations/")
    print(f"│       ├── voc_train.json")
    print(f"│       └── voc_val.json")
    print(f"├── nanodet-pytorch/data -> ../data")
    print(f"└── nanodet-jittor/data -> ../data")
    
    # 统计信息
    with open(train_ann_file, 'r') as f:
        train_data = json.load(f)
    with open(val_ann_file, 'r') as f:
        val_data = json.load(f)
    
    print(f"\n数据集统计:")
    print(f"训练集: {len(train_data['images'])} 图片, {len(train_data['annotations'])} 标注")
    print(f"验证集: {len(val_data['images'])} 图片, {len(val_data['annotations'])} 标注")
    print(f"类别数: {len(train_data['categories'])} (VOC 20类)")
    
    # 类别分布统计
    train_class_count = {}
    for ann in train_data['annotations']:
        cat_id = ann['category_id']
        cat_name = train_data['categories'][cat_id]['name']
        train_class_count[cat_name] = train_class_count.get(cat_name, 0) + 1
    
    print(f"\n训练集类别分布 (前10个):")
    sorted_classes = sorted(train_class_count.items(), key=lambda x: x[1], reverse=True)
    for class_name, count in sorted_classes[:10]:
        print(f"  {class_name}: {count}")
    
    print("\n" + "=" * 60)
    print("演示数据集准备完成！")
    print("=" * 60)
    print("\n现在可以开始训练框架开发:")
    print("Jittor版本:")
    print("  cd nanodet-jittor")
    print("  python tools/train.py config/nanodet-plus-m_320_voc.yml")
    print("\nPyTorch版本:")
    print("  cd nanodet-pytorch")
    print("  python tools/train.py config/nanodet-plus-m_320_voc.yml")
    
    print(f"\n📝 注意:")
    print(f"这是演示数据集，用于训练框架开发和测试")
    print(f"实际训练时建议使用真实的VOC数据集")


if __name__ == '__main__':
    main()
