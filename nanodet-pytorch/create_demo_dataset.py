#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
创建演示数据集
为PyTorch训练创建一个小规模的演示数据集
"""

import os
import json
import numpy as np
from PIL import Image
import random


def create_demo_images(output_dir, num_images=20):
    """创建演示图片"""
    img_dir = os.path.join(output_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)
    
    image_infos = []
    
    for i in range(num_images):
        # 创建320x320的随机图片
        img_array = np.random.randint(0, 256, (320, 320, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        filename = f'demo_{i:04d}.jpg'
        img_path = os.path.join(img_dir, filename)
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
        # 每张图片随机生成1-3个标注框
        num_boxes = random.randint(1, 3)
        
        for _ in range(num_boxes):
            # 随机生成边界框
            x = random.randint(10, 250)
            y = random.randint(10, 250)
            w = random.randint(20, 100)
            h = random.randint(20, 100)
            
            # 确保边界框在图片内
            x = min(x, 320 - w)
            y = min(y, 320 - h)
            
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


def create_coco_format_dataset(output_dir, split='train', num_images=20):
    """创建COCO格式的数据集"""
    print(f"Creating {split} dataset with {num_images} images...")
    
    # VOC类别
    categories = []
    voc_classes = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
        'bus', 'car', 'cat', 'chair', 'cow', 
        'diningtable', 'dog', 'horse', 'motorbike', 'person', 
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    
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
            'description': f'Demo VOC {split} dataset',
            'version': '1.0',
            'year': 2023,
            'contributor': 'NanoDet-PyTorch',
            'date_created': '2023-01-01'
        },
        'licenses': [{'id': 1, 'name': 'Unknown', 'url': ''}],
        'categories': categories,
        'images': image_infos,
        'annotations': annotations
    }
    
    # 保存标注文件
    ann_dir = os.path.join(output_dir, 'annotations')
    os.makedirs(ann_dir, exist_ok=True)
    
    ann_file = os.path.join(ann_dir, f'voc_{split}.json')
    with open(ann_file, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"✓ Created {split} dataset:")
    print(f"  Images: {len(image_infos)}")
    print(f"  Annotations: {len(annotations)}")
    print(f"  Saved to: {ann_file}")
    
    return ann_file


def main():
    """主函数"""
    print("=" * 60)
    print("Creating Demo VOC Dataset for PyTorch Training")
    print("=" * 60)
    
    output_dir = 'data/VOC_demo'
    
    # 创建训练集
    train_ann_file = create_coco_format_dataset(output_dir, 'train', num_images=50)
    
    # 创建验证集
    val_ann_file = create_coco_format_dataset(output_dir, 'val', num_images=20)
    
    print(f"\n✓ Demo dataset created successfully!")
    print(f"Dataset structure:")
    print(f"├── {output_dir}/")
    print(f"│   ├── images/")
    print(f"│   │   ├── demo_0000.jpg")
    print(f"│   │   ├── demo_0001.jpg")
    print(f"│   │   └── ...")
    print(f"│   └── annotations/")
    print(f"│       ├── voc_train.json")
    print(f"│       └── voc_val.json")
    
    print(f"\nNext steps:")
    print(f"1. Update config file paths to use: {os.path.abspath(output_dir)}")
    print(f"2. Run training: python tools/train.py config/nanodet-plus-m_320_voc.yml")
    
    return output_dir


if __name__ == '__main__':
    dataset_dir = main()
