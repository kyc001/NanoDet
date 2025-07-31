#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
创建小规模VOC数据集用于快速验证
从完整的VOC数据集中随机采样指定数量的图片和对应的标注
"""

import argparse
import json
import os
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description='Create mini VOC dataset')
    parser.add_argument('--src-dir', type=str, default='data/VOCdevkit',
                        help='Source VOC dataset directory')
    parser.add_argument('--dst-dir', type=str, default='data/VOC_mini',
                        help='Destination mini dataset directory')
    parser.add_argument('--train-samples', type=int, default=100,
                        help='Number of training samples')
    parser.add_argument('--val-samples', type=int, default=50,
                        help='Number of validation samples')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    return parser.parse_args()


# VOC类别名称
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
    'bus', 'car', 'cat', 'chair', 'cow', 
    'diningtable', 'dog', 'horse', 'motorbike', 'person', 
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]


def parse_voc_annotation(xml_file):
    """解析VOC XML标注文件"""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # 获取图片信息
    filename = root.find('filename').text
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    # 获取标注信息
    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name not in VOC_CLASSES:
            continue
            
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        
        # 转换为COCO格式 [x, y, width, height]
        x = xmin
        y = ymin
        w = xmax - xmin
        h = ymax - ymin
        
        objects.append({
            'category': name,
            'category_id': VOC_CLASSES.index(name),
            'bbox': [x, y, w, h],
            'area': w * h,
            'iscrowd': 0
        })
    
    return {
        'filename': filename,
        'width': width,
        'height': height,
        'objects': objects
    }


def collect_voc_samples(voc_dir, year, split):
    """收集VOC数据集样本"""
    voc_year_dir = Path(voc_dir) / f'VOC{year}'
    
    # 读取图片列表
    split_file = voc_year_dir / 'ImageSets' / 'Main' / f'{split}.txt'
    if not split_file.exists():
        print(f'警告: {split_file} 不存在')
        return []
    
    with open(split_file, 'r') as f:
        image_names = [line.strip() for line in f.readlines()]
    
    samples = []
    for img_name in image_names:
        img_path = voc_year_dir / 'JPEGImages' / f'{img_name}.jpg'
        xml_path = voc_year_dir / 'Annotations' / f'{img_name}.xml'
        
        if img_path.exists() and xml_path.exists():
            try:
                ann_data = parse_voc_annotation(xml_path)
                # 只保留有标注的图片
                if ann_data['objects']:
                    samples.append({
                        'year': year,
                        'name': img_name,
                        'img_path': img_path,
                        'xml_path': xml_path,
                        'annotation': ann_data
                    })
            except Exception as e:
                print(f'警告: 解析 {xml_path} 失败: {e}')
    
    return samples


def sample_balanced_data(samples, num_samples, seed=42):
    """平衡采样数据，确保每个类别都有代表"""
    random.seed(seed)
    
    # 按类别分组
    class_samples = defaultdict(list)
    for sample in samples:
        for obj in sample['annotation']['objects']:
            class_samples[obj['category']].append(sample)
    
    # 确保每个类别至少有一个样本
    selected_samples = []
    used_samples = set()
    
    # 每个类别至少选一个
    for class_name in VOC_CLASSES:
        if class_name in class_samples and class_samples[class_name]:
            sample = random.choice(class_samples[class_name])
            if id(sample) not in used_samples:
                selected_samples.append(sample)
                used_samples.add(id(sample))
    
    # 随机选择剩余样本
    remaining_samples = [s for s in samples if id(s) not in used_samples]
    remaining_count = num_samples - len(selected_samples)
    
    if remaining_count > 0 and remaining_samples:
        additional_samples = random.sample(
            remaining_samples, 
            min(remaining_count, len(remaining_samples))
        )
        selected_samples.extend(additional_samples)
    
    return selected_samples[:num_samples]


def create_coco_annotation(samples, split_name):
    """创建COCO格式标注"""
    coco_data = {
        'info': {
            'description': f'VOC {split_name} mini dataset in COCO format',
            'version': '1.0',
            'year': 2023,
            'contributor': 'NanoDet-Jittor',
            'date_created': '2023-01-01'
        },
        'licenses': [{'id': 1, 'name': 'Unknown', 'url': ''}],
        'categories': [
            {'id': i, 'name': name, 'supercategory': 'object'} 
            for i, name in enumerate(VOC_CLASSES)
        ],
        'images': [],
        'annotations': []
    }
    
    annotation_id = 0
    for image_id, sample in enumerate(samples):
        # 添加图片信息
        image_info = {
            'id': image_id,
            'file_name': f"{sample['year']}_{sample['name']}.jpg",
            'width': sample['annotation']['width'],
            'height': sample['annotation']['height']
        }
        coco_data['images'].append(image_info)
        
        # 添加标注信息
        for obj in sample['annotation']['objects']:
            annotation = {
                'id': annotation_id,
                'image_id': image_id,
                'category_id': obj['category_id'],
                'bbox': obj['bbox'],
                'area': obj['area'],
                'iscrowd': obj['iscrowd']
            }
            coco_data['annotations'].append(annotation)
            annotation_id += 1
    
    return coco_data


def copy_images(samples, dst_img_dir):
    """复制图片到目标目录"""
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    
    copied_count = 0
    for sample in samples:
        src_path = sample['img_path']
        dst_filename = f"{sample['year']}_{sample['name']}.jpg"
        dst_path = dst_img_dir / dst_filename
        
        try:
            shutil.copy2(src_path, dst_path)
            copied_count += 1
        except Exception as e:
            print(f'警告: 复制图片失败 {src_path} -> {dst_path}: {e}')
    
    print(f'复制了 {copied_count} 张图片到 {dst_img_dir}')
    return copied_count


def create_mini_voc_dataset(args):
    """创建小规模VOC数据集"""
    print(f"创建小规模VOC数据集: {args.src_dir} -> {args.dst_dir}")
    print(f"训练样本: {args.train_samples}, 验证样本: {args.val_samples}")
    
    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)
    
    # 创建目标目录
    dst_dir.mkdir(parents=True, exist_ok=True)
    (dst_dir / 'images').mkdir(exist_ok=True)
    (dst_dir / 'annotations').mkdir(exist_ok=True)
    
    # 收集训练数据 (VOC2007 trainval + VOC2012 trainval)
    print("\n收集训练数据...")
    train_samples = []
    
    for year, split in [('2007', 'trainval'), ('2012', 'trainval')]:
        samples = collect_voc_samples(src_dir, year, split)
        train_samples.extend(samples)
        print(f"  VOC{year} {split}: {len(samples)} 个样本")
    
    print(f"总训练样本: {len(train_samples)}")
    
    # 收集验证数据 (VOC2007 test)
    print("\n收集验证数据...")
    val_samples = collect_voc_samples(src_dir, '2007', 'test')
    print(f"总验证样本: {len(val_samples)}")
    
    # 采样数据
    print(f"\n采样数据...")
    selected_train = sample_balanced_data(train_samples, args.train_samples, args.seed)
    selected_val = sample_balanced_data(val_samples, args.val_samples, args.seed + 1)
    
    print(f"选择的训练样本: {len(selected_train)}")
    print(f"选择的验证样本: {len(selected_val)}")
    
    # 统计类别分布
    def count_classes(samples):
        class_count = defaultdict(int)
        for sample in samples:
            for obj in sample['annotation']['objects']:
                class_count[obj['category']] += 1
        return class_count
    
    train_class_count = count_classes(selected_train)
    val_class_count = count_classes(selected_val)
    
    print(f"\n训练集类别分布:")
    for class_name in VOC_CLASSES:
        count = train_class_count.get(class_name, 0)
        print(f"  {class_name}: {count}")
    
    print(f"\n验证集类别分布:")
    for class_name in VOC_CLASSES:
        count = val_class_count.get(class_name, 0)
        print(f"  {class_name}: {count}")
    
    # 复制图片
    print(f"\n复制图片...")
    copy_images(selected_train + selected_val, dst_dir / 'images')
    
    # 创建COCO格式标注
    print(f"\n创建标注文件...")
    train_coco = create_coco_annotation(selected_train, 'train')
    val_coco = create_coco_annotation(selected_val, 'val')
    
    # 保存标注文件
    train_ann_file = dst_dir / 'annotations' / 'voc_train.json'
    val_ann_file = dst_dir / 'annotations' / 'voc_val.json'
    
    with open(train_ann_file, 'w') as f:
        json.dump(train_coco, f, indent=2)
    
    with open(val_ann_file, 'w') as f:
        json.dump(val_coco, f, indent=2)
    
    print(f"训练标注保存到: {train_ann_file}")
    print(f"验证标注保存到: {val_ann_file}")
    
    # 打印数据集统计
    print(f"\n数据集统计:")
    print(f"├── images/: {len(os.listdir(dst_dir / 'images'))} 张图片")
    print(f"└── annotations/")
    print(f"    ├── voc_train.json: {len(train_coco['images'])} 图片, {len(train_coco['annotations'])} 标注")
    print(f"    └── voc_val.json: {len(val_coco['images'])} 图片, {len(val_coco['annotations'])} 标注")
    
    print(f"\n小规模VOC数据集创建完成: {dst_dir}")


if __name__ == '__main__':
    args = parse_args()
    create_mini_voc_dataset(args)
