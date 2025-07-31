#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
创建小规模COCO数据集用于快速验证和调试
从完整的COCO数据集中随机采样指定数量的图片和对应的标注
"""

import argparse
import json
import os
import random
import shutil
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description='Create mini COCO dataset')
    parser.add_argument('--src-dir', type=str, required=True,
                        help='Source COCO dataset directory')
    parser.add_argument('--dst-dir', type=str, required=True,
                        help='Destination mini dataset directory')
    parser.add_argument('--train-samples', type=int, default=100,
                        help='Number of training samples')
    parser.add_argument('--val-samples', type=int, default=50,
                        help='Number of validation samples')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    return parser.parse_args()


def load_coco_annotations(ann_file):
    """加载COCO标注文件"""
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    return coco_data


def sample_images_with_annotations(coco_data, num_samples, seed=42):
    """从COCO数据中采样指定数量的图片及其标注"""
    random.seed(seed)
    
    # 获取所有图片ID
    all_image_ids = [img['id'] for img in coco_data['images']]
    
    # 随机采样图片ID
    sampled_image_ids = random.sample(all_image_ids, min(num_samples, len(all_image_ids)))
    sampled_image_ids_set = set(sampled_image_ids)
    
    # 过滤图片
    sampled_images = [img for img in coco_data['images'] if img['id'] in sampled_image_ids_set]
    
    # 过滤标注
    sampled_annotations = [ann for ann in coco_data['annotations'] 
                          if ann['image_id'] in sampled_image_ids_set]
    
    # 构建新的COCO数据结构
    mini_coco_data = {
        'info': coco_data['info'],
        'licenses': coco_data['licenses'],
        'categories': coco_data['categories'],
        'images': sampled_images,
        'annotations': sampled_annotations
    }
    
    return mini_coco_data, sampled_image_ids


def copy_images(src_img_dir, dst_img_dir, image_ids, image_id_to_filename):
    """复制采样的图片到目标目录"""
    os.makedirs(dst_img_dir, exist_ok=True)
    
    copied_count = 0
    for image_id in image_ids:
        if image_id in image_id_to_filename:
            src_path = os.path.join(src_img_dir, image_id_to_filename[image_id])
            dst_path = os.path.join(dst_img_dir, image_id_to_filename[image_id])
            
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
                copied_count += 1
            else:
                print(f"Warning: Image {src_path} not found")
    
    print(f"Copied {copied_count} images to {dst_img_dir}")


def create_mini_dataset(args):
    """创建小规模数据集"""
    print(f"Creating mini dataset from {args.src_dir} to {args.dst_dir}")
    print(f"Train samples: {args.train_samples}, Val samples: {args.val_samples}")
    
    # 创建目标目录结构
    os.makedirs(args.dst_dir, exist_ok=True)
    os.makedirs(os.path.join(args.dst_dir, 'train2017'), exist_ok=True)
    os.makedirs(os.path.join(args.dst_dir, 'val2017'), exist_ok=True)
    os.makedirs(os.path.join(args.dst_dir, 'annotations'), exist_ok=True)
    
    # 处理训练集
    print("\nProcessing training set...")
    train_ann_file = os.path.join(args.src_dir, 'annotations', 'instances_train2017.json')
    if os.path.exists(train_ann_file):
        train_coco_data = load_coco_annotations(train_ann_file)
        mini_train_data, train_image_ids = sample_images_with_annotations(
            train_coco_data, args.train_samples, args.seed)
        
        # 保存训练集标注
        mini_train_ann_file = os.path.join(args.dst_dir, 'annotations', 'instances_train2017.json')
        with open(mini_train_ann_file, 'w') as f:
            json.dump(mini_train_data, f)
        
        # 创建图片ID到文件名的映射
        train_id_to_filename = {img['id']: img['file_name'] for img in train_coco_data['images']}
        
        # 复制训练图片
        copy_images(
            os.path.join(args.src_dir, 'train2017'),
            os.path.join(args.dst_dir, 'train2017'),
            train_image_ids,
            train_id_to_filename
        )
        
        print(f"Training set: {len(mini_train_data['images'])} images, "
              f"{len(mini_train_data['annotations'])} annotations")
    else:
        print(f"Training annotation file {train_ann_file} not found")
    
    # 处理验证集
    print("\nProcessing validation set...")
    val_ann_file = os.path.join(args.src_dir, 'annotations', 'instances_val2017.json')
    if os.path.exists(val_ann_file):
        val_coco_data = load_coco_annotations(val_ann_file)
        mini_val_data, val_image_ids = sample_images_with_annotations(
            val_coco_data, args.val_samples, args.seed + 1)  # 不同的随机种子
        
        # 保存验证集标注
        mini_val_ann_file = os.path.join(args.dst_dir, 'annotations', 'instances_val2017.json')
        with open(mini_val_ann_file, 'w') as f:
            json.dump(mini_val_data, f)
        
        # 创建图片ID到文件名的映射
        val_id_to_filename = {img['id']: img['file_name'] for img in val_coco_data['images']}
        
        # 复制验证图片
        copy_images(
            os.path.join(args.src_dir, 'val2017'),
            os.path.join(args.dst_dir, 'val2017'),
            val_image_ids,
            val_id_to_filename
        )
        
        print(f"Validation set: {len(mini_val_data['images'])} images, "
              f"{len(mini_val_data['annotations'])} annotations")
    else:
        print(f"Validation annotation file {val_ann_file} not found")
    
    print(f"\nMini dataset created successfully at {args.dst_dir}")
    
    # 打印数据集统计信息
    print("\nDataset Statistics:")
    print(f"├── train2017/: {len(os.listdir(os.path.join(args.dst_dir, 'train2017')))} images")
    print(f"├── val2017/: {len(os.listdir(os.path.join(args.dst_dir, 'val2017')))} images")
    print(f"└── annotations/")
    print(f"    ├── instances_train2017.json")
    print(f"    └── instances_val2017.json")


if __name__ == '__main__':
    args = parse_args()
    create_mini_dataset(args)
