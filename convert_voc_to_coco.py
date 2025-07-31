#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将VOC格式转换为COCO格式
为PyTorch训练准备数据
"""

import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
import argparse


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
            
        # 检查是否是困难样本
        difficult = obj.find('difficult')
        is_difficult = int(difficult.text) if difficult is not None else 0
        
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
            'iscrowd': 0,
            'difficult': is_difficult
        })
    
    return {
        'filename': filename,
        'width': width,
        'height': height,
        'objects': objects
    }


def convert_voc_to_coco(voc_dir, output_dir, split='trainval'):
    """将VOC格式转换为COCO格式"""
    voc_dir = Path(voc_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f'转换 VOC {split} 数据集到COCO格式...')
    
    # COCO格式数据结构
    coco_data = {
        'info': {
            'description': f'VOC2007 {split} dataset in COCO format',
            'version': '1.0',
            'year': 2007,
            'contributor': 'PASCAL VOC',
            'date_created': '2007-01-01'
        },
        'licenses': [{'id': 1, 'name': 'Unknown', 'url': ''}],
        'categories': [
            {'id': i, 'name': name, 'supercategory': 'object'} 
            for i, name in enumerate(VOC_CLASSES)
        ],
        'images': [],
        'annotations': []
    }
    
    # 读取图片列表
    split_file = voc_dir / 'ImageSets' / 'Main' / f'{split}.txt'
    if not split_file.exists():
        print(f'错误: {split_file} 不存在')
        return None
        
    with open(split_file, 'r') as f:
        image_names = [line.strip() for line in f.readlines()]
    
    print(f'处理 {len(image_names)} 张图片...')
    
    image_id = 0
    annotation_id = 0
    
    for img_name in image_names:
        # 图片路径
        img_path = voc_dir / 'JPEGImages' / f'{img_name}.jpg'
        xml_path = voc_dir / 'Annotations' / f'{img_name}.xml'
        
        if not img_path.exists() or not xml_path.exists():
            print(f'警告: {img_name} 的图片或标注文件不存在，跳过')
            continue
        
        # 解析标注
        try:
            ann_data = parse_voc_annotation(xml_path)
        except Exception as e:
            print(f'警告: 解析 {xml_path} 失败: {e}')
            continue
        
        # 添加图片信息
        image_info = {
            'id': image_id,
            'file_name': f'{img_name}.jpg',
            'width': ann_data['width'],
            'height': ann_data['height']
        }
        coco_data['images'].append(image_info)
        
        # 添加标注信息
        for obj in ann_data['objects']:
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
        
        image_id += 1
        
        if image_id % 1000 == 0:
            print(f'已处理 {image_id} 张图片...')
    
    # 保存COCO格式标注文件
    output_file = output_dir / f'voc_{split}.json'
    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f'转换完成!')
    print(f'保存到: {output_file}')
    print(f'图片数量: {len(coco_data["images"])}')
    print(f'标注数量: {len(coco_data["annotations"])}')
    
    # 统计类别分布
    class_count = defaultdict(int)
    for ann in coco_data['annotations']:
        cat_id = ann['category_id']
        cat_name = VOC_CLASSES[cat_id]
        class_count[cat_name] += 1
    
    print(f'类别分布:')
    for class_name, count in sorted(class_count.items()):
        print(f'  {class_name}: {count}')
    
    return output_file


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Convert VOC to COCO format')
    parser.add_argument('--voc_dir', default='VOCdevkit/VOC2007', help='VOC dataset directory')
    parser.add_argument('--output_dir', default='annotations', help='Output directory for COCO annotations')
    parser.add_argument('--splits', nargs='+', default=['trainval', 'test'], help='Dataset splits to convert')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("VOC to COCO Format Conversion")
    print("=" * 60)
    
    voc_dir = Path(args.voc_dir)
    if not voc_dir.exists():
        print(f'错误: VOC目录不存在: {voc_dir}')
        return
    
    # 转换指定的数据集分割
    for split in args.splits:
        print(f'\n转换 {split} 数据集...')
        output_file = convert_voc_to_coco(voc_dir, args.output_dir, split)
        if output_file is None:
            print(f'转换 {split} 失败')
            continue
    
    print(f'\n所有转换完成!')
    print(f'输出目录: {args.output_dir}')
    
    # 为训练创建train/val分割
    if 'trainval' in args.splits and 'test' in args.splits:
        print(f'\n建议的训练配置:')
        print(f'训练集: voc_trainval.json ({Path(args.output_dir) / "voc_trainval.json"})')
        print(f'验证集: voc_test.json ({Path(args.output_dir) / "voc_test.json"})')


if __name__ == '__main__':
    main()
