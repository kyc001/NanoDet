#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VOC数据集下载和准备脚本
自动下载VOC2007和VOC2012数据集，并转换为训练所需的格式
"""

import argparse
import os
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
import json
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description='Download and prepare VOC dataset')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directory to save VOC dataset')
    parser.add_argument('--download', action='store_true',
                        help='Download VOC dataset from official source')
    parser.add_argument('--convert', action='store_true', default=True,
                        help='Convert VOC annotations to COCO format')
    parser.add_argument('--verify', action='store_true', default=True,
                        help='Verify dataset integrity')
    return parser.parse_args()


# VOC数据集下载链接
VOC_URLS = {
    'VOC2007_trainval': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
    'VOC2007_test': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar',
    'VOC2012_trainval': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
}

# VOC类别名称
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
    'bus', 'car', 'cat', 'chair', 'cow', 
    'diningtable', 'dog', 'horse', 'motorbike', 'person', 
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]


def download_file(url, filepath):
    """下载文件并显示进度"""
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            print(f'\r下载进度: {percent:.1f}% ({downloaded}/{total_size} bytes)', end='')
        else:
            print(f'\r已下载: {downloaded} bytes', end='')
    
    print(f'正在下载: {url}')
    urllib.request.urlretrieve(url, filepath, progress_hook)
    print()  # 换行


def extract_tar(tar_path, extract_dir):
    """解压tar文件"""
    print(f'正在解压: {tar_path}')
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(extract_dir)
    print(f'解压完成: {extract_dir}')


def download_voc_dataset(data_dir):
    """下载VOC数据集"""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    for name, url in VOC_URLS.items():
        tar_path = data_dir / f'{name}.tar'
        
        # 检查是否已下载
        if tar_path.exists():
            print(f'{name} 已存在，跳过下载')
            continue
            
        # 下载
        try:
            download_file(url, tar_path)
        except Exception as e:
            print(f'下载 {name} 失败: {e}')
            continue
            
        # 解压
        try:
            extract_tar(tar_path, data_dir)
            # 删除tar文件以节省空间
            tar_path.unlink()
        except Exception as e:
            print(f'解压 {name} 失败: {e}')


def parse_voc_annotation(xml_file):
    """解析VOC XML标注文件"""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # 获取图片信息
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
        'width': width,
        'height': height,
        'objects': objects
    }


def convert_voc_to_coco(voc_dir, output_dir):
    """将VOC格式转换为COCO格式"""
    voc_dir = Path(voc_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 数据集分割
    splits = {
        'train': [('VOC2007', 'trainval'), ('VOC2012', 'trainval')],
        'val': [('VOC2007', 'test')]
    }
    
    for split_name, datasets in splits.items():
        print(f'转换 {split_name} 数据集...')
        
        # COCO格式数据结构
        coco_data = {
            'info': {
                'description': f'VOC {split_name} dataset in COCO format',
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
        
        image_id = 0
        annotation_id = 0
        
        for year, split in datasets:
            voc_year_dir = voc_dir / f'VOC{year}'
            if not voc_year_dir.exists():
                print(f'警告: {voc_year_dir} 不存在，跳过')
                continue
                
            # 读取图片列表
            split_file = voc_year_dir / 'ImageSets' / 'Main' / f'{split}.txt'
            if not split_file.exists():
                print(f'警告: {split_file} 不存在，跳过')
                continue
                
            with open(split_file, 'r') as f:
                image_names = [line.strip() for line in f.readlines()]
            
            print(f'  处理 VOC{year} {split}: {len(image_names)} 张图片')
            
            for img_name in image_names:
                # 图片路径
                img_path = voc_year_dir / 'JPEGImages' / f'{img_name}.jpg'
                xml_path = voc_year_dir / 'Annotations' / f'{img_name}.xml'
                
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
                    'file_name': f'{year}_{img_name}.jpg',  # 添加年份前缀避免重名
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
        
        # 保存COCO格式标注文件
        output_file = output_dir / f'voc_{split_name}.json'
        with open(output_file, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        print(f'  保存到: {output_file}')
        print(f'  图片数量: {len(coco_data["images"])}')
        print(f'  标注数量: {len(coco_data["annotations"])}')


def verify_dataset(data_dir):
    """验证数据集完整性"""
    data_dir = Path(data_dir)
    
    print('验证数据集完整性...')
    
    # 检查VOC目录结构
    required_dirs = [
        'VOCdevkit/VOC2007',
        'VOCdevkit/VOC2012'
    ]
    
    for dir_path in required_dirs:
        full_path = data_dir / dir_path
        if full_path.exists():
            print(f'✓ {dir_path} 存在')
            
            # 检查子目录
            subdirs = ['JPEGImages', 'Annotations', 'ImageSets/Main']
            for subdir in subdirs:
                subdir_path = full_path / subdir
                if subdir_path.exists():
                    print(f'  ✓ {subdir} 存在')
                else:
                    print(f'  ✗ {subdir} 不存在')
        else:
            print(f'✗ {dir_path} 不存在')
    
    # 检查转换后的标注文件
    annotations_dir = data_dir / 'annotations'
    if annotations_dir.exists():
        print(f'✓ annotations 目录存在')
        for split in ['train', 'val']:
            ann_file = annotations_dir / f'voc_{split}.json'
            if ann_file.exists():
                print(f'  ✓ voc_{split}.json 存在')
                # 读取并验证JSON格式
                try:
                    with open(ann_file, 'r') as f:
                        data = json.load(f)
                    print(f'    - 图片数量: {len(data["images"])}')
                    print(f'    - 标注数量: {len(data["annotations"])}')
                    print(f'    - 类别数量: {len(data["categories"])}')
                except Exception as e:
                    print(f'  ✗ voc_{split}.json 格式错误: {e}')
            else:
                print(f'  ✗ voc_{split}.json 不存在')


def main():
    args = parse_args()
    
    print('=' * 60)
    print('VOC数据集准备工具')
    print('=' * 60)
    
    data_dir = Path(args.data_dir)
    voc_dir = data_dir / 'VOCdevkit'
    
    if args.download:
        print('\n步骤1: 下载VOC数据集')
        download_voc_dataset(data_dir)
    
    if args.convert:
        print('\n步骤2: 转换为COCO格式')
        annotations_dir = data_dir / 'annotations'
        convert_voc_to_coco(voc_dir, annotations_dir)
    
    if args.verify:
        print('\n步骤3: 验证数据集')
        verify_dataset(data_dir)
    
    print('\n' + '=' * 60)
    print('VOC数据集准备完成！')
    print('=' * 60)
    print('\n使用方法:')
    print('1. 训练模型:')
    print('   python tools/train.py config/nanodet-plus-m_320_voc.yml')
    print('\n2. 测试模型:')
    print('   python tools/test.py config/nanodet-plus-m_320_voc.yml --checkpoint model.pkl')


if __name__ == '__main__':
    main()
