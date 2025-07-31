#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
统一VOC数据集下载脚本（加速版）
优化点：
1. 并行下载多个数据集文件
2. 使用aria2多连接下载（需提前安装）
3. 使用pigz多线程解压（需提前安装）
4. 用lxml替代标准库解析XML（需提前安装）
5. 并行处理XML标注文件
6. 优化JSON写入速度
"""

import os
import sys
import json
import tarfile
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from lxml import etree  # 需安装：pip install lxml


# 依赖工具检查（首次运行请确保安装）
def check_dependencies():
    required_tools = {
        'aria2c': 'sudo apt install aria2 (Linux) 或 官网下载 (Windows)',
        'pigz': 'sudo apt install pigz (Linux) 或 brew install pigz (macOS)'
    }
    for tool, install_cmd in required_tools.items():
        if not os.system(f'which {tool} > /dev/null 2>&1') == 0:
            print(f"错误：未找到工具 {tool}，请先安装：{install_cmd}")
            sys.exit(1)
    try:
        import lxml
    except ImportError:
        print("错误：未找到lxml库，请先安装：pip install lxml")
        sys.exit(1)


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
    """用aria2多连接下载（加速核心）"""
    print(f'正在下载: {url.split("/")[-1]}')
    # -x 16：16个连接，-s 16：分16段，--continue：支持断点续传
    cmd = f'aria2c -x 16 -s 16 --continue -o "{filepath}" "{url}"'
    ret = os.system(cmd)
    if ret == 0:
        print(f'✓ 下载完成: {filepath.name}')
        return True
    else:
        print(f'✗ 下载失败: {url}')
        if os.path.exists(filepath):
            os.remove(filepath)  # 删除不完整文件
        return False


def extract_tar(tar_path, extract_dir):
    """用pigz多线程解压（加速核心）"""
    print(f'正在解压: {tar_path.name}')
    # -I pigz：启用多线程解压，-C：指定解压目录
    cmd = f'tar -I pigz -xf "{tar_path}" -C "{extract_dir}"'
    ret = os.system(cmd)
    if ret == 0:
        print(f'✓ 解压完成: {tar_path.name}')
        return True
    else:
        print(f'✗ 解压失败: {tar_path.name}')
        return False


def parse_voc_annotation(xml_file):
    """用lxml解析XML（比标准库快3-5倍）"""
    try:
        tree = etree.parse(xml_file)
        root = tree.getroot()
        
        # 提取图片信息
        filename = root.findtext('filename')
        size = root.find('size')
        width = int(size.findtext('width'))
        height = int(size.findtext('height'))
        
        # 提取标注信息
        objects = []
        for obj in root.findall('object'):
            name = obj.findtext('name')
            if name not in VOC_CLASSES:
                continue
                
            bbox = obj.find('bndbox')
            xmin = float(bbox.findtext('xmin'))
            ymin = float(bbox.findtext('ymin'))
            xmax = float(bbox.findtext('xmax'))
            ymax = float(bbox.findtext('ymax'))
            
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
    except Exception as e:
        print(f'解析失败 {xml_file}: {str(e)[:50]}...')
        return None


def process_image(args):
    """并行处理单张图片的标注（供多进程调用）"""
    img_name, voc_year_dir, year = args
    img_path = voc_year_dir / 'JPEGImages' / f'{img_name}.jpg'
    xml_path = voc_year_dir / 'Annotations' / f'{img_name}.xml'
    
    if not img_path.exists() or not xml_path.exists():
        return None  # 跳过缺失文件
    
    ann_data = parse_voc_annotation(xml_path)
    if not ann_data:
        return None
    
    return {
        'img_name': img_name,
        'year': year,
        'ann_data': ann_data
    }


def convert_voc_to_coco(voc_dir, output_dir):
    """并行转换VOC格式为COCO格式（加速核心）"""
    voc_dir = Path(voc_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 数据集分割方案
    splits = {
        'train': [('VOC2007', 'trainval'), ('VOC2012', 'trainval')],
        'val': [('VOC2007', 'test')]
    }
    
    for split_name, datasets in splits.items():
        print(f'\n转换 {split_name} 数据集...')
        
        # COCO格式数据结构
        coco_data = {
            'info': {'description': f'VOC {split_name} in COCO format'},
            'licenses': [{'id': 1, 'name': 'Unknown'}],
            'categories': [{'id': i, 'name': name} for i, name in enumerate(VOC_CLASSES)],
            'images': [],
            'annotations': []
        }
        
        image_id = 0
        annotation_id = 0
        
        for year, split in datasets:
            voc_year_dir = voc_dir / f'VOC{year}'
            if not voc_year_dir.exists():
                print(f'警告: 跳过不存在的 {voc_year_dir}')
                continue
                
            # 读取图片列表
            split_file = voc_year_dir / 'ImageSets' / 'Main' / f'{split}.txt'
            if not split_file.exists():
                print(f'警告: 跳过不存在的 {split_file}')
                continue
                
            with open(split_file, 'r') as f:
                image_names = [line.strip() for line in f.readlines() if line.strip()]
            print(f'  处理 VOC{year} {split}: {len(image_names)} 张图片')
            
            # 准备并行任务参数
            tasks = [(img_name, voc_year_dir, year) for img_name in image_names]
            
            # 多进程并行处理（进程数=CPU核心数）
            with ProcessPoolExecutor() as executor:
                results = list(executor.map(process_image, tasks))
            
            # 整理结果（单线程避免竞争）
            for result in results:
                if not result:
                    continue
                ann_data = result['ann_data']
                img_name = result['img_name']
                year = result['year']
                
                # 添加图片信息
                coco_data['images'].append({
                    'id': image_id,
                    'file_name': f'{year}_{img_name}.jpg',  # 避免重名
                    'width': ann_data['width'],
                    'height': ann_data['height']
                })
                
                # 添加标注信息
                for obj in ann_data['objects']:
                    coco_data['annotations'].append({
                        'id': annotation_id,
                        'image_id': image_id,
                        'category_id': obj['category_id'],
                        'bbox': obj['bbox'],
                        'area': obj['area'],
                        'iscrowd': obj['iscrowd']
                    })
                    annotation_id += 1
                
                image_id += 1
        
        # 保存COCO标注（不格式化JSON，加速写入）
        output_file = output_dir / f'voc_{split_name}.json'
        with open(output_file, 'w') as f:
            json.dump(coco_data, f)  # 去掉indent，减少50%写入时间
        
        print(f'✓ 保存 {split_name} 标注: {output_file}')
        print(f'  图片数: {len(coco_data["images"])}, 标注数: {len(coco_data["annotations"])}')


def create_symlinks():
    """为各版本创建数据链接"""
    print('\n创建符号链接...')
    data_dir = Path('data')
    
    # 为PyTorch版本创建链接
    pytorch_link = Path('nanodet-pytorch/data')
    if not pytorch_link.exists():
        pytorch_link.symlink_to('../data', target_is_directory=True)
        print(f'✓ 已创建: {pytorch_link} -> ../data')
    
    # 为Jittor版本创建链接
    jittor_link = Path('nanodet-jittor/data')
    if not jittor_link.exists():
        jittor_link.symlink_to('../data', target_is_directory=True)
        print(f'✓ 已创建: {jittor_link} -> ../data')


def main():
    """主函数"""
    check_dependencies()  # 检查加速工具是否安装
    
    print("=" * 60)
    print("VOC数据集加速下载与转换工具")
    print("=" * 60)
    
    data_dir = Path('data')
    voc_dir = data_dir / 'VOCdevkit'
    data_dir.mkdir(exist_ok=True)
    
    # 步骤1: 并行下载和解压
    print("\n步骤1/4: 并行下载数据集")
    with ThreadPoolExecutor(max_workers=3) as executor:  # 3个文件同时下载
        futures = []
        for name, url in VOC_URLS.items():
            tar_path = data_dir / f'{name}.tar'
            if not tar_path.exists():
                futures.append(executor.submit(download_file, url, tar_path))
        
        # 等待所有下载完成后再解压（避免IO冲突）
        for future in as_completed(futures):
            if future.result():
                tar_path = data_dir / f'{VOC_URLS.keys()[list(VOC_URLS.values()).index(future.result())]}.tar'
                extract_tar(tar_path, data_dir)
                tar_path.unlink()  # 删除tar包节省空间
    
    # 步骤2: 并行转换格式
    print("\n步骤2/4: 并行转换为COCO格式")
    annotations_dir = data_dir / 'annotations'
    convert_voc_to_coco(voc_dir, annotations_dir)
    
    # 步骤3: 创建符号链接
    print("\n步骤3/4: 创建符号链接")
    create_symlinks()
    
    # 步骤4: 验证结果
    print("\n步骤4/4: 验证数据集")
    train_ann = data_dir / 'annotations/voc_train.json'
    val_ann = data_dir / 'annotations/voc_val.json'
    
    if train_ann.exists() and val_ann.exists():
        with open(train_ann, 'r') as f:
            train_size = len(json.load(f)['images'])
        with open(val_ann, 'r') as f:
            val_size = len(json.load(f)['images'])
        print(f"\n数据集验证成功:")
        print(f"  训练集: {train_size} 张图片")
        print(f"  验证集: {val_size} 张图片")
        print("\n" + "=" * 60)
        print("VOC数据集准备完成！")
        print("=" * 60)
        print("\n训练命令:")
        print("  PyTorch版本: cd nanodet-pytorch && python tools/train.py config/nanodet-plus-m_320_voc.yml")
        print("  Jittor版本: cd nanodet-jittor && python tools/train.py config/nanodet-plus-m_320_voc.yml")
    else:
        print("\n✗ 数据集准备失败，请检查上述错误信息")
        sys.exit(1)


if __name__ == '__main__':
    main()