#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将VOC数据集转换为COCO格式
用于PyTorch评估脚本
"""

import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime

# VOC类别映射到COCO格式
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# 创建类别ID映射
CLASS_TO_ID = {cls_name: i + 1 for i, cls_name in enumerate(VOC_CLASSES)}


def parse_voc_annotation(xml_path):
    """解析VOC XML标注文件"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # 获取图像信息
    filename = root.find('filename').text
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    # 获取所有目标
    objects = []
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        if class_name not in VOC_CLASSES:
            continue
        
        # 获取边界框
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        
        # 转换为COCO格式 [x, y, width, height]
        coco_bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
        area = (xmax - xmin) * (ymax - ymin)
        
        objects.append({
            'class_name': class_name,
            'class_id': CLASS_TO_ID[class_name],
            'bbox': coco_bbox,
            'area': area
        })
    
    return {
        'filename': filename,
        'width': width,
        'height': height,
        'objects': objects
    }


def convert_voc_to_coco(voc_root, split='val', output_file=None):
    """将VOC数据集转换为COCO格式"""
    print(f"🔍 转换VOC {split} 数据集为COCO格式...")
    
    # 读取图像列表
    split_file = os.path.join(voc_root, f"ImageSets/Main/{split}.txt")
    with open(split_file, 'r') as f:
        image_ids = [line.strip() for line in f.readlines()]
    
    print(f"找到 {len(image_ids)} 张图像")
    
    # 初始化COCO格式数据
    coco_data = {
        "info": {
            "description": f"VOC2007 {split} dataset in COCO format",
            "version": "1.0",
            "year": 2007,
            "contributor": "PASCAL VOC",
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "licenses": [
            {
                "id": 1,
                "name": "Unknown",
                "url": ""
            }
        ],
        "categories": [],
        "images": [],
        "annotations": []
    }
    
    # 添加类别信息
    for i, class_name in enumerate(VOC_CLASSES):
        coco_data["categories"].append({
            "id": i + 1,
            "name": class_name,
            "supercategory": "object"
        })
    
    annotation_id = 1
    
    # 处理每张图像
    for image_id, img_id in enumerate(image_ids, 1):
        # 检查图像文件是否存在
        image_path = os.path.join(voc_root, f"JPEGImages/{img_id}.jpg")
        if not os.path.exists(image_path):
            print(f"⚠️ 图像不存在: {image_path}")
            continue
        
        # 检查标注文件是否存在
        annotation_path = os.path.join(voc_root, f"Annotations/{img_id}.xml")
        if not os.path.exists(annotation_path):
            print(f"⚠️ 标注不存在: {annotation_path}")
            continue
        
        try:
            # 解析标注
            annotation_data = parse_voc_annotation(annotation_path)
            
            # 添加图像信息
            coco_data["images"].append({
                "id": image_id,
                "file_name": annotation_data['filename'],
                "width": annotation_data['width'],
                "height": annotation_data['height'],
                "license": 1,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": ""
            })
            
            # 添加标注信息
            for obj in annotation_data['objects']:
                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": obj['class_id'],
                    "bbox": obj['bbox'],
                    "area": obj['area'],
                    "iscrowd": 0,
                    "segmentation": []
                })
                annotation_id += 1
        
        except Exception as e:
            print(f"❌ 处理 {img_id} 失败: {e}")
            continue
        
        if (image_id) % 500 == 0:
            print(f"  已处理: {image_id}/{len(image_ids)}")
    
    print(f"✅ 转换完成:")
    print(f"  图像数: {len(coco_data['images'])}")
    print(f"  标注数: {len(coco_data['annotations'])}")
    print(f"  类别数: {len(coco_data['categories'])}")
    
    # 保存COCO格式文件
    if output_file is None:
        output_file = f"data/annotations/voc_{split}.json"
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"✅ COCO格式文件已保存: {output_file}")
    
    return output_file


def create_data_symlinks():
    """创建数据集软链接"""
    print("🔍 创建数据集软链接...")
    
    # 创建data目录
    os.makedirs("data", exist_ok=True)
    
    # 创建VOCdevkit软链接
    voc_source = "/home/kyc/project/nanodet/data/VOCdevkit"
    voc_target = "data/VOCdevkit"
    
    if not os.path.exists(voc_target):
        os.symlink(voc_source, voc_target)
        print(f"✅ 创建软链接: {voc_target} -> {voc_source}")
    else:
        print(f"✅ 软链接已存在: {voc_target}")


def main():
    """主函数"""
    print("🚀 开始VOC到COCO格式转换")
    print("=" * 60)
    
    try:
        # 1. 创建数据集软链接
        create_data_symlinks()
        
        # 2. 设置路径
        voc_root = "/home/kyc/project/nanodet/data/VOCdevkit/VOC2007"
        
        if not os.path.exists(voc_root):
            print(f"❌ VOC数据集不存在: {voc_root}")
            return
        
        # 3. 转换训练集
        print(f"\n转换训练集...")
        train_file = convert_voc_to_coco(voc_root, 'train', 'data/annotations/voc_train.json')
        
        # 4. 转换验证集
        print(f"\n转换验证集...")
        val_file = convert_voc_to_coco(voc_root, 'val', 'data/annotations/voc_val.json')
        
        # 5. 转换测试集（如果存在）
        test_split_file = os.path.join(voc_root, "ImageSets/Main/test.txt")
        if os.path.exists(test_split_file):
            print(f"\n转换测试集...")
            test_file = convert_voc_to_coco(voc_root, 'test', 'data/annotations/voc_test.json')
        else:
            print(f"\n⚠️ 测试集不存在，使用验证集作为测试集")
            # 复制验证集文件作为测试集
            import shutil
            shutil.copy(val_file, 'data/annotations/voc_test.json')
            print(f"✅ 复制验证集作为测试集: data/annotations/voc_test.json")
        
        print(f"\n🎯 转换完成！")
        print(f"现在可以使用PyTorch评估脚本进行测试了")
        
        # 6. 验证文件
        print(f"\n验证生成的文件:")
        for file_path in ['data/annotations/voc_train.json', 'data/annotations/voc_val.json', 'data/annotations/voc_test.json']:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path) / 1024 / 1024  # MB
                print(f"  ✅ {file_path} ({file_size:.1f} MB)")
            else:
                print(f"  ❌ {file_path} 不存在")
        
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
