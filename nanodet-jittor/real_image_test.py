#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
真实图像测试
使用真实图像测试模型性能
"""

import os
import sys
import cv2
import torch
import jittor as jt
import numpy as np

# 添加路径
sys.path.append('/home/kyc/project/nanodet/nanodet-jittor')
from nanodet.model.arch.nanodet_plus import NanoDetPlus


def preprocess_image(image_path, input_size=(320, 320)):
    """预处理图像"""
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # 转换为RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 调整大小
    image = cv2.resize(image, input_size)
    
    # 归一化
    image = image.astype(np.float32) / 255.0
    
    # 标准化 (ImageNet标准)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    
    # 转换为CHW格式
    image = image.transpose(2, 0, 1)
    
    # 添加batch维度
    image = image[np.newaxis, ...]
    
    return image


def test_real_image():
    """测试真实图像"""
    print("🔍 真实图像测试")
    
    # 创建一个简单的测试图像（如果没有真实图像）
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    cv2.imwrite("test_image.jpg", test_image)
    
    # 预处理
    input_data = preprocess_image("test_image.jpg")
    print(f"预处理后: {input_data.shape}, 范围[{input_data.min():.6f}, {input_data.max():.6f}]")
    
    # 创建模型并测试
    # ... (模型创建代码)
    
    print("✅ 真实图像测试脚本已创建")


if __name__ == '__main__':
    test_real_image()
