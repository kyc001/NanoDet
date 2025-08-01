#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PyTorch参考输出生成脚本
"""

import os
import sys
import cv2
import torch
import numpy as np

# 添加PyTorch版本路径
sys.path.insert(0, '/home/kyc/project/nanodet/nanodet-pytorch')

from nanodet.model.arch import build_model
from nanodet.util import cfg, load_config


def main():
    """生成PyTorch参考输出"""
    print("🚀 生成PyTorch参考输出")
    
    # 加载配置
    config_path = "/home/kyc/project/nanodet/nanodet-pytorch/config/nanodet-plus-m_320_voc.yml"
    load_config(cfg, config_path)
    
    # 创建模型
    model = build_model(cfg.model)
    
    # 加载权重
    checkpoint_path = "/home/kyc/project/nanodet/nanodet-pytorch/workspace/nanodet-plus-m_320_voc_bs64/model_best/model_best.ckpt"
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    # 移除前缀
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('model.', '') if key.startswith('model.') else key
        new_state_dict[new_key] = value
    
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    
    print("✓ PyTorch模型加载成功")
    
    # 加载测试图像
    if os.path.exists("test_image.npy"):
        test_img = np.load("test_image.npy")
        print("✓ 使用Jittor保存的测试图像")
    else:
        test_img_path = "data/VOCdevkit/VOC2007/JPEGImages/000001.jpg"
        if os.path.exists(test_img_path):
            test_img = cv2.imread(test_img_path)
            test_img = cv2.resize(test_img, (320, 320))
        else:
            test_img = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
        print("✓ 使用新的测试图像")
    
    # 预处理
    img_tensor = torch.from_numpy(test_img.transpose(2, 0, 1)).unsqueeze(0).float()
    mean = torch.tensor([123.675, 116.28, 103.53]).reshape(1, 3, 1, 1)
    std = torch.tensor([58.395, 57.12, 57.375]).reshape(1, 3, 1, 1)
    img_normalized = (img_tensor - mean) / std
    
    # 推理
    with torch.no_grad():
        output = model(img_normalized)
    
    # 保存输出
    output_np = output.detach().numpy()
    np.save("pytorch_reference_output.npy", output_np)
    
    print(f"✓ PyTorch输出已保存: {output.shape}")
    print(f"   输出范围: [{output.min():.6f}, {output.max():.6f}]")
    
    # 分析输出
    cls_preds = output[:, :, :20]
    reg_preds = output[:, :, 20:]
    cls_scores = torch.sigmoid(cls_preds)
    
    print(f"   分类预测范围: [{cls_preds.min():.6f}, {cls_preds.max():.6f}]")
    print(f"   回归预测范围: [{reg_preds.min():.6f}, {reg_preds.max():.6f}]")
    print(f"   最高置信度: {cls_scores.max():.6f}")


if __name__ == '__main__':
    main()
