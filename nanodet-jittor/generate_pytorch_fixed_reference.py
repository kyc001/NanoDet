#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PyTorch固定输入参考输出生成脚本
"""

import os
import sys
import torch
import numpy as np

# 添加PyTorch版本路径
sys.path.insert(0, '/home/kyc/project/nanodet/nanodet-pytorch')

from nanodet.model.arch import build_model
from nanodet.util import cfg, load_config


def main():
    """生成PyTorch固定输入参考输出"""
    print("🚀 生成PyTorch固定输入参考输出")
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
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
    
    # 加载固定输入
    if os.path.exists("fixed_input_data.npy"):
        input_data = np.load("fixed_input_data.npy")
        print("✓ 使用Jittor保存的固定输入")
    else:
        # 创建相同的固定输入
        input_data = np.random.randn(1, 3, 320, 320).astype(np.float32)
        print("✓ 创建新的固定输入")
    
    print(f"输入形状: {input_data.shape}")
    print(f"输入范围: [{input_data.min():.6f}, {input_data.max():.6f}]")
    
    # 推理
    input_tensor = torch.from_numpy(input_data)
    with torch.no_grad():
        output = model(input_tensor)
    
    # 保存输出
    output_np = output.detach().numpy()
    np.save("pytorch_fixed_output.npy", output_np)
    
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
