#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PyTorch逐层调试脚本
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
    """PyTorch逐层调试"""
    print("🚀 PyTorch逐层调试")
    
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
        print("✓ 使用固定输入")
    else:
        input_data = np.random.randn(1, 3, 320, 320).astype(np.float32)
        print("✓ 创建新的固定输入")
    
    input_tensor = torch.from_numpy(input_data)
    
    print(f"\n🔍 逐层分析PyTorch模型...")
    
    # Backbone
    print(f"\n🔍 Backbone:")
    with torch.no_grad():
        backbone_features = model.backbone(input_tensor)
    
    for i, feat in enumerate(backbone_features):
        print(f"   特征{i}: {feat.shape}, 范围[{feat.min():.6f}, {feat.max():.6f}]")
        np.save(f"pytorch_backbone_feat_{i}.npy", feat.detach().numpy())
    
    # FPN
    print(f"\n🔍 FPN:")
    with torch.no_grad():
        fpn_features = model.fpn(backbone_features)
    
    for i, feat in enumerate(fpn_features):
        print(f"   FPN特征{i}: {feat.shape}, 范围[{feat.min():.6f}, {feat.max():.6f}]")
        np.save(f"pytorch_fpn_feat_{i}.npy", feat.detach().numpy())
    
    # Head
    print(f"\n🔍 Head:")
    with torch.no_grad():
        head_output = model.head(fpn_features)
    
    print(f"   Head输出: {head_output.shape}, 范围[{head_output.min():.6f}, {head_output.max():.6f}]")
    np.save("pytorch_head_output.npy", head_output.detach().numpy())
    
    # 完整模型
    print(f"\n🔍 完整模型:")
    with torch.no_grad():
        full_output = model(input_tensor)
    
    print(f"   完整输出: {full_output.shape}, 范围[{full_output.min():.6f}, {full_output.max():.6f}]")
    
    print(f"\n✅ PyTorch逐层调试完成")


if __name__ == '__main__':
    main()
