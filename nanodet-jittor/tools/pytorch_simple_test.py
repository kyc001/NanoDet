#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
简单的PyTorch测试
获取PyTorch版本的参考输出
"""

import os
import sys
import torch
import numpy as np

# 添加PyTorch项目路径
sys.path.append('/home/kyc/project/nanodet/nanodet-pytorch')

try:
    from nanodet.model.arch.nanodet_plus import NanoDetPlus
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("❌ PyTorch版本不可用")


def main():
    """主函数"""
    if not PYTORCH_AVAILABLE:
        print("请在PyTorch环境中运行此脚本")
        return
    
    print("🔍 PyTorch简单测试")
    
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 加载输入数据
    input_data = np.load("/home/kyc/project/nanodet/nanodet-jittor/fixed_input_data.npy")
    pytorch_input = torch.from_numpy(input_data)
    
    print(f"输入数据: {input_data.shape}, 范围[{input_data.min():.6f}, {input_data.max():.6f}]")
    
    # 创建模型配置
    backbone_cfg = {
        'name': 'ShuffleNetV2',
        'model_size': '1.0x',
        'out_stages': [2, 3, 4],
        'activation': 'LeakyReLU',
        'pretrain': True
    }
    
    fpn_cfg = {
        'name': 'GhostPAN',
        'in_channels': [116, 232, 464],
        'out_channels': 96,
        'kernel_size': 5,
        'num_extra_level': 1,
        'use_depthwise': True,
        'activation': 'LeakyReLU'
    }
    
    head_cfg = {
        'name': 'NanoDetPlusHead',
        'num_classes': 20,
        'input_channel': 96,
        'feat_channels': 96,
        'stacked_convs': 2,
        'kernel_size': 5,
        'strides': [8, 16, 32, 64],
        'activation': 'LeakyReLU',
        'reg_max': 7,
        'norm_cfg': {'type': 'BN'},
        'loss': {
            'loss_qfl': {
                'name': 'QualityFocalLoss',
                'use_sigmoid': True,
                'beta': 2.0,
                'loss_weight': 1.0
            },
            'loss_dfl': {
                'name': 'DistributionFocalLoss',
                'loss_weight': 0.25
            },
            'loss_bbox': {
                'name': 'GIoULoss',
                'loss_weight': 2.0
            }
        }
    }
    
    aux_head_cfg = {
        'name': 'SimpleConvHead',
        'num_classes': 20,
        'input_channel': 192,
        'feat_channels': 192,
        'stacked_convs': 4,
        'strides': [8, 16, 32, 64],
        'activation': 'LeakyReLU',
        'reg_max': 7
    }
    
    # 创建模型
    model = NanoDetPlus(backbone_cfg, fpn_cfg, aux_head_cfg, head_cfg)
    
    # 加载权重
    checkpoint_path = "/home/kyc/project/nanodet/nanodet-pytorch/workspace/nanodet-plus-m_320_voc_bs64/model_best/model_best.ckpt"
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    # 推理
    with torch.no_grad():
        output = model(pytorch_input)
        
        # 分析输出
        cls_preds = output[:, :, :20]
        reg_preds = output[:, :, 20:]
        cls_scores = torch.sigmoid(cls_preds)
        
        print(f"PyTorch模型输出:")
        print(f"  输出形状: {output.shape}")
        print(f"  输出范围: [{output.min():.6f}, {output.max():.6f}]")
        print(f"  分类预测: 范围[{cls_preds.min():.6f}, {cls_preds.max():.6f}]")
        print(f"  回归预测: 范围[{reg_preds.min():.6f}, {reg_preds.max():.6f}]")
        print(f"  分类置信度: 范围[{cls_scores.min():.6f}, {cls_scores.max():.6f}]")
        print(f"  最高置信度: {cls_scores.max():.6f}")
        
        # 置信度统计
        print(f"  置信度统计:")
        print(f"    均值: {cls_scores.mean():.6f}")
        print(f"    >0.1的比例: {(cls_scores > 0.1).float().mean():.4f}")
        print(f"    >0.5的比例: {(cls_scores > 0.5).float().mean():.4f}")
        
        # 保存结果
        results = {
            'output': output.detach().numpy(),
            'cls_scores': cls_scores.detach().numpy(),
            'max_confidence': cls_scores.max().item()
        }
        
        np.save("/home/kyc/project/nanodet/nanodet-jittor/pytorch_simple_results.npy", results)
        print(f"  ✅ PyTorch结果已保存")


if __name__ == '__main__':
    main()
