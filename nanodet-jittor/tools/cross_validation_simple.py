#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
简化的交叉验证工具
对比Jittor模型与预期的PyTorch输出
"""

import os
import sys
import torch
import jittor as jt
import numpy as np

# 添加路径
sys.path.append('/home/kyc/project/nanodet/nanodet-jittor')
from nanodet.model.arch.nanodet_plus import NanoDetPlus


def create_test_input():
    """创建固定的测试输入"""
    np.random.seed(42)
    torch.manual_seed(42)
    jt.set_global_seed(42)
    
    # 使用固定的测试数据
    if os.path.exists("fixed_input_data.npy"):
        input_data = np.load("fixed_input_data.npy")
    else:
        input_data = np.random.randn(1, 3, 320, 320).astype(np.float32)
        np.save("fixed_input_data.npy", input_data)
    
    return input_data


def create_jittor_model():
    """创建Jittor模型"""
    print("🔍 创建Jittor模型...")
    
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
    
    model = NanoDetPlus(backbone_cfg, fpn_cfg, aux_head_cfg, head_cfg)
    
    # 加载权重
    print("加载PyTorch权重...")
    checkpoint_path = "/home/kyc/project/nanodet/nanodet-pytorch/workspace/nanodet-plus-m_320_voc_bs64/model_best/model_best.ckpt"
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)
    
    jittor_state_dict = {}
    for name, param in model.named_parameters():
        jittor_state_dict[name] = param
    
    # 权重加载
    for pytorch_name, pytorch_param in state_dict.items():
        jittor_name = pytorch_name
        if jittor_name.startswith("model."):
            jittor_name = jittor_name[6:]
        
        if "num_batches_tracked" in jittor_name or jittor_name.startswith("avg_"):
            continue
        
        if "distribution_project.project" in jittor_name:
            continue
        
        if jittor_name in jittor_state_dict:
            jittor_param = jittor_state_dict[jittor_name]
            
            if list(pytorch_param.shape) == list(jittor_param.shape):
                jittor_param.assign(jt.array(pytorch_param.detach().numpy()))
            elif "scale" in jittor_name and len(pytorch_param.shape) == 0 and list(jittor_param.shape) == [1]:
                jittor_param.assign(jt.array([pytorch_param.detach().numpy()]))
    
    model.eval()
    return model


def run_jittor_inference():
    """运行Jittor推理"""
    print("🔍 运行Jittor推理...")
    
    input_data = create_test_input()
    jittor_input = jt.array(input_data)
    
    jittor_model = create_jittor_model()
    
    with jt.no_grad():
        jittor_output = jittor_model(jittor_input)
        
        # 分析输出
        cls_preds = jittor_output[:, :, :20]
        reg_preds = jittor_output[:, :, 20:]
        cls_scores = jt.sigmoid(cls_preds)
        
        print(f"  Jittor输出形状: {jittor_output.shape}")
        print(f"  Jittor输出范围: [{jittor_output.min():.6f}, {jittor_output.max():.6f}]")
        print(f"  Jittor分类预测: 范围[{cls_preds.min():.6f}, {cls_preds.max():.6f}]")
        print(f"  Jittor回归预测: 范围[{reg_preds.min():.6f}, {reg_preds.max():.6f}]")
        print(f"  Jittor分类置信度: 范围[{cls_scores.min():.6f}, {cls_scores.max():.6f}]")
        print(f"  Jittor最高置信度: {cls_scores.max():.6f}")
        
        jittor_results = {
            'output': jittor_output.numpy(),
            'cls_scores': cls_scores.numpy(),
            'max_confidence': float(cls_scores.max().numpy())
        }
        
        # 保存结果
        np.save("jittor_inference_results.npy", jittor_results)
        print(f"  ✅ Jittor结果已保存")
        
        return jittor_results


def create_pytorch_inference_script():
    """创建PyTorch推理脚本"""
    print("🔍 创建PyTorch推理脚本...")
    
    pytorch_script = '''#!/usr/bin/env python3
import os
import sys
import torch
import numpy as np

# 添加路径
sys.path.append('/home/kyc/project/nanodet/nanodet-pytorch')
from nanodet.model.arch.nanodet_plus import NanoDetPlus

def main():
    print("🔍 运行PyTorch推理...")
    
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 加载输入数据
    input_data = np.load("/home/kyc/project/nanodet/nanodet-jittor/fixed_input_data.npy")
    pytorch_input = torch.from_numpy(input_data)
    
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
        
        print(f"  PyTorch输出形状: {output.shape}")
        print(f"  PyTorch输出范围: [{output.min():.6f}, {output.max():.6f}]")
        print(f"  PyTorch分类预测: 范围[{cls_preds.min():.6f}, {cls_preds.max():.6f}]")
        print(f"  PyTorch回归预测: 范围[{reg_preds.min():.6f}, {reg_preds.max():.6f}]")
        print(f"  PyTorch分类置信度: 范围[{cls_scores.min():.6f}, {cls_scores.max():.6f}]")
        print(f"  PyTorch最高置信度: {cls_scores.max():.6f}")
        
        # 保存结果
        results = {
            'output': output.detach().numpy(),
            'cls_scores': cls_scores.detach().numpy(),
            'max_confidence': cls_scores.max().item()
        }
        
        np.save("/home/kyc/project/nanodet/nanodet-jittor/pytorch_inference_results.npy", results)
        print(f"  ✅ PyTorch结果已保存")

if __name__ == '__main__':
    main()
'''
    
    # 保存脚本到PyTorch目录
    script_path = "/home/kyc/project/nanodet/nanodet-pytorch/pytorch_inference.py"
    with open(script_path, "w") as f:
        f.write(pytorch_script)
    
    print(f"✅ PyTorch推理脚本已创建: {script_path}")
    return script_path


def main():
    """主函数"""
    print("🚀 开始交叉验证")
    
    # 1. 运行Jittor推理
    jittor_results = run_jittor_inference()
    
    # 2. 创建PyTorch推理脚本
    pytorch_script_path = create_pytorch_inference_script()
    
    print(f"\n💡 下一步操作:")
    print(f"   cd /home/kyc/project/nanodet/nanodet-pytorch")
    print(f"   python pytorch_inference.py")
    print(f"   cd /home/kyc/project/nanodet/nanodet-jittor")
    print(f"   python tools/compare_results.py")
    
    print(f"\n✅ 交叉验证准备完成")


if __name__ == '__main__':
    main()
