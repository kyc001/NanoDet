#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PyTorch vs Jittor 直接对比工具
在PyTorch环境中运行，直接对比两个版本的输出
"""

import os
import sys
import cv2
import torch
import numpy as np
import subprocess

def run_pytorch_inference():
    """运行PyTorch版本的推理"""
    print("🔍 运行PyTorch版本推理...")
    
    # 创建PyTorch推理脚本
    pytorch_script = """
import sys
import torch
import numpy as np
sys.path.append('/home/kyc/project/nanodet/nanodet-pytorch')

# 导入PyTorch版本
from nanodet.model.arch.nanodet_plus import NanoDetPlus

def create_pytorch_model():
    backbone_cfg = {
        'name': 'ShuffleNetV2',
        'model_size': '1.0x',
        'out_stages': [2, 3, 4],
        'activation': 'LeakyReLU',
        'pretrain': False
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
    checkpoint_path = "/home/kyc/project/nanodet/nanodet-pytorch/workspace/nanodet-plus-m_320_voc_bs64/model_best/model_best.ckpt"
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)
    
    # 移除'model.'前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            new_state_dict[k[6:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    
    return model

def test_pytorch_model():
    model = create_pytorch_model()
    
    # 创建相同的测试输入
    np.random.seed(42)
    input_data = np.random.randn(1, 3, 320, 320).astype(np.float32)
    torch_input = torch.tensor(input_data)
    
    print(f"PyTorch输入: {input_data.shape}, 范围[{input_data.min():.6f}, {input_data.max():.6f}]")
    
    with torch.no_grad():
        # 分析各组件输出
        print("1. PyTorch Backbone输出:")
        backbone_features = model.backbone(torch_input)
        for i, feat in enumerate(backbone_features):
            print(f"  层{i}: {feat.shape}, 范围[{feat.min():.6f}, {feat.max():.6f}]")
        
        print("2. PyTorch FPN输出:")
        fpn_features = model.fpn(backbone_features)
        for i, feat in enumerate(fpn_features):
            print(f"  层{i}: {feat.shape}, 范围[{feat.min():.6f}, {feat.max():.6f}]")
        
        print("3. PyTorch Head输出:")
        head_output = model.head(fpn_features)
        print(f"  Head输出: {head_output.shape}, 范围[{head_output.min():.6f}, {head_output.max():.6f}]")
        
        # 分析分类预测
        cls_preds = head_output[:, :, :20]
        cls_scores = torch.sigmoid(cls_preds)
        
        max_conf = float(cls_scores.max().item())
        mean_conf = float(cls_scores.mean().item())
        
        print(f"  分类预测: 范围[{cls_preds.min():.6f}, {cls_preds.max():.6f}]")
        print(f"  最高置信度: {max_conf:.6f}")
        print(f"  平均置信度: {mean_conf:.6f}")
        
        # 统计检测数量
        for threshold in [0.01, 0.05, 0.1]:
            max_scores = torch.max(cls_scores, dim=2)[0]
            valid_detections = int((max_scores > threshold).sum().item())
            print(f"  阈值{threshold}: {valid_detections}个检测")
        
        # 检查关键权重
        print("4. PyTorch关键权重:")
        for name, param in model.named_parameters():
            if 'head.gfl_cls.0.bias' in name:
                weight = param.detach().numpy()
                print(f"  {name}: {weight.shape}, 范围[{weight.min():.6f}, {weight.max():.6f}]")
                print(f"    前5个值: {weight[:5]}")

if __name__ == '__main__':
    test_pytorch_model()
"""
    
    # 保存PyTorch脚本
    with open('/tmp/pytorch_test.py', 'w') as f:
        f.write(pytorch_script)
    
    # 运行PyTorch脚本
    try:
        result = subprocess.run([
            '/home/kyc/miniconda3/envs/nano/bin/python', 
            '/tmp/pytorch_test.py'
        ], 
        capture_output=True, 
        text=True, 
        timeout=120,
        cwd='/home/kyc/project/nanodet/nanodet-pytorch'
        )
        
        if result.returncode == 0:
            print("✅ PyTorch推理成功")
            print(result.stdout)
            return result.stdout
        else:
            print("❌ PyTorch推理失败")
            print("STDERR:", result.stderr)
            return None
            
    except subprocess.TimeoutExpired:
        print("❌ PyTorch推理超时")
        return None
    except Exception as e:
        print(f"❌ PyTorch推理异常: {e}")
        return None


def run_jittor_inference():
    """运行Jittor版本的推理"""
    print("🔍 运行Jittor版本推理...")
    
    # 这里直接调用我们之前的代码
    import jittor as jt
    sys.path.append('/home/kyc/project/nanodet/nanodet-jittor')
    from nanodet.model.arch.nanodet_plus import NanoDetPlus
    
    # 创建Jittor模型
    backbone_cfg = {
        'name': 'ShuffleNetV2',
        'model_size': '1.0x',
        'out_stages': [2, 3, 4],
        'activation': 'LeakyReLU',
        'pretrain': False
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
    import torch
    checkpoint_path = "/home/kyc/project/nanodet/nanodet-pytorch/workspace/nanodet-plus-m_320_voc_bs64/model_best/model_best.ckpt"
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)
    
    jittor_state_dict = {}
    for name, param in model.named_parameters():
        jittor_state_dict[name] = param
    
    loaded_count = 0
    total_count = 0
    
    for pytorch_name, pytorch_param in state_dict.items():
        jittor_name = pytorch_name
        if jittor_name.startswith("model."):
            jittor_name = jittor_name[6:]
        
        if "num_batches_tracked" in jittor_name or jittor_name.startswith("avg_"):
            continue
        
        if "distribution_project.project" in jittor_name:
            continue
        
        total_count += 1
        
        if jittor_name in jittor_state_dict:
            jittor_param = jittor_state_dict[jittor_name]
            
            if list(pytorch_param.shape) == list(jittor_param.shape):
                jittor_param.assign(jt.array(pytorch_param.detach().numpy()))
                loaded_count += 1
            elif "scale" in jittor_name and len(pytorch_param.shape) == 0 and list(jittor_param.shape) == [1]:
                jittor_param.assign(jt.array([pytorch_param.detach().numpy()]))
                loaded_count += 1
    
    print(f"✅ Jittor权重加载: {loaded_count}/{total_count}")
    model.eval()
    
    # 创建相同的测试输入
    np.random.seed(42)
    input_data = np.random.randn(1, 3, 320, 320).astype(np.float32)
    jittor_input = jt.array(input_data)
    
    print(f"Jittor输入: {input_data.shape}, 范围[{input_data.min():.6f}, {input_data.max():.6f}]")
    
    with jt.no_grad():
        # 分析各组件输出
        print("1. Jittor Backbone输出:")
        backbone_features = model.backbone(jittor_input)
        for i, feat in enumerate(backbone_features):
            print(f"  层{i}: {feat.shape}, 范围[{feat.min():.6f}, {feat.max():.6f}]")
        
        print("2. Jittor FPN输出:")
        fpn_features = model.fpn(backbone_features)
        for i, feat in enumerate(fpn_features):
            print(f"  层{i}: {feat.shape}, 范围[{feat.min():.6f}, {feat.max():.6f}]")
        
        print("3. Jittor Head输出:")
        head_output = model.head(fpn_features)
        print(f"  Head输出: {head_output.shape}, 范围[{head_output.min():.6f}, {head_output.max():.6f}]")
        
        # 分析分类预测
        cls_preds = head_output[:, :, :20]
        cls_scores = jt.sigmoid(cls_preds)
        
        max_conf = float(cls_scores.max().numpy())
        mean_conf = float(cls_scores.mean().numpy())
        
        print(f"  分类预测: 范围[{cls_preds.min():.6f}, {cls_preds.max():.6f}]")
        print(f"  最高置信度: {max_conf:.6f}")
        print(f"  平均置信度: {mean_conf:.6f}")
        
        # 统计检测数量
        for threshold in [0.01, 0.05, 0.1]:
            max_scores = jt.max(cls_scores, dim=2)[0]
            valid_detections = int((max_scores > threshold).sum().numpy())
            print(f"  阈值{threshold}: {valid_detections}个检测")
        
        # 检查关键权重
        print("4. Jittor关键权重:")
        for name, param in model.named_parameters():
            if 'head.gfl_cls.0.bias' in name:
                weight = param.numpy()
                print(f"  {name}: {weight.shape}, 范围[{weight.min():.6f}, {weight.max():.6f}]")
                print(f"    前5个值: {weight[:5]}")
    
    return max_conf


def compare_results(pytorch_output, jittor_max_conf):
    """对比结果"""
    print("\n📊 PyTorch vs Jittor 对比结果:")
    print("=" * 80)
    
    if pytorch_output is None:
        print("❌ 无法获取PyTorch结果进行对比")
        return
    
    # 从PyTorch输出中提取关键信息
    lines = pytorch_output.split('\n')
    pytorch_max_conf = None
    
    for line in lines:
        if '最高置信度:' in line:
            try:
                pytorch_max_conf = float(line.split(':')[1].strip())
                break
            except:
                pass
    
    if pytorch_max_conf is not None:
        print(f"PyTorch最高置信度: {pytorch_max_conf:.6f}")
        print(f"Jittor最高置信度: {jittor_max_conf:.6f}")
        
        diff = abs(pytorch_max_conf - jittor_max_conf)
        relative_diff = diff / pytorch_max_conf * 100 if pytorch_max_conf > 0 else 0
        
        print(f"绝对差异: {diff:.6f}")
        print(f"相对差异: {relative_diff:.2f}%")
        
        if relative_diff < 1:
            print("✅ 结果高度一致")
        elif relative_diff < 5:
            print("⚠️ 结果基本一致")
        else:
            print("❌ 结果存在显著差异")
    else:
        print("⚠️ 无法从PyTorch输出中提取置信度信息")


def main():
    """主函数"""
    print("🚀 开始PyTorch vs Jittor 直接对比")
    print("=" * 80)
    
    try:
        # 1. 运行PyTorch推理
        pytorch_output = run_pytorch_inference()
        
        print("\n" + "="*80 + "\n")
        
        # 2. 运行Jittor推理
        jittor_max_conf = run_jittor_inference()
        
        # 3. 对比结果
        compare_results(pytorch_output, jittor_max_conf)
        
        print(f"\n✅ 对比完成")
        
    except Exception as e:
        print(f"❌ 对比失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
