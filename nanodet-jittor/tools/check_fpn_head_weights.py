#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
检查FPN和Head的权重加载情况
找出权重加载的具体问题
"""

import os
import sys
import torch
import jittor as jt
import numpy as np

# 添加路径
sys.path.append('/home/kyc/project/nanodet/nanodet-jittor')
from nanodet.model.arch.nanodet_plus import NanoDetPlus


def check_weight_loading_details():
    """检查权重加载的详细情况"""
    print("🔍 检查FPN和Head权重加载详细情况")
    print("=" * 60)
    
    # 创建模型
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
    
    print("1️⃣ 创建Jittor模型...")
    model = NanoDetPlus(backbone_cfg, fpn_cfg, aux_head_cfg, head_cfg)
    
    # 加载PyTorch权重
    print("\n2️⃣ 加载PyTorch权重...")
    checkpoint_path = "/home/kyc/project/nanodet/nanodet-pytorch/workspace/nanodet-plus-m_320_voc_bs64/model_best/model_best.ckpt"
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)
    
    print(f"PyTorch checkpoint包含 {len(state_dict)} 个参数")
    
    # 分析PyTorch权重
    backbone_params = {}
    fpn_params = {}
    aux_fpn_params = {}
    head_params = {}
    aux_head_params = {}
    other_params = {}
    
    for name, param in state_dict.items():
        clean_name = name.replace('model.', '') if name.startswith('model.') else name
        
        if clean_name.startswith('backbone.'):
            backbone_params[clean_name] = param
        elif clean_name.startswith('fpn.'):
            fpn_params[clean_name] = param
        elif clean_name.startswith('aux_fpn.'):
            aux_fpn_params[clean_name] = param
        elif clean_name.startswith('head.'):
            head_params[clean_name] = param
        elif clean_name.startswith('aux_head.'):
            aux_head_params[clean_name] = param
        else:
            other_params[clean_name] = param
    
    print(f"\nPyTorch权重分布:")
    print(f"  backbone: {len(backbone_params)} 个参数")
    print(f"  fpn: {len(fpn_params)} 个参数")
    print(f"  aux_fpn: {len(aux_fpn_params)} 个参数")
    print(f"  head: {len(head_params)} 个参数")
    print(f"  aux_head: {len(aux_head_params)} 个参数")
    print(f"  other: {len(other_params)} 个参数")
    
    # 获取Jittor模型的参数
    print("\n3️⃣ 分析Jittor模型参数...")
    jittor_backbone_params = {}
    jittor_fpn_params = {}
    jittor_aux_fpn_params = {}
    jittor_head_params = {}
    jittor_aux_head_params = {}
    
    for name, param in model.named_parameters():
        if name.startswith('backbone.'):
            jittor_backbone_params[name] = param
        elif name.startswith('fpn.'):
            jittor_fpn_params[name] = param
        elif name.startswith('aux_fpn.'):
            jittor_aux_fpn_params[name] = param
        elif name.startswith('head.'):
            jittor_head_params[name] = param
        elif name.startswith('aux_head.'):
            jittor_aux_head_params[name] = param
    
    print(f"Jittor模型参数分布:")
    print(f"  backbone: {len(jittor_backbone_params)} 个参数")
    print(f"  fpn: {len(jittor_fpn_params)} 个参数")
    print(f"  aux_fpn: {len(jittor_aux_fpn_params)} 个参数")
    print(f"  head: {len(jittor_head_params)} 个参数")
    print(f"  aux_head: {len(jittor_aux_head_params)} 个参数")
    
    # 详细检查权重加载
    print("\n4️⃣ 详细检查权重加载...")
    
    def check_module_weights(pytorch_params, jittor_params, module_name):
        print(f"\n🔍 检查{module_name}权重加载:")
        
        loaded_count = 0
        failed_count = 0
        missing_count = 0
        
        # 检查PyTorch -> Jittor的映射
        for pytorch_name, pytorch_param in pytorch_params.items():
            if pytorch_name in jittor_params:
                jittor_param = jittor_params[pytorch_name]
                
                if list(pytorch_param.shape) == list(jittor_param.shape):
                    loaded_count += 1
                    # 检查权重数值
                    pytorch_np = pytorch_param.detach().numpy()
                    jittor_np = jittor_param.numpy()
                    diff = np.abs(pytorch_np - jittor_np).max()
                    
                    if diff > 1e-6:
                        print(f"    ⚠️ {pytorch_name}: 权重数值不一致 (差异{diff:.8f})")
                else:
                    print(f"    ❌ {pytorch_name}: 形状不匹配 PyTorch{pytorch_param.shape} vs Jittor{jittor_param.shape}")
                    failed_count += 1
            else:
                print(f"    ❌ {pytorch_name}: 在Jittor中不存在")
                missing_count += 1
        
        # 检查Jittor中多余的参数
        extra_count = 0
        for jittor_name in jittor_params.keys():
            if jittor_name not in pytorch_params:
                print(f"    ⚠️ {jittor_name}: Jittor中多余的参数")
                extra_count += 1
        
        print(f"  📊 {module_name}权重加载统计:")
        print(f"    成功加载: {loaded_count}")
        print(f"    加载失败: {failed_count}")
        print(f"    缺失参数: {missing_count}")
        print(f"    多余参数: {extra_count}")
        
        return loaded_count, failed_count, missing_count, extra_count
    
    # 检查各个模块
    backbone_stats = check_module_weights(backbone_params, jittor_backbone_params, "Backbone")
    fpn_stats = check_module_weights(fpn_params, jittor_fpn_params, "FPN")
    aux_fpn_stats = check_module_weights(aux_fpn_params, jittor_aux_fpn_params, "Aux_FPN")
    head_stats = check_module_weights(head_params, jittor_head_params, "Head")
    aux_head_stats = check_module_weights(aux_head_params, jittor_aux_head_params, "Aux_Head")
    
    # 总结
    print(f"\n📊 权重加载总结:")
    total_loaded = sum(stats[0] for stats in [backbone_stats, fpn_stats, aux_fpn_stats, head_stats, aux_head_stats])
    total_failed = sum(stats[1] for stats in [backbone_stats, fpn_stats, aux_fpn_stats, head_stats, aux_head_stats])
    total_missing = sum(stats[2] for stats in [backbone_stats, fpn_stats, aux_fpn_stats, head_stats, aux_head_stats])
    total_extra = sum(stats[3] for stats in [backbone_stats, fpn_stats, aux_fpn_stats, head_stats, aux_head_stats])
    
    print(f"  总成功加载: {total_loaded}")
    print(f"  总加载失败: {total_failed}")
    print(f"  总缺失参数: {total_missing}")
    print(f"  总多余参数: {total_extra}")
    
    if total_failed > 0 or total_missing > 0:
        print(f"  ❌ 权重加载存在问题")
        return False
    else:
        print(f"  ✅ 权重加载完全正确")
        return True


def test_segmented_model():
    """测试分段模型"""
    print("\n5️⃣ 测试分段模型...")
    
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    jt.set_global_seed(42)
    
    # 创建模型
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
    
    # 加载权重（简化版本）
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
    
    # 使用固定输入
    if os.path.exists("fixed_input_data.npy"):
        input_data = np.load("fixed_input_data.npy")
    else:
        input_data = np.random.randn(1, 3, 320, 320).astype(np.float32)
    
    jittor_input = jt.array(input_data)
    
    print(f"\n🔍 分段测试:")
    
    with jt.no_grad():
        # 1. Backbone输出
        backbone_features = model.backbone(jittor_input)
        print(f"  Backbone输出:")
        for i, feat in enumerate(backbone_features):
            print(f"    特征{i}: {feat.shape}, 范围[{feat.min():.6f}, {feat.max():.6f}]")
        
        # 2. FPN输出
        fpn_features = model.fpn(backbone_features)
        print(f"  FPN输出:")
        for i, feat in enumerate(fpn_features):
            print(f"    FPN特征{i}: {feat.shape}, 范围[{feat.min():.6f}, {feat.max():.6f}]")
        
        # 3. Head输出
        head_output = model.head(fpn_features)
        print(f"  Head输出:")
        print(f"    Head输出: {head_output.shape}, 范围[{head_output.min():.6f}, {head_output.max():.6f}]")
        
        # 分析Head输出
        cls_preds = head_output[:, :, :20]
        reg_preds = head_output[:, :, 20:]
        cls_scores = jt.sigmoid(cls_preds)
        
        print(f"    分类预测: 范围[{cls_preds.min():.6f}, {cls_preds.max():.6f}]")
        print(f"    回归预测: 范围[{reg_preds.min():.6f}, {reg_preds.max():.6f}]")
        print(f"    最高置信度: {cls_scores.max():.6f}")


def main():
    """主函数"""
    print("🚀 开始检查FPN和Head权重加载")
    
    # 检查权重加载详情
    weight_ok = check_weight_loading_details()
    
    # 测试分段模型
    test_segmented_model()
    
    print(f"\n✅ 检查完成")
    
    if weight_ok:
        print(f"权重加载正确，问题可能在模型实现细节")
    else:
        print(f"发现权重加载问题，需要修复")


if __name__ == '__main__':
    main()
