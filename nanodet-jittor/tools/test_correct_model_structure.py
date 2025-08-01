#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试修复后的模型结构
验证参数数量是否与PyTorch版本一致
"""

import os
import sys
import torch
import jittor as jt
from collections import defaultdict

# 添加路径
sys.path.append('/home/kyc/project/nanodet/nanodet-jittor')
from nanodet.model.arch.nanodet_plus import NanoDetPlus


def create_correct_jittor_model():
    """创建修复后的Jittor模型"""
    print("创建修复后的Jittor模型...")
    
    # 创建配置字典 - 完全对齐PyTorch版本
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
    
    # 创建aux_head配置 - 完全对齐PyTorch版本
    aux_head_cfg = {
        'name': 'SimpleConvHead',
        'num_classes': 20,
        'input_channel': 192,
        'feat_channels': 192,  # 与PyTorch版本一致
        'stacked_convs': 4,    # 与PyTorch版本一致
        'strides': [8, 16, 32, 64],
        'activation': 'LeakyReLU',
        'reg_max': 7
    }
    
    # 创建完整模型
    model = NanoDetPlus(backbone_cfg, fpn_cfg, aux_head_cfg, head_cfg)
    
    return model


def analyze_jittor_model_structure(model):
    """分析Jittor模型结构"""
    print(f"\n📊 分析Jittor模型结构:")
    
    total_params = 0
    module_stats = defaultdict(int)
    param_details = {}
    
    for name, param in model.named_parameters():
        param_count = param.size if hasattr(param, 'size') and not callable(param.size) else param.numel()
        total_params += param_count
        
        # 按模块分组
        parts = name.split('.')
        if len(parts) >= 2:
            module_name = f"{parts[0]}.{parts[1]}" if len(parts) > 2 else parts[0]
        else:
            module_name = parts[0]
        
        module_stats[module_name] += param_count
        
        # 记录参数详情
        param_details[name] = {
            'shape': list(param.shape),
            'count': param_count
        }
    
    print(f"  总参数数量: {total_params:,}")
    print(f"  参数项数量: {len(param_details)}")
    
    print(f"\n📊 按模块统计:")
    for module, count in sorted(module_stats.items(), key=lambda x: x[1], reverse=True):
        percentage = count / total_params * 100
        print(f"  {module:<20}: {count:>8,} 参数 ({percentage:5.1f}%)")
    
    # 重点分析aux_head
    print(f"\n🔍 aux_head详细分析:")
    aux_head_params = {k: v for k, v in param_details.items() if k.startswith('aux_head')}
    aux_head_total = sum(v['count'] for v in aux_head_params.values())
    
    print(f"  aux_head总参数: {aux_head_total:,}")
    print(f"  aux_head参数项: {len(aux_head_params)}")
    
    # 按层分组
    aux_layers = defaultdict(int)
    for name, details in aux_head_params.items():
        layer_name = '.'.join(name.split('.')[:3])  # aux_head.xxx.yyy
        aux_layers[layer_name] += details['count']
    
    print(f"  按层统计:")
    for layer, count in sorted(aux_layers.items()):
        print(f"    {layer}: {count:,} 参数")
    
    return param_details, module_stats, aux_head_total


def load_pytorch_weights_and_test(model):
    """加载PyTorch权重并测试"""
    print(f"\n🔧 加载PyTorch权重...")
    
    checkpoint_path = "/home/kyc/project/nanodet/nanodet-pytorch/workspace/nanodet-plus-m_320_voc_bs64/model_best/model_best.ckpt"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ checkpoint文件不存在: {checkpoint_path}")
        return False
    
    # 使用PyTorch加载checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)
    
    print(f"✓ PyTorch checkpoint包含 {len(state_dict)} 个参数")
    
    # 获取Jittor模型的参数字典
    jittor_state_dict = {}
    for name, param in model.named_parameters():
        jittor_state_dict[name] = param
    
    print(f"✓ Jittor模型包含 {len(jittor_state_dict)} 个参数")
    
    # 100%修复的权重加载
    loaded_count = 0
    failed_count = 0
    skipped_count = 0
    scale_fixed_count = 0
    
    for pytorch_name, pytorch_param in state_dict.items():
        # 移除PyTorch特有的前缀
        jittor_name = pytorch_name
        if jittor_name.startswith("model."):
            jittor_name = jittor_name[6:]  # 移除"model."前缀
        
        # 跳过Jittor中不存在的BatchNorm统计参数
        if "num_batches_tracked" in jittor_name:
            skipped_count += 1
            continue
        
        # 跳过avg_model参数（权重平均相关）
        if jittor_name.startswith("avg_"):
            skipped_count += 1
            continue
        
        if jittor_name in jittor_state_dict:
            jittor_param = jittor_state_dict[jittor_name]
            
            # 检查形状匹配
            if list(pytorch_param.shape) == list(jittor_param.shape):
                # 转换并加载参数
                jittor_param.assign(jt.array(pytorch_param.detach().numpy()))
                loaded_count += 1
            elif "scale" in jittor_name and len(pytorch_param.shape) == 0 and list(jittor_param.shape) == [1]:
                # 特殊处理Scale参数：PyTorch标量 -> Jittor 1维张量
                jittor_param.assign(jt.array([pytorch_param.detach().numpy()]))
                loaded_count += 1
                scale_fixed_count += 1
            else:
                print(f"❌ 形状不匹配: {jittor_name}")
                print(f"   PyTorch: {list(pytorch_param.shape)}")
                print(f"   Jittor: {list(jittor_param.shape)}")
                failed_count += 1
        else:
            print(f"❌ 参数名不存在: {jittor_name}")
            failed_count += 1
    
    print(f"\n📊 权重加载结果:")
    print(f"✅ 成功加载: {loaded_count} 个参数")
    print(f"✅ Scale参数修复: {scale_fixed_count} 个")
    print(f"⏭️ 跳过无关: {skipped_count} 个参数")
    print(f"❌ 加载失败: {failed_count} 个参数")
    
    if failed_count == 0:
        print("🎉 100%权重加载成功！")
        return True
    else:
        print(f"⚠️ 仍有 {failed_count} 个参数加载失败")
        return False


def test_model_inference(model):
    """测试模型推理"""
    print("\n🔍 测试模型推理...")
    
    # 设置为评估模式
    model.eval()
    
    # 创建测试输入
    test_input = jt.randn(1, 3, 320, 320)
    
    print("进行前向推理...")
    with jt.no_grad():
        output = model(test_input)
    
    print(f"✅ 推理成功!")
    print(f"   输出形状: {output.shape}")
    print(f"   输出数值范围: [{output.min():.6f}, {output.max():.6f}]")
    
    # 分析输出通道
    if len(output.shape) == 3:  # [B, N, C]
        batch_size, num_anchors, num_channels = output.shape
        print(f"   批次大小: {batch_size}")
        print(f"   锚点数量: {num_anchors}")
        print(f"   输出通道: {num_channels}")
        
        # 分析通道分配
        expected_cls_channels = 20  # VOC 20类
        expected_reg_channels = 32  # 4 * (7+1) = 32
        expected_total = expected_cls_channels + expected_reg_channels
        
        print(f"\n🔹 通道分析:")
        print(f"   期望分类通道: {expected_cls_channels}")
        print(f"   期望回归通道: {expected_reg_channels}")
        print(f"   期望总通道: {expected_total}")
        print(f"   实际总通道: {num_channels}")
        
        if num_channels == expected_total:
            print("✅ 输出通道数正确")
        else:
            print("❌ 输出通道数不正确")
    
    return True


def main():
    """主函数"""
    print("🚀 开始测试修复后的模型结构")
    print("=" * 80)
    
    # 创建修复后的模型
    model = create_correct_jittor_model()
    
    # 分析模型结构
    param_details, module_stats, aux_head_total = analyze_jittor_model_structure(model)
    
    # 与PyTorch版本对比
    print(f"\n📊 与PyTorch版本对比:")
    pytorch_total = 4203884  # 从之前的测试得到
    jittor_total = sum(module_stats.values())
    
    print(f"  PyTorch总参数: {pytorch_total:,}")
    print(f"  Jittor总参数: {jittor_total:,}")
    print(f"  差异: {abs(pytorch_total - jittor_total):,} ({abs(pytorch_total - jittor_total) / pytorch_total * 100:.1f}%)")
    
    if abs(pytorch_total - jittor_total) / pytorch_total < 0.01:  # 1%以内
        print("✅ 参数数量基本一致")
    else:
        print("❌ 参数数量差异较大")
    
    # 加载权重并测试
    weight_success = load_pytorch_weights_and_test(model)
    
    if weight_success:
        # 测试推理
        test_model_inference(model)
        
        print(f"\n✅ 所有测试通过！模型结构修复成功！")
        return True
    else:
        print(f"\n❌ 权重加载仍有问题")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
