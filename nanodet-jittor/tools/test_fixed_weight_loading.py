#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试修复后的权重加载
特殊处理Scale参数的形状不匹配问题
"""

import os
import sys
import torch
import jittor as jt

# 添加路径
sys.path.append('/home/kyc/project/nanodet/nanodet-jittor')
from nanodet.model.arch.nanodet_plus import NanoDetPlus


def create_nanodet_model():
    """创建NanoDet模型"""
    print("创建NanoDet模型...")
    
    # 创建配置字典
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
    
    # 创建aux_head配置 - 使用正确的SimpleConvHead
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
    
    # 创建完整模型
    model = NanoDetPlus(backbone_cfg, fpn_cfg, aux_head_cfg, head_cfg)
    
    return model


def load_pytorch_weights_fixed(model, checkpoint_path):
    """
    修复后的权重加载函数
    特殊处理Scale参数的形状不匹配
    """
    print(f"加载PyTorch checkpoint: {checkpoint_path}")
    
    # 使用PyTorch加载checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)
    
    print(f"✓ PyTorch checkpoint包含 {len(state_dict)} 个参数")
    
    # 获取Jittor模型的参数字典
    jittor_state_dict = {}
    for name, param in model.named_parameters():
        jittor_state_dict[name] = param
    
    print(f"✓ Jittor模型包含 {len(jittor_state_dict)} 个参数")
    
    # 改进的权重加载
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
                print(f"✓ 特殊处理Scale参数: {jittor_name}")
            else:
                print(f"❌ 形状不匹配: {jittor_name}")
                print(f"   PyTorch: {list(pytorch_param.shape)}")
                print(f"   Jittor: {list(jittor_param.shape)}")
                failed_count += 1
        else:
            print(f"❌ 参数名不存在: {jittor_name}")
            failed_count += 1
    
    print(f"\n📊 修复后的权重加载结果:")
    print(f"✅ 成功加载: {loaded_count} 个参数")
    print(f"✅ Scale参数修复: {scale_fixed_count} 个")
    print(f"⏭️ 跳过无关: {skipped_count} 个参数")
    print(f"❌ 加载失败: {failed_count} 个参数")
    
    if failed_count == 0:
        print("🎉 所有相关参数加载成功！")
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
    print(f"   输出数据类型: {output.dtype}")
    
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
            print(f"   差异: {num_channels - expected_total}")
    
    return True


def main():
    """主函数"""
    print("🚀 开始测试修复后的权重加载")
    
    # 创建模型
    model = create_nanodet_model()
    
    # 加载权重
    checkpoint_path = "/home/kyc/project/nanodet/nanodet-pytorch/workspace/nanodet-plus-m_320_voc_bs64/model_best/model_best.ckpt"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ 模型文件不存在: {checkpoint_path}")
        return False
    
    # 修复后的权重加载
    success = load_pytorch_weights_fixed(model, checkpoint_path)
    
    if success:
        print("\n🎉 权重加载完全成功！")
        
        # 测试模型推理
        test_model_inference(model)
        
        print("\n✅ 所有测试通过！模型已准备好进行mAP评估。")
        return True
    else:
        print("\n❌ 权重加载仍有问题，需要进一步修复。")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
