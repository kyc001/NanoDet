#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
统一的FPN验证工具
确保两次测试使用相同的输入和模型状态
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


def unified_fpn_verification():
    """统一的FPN验证"""
    print(f"🔍 统一的FPN验证")
    print("=" * 60)
    
    # 创建模型
    model = create_jittor_model()
    
    # 创建测试输入
    input_data = create_test_input()
    jittor_input = jt.array(input_data)
    
    print(f"输入数据: {input_data.shape}, 范围[{input_data.min():.6f}, {input_data.max():.6f}]")
    
    with jt.no_grad():
        # 获取backbone特征
        backbone_features = model.backbone(jittor_input)
        print(f"\nBackbone特征:")
        for i, feat in enumerate(backbone_features):
            print(f"  特征{i}: {feat.shape}, 范围[{feat.min():.6f}, {feat.max():.6f}]")
        
        # 方法1: 使用完整FPN
        print(f"\n方法1: 使用完整FPN")
        fpn_features_full = model.fpn(backbone_features)
        for i, feat in enumerate(fpn_features_full):
            print(f"  FPN特征{i}: {feat.shape}, 范围[{feat.min():.6f}, {feat.max():.6f}]")
        
        # 方法2: 手动执行FPN的每一步（简化版）
        print(f"\n方法2: 手动执行FPN（简化版）")
        fpn = model.fpn
        
        # 1x1 conv to reduce channels
        reduced_inputs = []
        for i, (input_x, reduce) in enumerate(zip(backbone_features, fpn.reduce_layers)):
            reduced = reduce(input_x)
            reduced_inputs.append(reduced)
        
        # top-down path
        inner_outs = [reduced_inputs[-1]]
        
        for idx in range(len(fpn.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = reduced_inputs[idx - 1]
            
            inner_outs[0] = feat_high
            
            # upsample
            upsample_feat = fpn.upsample(feat_high)
            
            # concat
            concat_feat = jt.concat([upsample_feat, feat_low], dim=1)
            
            # top_down_block处理
            block_idx = len(fpn.in_channels) - 1 - idx
            inner_out = fpn.top_down_blocks[block_idx](concat_feat)
            
            # 特别关注top_down_block1
            if block_idx == 1:
                print(f"    🔍 top_down_block1详细:")
                print(f"      输入concat: {concat_feat.shape}, 范围[{concat_feat.min():.6f}, {concat_feat.max():.6f}]")
                print(f"      输出: {inner_out.shape}, 范围[{inner_out.min():.6f}, {inner_out.max():.6f}]")
                
                # 手动执行top_down_block1
                top_down_block1 = fpn.top_down_blocks[1]
                manual_output = top_down_block1(concat_feat)
                diff = jt.abs(inner_out - manual_output).max()
                print(f"      手动vs自动差异: {diff:.10f}")
            
            inner_outs.insert(0, inner_out)
        
        # bottom-up path
        outs = [inner_outs[0]]
        
        for idx in range(len(fpn.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            
            # downsample
            downsample_feat = fpn.downsamples[idx](feat_low)
            
            # concat
            concat_feat = jt.concat([downsample_feat, feat_high], dim=1)
            
            # bottom_up_block处理
            out = fpn.bottom_up_blocks[idx](concat_feat)
            
            outs.append(out)
        
        # extra layers
        for i, (extra_in_layer, extra_out_layer) in enumerate(zip(fpn.extra_lvl_in_conv, fpn.extra_lvl_out_conv)):
            extra_in = extra_in_layer(reduced_inputs[-1])
            extra_out = extra_out_layer(outs[-1])
            extra_final = extra_in + extra_out
            
            outs.append(extra_final)
        
        print(f"\n手动FPN输出:")
        for i, out in enumerate(outs):
            print(f"  手动FPN特征{i}: {out.shape}, 范围[{out.min():.6f}, {out.max():.6f}]")
        
        # 对比两种方法
        print(f"\n方法对比:")
        for i, (full_out, manual_out) in enumerate(zip(fpn_features_full, outs)):
            diff = jt.abs(full_out - manual_out).max()
            print(f"  特征{i}差异: {diff:.10f}")
            
            if diff < 1e-6:
                print(f"    ✅ 两种方法一致")
            else:
                print(f"    ❌ 两种方法不一致")
                print(f"      完整FPN: 范围[{full_out.min():.6f}, {full_out.max():.6f}]")
                print(f"      手动FPN: 范围[{manual_out.min():.6f}, {manual_out.max():.6f}]")
        
        # 方法3: 直接测试top_down_block1
        print(f"\n方法3: 直接测试top_down_block1")
        
        # 重新获取top_down_block1的输入
        reduced_feat_high = reduced_inputs[-1]  # [1,96,10,10]
        reduced_feat_low = reduced_inputs[1]    # [1,96,20,20]
        
        # upsample
        upsample_feat = fpn.upsample(reduced_feat_high)
        
        # concat
        test_concat_feat = jt.concat([upsample_feat, reduced_feat_low], dim=1)
        
        # 直接调用top_down_block1
        top_down_block1 = fpn.top_down_blocks[1]
        direct_output = top_down_block1(test_concat_feat)
        
        print(f"  直接测试输入: {test_concat_feat.shape}, 范围[{test_concat_feat.min():.6f}, {test_concat_feat.max():.6f}]")
        print(f"  直接测试输出: {direct_output.shape}, 范围[{direct_output.min():.6f}, {direct_output.max():.6f}]")
        
        # 与之前的结果对比
        # 这里我们需要找到对应的输出进行对比
        # 在完整FPN中，top_down_block1的输出应该对应某个中间结果
        
        print(f"\n✅ 统一验证完成")


def main():
    """主函数"""
    print("🚀 开始统一FPN验证")
    
    # 统一FPN验证
    unified_fpn_verification()
    
    print(f"\n✅ 验证完成")


if __name__ == '__main__':
    main()
