#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FPN逐步分析工具
深入检查FPN内部每一个操作的输出
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


def analyze_fpn_step_by_step():
    """逐步分析FPN的每一个操作"""
    print(f"🔍 FPN逐步分析")
    print("=" * 60)
    
    # 创建模型
    model = create_jittor_model()
    
    # 创建测试输入
    input_data = create_test_input()
    jittor_input = jt.array(input_data)
    
    with jt.no_grad():
        # 获取backbone特征
        backbone_features = model.backbone(jittor_input)
        print(f"Backbone特征:")
        for i, feat in enumerate(backbone_features):
            print(f"  特征{i}: {feat.shape}, 范围[{feat.min():.6f}, {feat.max():.6f}]")
        
        # 手动执行FPN的每一步
        fpn = model.fpn
        
        print(f"\n🔍 FPN步骤1: 1x1卷积降维")
        # 1. 1x1 conv to reduce channels
        reduced_inputs = []
        for i, (input_x, reduce) in enumerate(zip(backbone_features, fpn.reduce_layers)):
            reduced = reduce(input_x)
            reduced_inputs.append(reduced)
            print(f"  reduce{i}输出: {reduced.shape}, 范围[{reduced.min():.6f}, {reduced.max():.6f}]")
        
        print(f"\n🔍 FPN步骤2: Top-down路径")
        # 2. top-down path
        inner_outs = [reduced_inputs[-1]]
        print(f"  初始inner_out: {inner_outs[0].shape}, 范围[{inner_outs[0].min():.6f}, {inner_outs[0].max():.6f}]")
        
        for idx in range(len(fpn.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = reduced_inputs[idx - 1]
            
            print(f"\n  Top-down步骤 {len(fpn.in_channels) - idx}:")
            print(f"    feat_high: {feat_high.shape}, 范围[{feat_high.min():.6f}, {feat_high.max():.6f}]")
            print(f"    feat_low: {feat_low.shape}, 范围[{feat_low.min():.6f}, {feat_low.max():.6f}]")
            
            inner_outs[0] = feat_high
            
            # upsample
            upsample_feat = fpn.upsample(feat_high)
            print(f"    upsample后: {upsample_feat.shape}, 范围[{upsample_feat.min():.6f}, {upsample_feat.max():.6f}]")
            
            # concat
            concat_feat = jt.concat([upsample_feat, feat_low], dim=1)
            print(f"    concat后: {concat_feat.shape}, 范围[{concat_feat.min():.6f}, {concat_feat.max():.6f}]")
            
            # top_down_block处理
            block_idx = len(fpn.in_channels) - 1 - idx
            inner_out = fpn.top_down_blocks[block_idx](concat_feat)
            print(f"    top_down_block{block_idx}后: {inner_out.shape}, 范围[{inner_out.min():.6f}, {inner_out.max():.6f}]")
            
            inner_outs.insert(0, inner_out)
        
        print(f"\n  Top-down完成后的inner_outs:")
        for i, out in enumerate(inner_outs):
            print(f"    inner_out{i}: {out.shape}, 范围[{out.min():.6f}, {out.max():.6f}]")
        
        print(f"\n🔍 FPN步骤3: Bottom-up路径")
        # 3. bottom-up path
        outs = [inner_outs[0]]
        print(f"  初始out: {outs[0].shape}, 范围[{outs[0].min():.6f}, {outs[0].max():.6f}]")
        
        for idx in range(len(fpn.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            
            print(f"\n  Bottom-up步骤 {idx + 1}:")
            print(f"    feat_low: {feat_low.shape}, 范围[{feat_low.min():.6f}, {feat_low.max():.6f}]")
            print(f"    feat_high: {feat_high.shape}, 范围[{feat_high.min():.6f}, {feat_high.max():.6f}]")
            
            # downsample
            downsample_feat = fpn.downsamples[idx](feat_low)
            print(f"    downsample后: {downsample_feat.shape}, 范围[{downsample_feat.min():.6f}, {downsample_feat.max():.6f}]")
            
            # concat
            concat_feat = jt.concat([downsample_feat, feat_high], dim=1)
            print(f"    concat后: {concat_feat.shape}, 范围[{concat_feat.min():.6f}, {concat_feat.max():.6f}]")
            
            # bottom_up_block处理
            out = fpn.bottom_up_blocks[idx](concat_feat)
            print(f"    bottom_up_block{idx}后: {out.shape}, 范围[{out.min():.6f}, {out.max():.6f}]")
            
            outs.append(out)
        
        print(f"\n🔍 FPN步骤4: Extra layers")
        # 4. extra layers
        for i, (extra_in_layer, extra_out_layer) in enumerate(zip(fpn.extra_lvl_in_conv, fpn.extra_lvl_out_conv)):
            extra_in = extra_in_layer(reduced_inputs[-1])
            extra_out = extra_out_layer(outs[-1])
            extra_final = extra_in + extra_out
            
            print(f"  extra_in{i}: {extra_in.shape}, 范围[{extra_in.min():.6f}, {extra_in.max():.6f}]")
            print(f"  extra_out{i}: {extra_out.shape}, 范围[{extra_out.min():.6f}, {extra_out.max():.6f}]")
            print(f"  extra_final{i}: {extra_final.shape}, 范围[{extra_final.min():.6f}, {extra_final.max():.6f}]")
            
            outs.append(extra_final)
        
        print(f"\n🔍 FPN最终输出:")
        for i, out in enumerate(outs):
            print(f"  FPN输出{i}: {out.shape}, 范围[{out.min():.6f}, {out.max():.6f}]")
        
        # 对比完整FPN输出
        print(f"\n🔍 对比完整FPN输出:")
        fpn_full_output = fpn(backbone_features)
        for i, (manual_out, full_out) in enumerate(zip(outs, fpn_full_output)):
            diff = jt.abs(manual_out - full_out).max()
            print(f"  输出{i}差异: {diff:.10f}")
            
            if diff < 1e-6:
                print(f"    ✅ 手动计算与完整FPN一致")
            else:
                print(f"    ❌ 手动计算与完整FPN不一致")


def main():
    """主函数"""
    print("🚀 开始FPN逐步分析")
    
    # FPN逐步分析
    analyze_fpn_step_by_step()
    
    print(f"\n✅ FPN逐步分析完成")


if __name__ == '__main__':
    main()
