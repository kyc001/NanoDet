#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
深度调试GhostBlocks
找出top_down_block1内部的具体问题
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


def debug_ghost_blocks_internal():
    """深度调试GhostBlocks内部"""
    print(f"🔍 深度调试GhostBlocks内部")
    print("=" * 60)
    
    # 创建模型
    model = create_jittor_model()
    
    # 创建测试输入
    input_data = create_test_input()
    jittor_input = jt.array(input_data)
    
    with jt.no_grad():
        # 获取到top_down_block1的输入
        backbone_features = model.backbone(jittor_input)
        fpn = model.fpn
        
        # 1x1 conv to reduce channels
        reduced_inputs = []
        for i, (input_x, reduce) in enumerate(zip(backbone_features, fpn.reduce_layers)):
            reduced = reduce(input_x)
            reduced_inputs.append(reduced)
        
        # 获取到top_down_block1的输入
        feat_high = reduced_inputs[-1]  # [1,96,10,10]
        feat_low = reduced_inputs[1]    # [1,96,20,20]
        
        # upsample
        upsample_feat = fpn.upsample(feat_high)
        
        # concat - 这是top_down_block1的输入
        concat_feat = jt.concat([upsample_feat, feat_low], dim=1)
        print(f"top_down_block1输入: {concat_feat.shape}, 范围[{concat_feat.min():.6f}, {concat_feat.max():.6f}]")
        
        # 获取top_down_block1
        top_down_block1 = fpn.top_down_blocks[1]
        print(f"top_down_block1类型: {type(top_down_block1)}")
        
        # 检查top_down_block1的结构
        print(f"\ntop_down_block1结构:")
        for name, module in top_down_block1.named_modules():
            if name:  # 跳过根模块
                print(f"  {name}: {type(module)}")
        
        # 检查top_down_block1的参数
        print(f"\ntop_down_block1参数:")
        for name, param in top_down_block1.named_parameters():
            print(f"  {name}: {param.shape}, 范围[{param.min():.6f}, {param.max():.6f}]")
        
        # 手动执行top_down_block1的每一步
        print(f"\n🔍 手动执行top_down_block1:")
        
        # top_down_block1是GhostBlocks
        ghost_blocks = top_down_block1
        x = concat_feat
        
        print(f"  输入: {x.shape}, 范围[{x.min():.6f}, {x.max():.6f}]")
        
        # 检查是否有reduce_conv
        if hasattr(ghost_blocks, 'reduce_conv') and ghost_blocks.use_res:
            reduce_conv_out = ghost_blocks.reduce_conv(x)
            print(f"  reduce_conv输出: {reduce_conv_out.shape}, 范围[{reduce_conv_out.min():.6f}, {reduce_conv_out.max():.6f}]")
        
        # 执行blocks
        blocks_input = x
        print(f"  blocks输入: {blocks_input.shape}, 范围[{blocks_input.min():.6f}, {blocks_input.max():.6f}]")
        
        # 逐个执行GhostBottleneck
        current_x = blocks_input
        for i, block in enumerate(ghost_blocks.blocks):
            print(f"\n  🔍 GhostBottleneck {i}:")
            print(f"    输入: {current_x.shape}, 范围[{current_x.min():.6f}, {current_x.max():.6f}]")
            
            # 手动执行GhostBottleneck的每一步
            residual = current_x
            
            # 1st ghost bottleneck
            ghost1_out = block.ghost1(current_x)
            print(f"    ghost1输出: {ghost1_out.shape}, 范围[{ghost1_out.min():.6f}, {ghost1_out.max():.6f}]")
            
            # Depth-wise convolution (如果stride > 1)
            if block.stride > 1:
                dw_out = block.conv_dw(ghost1_out)
                dw_out = block.bn_dw(dw_out)
                print(f"    dw_conv输出: {dw_out.shape}, 范围[{dw_out.min():.6f}, {dw_out.max():.6f}]")
                ghost1_out = dw_out
            
            # Squeeze-and-excitation (如果有)
            if block.se is not None:
                se_out = block.se(ghost1_out)
                print(f"    se输出: {se_out.shape}, 范围[{se_out.min():.6f}, {se_out.max():.6f}]")
                ghost1_out = se_out
            
            # 2nd ghost bottleneck
            ghost2_out = block.ghost2(ghost1_out)
            print(f"    ghost2输出: {ghost2_out.shape}, 范围[{ghost2_out.min():.6f}, {ghost2_out.max():.6f}]")
            
            # shortcut
            shortcut_out = block.shortcut(residual)
            print(f"    shortcut输出: {shortcut_out.shape}, 范围[{shortcut_out.min():.6f}, {shortcut_out.max():.6f}]")
            
            # 相加
            block_out = ghost2_out + shortcut_out
            print(f"    block输出: {block_out.shape}, 范围[{block_out.min():.6f}, {block_out.max():.6f}]")
            
            # 对比完整block输出
            full_block_out = block(current_x)
            diff = jt.abs(block_out - full_block_out).max()
            print(f"    手动vs完整差异: {diff:.10f}")
            
            current_x = block_out
        
        blocks_out = current_x
        print(f"\n  blocks完整输出: {blocks_out.shape}, 范围[{blocks_out.min():.6f}, {blocks_out.max():.6f}]")
        
        # 最终输出
        if ghost_blocks.use_res:
            final_out = blocks_out + ghost_blocks.reduce_conv(concat_feat)
        else:
            final_out = blocks_out
        
        print(f"  最终输出: {final_out.shape}, 范围[{final_out.min():.6f}, {final_out.max():.6f}]")
        
        # 对比完整GhostBlocks输出
        full_ghost_blocks_out = ghost_blocks(concat_feat)
        diff = jt.abs(final_out - full_ghost_blocks_out).max()
        print(f"  手动vs完整GhostBlocks差异: {diff:.10f}")
        
        if diff < 1e-6:
            print(f"  ✅ 手动计算与完整GhostBlocks一致")
        else:
            print(f"  ❌ 手动计算与完整GhostBlocks不一致")


def check_ghost_blocks_weights():
    """检查GhostBlocks的权重加载情况"""
    print(f"\n🔍 检查GhostBlocks的权重加载情况")
    print("=" * 60)
    
    # 创建模型
    model = create_jittor_model()
    
    # 获取top_down_block1
    top_down_block1 = model.fpn.top_down_blocks[1]
    
    # 检查权重
    print(f"top_down_block1权重:")
    for name, param in top_down_block1.named_parameters():
        print(f"  {name}: {param.shape}, 范围[{param.min():.6f}, {param.max():.6f}], 均值{param.mean():.6f}")
    
    # 加载PyTorch权重并对比
    checkpoint_path = "/home/kyc/project/nanodet/nanodet-pytorch/workspace/nanodet-plus-m_320_voc_bs64/model_best/model_best.ckpt"
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)
    
    print(f"\n对应的PyTorch权重:")
    for pytorch_name, pytorch_param in state_dict.items():
        if "fpn.top_down_blocks.1" in pytorch_name:
            jittor_name = pytorch_name
            if jittor_name.startswith("model."):
                jittor_name = jittor_name[6:]
            
            pytorch_data = pytorch_param.detach().numpy()
            print(f"  {pytorch_name} -> {jittor_name}: {pytorch_data.shape}, 范围[{pytorch_data.min():.6f}, {pytorch_data.max():.6f}], 均值{pytorch_data.mean():.6f}")
            
            # 检查是否在Jittor模型中
            jittor_param = None
            for name, param in top_down_block1.named_parameters():
                full_name = f"fpn.top_down_blocks.1.{name}"
                if full_name == jittor_name:
                    jittor_param = param
                    break
            
            if jittor_param is not None:
                diff = np.abs(pytorch_data - jittor_param.numpy()).max()
                print(f"    权重差异: {diff:.10f}")
                if diff < 1e-6:
                    print(f"    ✅ 权重一致")
                else:
                    print(f"    ❌ 权重不一致")
            else:
                print(f"    ❌ 在Jittor中未找到对应参数")


def main():
    """主函数"""
    print("🚀 开始深度调试GhostBlocks")
    
    # 深度调试GhostBlocks内部
    debug_ghost_blocks_internal()
    
    # 检查权重加载
    check_ghost_blocks_weights()
    
    print(f"\n✅ 深度调试完成")


if __name__ == '__main__':
    main()
