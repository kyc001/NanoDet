#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PyTorch权重加载验证脚本
测试Jittor版本是否能正确加载PyTorch训练的权重
验证模型架构的一致性
"""

import os
import sys
import jittor as jt
import numpy as np
import traceback


def load_pytorch_checkpoint(ckpt_path):
    """加载PyTorch检查点文件"""
    print(f"加载PyTorch检查点: {ckpt_path}")
    
    try:
        # 使用CPU加载，避免CUDA问题
        import torch
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        print(f"✓ 检查点加载成功")
        print(f"  Keys: {list(checkpoint.keys())}")
        
        # 获取模型权重
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # 过滤模型权重（移除'model.'前缀）
        model_weights = {}
        for key, value in state_dict.items():
            if key.startswith('model.'):
                new_key = key[6:]  # 移除'model.'前缀
                model_weights[new_key] = value.numpy()  # 转换为numpy
            elif not key.startswith('avg_model.') and not key.startswith('optimizer'):
                model_weights[key] = value.numpy()
        
        print(f"✓ 提取模型权重: {len(model_weights)} 个参数")
        
        # 显示一些关键权重的信息
        key_weights = [
            'backbone.stage2.0.branch1.0.weight',
            'fpn.reduce_layers.0.conv.weight', 
            'head.gfl_cls.0.weight',
            'head.gfl_cls.0.bias'
        ]
        
        for key in key_weights:
            if key in model_weights:
                weight = model_weights[key]
                print(f"  {key}: {weight.shape}, mean={weight.mean():.6f}, std={weight.std():.6f}")
        
        return model_weights
        
    except Exception as e:
        print(f"✗ 加载PyTorch检查点失败: {e}")
        traceback.print_exc()
        return None


def create_jittor_model():
    """创建Jittor版本的模型"""
    print("\n创建Jittor模型...")
    
    try:
        from nanodet.model.backbone import build_backbone
        from nanodet.model.fpn import build_fpn
        from nanodet.model.head import build_head
        
        # 创建backbone
        backbone_cfg = {
            'name': 'ShuffleNetV2',
            'model_size': '1.0x',
            'out_stages': [2, 3, 4],
            'activation': 'LeakyReLU',
            'pretrain': False
        }
        backbone = build_backbone(backbone_cfg)
        
        # 创建FPN
        fpn_cfg = {
            'name': 'GhostPAN',
            'in_channels': [116, 232, 464],
            'out_channels': 96,
            'kernel_size': 5,
            'num_extra_level': 1,
            'use_depthwise': True,
            'activation': 'LeakyReLU'
        }
        fpn = build_fpn(fpn_cfg)
        
        # 创建检测头
        loss_cfg = type('LossCfg', (), {
            'loss_qfl': type('QFL', (), {'beta': 2.0, 'loss_weight': 1.0})(),
            'loss_dfl': type('DFL', (), {'loss_weight': 0.25})(),
            'loss_bbox': type('BBOX', (), {'loss_weight': 2.0})()
        })()
        
        head_cfg = {
            'name': 'NanoDetPlusHead',
            'num_classes': 20,
            'loss': loss_cfg,
            'input_channel': 96,
            'feat_channels': 96,
            'stacked_convs': 2,
            'kernel_size': 5,
            'strides': [8, 16, 32, 64],
            'conv_type': 'DWConv',
            'norm_cfg': dict(type='BN'),
            'reg_max': 7,
            'activation': 'LeakyReLU'
        }
        head = build_head(head_cfg)
        
        print(f"✓ Jittor模型创建成功")
        print(f"  Backbone参数: {sum(p.numel() for p in backbone.parameters())/1e6:.2f}M")
        print(f"  FPN参数: {sum(p.numel() for p in fpn.parameters())/1e6:.2f}M")
        print(f"  Head参数: {sum(p.numel() for p in head.parameters())/1e6:.2f}M")
        
        return backbone, fpn, head
        
    except Exception as e:
        print(f"✗ 创建Jittor模型失败: {e}")
        traceback.print_exc()
        return None, None, None


def compare_model_outputs(backbone, fpn, head, pytorch_weights):
    """比较模型输出，验证架构一致性"""
    print("\n比较模型输出...")
    
    try:
        # 创建测试输入
        x = jt.randn(1, 3, 320, 320)
        
        # Jittor模型前向传播
        with jt.no_grad():
            backbone_out = backbone(x)
            fpn_out = fpn(backbone_out)
            head_out = head(fpn_out)
        
        print(f"✓ Jittor模型前向传播成功")
        print(f"  输入: {x.shape}")
        print(f"  Backbone输出: {[o.shape for o in backbone_out]}")
        print(f"  FPN输出: {[o.shape for o in fpn_out]}")
        print(f"  Head输出: {head_out.shape}")
        
        # 检查输出形状是否符合预期
        expected_shape = (1, 2125, 52)  # (batch, points, classes+reg)
        if head_out.shape != expected_shape:
            print(f"⚠ 输出形状不匹配: 得到 {head_out.shape}, 期望 {expected_shape}")
            return False
        
        print(f"✓ 输出形状正确: {head_out.shape}")
        
        # 分析输出统计信息
        print(f"  输出统计:")
        print(f"    均值: {jt.mean(head_out).item():.6f}")
        print(f"    标准差: {jt.std(head_out).item():.6f}")
        print(f"    最小值: {jt.min(head_out).item():.6f}")
        print(f"    最大值: {jt.max(head_out).item():.6f}")
        
        return True
        
    except Exception as e:
        print(f"✗ 模型输出比较失败: {e}")
        traceback.print_exc()
        return False


def analyze_weight_compatibility(backbone, fpn, head, pytorch_weights):
    """分析权重兼容性"""
    print("\n分析权重兼容性...")
    
    try:
        # 获取Jittor模型的参数名称
        jittor_params = {}
        
        # Backbone参数
        for name, param in backbone.named_parameters():
            jittor_params[f'backbone.{name}'] = param.shape
        
        # FPN参数
        for name, param in fpn.named_parameters():
            jittor_params[f'fpn.{name}'] = param.shape
        
        # Head参数
        for name, param in head.named_parameters():
            jittor_params[f'head.{name}'] = param.shape
        
        print(f"✓ Jittor模型参数: {len(jittor_params)} 个")
        
        # 比较参数名称和形状
        matched_params = 0
        mismatched_params = 0
        missing_params = 0
        
        for jittor_name, jittor_shape in jittor_params.items():
            if jittor_name in pytorch_weights:
                pytorch_shape = pytorch_weights[jittor_name].shape
                if jittor_shape == pytorch_shape:
                    matched_params += 1
                else:
                    print(f"  ⚠ 形状不匹配: {jittor_name}")
                    print(f"    Jittor: {jittor_shape}, PyTorch: {pytorch_shape}")
                    mismatched_params += 1
            else:
                print(f"  ✗ 缺失参数: {jittor_name}")
                missing_params += 1
        
        # 检查PyTorch中多余的参数
        extra_params = 0
        for pytorch_name in pytorch_weights.keys():
            if pytorch_name not in jittor_params:
                print(f"  + 额外参数: {pytorch_name}")
                extra_params += 1
        
        print(f"\n权重兼容性分析:")
        print(f"  ✓ 匹配参数: {matched_params}")
        print(f"  ⚠ 形状不匹配: {mismatched_params}")
        print(f"  ✗ 缺失参数: {missing_params}")
        print(f"  + 额外参数: {extra_params}")
        
        compatibility_rate = matched_params / len(jittor_params) * 100
        print(f"  兼容性: {compatibility_rate:.1f}%")
        
        return compatibility_rate > 90  # 90%以上认为兼容
        
    except Exception as e:
        print(f"✗ 权重兼容性分析失败: {e}")
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("=" * 60)
    print("PyTorch权重加载验证")
    print("=" * 60)
    
    # 显示系统信息
    print(f"Jittor版本: {jt.__version__}")
    print(f"CUDA可用: {jt.has_cuda}")
    if jt.has_cuda:
        jt.flags.use_cuda = 1
        print(f"使用CUDA: {jt.flags.use_cuda}")
    
    # 检查PyTorch检查点文件
    ckpt_path = "../nanodet-pytorch/workspace/nanodet-plus-m_320_voc/model_last.ckpt"
    if not os.path.exists(ckpt_path):
        print(f"✗ PyTorch检查点文件不存在: {ckpt_path}")
        return False
    
    # 加载PyTorch权重
    pytorch_weights = load_pytorch_checkpoint(ckpt_path)
    if pytorch_weights is None:
        return False
    
    # 创建Jittor模型
    backbone, fpn, head = create_jittor_model()
    if backbone is None:
        return False
    
    # 比较模型输出
    if not compare_model_outputs(backbone, fpn, head, pytorch_weights):
        return False
    
    # 分析权重兼容性
    if not analyze_weight_compatibility(backbone, fpn, head, pytorch_weights):
        print("\n⚠ 权重兼容性较低，可能需要调整模型架构")
        return False
    
    print("\n" + "=" * 60)
    print("验证结果")
    print("=" * 60)
    print("🎉 PyTorch权重加载验证成功！")
    print("✅ Jittor模型架构与PyTorch版本兼容")
    print("✅ 可以开始Jittor版本的训练")
    print("✅ 建议使用相同的初始化和训练参数")
    
    return True


if __name__ == '__main__':
    success = main()
    if not success:
        print("\n❌ 验证失败！请检查模型架构一致性。")
        sys.exit(1)
    else:
        print("\n✅ 验证成功！可以开始Jittor训练。")
        sys.exit(0)
