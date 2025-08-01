#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
逐层调试工具
找出PyTorch和Jittor模型差异的具体层
"""

import os
import sys
import torch
import jittor as jt
import numpy as np

# 添加路径
sys.path.append('/home/kyc/project/nanodet/nanodet-jittor')
from nanodet.model.arch.nanodet_plus import NanoDetPlus


def create_jittor_model():
    """创建Jittor模型"""
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
    return model


def load_weights_to_jittor_model(model):
    """加载权重到Jittor模型"""
    checkpoint_path = "/home/kyc/project/nanodet/nanodet-pytorch/workspace/nanodet-plus-m_320_voc_bs64/model_best/model_best.ckpt"
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)
    
    # 获取Jittor模型的参数字典
    jittor_state_dict = {}
    for name, param in model.named_parameters():
        jittor_state_dict[name] = param
    
    # 权重加载
    loaded_count = 0
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
                loaded_count += 1
            elif "scale" in jittor_name and len(pytorch_param.shape) == 0 and list(jittor_param.shape) == [1]:
                jittor_param.assign(jt.array([pytorch_param.detach().numpy()]))
                loaded_count += 1
    
    return model


def layer_by_layer_debug():
    """逐层调试"""
    print("🔍 开始逐层调试")
    print("=" * 60)
    
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    jt.set_global_seed(42)
    
    # 加载固定输入
    if os.path.exists("fixed_input_data.npy"):
        input_data = np.load("fixed_input_data.npy")
        print(f"✓ 使用固定输入: {input_data.shape}")
    else:
        input_data = np.random.randn(1, 3, 320, 320).astype(np.float32)
        print(f"✓ 创建新的固定输入: {input_data.shape}")
    
    # 创建Jittor模型
    print(f"\n1️⃣ 创建Jittor模型...")
    jittor_model = create_jittor_model()
    jittor_model = load_weights_to_jittor_model(jittor_model)
    jittor_model.eval()
    
    # 准备输入
    jittor_input = jt.array(input_data)
    
    print(f"\n2️⃣ 逐层分析Jittor模型...")
    
    # Backbone
    print(f"\n🔍 Backbone (ShuffleNetV2):")
    with jt.no_grad():
        backbone_features = jittor_model.backbone(jittor_input)
    
    for i, feat in enumerate(backbone_features):
        print(f"   特征{i}: {feat.shape}, 范围[{feat.min():.6f}, {feat.max():.6f}]")
        # 保存backbone特征
        np.save(f"jittor_backbone_feat_{i}.npy", feat.numpy())
    
    # FPN
    print(f"\n🔍 FPN (GhostPAN):")
    with jt.no_grad():
        fpn_features = jittor_model.fpn(backbone_features)
    
    for i, feat in enumerate(fpn_features):
        print(f"   FPN特征{i}: {feat.shape}, 范围[{feat.min():.6f}, {feat.max():.6f}]")
        # 保存FPN特征
        np.save(f"jittor_fpn_feat_{i}.npy", feat.numpy())
    
    # Head
    print(f"\n🔍 Head (NanoDetPlusHead):")
    with jt.no_grad():
        head_output = jittor_model.head(fpn_features)
    
    print(f"   Head输出: {head_output.shape}, 范围[{head_output.min():.6f}, {head_output.max():.6f}]")
    
    # 分析Head输出
    cls_preds = head_output[:, :, :20]
    reg_preds = head_output[:, :, 20:]
    cls_scores = jt.sigmoid(cls_preds)
    
    print(f"   分类预测: 范围[{cls_preds.min():.6f}, {cls_preds.max():.6f}]")
    print(f"   回归预测: 范围[{reg_preds.min():.6f}, {reg_preds.max():.6f}]")
    print(f"   最高置信度: {cls_scores.max():.6f}")
    
    # 保存Head输出
    np.save("jittor_head_output.npy", head_output.numpy())
    
    # 完整模型
    print(f"\n🔍 完整模型:")
    with jt.no_grad():
        full_output = jittor_model(jittor_input)
    
    print(f"   完整输出: {full_output.shape}, 范围[{full_output.min():.6f}, {full_output.max():.6f}]")
    
    # 检查Head输出和完整输出是否一致
    head_vs_full_diff = np.abs(head_output.numpy() - full_output.numpy()).max()
    print(f"   Head vs 完整输出差异: {head_vs_full_diff:.8f}")
    
    if head_vs_full_diff < 1e-6:
        print(f"   ✅ Head输出与完整输出一致")
    else:
        print(f"   ❌ Head输出与完整输出不一致")
    
    print(f"\n3️⃣ 与PyTorch参考对比...")
    
    # 对比backbone特征
    backbone_diffs = []
    for i in range(len(backbone_features)):
        pytorch_file = f"pytorch_backbone_feat_{i}.npy"
        if os.path.exists(pytorch_file):
            pytorch_feat = np.load(pytorch_file)
            jittor_feat = np.load(f"jittor_backbone_feat_{i}.npy")
            
            diff = np.abs(pytorch_feat - jittor_feat)
            max_diff = diff.max()
            mean_diff = diff.mean()
            
            backbone_diffs.append(max_diff)
            print(f"   Backbone特征{i}差异: 最大{max_diff:.6f}, 平均{mean_diff:.6f}")
        else:
            print(f"   ⚠️ 没有找到PyTorch Backbone特征{i}")
    
    # 对比FPN特征
    fpn_diffs = []
    for i in range(len(fpn_features)):
        pytorch_file = f"pytorch_fpn_feat_{i}.npy"
        if os.path.exists(pytorch_file):
            pytorch_feat = np.load(pytorch_file)
            jittor_feat = np.load(f"jittor_fpn_feat_{i}.npy")
            
            diff = np.abs(pytorch_feat - jittor_feat)
            max_diff = diff.max()
            mean_diff = diff.mean()
            
            fpn_diffs.append(max_diff)
            print(f"   FPN特征{i}差异: 最大{max_diff:.6f}, 平均{mean_diff:.6f}")
        else:
            print(f"   ⚠️ 没有找到PyTorch FPN特征{i}")
    
    # 对比Head输出
    if os.path.exists("pytorch_head_output.npy"):
        pytorch_head = np.load("pytorch_head_output.npy")
        jittor_head = np.load("jittor_head_output.npy")
        
        diff = np.abs(pytorch_head - jittor_head)
        max_diff = diff.max()
        mean_diff = diff.mean()
        
        print(f"   Head输出差异: 最大{max_diff:.6f}, 平均{mean_diff:.6f}")
    else:
        print(f"   ⚠️ 没有找到PyTorch Head输出")
    
    print(f"\n📊 差异分析总结:")
    if backbone_diffs:
        print(f"   Backbone最大差异: {max(backbone_diffs):.6f}")
    if fpn_diffs:
        print(f"   FPN最大差异: {max(fpn_diffs):.6f}")
    
    # 判断问题出现在哪一层
    if backbone_diffs and max(backbone_diffs) > 1e-3:
        print(f"   🚨 问题可能出现在Backbone层")
    elif fpn_diffs and max(fpn_diffs) > 1e-3:
        print(f"   🚨 问题可能出现在FPN层")
    else:
        print(f"   🚨 问题可能出现在Head层")
    
    print(f"\n✅ 逐层调试完成")


def create_pytorch_layer_debug_script():
    """创建PyTorch逐层调试脚本"""
    pytorch_script = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PyTorch逐层调试脚本
"""

import os
import sys
import torch
import numpy as np

# 添加PyTorch版本路径
sys.path.insert(0, '/home/kyc/project/nanodet/nanodet-pytorch')

from nanodet.model.arch import build_model
from nanodet.util import cfg, load_config


def main():
    """PyTorch逐层调试"""
    print("🚀 PyTorch逐层调试")
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 加载配置
    config_path = "/home/kyc/project/nanodet/nanodet-pytorch/config/nanodet-plus-m_320_voc.yml"
    load_config(cfg, config_path)
    
    # 创建模型
    model = build_model(cfg.model)
    
    # 加载权重
    checkpoint_path = "/home/kyc/project/nanodet/nanodet-pytorch/workspace/nanodet-plus-m_320_voc_bs64/model_best/model_best.ckpt"
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    # 移除前缀
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('model.', '') if key.startswith('model.') else key
        new_state_dict[new_key] = value
    
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    
    print("✓ PyTorch模型加载成功")
    
    # 加载固定输入
    if os.path.exists("fixed_input_data.npy"):
        input_data = np.load("fixed_input_data.npy")
        print("✓ 使用固定输入")
    else:
        input_data = np.random.randn(1, 3, 320, 320).astype(np.float32)
        print("✓ 创建新的固定输入")
    
    input_tensor = torch.from_numpy(input_data)
    
    print(f"\\n🔍 逐层分析PyTorch模型...")
    
    # Backbone
    print(f"\\n🔍 Backbone:")
    with torch.no_grad():
        backbone_features = model.backbone(input_tensor)
    
    for i, feat in enumerate(backbone_features):
        print(f"   特征{i}: {feat.shape}, 范围[{feat.min():.6f}, {feat.max():.6f}]")
        np.save(f"pytorch_backbone_feat_{i}.npy", feat.detach().numpy())
    
    # FPN
    print(f"\\n🔍 FPN:")
    with torch.no_grad():
        fpn_features = model.fpn(backbone_features)
    
    for i, feat in enumerate(fpn_features):
        print(f"   FPN特征{i}: {feat.shape}, 范围[{feat.min():.6f}, {feat.max():.6f}]")
        np.save(f"pytorch_fpn_feat_{i}.npy", feat.detach().numpy())
    
    # Head
    print(f"\\n🔍 Head:")
    with torch.no_grad():
        head_output = model.head(fpn_features)
    
    print(f"   Head输出: {head_output.shape}, 范围[{head_output.min():.6f}, {head_output.max():.6f}]")
    np.save("pytorch_head_output.npy", head_output.detach().numpy())
    
    # 完整模型
    print(f"\\n🔍 完整模型:")
    with torch.no_grad():
        full_output = model(input_tensor)
    
    print(f"   完整输出: {full_output.shape}, 范围[{full_output.min():.6f}, {full_output.max():.6f}]")
    
    print(f"\\n✅ PyTorch逐层调试完成")


if __name__ == '__main__':
    main()
'''
    
    with open('pytorch_layer_debug.py', 'w') as f:
        f.write(pytorch_script)
    
    print("✓ 创建了PyTorch逐层调试脚本: pytorch_layer_debug.py")


def main():
    """主函数"""
    print("🚀 开始逐层调试")
    
    # 创建PyTorch调试脚本
    create_pytorch_layer_debug_script()
    
    # 进行逐层调试
    layer_by_layer_debug()
    
    print(f"\n📝 下一步:")
    print(f"1. 运行: python pytorch_layer_debug.py")
    print(f"2. 重新运行此脚本查看详细对比")
    
    print(f"\n✅ 逐层调试完成")


if __name__ == '__main__':
    main()
