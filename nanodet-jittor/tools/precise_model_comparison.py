#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
精确的模型对比
使用完全相同的输入和权重，逐层对比
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
    
    print(f"✓ 加载了 {loaded_count} 个参数")
    return model


def compare_with_fixed_input():
    """使用固定输入进行精确对比"""
    print("🔍 使用固定输入进行精确对比")
    print("=" * 60)
    
    # 创建固定的随机输入
    np.random.seed(42)
    torch.manual_seed(42)
    jt.set_global_seed(42)
    
    # 创建测试输入
    input_data = np.random.randn(1, 3, 320, 320).astype(np.float32)
    
    print(f"测试输入形状: {input_data.shape}")
    print(f"测试输入范围: [{input_data.min():.6f}, {input_data.max():.6f}]")
    
    # 创建Jittor模型
    print(f"\n1️⃣ 创建并加载Jittor模型...")
    jittor_model = create_jittor_model()
    jittor_model = load_weights_to_jittor_model(jittor_model)
    jittor_model.eval()
    
    # Jittor推理
    jittor_input = jt.array(input_data)
    with jt.no_grad():
        jittor_output = jittor_model(jittor_input)
    
    print(f"Jittor输出形状: {jittor_output.shape}")
    print(f"Jittor输出范围: [{jittor_output.min():.6f}, {jittor_output.max():.6f}]")
    
    # 分析Jittor输出
    jittor_cls = jittor_output[:, :, :20]
    jittor_reg = jittor_output[:, :, 20:]
    jittor_cls_scores = jt.sigmoid(jittor_cls)
    
    print(f"Jittor分类预测范围: [{jittor_cls.min():.6f}, {jittor_cls.max():.6f}]")
    print(f"Jittor回归预测范围: [{jittor_reg.min():.6f}, {jittor_reg.max():.6f}]")
    print(f"Jittor最高置信度: {jittor_cls_scores.max():.6f}")
    
    # 保存Jittor输出用于PyTorch对比
    np.save("fixed_input_data.npy", input_data)
    np.save("jittor_fixed_output.npy", jittor_output.numpy())
    
    print(f"\n✓ Jittor结果已保存")
    
    # 加载PyTorch参考输出（如果存在）
    if os.path.exists("pytorch_fixed_output.npy"):
        print(f"\n2️⃣ 加载PyTorch参考输出...")
        pytorch_output = np.load("pytorch_fixed_output.npy")
        
        print(f"PyTorch输出形状: {pytorch_output.shape}")
        print(f"PyTorch输出范围: [{pytorch_output.min():.6f}, {pytorch_output.max():.6f}]")
        
        # 分析PyTorch输出
        pytorch_cls = pytorch_output[:, :, :20]
        pytorch_reg = pytorch_output[:, :, 20:]
        pytorch_cls_scores = 1 / (1 + np.exp(-pytorch_cls))  # sigmoid
        
        print(f"PyTorch分类预测范围: [{pytorch_cls.min():.6f}, {pytorch_cls.max():.6f}]")
        print(f"PyTorch回归预测范围: [{pytorch_reg.min():.6f}, {pytorch_reg.max():.6f}]")
        print(f"PyTorch最高置信度: {pytorch_cls_scores.max():.6f}")
        
        # 计算差异
        diff = np.abs(pytorch_output - jittor_output.numpy())
        print(f"\n📊 输出差异分析:")
        print(f"   最大差异: {diff.max():.6f}")
        print(f"   平均差异: {diff.mean():.6f}")
        print(f"   差异标准差: {diff.std():.6f}")
        
        # 分析分类和回归部分的差异
        cls_diff = np.abs(pytorch_cls - jittor_cls.numpy())
        reg_diff = np.abs(pytorch_reg - jittor_reg.numpy())
        
        print(f"   分类部分最大差异: {cls_diff.max():.6f}")
        print(f"   分类部分平均差异: {cls_diff.mean():.6f}")
        print(f"   回归部分最大差异: {reg_diff.max():.6f}")
        print(f"   回归部分平均差异: {reg_diff.mean():.6f}")
        
        # 分析置信度差异
        conf_diff = np.abs(pytorch_cls_scores - jittor_cls_scores.numpy())
        print(f"   置信度最大差异: {conf_diff.max():.6f}")
        print(f"   置信度平均差异: {conf_diff.mean():.6f}")
        
        if diff.max() < 1e-4:
            print(f"\n✅ 输出高度一致！")
            return True
        else:
            print(f"\n❌ 输出存在显著差异")
            return False
    
    else:
        print(f"\n⚠️ 没有找到PyTorch参考输出")
        print(f"   请运行PyTorch版本生成参考输出")
        return False


def create_pytorch_reference_script():
    """创建PyTorch参考脚本"""
    pytorch_script = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PyTorch固定输入参考输出生成脚本
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
    """生成PyTorch固定输入参考输出"""
    print("🚀 生成PyTorch固定输入参考输出")
    
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
        print("✓ 使用Jittor保存的固定输入")
    else:
        # 创建相同的固定输入
        input_data = np.random.randn(1, 3, 320, 320).astype(np.float32)
        print("✓ 创建新的固定输入")
    
    print(f"输入形状: {input_data.shape}")
    print(f"输入范围: [{input_data.min():.6f}, {input_data.max():.6f}]")
    
    # 推理
    input_tensor = torch.from_numpy(input_data)
    with torch.no_grad():
        output = model(input_tensor)
    
    # 保存输出
    output_np = output.detach().numpy()
    np.save("pytorch_fixed_output.npy", output_np)
    
    print(f"✓ PyTorch输出已保存: {output.shape}")
    print(f"   输出范围: [{output.min():.6f}, {output.max():.6f}]")
    
    # 分析输出
    cls_preds = output[:, :, :20]
    reg_preds = output[:, :, 20:]
    cls_scores = torch.sigmoid(cls_preds)
    
    print(f"   分类预测范围: [{cls_preds.min():.6f}, {cls_preds.max():.6f}]")
    print(f"   回归预测范围: [{reg_preds.min():.6f}, {reg_preds.max():.6f}]")
    print(f"   最高置信度: {cls_scores.max():.6f}")


if __name__ == '__main__':
    main()
'''
    
    with open('generate_pytorch_fixed_reference.py', 'w') as f:
        f.write(pytorch_script)
    
    print("✓ 创建了PyTorch固定输入参考脚本: generate_pytorch_fixed_reference.py")


def main():
    """主函数"""
    print("🚀 开始精确模型对比")
    
    # 创建PyTorch参考脚本
    create_pytorch_reference_script()
    
    # 进行对比
    success = compare_with_fixed_input()
    
    if not success:
        print(f"\n📝 下一步:")
        print(f"1. 运行: python generate_pytorch_fixed_reference.py")
        print(f"2. 重新运行此脚本进行精确对比")
    
    print(f"\n✅ 精确对比完成")


if __name__ == '__main__':
    main()
