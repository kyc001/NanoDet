#!/usr/bin/env python3
"""
将PyTorch权重转换为Jittor格式
"""

import sys
import os
import argparse

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.abspath(os.path.join(ROOT, ".."))
JT_ROOT = os.path.join(PROJ, "nanodet-jittor")
if JT_ROOT not in sys.path:
    sys.path.insert(0, JT_ROOT)

import jittor as jt
import torch
from nanodet.util.check_point import pt_to_jt_checkpoint
from nanodet.model.arch import build_model
from nanodet.util.config import load_config, cfg

def convert_pytorch_to_jittor(args):
    """转换PyTorch权重为Jittor格式"""

    # 加载配置
    config_path = args.config
    load_config(cfg, config_path)

    # 构建Jittor模型
    print("🔧 构建Jittor模型...")
    model = build_model(cfg.model)

    # 加载PyTorch权重
    pt_weight_path = args.pt_ckpt
    print(f"📥 加载PyTorch权重: {pt_weight_path}")

    if not os.path.exists(pt_weight_path):
        raise FileNotFoundError(f"PyTorch权重文件不存在: {pt_weight_path}")

    # 加载PyTorch检查点
    pt_ckpt = torch.load(pt_weight_path, map_location='cpu')
    print(f"✅ PyTorch权重加载成功，键数: {len(pt_ckpt.get('state_dict', pt_ckpt))}")

    # 转换为Jittor格式
    print("🔄 转换权重格式...")
    jt_ckpt = pt_to_jt_checkpoint(pt_ckpt, model)

    # 创建完整的Jittor检查点
    jittor_checkpoint = {
        'state_dict': jt_ckpt['state_dict'] if 'state_dict' in jt_ckpt else jt_ckpt,
        'epoch': args.epoch,
        'best_map': args.best_map,
        'optimizer': None,  # 不保存优化器状态
        'lr_scheduler': None,
        'meta': {
            'framework': 'Jittor',
            'model_name': 'NanoDet-Plus-m',
            'dataset': 'VOC2007',
            'input_size': [320, 320],
            'converted_from': 'PyTorch',
            'original_path': pt_weight_path,
            'config': config_path,
        }
    }

    # 保存Jittor权重
    output_path = args.jt_ckpt
    print(f"💾 保存Jittor权重: {output_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    jt.save(jittor_checkpoint, output_path)

    print("✅ 权重转换完成！")
    print(f"📊 输出文件: {output_path}")
    print(f"📈 记录epoch: {args.epoch}")
    print(f"🎯 best_map: {args.best_map}")

    return output_path


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default="nanodet-jittor/config/nanodet-plus-m_320_voc.yml",
        help="Jittor 配置文件路径",
    )
    ap.add_argument(
        "--pt_ckpt",
        default="nanodet-pytorch/workspace/nanodet-plus-m_320_voc/model_best/nanodet_model_best.pth",
        help="PyTorch 权重路径(.pth/.ckpt)",
    )
    ap.add_argument(
        "--jt_ckpt",
        default="workspace/pt2jt_model_best.pkl",
        help="输出 Jittor 权重路径(.pkl)",
    )
    ap.add_argument("--epoch", type=int, default=0, help="记录 epoch")
    ap.add_argument("--best_map", type=float, default=0.0, help="记录 best mAP")
    return ap.parse_args()

if __name__ == "__main__":
    try:
        args = parse_args()
        output_path = convert_pytorch_to_jittor(args)
        print(f"\n🎉 成功！Jittor权重已保存到: {output_path}")
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        sys.exit(1)
