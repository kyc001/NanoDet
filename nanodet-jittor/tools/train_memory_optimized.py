#!/usr/bin/env python3
"""
内存优化的训练脚本
专门针对8GB显存进行优化
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import jittor as jt
from jittor import nn
import time
import yaml
from copy import deepcopy

from nanodet.util import Logger, cfg, load_config
from nanodet.trainer.task import TrainingTask
from nanodet.data.dataset import build_dataset
from nanodet.evaluator import build_evaluator
from nanodet.util.logger import NanoDetLightningLogger


def setup_memory_optimization():
    """
    设置内存优化环境
    """
    # 🔧 激进的内存优化设置
    jt.flags.use_cuda = 1
    
    # 🔧 限制内存池大小
    os.environ['JITTOR_MEMORY_POOL'] = '1'
    os.environ['JITTOR_MEMORY_POOL_SIZE'] = '4GB'  # 限制为4GB
    
    # 🔧 启用内存优化选项
    jt.flags.use_cuda_managed_allocator = 1
    
    # 🔧 启用梯度检查点
    os.environ['JITTOR_ENABLE_GRAD_CHECKPOINT'] = '1'
    
    # 🔧 减少编译缓存
    os.environ['JITTOR_CACHE_SIZE'] = '1GB'
    
    print("✅ 内存优化设置完成")


def optimize_config_for_memory(config_path):
    """
    为内存优化调整配置
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 🔧 强制最小内存设置
    if 'device' in config:
        config['device']['batchsize_per_gpu'] = 1  # 最小batch size
        config['device']['workers_per_gpu'] = 1   # 最小worker数
        config['device']['precision'] = 16        # 混合精度
    
    # 🔧 减小模型尺寸
    if 'model' in config and 'arch' in config['model']:
        arch = config['model']['arch']
        
        # 减小FPN通道数
        if 'fpn' in arch:
            arch['fpn']['out_channels'] = min(arch['fpn'].get('out_channels', 96), 64)
        
        # 减小Head通道数
        if 'head' in arch:
            arch['head']['feat_channels'] = min(arch['head'].get('feat_channels', 96), 64)
            arch['head']['input_channel'] = min(arch['head'].get('input_channel', 96), 64)
        
        # 减小aux_head通道数
        if 'aux_head' in arch:
            arch['aux_head']['feat_channels'] = min(arch['aux_head'].get('feat_channels', 96), 64)
            arch['aux_head']['input_channel'] = min(arch['aux_head'].get('input_channel', 96), 64)
    
    # 🔧 减小数据增强强度
    if 'data' in config and 'train' in config['data']:
        train_cfg = config['data']['train']
        if 'pipeline' in train_cfg:
            # 减小图像尺寸
            if 'input_size' in train_cfg:
                train_cfg['input_size'] = [256, 256]  # 从320减小到256
    
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank for distributed training")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    args = parser.parse_args()

    # 🔧 设置内存优化
    setup_memory_optimization()
    
    # 🔧 优化配置
    optimized_config = optimize_config_for_memory(args.config)
    
    # 保存优化后的配置
    optimized_config_path = args.config.replace('.yml', '_memory_optimized.yml')
    with open(optimized_config_path, 'w') as f:
        yaml.dump(optimized_config, f, default_flow_style=False)
    
    print(f"✅ 优化配置已保存到: {optimized_config_path}")
    
    # 加载优化后的配置
    load_config(cfg, optimized_config_path)
    
    if args.seed is not None:
        print(f"设置随机种子为 {args.seed}")
        jt.set_global_seed(args.seed)

    # 🔧 强制垃圾回收
    import gc
    gc.collect()

    # 构建训练器
    val_dataset = build_dataset(cfg.data.val, 'val')
    evaluator = build_evaluator(cfg.evaluator, val_dataset)
    logger = NanoDetLightningLogger(cfg.save_dir)

    trainer = TrainingTask(cfg, evaluator, logger)

    print("🚀 开始内存优化训练...")
    trainer.fit()


if __name__ == "__main__":
    main()
