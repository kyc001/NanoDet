#!/usr/bin/env python3
"""
优化的训练脚本
集成了所有 jittordet 组件优化和内存修复
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 首先修复 DepthwiseConv 问题
from tools.fix_depthwise_conv import patch_depthwise_conv
patch_depthwise_conv()

import jittor as jt
from jittor import nn
import time
import yaml
from copy import deepcopy

from nanodet.util import Logger, cfg, load_config
from nanodet.trainer import build_trainer


def setup_environment():
    """
    设置训练环境
    """
    # 设置 Jittor 内存管理
    jt.flags.use_cuda = 1
    
    # 启用内存优化
    os.environ['JITTOR_MEMORY_POOL'] = '1'
    os.environ['JITTOR_MEMORY_POOL_SIZE'] = '6GB'  # 限制内存池大小
    
    # 启用梯度检查点以节省内存
    jt.flags.use_cuda_managed_allocator = 1
    
    print("✅ 环境设置完成")


def optimize_config(config_path):
    """
    优化配置文件以减少内存使用
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 内存优化设置
    if 'device' in config:
        config['device']['batchsize_per_gpu'] = 1  # 最小 batch size
        config['device']['workers_per_gpu'] = 1   # 减少 worker 数量
        config['device']['precision'] = 16        # 使用混合精度
    
    # 模型优化设置
    if 'model' in config and 'arch' in config['model']:
        arch = config['model']['arch']
        
        # 禁用 depthwise 卷积
        if 'fpn' in arch:
            arch['fpn']['use_depthwise'] = False
        
        # 减少特征通道数
        if 'head' in arch:
            arch['head']['feat_channels'] = min(arch['head'].get('feat_channels', 96), 96)
    
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="配置文件路径")
    parser.add_argument("--local_rank", default=0, type=int, help="local rank")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    args = parser.parse_args()
    
    print("=== NanoDet 优化训练脚本 ===")
    
    # 设置环境
    setup_environment()
    
    # 优化配置
    print(f"正在加载配置: {args.config}")
    optimized_config = optimize_config(args.config)
    
    # 保存优化后的配置
    optimized_config_path = args.config.replace('.yml', '_optimized.yml')
    with open(optimized_config_path, 'w') as f:
        yaml.dump(optimized_config, f, default_flow_style=False)
    print(f"优化配置已保存: {optimized_config_path}")
    
    # 加载配置
    load_config(cfg, optimized_config_path)
    
    # 设置随机种子
    if args.seed is not None:
        jt.set_global_seed(args.seed)
    
    # 创建 logger
    logger = Logger(args.local_rank, cfg.save_dir)
    
    # 显示内存信息
    print("\n=== 训练前内存状态 ===")
    jt.display_memory_info()
    
    try:
        # 创建训练器
        print("正在创建训练器...")
        trainer = build_trainer(args.local_rank, cfg)
        
        # 开始训练
        print("开始训练...")
        trainer.run()
        
    except RuntimeError as e:
        if "memory" in str(e).lower():
            print(f"\n❌ 内存不足错误: {e}")
            print("\n建议解决方案:")
            print("1. 进一步减少 batch_size")
            print("2. 减少模型特征通道数")
            print("3. 使用更小的输入分辨率")
            print("4. 启用梯度累积")
            
            # 显示内存信息
            jt.display_memory_info()
            
        else:
            print(f"\n❌ 训练错误: {e}")
            import traceback
            traceback.print_exc()
        
        return 1
    
    except Exception as e:
        print(f"\n❌ 未知错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("🎉 训练完成！")
    return 0


if __name__ == "__main__":
    sys.exit(main())
