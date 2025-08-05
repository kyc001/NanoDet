#!/usr/bin/env python3
"""
集成修复的训练脚本
在训练开始前自动应用 DepthwiseConv 修复
"""

import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 首先应用修复
print("🔧 正在应用 DepthwiseConv 修复...")
from tools.fix_depthwise_conv_jittordet import patch_depthwise_conv
if not patch_depthwise_conv():
    print("❌ DepthwiseConv 修复失败，退出训练")
    sys.exit(1)
print("✅ DepthwiseConv 修复成功，开始训练...")

# 然后导入训练相关模块
import argparse
import warnings
import jittor as jt
from jittor import nn

from nanodet.data import build_dataloader
from nanodet.model import build_model
from nanodet.trainer import build_trainer
from nanodet.util import (
    Logger,
    cfg,
    load_config,
    mkdir,
    set_random_seed,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank for distributed training")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    args = parser.parse_args()
    return args


def main(args):
    print("🚀 开始 NanoDet 训练（已修复 DepthwiseConv）...")
    
    # 加载配置
    load_config(cfg, args.config)
    
    # 设置随机种子
    if args.seed is not None:
        set_random_seed(args.seed)
    
    # 创建工作目录
    mkdir(cfg.save_dir)
    
    # 初始化日志
    logger = Logger(cfg.save_dir, use_tensorboard=cfg.log.use_tensorboard)
    logger.log_cfg(cfg)
    
    # 构建模型
    print("📦 正在构建模型...")
    model = build_model(cfg.model)
    
    # 构建数据加载器
    print("📊 正在构建数据加载器...")
    train_dataloader = build_dataloader(cfg.data.train, cfg.device.batchsize_per_gpu, cfg.device.workers_per_gpu, is_train=True)
    val_dataloader = build_dataloader(cfg.data.val, 1, cfg.device.workers_per_gpu, is_train=False)
    
    # 构建训练器
    print("🏃 正在构建训练器...")
    trainer = build_trainer(cfg.schedule, model, train_dataloader, val_dataloader, logger)
    
    # 开始训练
    print("🎯 开始训练...")
    trainer.run()
    
    print("🎉 训练完成！")


if __name__ == "__main__":
    args = parse_args()
    main(args)
