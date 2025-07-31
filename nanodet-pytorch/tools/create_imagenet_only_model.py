#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
创建只有ImageNet预训练权重的模型
用于测试ImageNet预训练权重在VOC数据集上的性能
"""

import argparse
import datetime
import os
import warnings

import pytorch_lightning as pl
import torch

from nanodet.trainer.task import TrainingTask
from nanodet.util import (
    NanoDetLightningLogger,
    cfg,
    load_config,
    mkdir,
)


def create_imagenet_only_model(config_path, output_path):
    """创建只有ImageNet预训练权重的模型"""
    
    # 加载配置
    load_config(cfg, config_path)
    
    # 创建保存目录
    save_dir = os.path.dirname(output_path)
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建logger
    logger = NanoDetLightningLogger(save_dir)
    logger.info("创建只有ImageNet预训练权重的模型...")
    
    # 创建模型
    logger.info("创建模型...")
    task = TrainingTask(cfg, None)  # 不需要evaluator
    
    # 模型会自动加载ImageNet预训练权重（在backbone中）
    logger.info("模型创建完成，ImageNet预训练权重已自动加载")
    
    # 保存模型状态
    checkpoint = {
        'epoch': 0,
        'global_step': 0,
        'pytorch-lightning_version': pl.__version__,
        'state_dict': task.state_dict(),
        'lr_schedulers': [],
        'optimizer_states': [],
        'lr_scheduler_states': [],
        'epoch_loop.state_dict': {},
        'global_step': 0,
        'pytorch-lightning_version': pl.__version__,
        'hyper_parameters': {}
    }
    
    # 保存检查点
    torch.save(checkpoint, output_path)
    logger.info(f"ImageNet预训练模型已保存到: {output_path}")
    
    # 验证模型
    logger.info("验证模型...")
    test_input = torch.randn(1, 3, 320, 320)
    if torch.cuda.is_available():
        test_input = test_input.cuda()
        task = task.cuda()
    
    task.eval()
    with torch.no_grad():
        output = task(test_input)
        logger.info(f"模型验证成功，输出形状: {output.shape}")
    
    logger.info("ImageNet预训练模型创建完成！")
    return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='创建ImageNet预训练模型')
    parser.add_argument('--config', required=True, help='配置文件路径')
    parser.add_argument('--output', required=True, help='输出模型路径')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("创建ImageNet预训练模型")
    print("=" * 60)
    print(f"配置文件: {args.config}")
    print(f"输出路径: {args.output}")
    
    # 设置CUDA
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    
    # 创建模型
    success = create_imagenet_only_model(args.config, args.output)
    
    if success:
        print("\n" + "=" * 60)
        print("ImageNet预训练模型创建成功！")
        print("=" * 60)
        print(f"模型已保存到: {args.output}")
        print("现在可以使用此模型进行测试")
    else:
        print("\n" + "=" * 60)
        print("ImageNet预训练模型创建失败！")
        print("=" * 60)
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
