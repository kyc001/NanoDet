# JITTOR MIGRATION by Google LLC.
import sys
import os


# 将项目根目录（nanodet-jittor）添加到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import random
import warnings
from types import SimpleNamespace

# JITTOR MIGRATION: 导入 jittor 和 numpy
import jittor as jt
import numpy as np

# JITTOR MIGRATION: 导入已迁移的工具函数和类
from nanodet.data.collate import naive_collate
from nanodet.data.dataset import build_dataset
from nanodet.evaluator import build_evaluator
from nanodet.trainer.task import TrainingTask
# JITTOR MIGRATION FIX: 直接从 logger 模块导入正确的类名，并调整其他导入
from nanodet.util.logger import NanoDetLightningLogger
from nanodet.util import (
    cfg,
    convert_old_model,
    env_utils,
    load_config,
    load_model_weight,
    mkdir,
)

def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="训练配置文件的路径")
    parser.add_argument(
        "--local_rank", default=-1, type=int, help="用于分布式训练的节点排名"
    )
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    args = parser.parse_args()
    return args

def set_seed(seed):
    """设置随机种子以确保可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    jt.seed(seed)

def main(args):
    # 加载配置
    load_config(cfg, args.config)
    if cfg.model.arch.head.num_classes != len(cfg.class_names):
        raise ValueError(
            "cfg.model.arch.head.num_classes 必须等于 len(cfg.class_names), "
            f"但得到了 {cfg.model.arch.head.num_classes} 和 {len(cfg.class_names)}"
        )
    
    # JITTOR MIGRATION: 使用 jt.rank 获取排名，并设置 GPU
    local_rank = jt.rank if jt.world_size > 1 else 0
    if jt.has_cuda:
        jt.flags.use_cuda = 1

    # 创建保存目录和日志记录器
    # JITTOR MIGRATION FIX: 修复 mkdir 的调用，传入 local_rank
    mkdir(local_rank, cfg.save_dir)
    # JITTOR MIGRATION FIX: 使用正确的类名实例化 Logger
    logger = NanoDetLightningLogger(cfg.save_dir)
    logger.dump_cfg(cfg)

    # 设置随机种子
    if args.seed is not None:
        logger.info(f"设置随机种子为 {args.seed}")
        set_seed(args.seed)

    # JITTOR MIGRATION: 移除 PyTorch-Lightning 和 torch.backends.cudnn 的设置
    
    logger.info("正在设置数据...")
    train_dataset = build_dataset(cfg.data.train, "train")
    val_dataset = build_dataset(cfg.data.val, "val") # 模式应为 'val'

    # JITTOR MIGRATION: Jittor 的 DataLoader 直接在 Dataset 对象上配置
    train_dataloader = train_dataset.set_attrs(
        batch_size=cfg.device.batchsize_per_gpu,
        shuffle=True,
        num_workers=cfg.device.workers_per_gpu,
        collate_batch=naive_collate,
        drop_last=True,
    )
    val_dataloader = val_dataset.set_attrs(
        batch_size=cfg.device.batchsize_per_gpu,
        shuffle=False,
        num_workers=cfg.device.workers_per_gpu,
        collate_batch=naive_collate,
        drop_last=False,
    )

    evaluator = build_evaluator(cfg.evaluator, val_dataset)

    logger.info("正在创建模型...")
    task = TrainingTask(cfg, evaluator, logger)
    
    # 加载预训练模型权重
    if "load_model" in cfg.schedule:
        ckpt = jt.load(cfg.schedule.load_model)
        if "pytorch-lightning_version" in ckpt:
             warnings.warn(
                "警告！您正在加载一个 PyTorch Lightning 检查点。请确保其与当前模型兼容。"
            )
        elif "state_dict" not in ckpt:
            # 假设是旧格式
            warnings.warn(
                "警告！旧的 .pth 检查点格式已弃用。请使用 tools/convert_old_checkpoint.py 进行转换。"
            )
            ckpt = convert_old_model(ckpt)
        load_model_weight(task.model, ckpt, logger)
        logger.info(f"从 {cfg.schedule.load_model} 加载了模型权重")

    # JITTOR MIGRATION: 替换 PyTorch Lightning Trainer 为手动训练循环
    
    # 配置优化器和学习率调度器
    optimizer, scheduler = task.configure_optimizers()
    
    # 如果是多 GPU 训练，使用 DataParallel 包装模型
    if jt.world_size > 1:
        task.model = jt.DataParallel(task.model)
        env_utils.set_multi_processing(distributed=True)

    logger.info("开始训练...")
    global_step = 0
    start_epoch = 0 # TODO: 从检查点恢复 epoch

    for epoch in range(start_epoch, cfg.schedule.total_epochs):
        task.on_train_epoch_start(epoch)
        
        # 模拟一个 trainer 对象，以保持与 task 方法的兼容性
        trainer_mock = SimpleNamespace(
            current_epoch=epoch,
            global_step=global_step,
            num_training_batches=len(train_dataloader),
            num_val_batches=len(val_dataloader),
            optimizer=optimizer
        )

        # 训练循环
        for i, batch in enumerate(train_dataloader):
            trainer_mock.global_step = global_step
            loss = task.training_step(batch, i, trainer_mock)
            optimizer.step(loss) # Jittor 的 optimizer.step 会自动处理梯度
            task.on_train_batch_end(global_step)
            global_step += 1
        
        # 更新学习率
        if scheduler:
            scheduler.step()
        
        # 验证循环
        if (epoch + 1) % cfg.schedule.val_intervals == 0:
            logger.info(f"Epoch {epoch + 1} 开始验证...")
            task.on_validation_epoch_start()
            val_outputs = []
            for i, batch in enumerate(val_dataloader):
                trainer_mock.global_step = global_step
                dets = task.validation_step(batch, i, trainer_mock)
                val_outputs.append(dets)
            
            task.validation_epoch_end(val_outputs, epoch)

        # 保存最后一个 epoch 的模型
        if jt.rank == 0:
            task.model.save(os.path.join(cfg.save_dir, "model_last.ckpt"))
            logger.info(f"Epoch {epoch + 1} 已完成，模型已保存。")

    logger.info("训练结束。")


if __name__ == "__main__":
    args = parse_args()
    main(args)
