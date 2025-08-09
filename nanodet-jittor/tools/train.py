# JITTOR MIGRATION by Google LLC.
import sys
import os


# 将项目根目录（nanodet-jittor）添加到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import random
import warnings
from types import SimpleNamespace
import traceback

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
    # 便于快速调试的可选参数
    parser.add_argument("--max_epochs", type=int, default=None, help="最多训练的 epoch 数（可选）")
    parser.add_argument("--max_train_batches", type=int, default=None, help="每个 epoch 训练的最大 batch 数（可选）")
    parser.add_argument("--max_val_batches", type=int, default=None, help="每次验证的最大 batch 数（可选）")
    parser.add_argument("--warmup_steps", type=int, default=0, help="预热步数，仅用于触发JIT编译缓存（不做验证与保存）")
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

    # JITTOR MIGRATION: 移除 PyTorch-Lightning 和 jt.backends.cudnn 的设置
    
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
    logger.info(f"总共 {cfg.schedule.total_epochs} 个 epoch，每个 epoch {len(train_dataloader)} 个批次")
    logger.info("=" * 80)

    # 预热阶段：仅用于触发 JIT 编译缓存
    if args.warmup_steps and args.warmup_steps > 0:
        logger.info(f"🚀 开始预热 warmup，共 {args.warmup_steps} 步（不保存模型/不验证）...")
        task.model.train()
        warmup_losses = []
        warmup_batches = iter(train_dataloader)
        for wi in range(args.warmup_steps):
            try:
                batch = next(warmup_batches)
            except StopIteration:
                warmup_batches = iter(train_dataloader)
                batch = next(warmup_batches)
            try:
                res = task.training_step(batch, wi, SimpleNamespace(current_epoch=0, global_step=wi, optimizer=None, num_training_batches=len(train_dataloader), num_val_batches=len(val_dataloader)))
                warmup_loss = res['loss'] if isinstance(res, dict) else res
                # 只做一次 optimizer.step 以确保反向路径被编译
                optimizer.step(warmup_loss)
                warmup_losses.append(float(warmup_loss))
                if (wi+1) % max(1, args.warmup_steps//5) == 0:
                    logger.info(f"Warmup {wi+1}/{args.warmup_steps} | loss:{np.mean(warmup_losses):.4f}")
            except Exception as e:
                logger.warning(f"Warmup 第 {wi} 步失败: {e}")
                traceback.print_exc()
                continue
        logger.info("✅ 预热完成，开始正式训练")

    global_step = 0
    start_epoch = 0
    best_ap = 0.0

    # 导入时间模块
    import time

    total_epochs = cfg.schedule.total_epochs if args.max_epochs is None else min(cfg.schedule.total_epochs, args.max_epochs)
    for epoch in range(start_epoch, total_epochs):
        epoch_start_time = time.time()
        task.on_train_epoch_start(epoch)

        # 模拟一个 trainer 对象
        trainer_mock = SimpleNamespace(
            current_epoch=epoch,
            global_step=global_step,
            num_training_batches=len(train_dataloader),
            num_val_batches=len(val_dataloader),
            optimizer=optimizer
        )

        # 🎯 PyTorch 风格的训练循环 - 带实时进度显示
        task.model.train()
        epoch_losses = []

        # 获取当前学习率 - Jittor 优化器兼容性修复
        current_lr = 0.001  # 默认值
        try:
            if hasattr(optimizer, 'param_groups') and len(optimizer.param_groups) > 0:
                current_lr = optimizer.param_groups[0]['lr']
            elif hasattr(optimizer, 'lr'):
                current_lr = optimizer.lr
            elif hasattr(optimizer, 'learning_rate'):
                current_lr = optimizer.learning_rate
        except:
            current_lr = 0.001  # 保持默认值

        logger.info(f"开始 Epoch {epoch+1}/{cfg.schedule.total_epochs} | LR: {current_lr:.6f}")

        # 使用简单的进度显示，避免 tqdm 兼容性问题
        batch_count = len(train_dataloader)
        print_interval = max(1, batch_count // 20)  # 每5%显示一次进度

        for i, batch in enumerate(train_dataloader):
            trainer_mock.global_step = global_step

            try:
                # 调试：检查并打印 batch 的关键字段形状，避免动态形状导致的缓存爆炸
                if i < 3:  # 仅前几个batch打印
                    try:
                        img = batch.get("img")
                        if isinstance(img, jt.Var):
                            logger.info(f"Batch {i} img shape: {tuple(img.shape)}")
                        else:
                            logger.info(f"Batch {i} img type: {type(img)}")
                        for k in ["gt_bboxes", "gt_labels", "gt_bboxes_ignore"]:
                            v = batch.get(k)
                            if v is None:
                                continue
                            if isinstance(v, jt.Var):
                                logger.info(f"  {k} jt.Var shape: {tuple(v.shape)}")
                            elif isinstance(v, np.ndarray):
                                logger.info(f"  {k} np shape: {v.shape} dtype:{v.dtype}")
                            else:
                                try:
                                    logger.info(f"  {k} type:{type(v)} len:{len(v)}")
                                except Exception:
                                    logger.info(f"  {k} type:{type(v)}")
                    except Exception as e_dbg:
                        logger.warning(f"Batch {i} debug inspect failed: {e_dbg}")

                # 前向传播和损失计算
                training_result = task.training_step(batch, i, trainer_mock)
                # 🔧 修复：正确处理返回值
                if isinstance(training_result, dict):
                    training_loss = training_result['loss']
                else:
                    training_loss = training_result
                # 🔧 修复：避免计算图断裂，延迟获取损失值用于记录
                loss_value = float(training_loss)  # Jittor 会自动处理
                epoch_losses.append(loss_value)

                # 反向传播
                optimizer.step(training_loss)
                task.on_train_batch_end(global_step)
                global_step += 1

                # 🎯 实时进度显示
                if (i + 1) % print_interval == 0 or i == 0:
                    progress = (i + 1) / batch_count * 100
                    avg_loss = np.mean(epoch_losses)
                    # 遵循用户要求的日志格式
                    # [NanoDet][MM-DD HH:MM:SS]INFO: Train|Epoch1/50|Iter0(1/108)| mem:5.06G| lr:1.00e-06| loss_qfl:...|
                    # Jittor 当前无 used_cuda_mem，可安全置0避免 AttributeError
                    mem_gb = 0.0
                    iter_str = f"Iter{trainer_mock.global_step}({i+1}/{batch_count})"
                    # 当 loss_states 可用时逐项打印；否则回退为 Loss/Avg
                    base = f"Train|Epoch{epoch+1}/{cfg.schedule.total_epochs}|{iter_str}| mem:{mem_gb:.2f}G| lr:{current_lr:.2e}| "
                    try:
                        if isinstance(training_result, dict) and 'loss_states' in training_result:
                            loss_states = training_result['loss_states']
                            for k, v in loss_states.items():
                                try:
                                    v_mean = v.mean() if hasattr(v, 'numel') and v.numel()>1 else v
                                    base += f"{k}:{float(v_mean):.4f}| "
                                except Exception:
                                    pass
                        else:
                            base += f"loss:{loss_value:.4f}| avg:{avg_loss:.4f}| "
                    except Exception:
                        base += f"loss:{loss_value:.4f}| avg:{avg_loss:.4f}| "
                    logger.info(base)

            except Exception as e:
                # 输出更详细的批次关键信息，帮助定位问题
                try:
                    shapes = {}
                    if isinstance(batch, dict):
                        for k, v in batch.items():
                            if isinstance(v, jt.Var):
                                shapes[k] = tuple(v.shape)
                            elif isinstance(v, np.ndarray):
                                shapes[k] = v.shape
                            else:
                                shapes[k] = type(v).__name__
                    logger.error(f"训练批次 {i} 失败: {e}. batch keys: {list(batch.keys()) if isinstance(batch, dict) else type(batch)} shapes: {shapes}")
                except Exception:
                    logger.error(f"训练批次 {i} 失败: {e}")
                # 无论如何都打印完整堆栈，便于定位
                traceback.print_exc()
                continue

            # 早停：限制每个 epoch 的训练 batch 数
            if args.max_train_batches is not None and (i + 1) >= args.max_train_batches:
                break

        # 更新学习率
        if scheduler:
            scheduler.step()

        # 计算 epoch 统计信息
        epoch_time = time.time() - epoch_start_time
        avg_loss = np.mean(epoch_losses)

        # 🎯 PyTorch 风格的 epoch 总结
        logger.info(f"Epoch {epoch+1:3d}/{cfg.schedule.total_epochs} | "
                   f"Loss: {avg_loss:.4f} | "
                   f"Time: {epoch_time:.1f}s | "
                   f"LR: {current_lr:.6f}")

        # 验证和测评
        if (epoch + 1) % cfg.schedule.val_intervals == 0:
            logger.info(f"🔍 开始验证 Epoch {epoch + 1}...")

            task.model.eval()
            task.on_validation_epoch_start()
            val_outputs = []

            # 验证循环 - 简单进度显示
            val_batch_count = len(val_dataloader)
            val_print_interval = max(1, val_batch_count // 10)

            for i, batch in enumerate(val_dataloader):
                trainer_mock.global_step = global_step
                try:
                    with jt.no_grad():
                        dets = task.validation_step(batch, i, trainer_mock)
                        val_outputs.append(dets)

                    # 验证进度显示
                    if (i + 1) % val_print_interval == 0 or i == 0:
                        val_progress = (i + 1) / val_batch_count * 100
                        print(f"  验证进度: [{i+1:4d}/{val_batch_count}] ({val_progress:5.1f}%)", flush=True)

                except Exception as e:
                    logger.error(f"验证批次 {i} 失败: {e}")
                    continue

                # 早停：限制验证 batch 数
                if args.max_val_batches is not None and (i + 1) >= args.max_val_batches:
                    break

            # 🎯 自动调用测评工具
            try:
                metrics = task.validation_epoch_end(val_outputs, epoch)

                # 提取关键指标
                if metrics and 'mAP' in metrics:
                    current_ap = metrics['mAP']
                    logger.info(f"📊 验证结果 | mAP: {current_ap:.4f}")

                    # 保存最佳模型
                    if current_ap > best_ap:
                        best_ap = current_ap
                        if jt.rank == 0:
                            best_model_path = os.path.join(cfg.save_dir, "model_best.ckpt")
                            task.model.save(best_model_path)
                            logger.info(f"🏆 新的最佳模型！mAP: {best_ap:.4f} -> {best_model_path}")
                else:
                    logger.info("📊 验证完成，但未获取到 mAP 指标")

            except Exception as e:
                logger.error(f"验证评估失败: {e}")

            logger.info("-" * 80)

        # 保存最新模型
        if jt.rank == 0:
            task.model.save(os.path.join(cfg.save_dir, "model_last.ckpt"))

    # 🎯 训练完成总结
    logger.info("=" * 80)
    logger.info("🎉 训练完成！")
    logger.info(f"📊 最佳 mAP: {best_ap:.4f}")
    logger.info(f"💾 模型保存在: {cfg.save_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    args = parse_args()
    main(args)
