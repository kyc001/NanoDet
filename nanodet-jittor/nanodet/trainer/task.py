import copy
import json
import os
import warnings
from typing import Any, Dict, List
import logging

# JITTOR HIGH-FIDELITY MOD: 导入 Jittor 核心库
import jittor as jt

# 假设这些外部函数都已经被正确地迁移到了 Jittor
from nanodet.data.batch_process import stack_batch_img
from nanodet.optim import build_optimizer
from nanodet.util import convert_avg_params, gather_results_jittor as gather_results, mkdir

from ..model.arch import build_model
from ..model.weight_averager import build_weight_averager

# JITTOR HIGH-FIDELITY MOD: 定义一个 rank_zero_only 装饰器，模拟 Lightning 的功能，代码更清晰
def rank_zero_only(fn):
    """一个装饰器，确保函数只在 rank 0 的进程上执行。"""
    def wrapper(*args, **kwargs):
        if not (jt.world_size > 1 and jt.rank != 0):
            return fn(*args, **kwargs)
    return wrapper

# JITTOR HIGH-FIDELITY MOD: 继承自 jt.Module，这是 Jittor 中所有模型的基类
class TrainingTask(jt.Module):
    """
    一个通用的 Jittor 训练任务模块。
    此版本旨在高度复原 PyTorch Lightning 的设计，将训练、验证等步骤定义为独立方法，
    由外部的训练循环（即模拟的 Trainer）来调用。

    Args:
        cfg: 训练设定。
        evaluator: 用于评估模型性能的评估器。
        logger (optional): 一个配置好的日志记录器，用于输出信息和指标。
    """

    def __init__(self, cfg, evaluator=None, logger=None):
        super(TrainingTask, self).__init__()
        self.cfg = cfg
        self.model = build_model(cfg.model)
        self.evaluator = evaluator
        self.save_flag = -10
        self.log_style = "NanoDet"
        
        # JITTOR HIGH-FIDELITY MOD: 使用标准日志模块，或从外部传入一个更复杂的 logger (如 Tensorboard)
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
        
        self.weight_averager = None
        if "weight_averager" in cfg.model:
            self.weight_averager = build_weight_averager(cfg.model.weight_averager)
            # JITTOR HIGH-FIDELITY MOD: 使用 copy.deepcopy 来创建 avg_model，确保完全独立，这比 .clone() 更可靠
            self.avg_model = copy.deepcopy(self.model)

    def _preprocess_batch_input(self, batch):
        """预处理批次输入。在 Jittor 中，数据已在正确的设备上，无需手动 .to(device)。"""
        batch_imgs = batch["img"]
        if isinstance(batch_imgs, list):
            batch_img_tensor = stack_batch_img(batch_imgs, divisible=32)
            batch["img"] = batch_img_tensor
        return batch

    # JITTOR HIGH-FIDELITY MOD: Jittor 的 forward 函数标准名称是 execute
    def execute(self, x):
        """模型的前向传播。"""
        return self.model(x)

    # JITTOR HIGH-FIDELITY MOD: 使用 @jt.no_grad() 装饰器来关闭梯度计算
    @jt.no_grad()
    def predict(self, batch):
        """预测函数，在无梯度的上下文中执行。"""
        batch = self._preprocess_batch_input(batch)
        preds = self.execute(batch["img"])
        results = self.model.head.post_process(preds, batch)
        return results

    def training_step(self, batch, batch_idx, trainer):
        """
        单个训练步骤，职责与 Lightning 完全对齐：仅计算并返回损失。
        日志记录也在此处完成。
        Args:
            trainer: 一个模拟的 Trainer 对象，用于获取当前状态如 epoch, global_step, optimizer 等。
        """
        batch = self._preprocess_batch_input(batch)
        preds, loss, loss_states = self.model.forward_train(batch)

        # JITTOR HIGH-FIDELITY MOD: 日志记录逻辑保持不变，但依赖外部 trainer 提供状态
        if trainer.global_step % self.cfg.log.interval == 0:
            # memory = jt.flags.used_cuda_mem / 1e9 if jt.flags.use_cuda else 0
            memory = 0
            lr = trainer.optimizer.lr
            log_msg = "Train|Epoch{}/{}|Iter{}({}/{})| mem:{:.3g}G| lr:{:.2e}| ".format(
                trainer.current_epoch + 1,
                self.cfg.schedule.total_epochs,
                trainer.global_step,
                batch_idx + 1,
                trainer.num_training_batches,
                memory,
                lr,
            )
            self.scalar_summary("Train_loss/lr", "Train", lr, trainer.global_step)
            for loss_name in loss_states:
                loss_value = loss_states[loss_name].mean().item()
                log_msg += "{}:{:.4f}| ".format(loss_name, loss_value)
                self.scalar_summary(
                    "Train_loss/" + loss_name, "Train", loss_value, trainer.global_step
                )
            self.info(log_msg)

        return loss

    def validation_step(self, batch, batch_idx, trainer):
        """单个验证步骤。在无梯度下执行，并返回检测结果。"""
        batch = self._preprocess_batch_input(batch)
        
        model_to_eval = self.avg_model if self.weight_averager else self.model
        
        with jt.no_grad():
            preds, loss, loss_states = model_to_eval.forward_train(batch)
            dets = model_to_eval.head.post_process(preds, batch)

        # 日志记录
        if batch_idx % self.cfg.log.interval == 0:
            memory = jt.flags.used_cuda_mem / 1e9 if jt.flags.use_cuda else 0
            lr = trainer.optimizer.param_groups[0]["lr"]
            log_msg = "Val|Epoch{}/{}|Iter{}({}/{})| mem:{:.3g}G| lr:{:.2e}| ".format(
                trainer.current_epoch + 1,
                self.cfg.schedule.total_epochs,
                trainer.global_step,
                batch_idx + 1,
                trainer.num_val_batches,
                memory,
                lr,
            )
            for loss_name in loss_states:
                log_msg += "{}:{:.4f}| ".format(loss_name, loss_states[loss_name].mean().item())
            self.info(log_msg)

        return dets

    def validation_epoch_end(self, validation_step_outputs, current_epoch):
        """验证 epoch 结束后的操作，汇总结果、评估并保存最佳模型。"""
        results = {}
        for res in validation_step_outputs:
            results.update(res)
        
        # JITTOR HIGH-FIDELITY MOD: 使用 jt.world_size > 1 判断是否为分布式环境
        all_results = gather_results(results) if jt.world_size > 1 else results
        
        if jt.rank == 0 and all_results:
            eval_results = self.evaluator.evaluate(all_results, self.cfg.save_dir)
            self.log_metrics(eval_results, current_epoch + 1)
            
            metric = eval_results.get(self.cfg.evaluator.save_key)
            if metric is None:
                warnings.warn(f"Warning! Save_key '{self.cfg.evaluator.save_key}' is not in eval results! Only save model last!")
                return

            if metric > self.save_flag:
                self.save_flag = metric
                best_save_path = os.path.join(self.cfg.save_dir, "model_best")
                mkdir(best_save_path) # mkdir 只在 rank 0 执行
                
                # JITTOR HIGH-FIDELITY MOD: 保存模型状态和检查点
                self.save_model_state(os.path.join(best_save_path, "nanodet_model_best.pth"))
                self.model.save(os.path.join(best_save_path, "model_best.ckpt")) # 保存Jittor的检查点
                
                txt_path = os.path.join(best_save_path, "eval_results.txt")
                with open(txt_path, "a") as f:
                    f.write(f"Epoch:{current_epoch + 1}\n")
                    for k, v in eval_results.items():
                        f.write(f"{k}: {v}\n")
        elif not all_results:
            self.info(f"Skip val on rank {jt.rank}")

    def test_step(self, batch):
        """单个测试步骤。"""
        return self.predict(batch)

    def test_epoch_end(self, test_step_outputs):
        """测试 epoch 结束后的操作，保存结果为 json 并评估。"""
        results = {}
        for res in test_step_outputs:
            results.update(res)
        
        all_results = gather_results(results) if jt.world_size > 1 else results
        
        if jt.rank == 0 and all_results:
            res_json = self.evaluator.results2json(all_results)
            json_path = os.path.join(self.cfg.save_dir, "results.json")
            with open(json_path, "w") as f:
                json.dump(res_json, f)

            if self.cfg.test_mode == "val":
                eval_results = self.evaluator.evaluate(all_results, self.cfg.save_dir)
                txt_path = os.path.join(self.cfg.save_dir, "eval_results.txt")
                with open(txt_path, "a") as f:
                    for k, v in eval_results.items():
                        f.write(f"{k}: {v}\n")
        elif not all_results:
            self.info(f"Skip test on rank {jt.rank}")

    # JITTOR HIGH-FIDELITY MOD: 将优化器和调度器的配置分开，与 Lightning 的设计更一致
    def configure_optimizers(self):
        """配置优化器和学习率调度器。"""
        optimizer_cfg = copy.deepcopy(self.cfg.schedule.optimizer)
        optimizer = build_optimizer(self.model, optimizer_cfg)

        schedule_cfg = copy.deepcopy(self.cfg.schedule.lr_schedule)
        name = schedule_cfg.pop("name")
        build_scheduler = getattr(jt.lr_scheduler, name)
        scheduler = build_scheduler(optimizer=optimizer, **schedule_cfg)
        
        return optimizer, scheduler

    def optimizer_step(self, optimizer, loss, global_step):
        """
        执行单个优化步骤，完整复刻了 Lightning 的 optimizer_step hook 的功能，
        包括学习率预热和参数更新。
        """
        # 学习率预热 (Warm up)
        if global_step <= self.cfg.schedule.warmup.steps:
            warmup_cfg = self.cfg.schedule.warmup
            if warmup_cfg.name == "constant":
                k = warmup_cfg.ratio
            elif warmup_cfg.name == "linear":
                k = 1 - (1 - global_step / warmup_cfg.steps) * (1 - warmup_cfg.ratio)
            elif warmup_cfg.name == "exp":
                k = warmup_cfg.ratio ** (1 - global_step / warmup_cfg.steps)
            else:
                raise Exception("Unsupported warm up type!")
            
            for pg in optimizer.param_groups:
                # 首次执行时，保存初始学习率
                if "initial_lr" not in pg:
                    pg["initial_lr"] = pg["lr"]
                pg["lr"] = pg["initial_lr"] * k
        
        # JITTOR HIGH-FIDELITY MOD: Jittor 的 optimizer.step(loss) 会自动完成梯度计算、参数更新和梯度清零
        optimizer.step(loss)

    @rank_zero_only
    def scalar_summary(self, tag, phase, value, step):
        """
        写入 Tensorboard 标量日志。
        依赖于传入的 logger 对象有一个名为 'experiment' 的 SummaryWriter 实例。
        """
        if hasattr(self.logger, 'experiment') and hasattr(self.logger.experiment, 'add_scalars'):
            self.logger.experiment.add_scalars(tag, {phase: value}, step)

    @rank_zero_only
    def log_metrics(self, metrics, step):
        """记录评估指标。"""
        if hasattr(self.logger, 'experiment') and hasattr(self.logger.experiment, 'add_scalars'):
            self.logger.experiment.add_scalars("Val_metrics", metrics, step)

    @rank_zero_only
    def info(self, string):
        self.logger.info(string)

    @rank_zero_only
    def save_model_state(self, path):
        """
        只在 rank 0 进程上储存模型状态。
        JITTOR HIGH-FIDELITY MOD: 严格按照 Lightning 的格式保存，以备后续使用。
        """
        self.info(f"Saving model to {path}")
        state_dict = (
            self.weight_averager.state_dict()
            if self.weight_averager
            else self.model.state_dict()
        )
        # JITTOR HIGH-FIDELITY MOD: 保存为字典格式，与 PyTorch 保持一致
        jt.save({"state_dict": state_dict}, path)

    # ------------模拟 Hooks (由外部训练循环调用)-----------------
    # JITTOR HIGH-FIDELITY MOD: 这些方法模拟了 Lightning 的回调钩子，应由外部循环在正确时机调用
    
    def on_fit_start(self):
        """训练开始前调用。"""
        if "weight_averager" in self.cfg.model:
            self.info("Weight Averaging is enabled")
            if self.weight_averager and self.weight_averager.has_inited():
                return
            self.weight_averager = build_weight_averager(self.cfg.model.weight_averager)
            self.weight_averager.load_from(self.model)

    def on_train_epoch_start(self, current_epoch):
        """训练 epoch 开始前调用。"""
        if hasattr(self.model, 'set_epoch'):
            self.model.set_epoch(current_epoch)

    def on_train_batch_end(self, global_step):
        """训练 batch 结束后调用。"""
        if self.weight_averager:
            self.weight_averager.update(self.model, global_step)

    def on_validation_epoch_start(self):
        """验证 epoch 开始前调用。"""
        if self.weight_averager:
            self.weight_averager.apply_to(self.avg_model)

    def on_test_epoch_start(self):
        """测试 epoch 开始前调用。"""
        if self.weight_averager:
            # JITTOR HIGH-FIDELITY MOD: 从模型当前状态加载，而不是整个检查点
            self.on_load_checkpoint({"state_dict": self.model.state_dict()})
            self.weight_averager.apply_to(self.model)

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]):
        """加载检查点时调用。"""
        if "weight_averager" in self.cfg.model:
            # JITTOR HIGH-FIDELITY MOD: 传入的是整个检查点字典
            avg_params = convert_avg_params(checkpoint)
            if not avg_params:
                self.info("Weight averaging is enabled, but no average state found in checkpoint.")
                return

            if len(avg_params) != len(self.model.state_dict()):
                self.info("Weight averaging is enabled but average state does not match the model")
            else:
                if self.weight_averager is None:
                    self.weight_averager = build_weight_averager(self.cfg.model.weight_averager)
                self.weight_averager.load_state_dict(avg_params)
                self.info("Loaded average state from checkpoint.")
