import logging
import os
import time
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import jittor as jt

from .path import mkdir

def rank_zero_only(fn):
    """装饰器：确保函数只在主进程上执行"""
    def wrapper(*args, **kwargs):
        # Jittor分布式环境检查
        if hasattr(jt, 'world_size') and jt.world_size > 1:
            if hasattr(jt, 'rank') and jt.rank != 0:
                return None
        return fn(*args, **kwargs)
    return wrapper


class Logger:
    """Jittor版本的简洁日志记录器，移除tensorboard依赖"""
    def __init__(self, save_dir="./", name="NanoDet"):
        self.rank = getattr(jt, 'rank', 0)
        self.save_dir = save_dir
        self.name = name

        # 只在主进程创建目录和文件
        if self.rank == 0:
            mkdir(save_dir)
            self.log_file = os.path.join(save_dir, "training.log")
            self.metrics_file = os.path.join(save_dir, "metrics.json")
            self.metrics_data = []

        # 设置日志格式
        self._setup_logging()

    def _setup_logging(self):
        """设置日志系统"""
        # 创建logger
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.INFO)

        # 避免重复添加handler
        if self.logger.handlers:
            return

        # 控制台输出格式
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # 简化的格式，不依赖termcolor
        console_format = "[%(name)s][%(asctime)s] %(levelname)s: %(message)s"
        console_formatter = logging.Formatter(console_format, datefmt="%m-%d %H:%M:%S")
        console_handler.setFormatter(console_formatter)

        self.logger.addHandler(console_handler)

        # 文件输出（只在主进程）
        if self.rank == 0:
            file_handler = logging.FileHandler(self.log_file, mode='w')
            file_handler.setLevel(logging.INFO)
            file_formatter = logging.Formatter(
                "[%(asctime)s] %(levelname)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

    @rank_zero_only
    def info(self, message):
        """记录信息日志"""
        self.logger.info(message)

    @rank_zero_only
    def log(self, message):
        """记录日志（兼容旧接口）"""
        self.info(message)

    @rank_zero_only
    def warning(self, message):
        """记录警告日志"""
        self.logger.warning(message)

    @rank_zero_only
    def error(self, message):
        """记录错误日志"""
        self.logger.error(message)

    @rank_zero_only
    def log_metrics(self, metrics, step):
        """记录训练指标到JSON文件"""
        if self.rank == 0:
            metric_entry = {
                'step': step,
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics
            }
            self.metrics_data.append(metric_entry)

            # 保存到JSON文件
            with open(self.metrics_file, 'w') as f:
                json.dump(self.metrics_data, f, indent=2)

            # 同时记录到日志
            metrics_str = ', '.join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            self.info(f"Step {step} - {metrics_str}")

    @rank_zero_only
    def scalar_summary(self, tag, phase, value, step):
        """记录标量指标（兼容旧接口）"""
        self.log_metrics({f"{tag}_{phase}": value}, step)


class MovingAverage(object):
    """计算移动平均值的类。框架无关，无需修改。"""
    def __init__(self, val, window_size=50):
        self.window_size = window_size
        self.reset()
        self.push(val)

    def reset(self):
        self.queue = []

    def push(self, val):
        self.queue.append(val)
        if len(self.queue) > self.window_size:
            self.queue.pop(0)

    def avg(self):
        return np.mean(self.queue)


class AverageMeter(object):
    """计算并存储平均值和当前值的类。框架无关，无需修改。"""
    def __init__(self, val=0): # 允许无初始值创建
        self.reset()
        if val != 0:
            self.update(val)

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count


class NanoDetLightningLogger:
    """
    Jittor版本的NanoDet日志记录器，移除Lightning依赖
    """
    def __init__(self, save_dir="./", name="NanoDet", **kwargs):
        self._name = name
        self._version = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self._save_dir = os.path.join(save_dir, f"logs-{self._version}")
        self.rank = getattr(jt, 'rank', 0)

        # 只在主进程创建目录
        if self.rank == 0:
            os.makedirs(self._save_dir, exist_ok=True)

        self._init_logger()
        self._metrics_data = []
        self._kwargs = kwargs

    @property
    def name(self):
        return self._name

    @property
    def version(self):
        return self._version

    @property
    def save_dir(self):
        return self._save_dir

    def _init_logger(self):
        """初始化日志系统"""
        self.logger = logging.getLogger(name=self.name)
        self.logger.setLevel(logging.INFO)

        # 避免重复添加handlers
        if self.logger.handlers:
            return

        # 文件处理器（只在主进程）
        if self.rank == 0:
            fh = logging.FileHandler(os.path.join(self._save_dir, "training.log"))
            fh.setLevel(logging.INFO)
            f_fmt = "[%(name)s][%(asctime)s] %(levelname)s: %(message)s"
            file_formatter = logging.Formatter(f_fmt, datefmt="%Y-%m-%d %H:%M:%S")
            fh.setFormatter(file_formatter)
            self.logger.addHandler(fh)

        # 控制台处理器
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        # 简化格式，不依赖termcolor
        c_fmt = "[%(name)s][%(asctime)s] %(levelname)s: %(message)s"
        console_formatter = logging.Formatter(c_fmt, datefmt="%m-%d %H:%M:%S")
        ch.setFormatter(console_formatter)
        self.logger.addHandler(ch)

    @rank_zero_only
    def info(self, message):
        """记录信息日志"""
        self.logger.info(message)

    @rank_zero_only
    def log(self, message):
        """记录日志"""
        self.logger.info(message)

    @rank_zero_only
    def warning(self, message):
        """记录警告日志"""
        self.logger.warning(message)

    @rank_zero_only
    def error(self, message):
        """记录错误日志"""
        self.logger.error(message)

    @rank_zero_only
    def dump_cfg(self, cfg_node):
        """保存配置文件"""
        try:
            config_file = os.path.join(self._save_dir, "train_config.yml")
            if hasattr(cfg_node, 'dump'):
                with open(config_file, "w") as f:
                    cfg_node.dump(stream=f)
            else:
                # 如果没有dump方法，尝试转换为字符串保存
                with open(config_file, "w") as f:
                    f.write(str(cfg_node))
            self.info(f"配置文件已保存到: {config_file}")
        except Exception as e:
            self.warning(f"保存配置文件失败: {e}")

    @rank_zero_only
    def log_hyperparams(self, params):
        """记录超参数"""
        self.info(f"超参数: {params}")

        # 保存超参数到JSON文件
        try:
            hyperparams_file = os.path.join(self._save_dir, "hyperparams.json")
            with open(hyperparams_file, 'w') as f:
                json.dump(params, f, indent=2, default=str)
        except Exception as e:
            self.warning(f"保存超参数失败: {e}")

    @rank_zero_only
    def log_metrics(self, metrics, step):
        """记录验证指标"""
        # 记录到日志
        metrics_str = ', '.join([f"{k}: {v:.4f}" if isinstance(v, (int, float)) else f"{k}: {v}"
                                for k, v in metrics.items()])
        self.info(f"Step {step} - 验证指标: {metrics_str}")

        # 保存到JSON文件
        try:
            metric_entry = {
                'step': step,
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics
            }
            self._metrics_data.append(metric_entry)

            metrics_file = os.path.join(self._save_dir, "validation_metrics.json")
            with open(metrics_file, 'w') as f:
                json.dump(self._metrics_data, f, indent=2, default=str)
        except Exception as e:
            self.warning(f"保存验证指标失败: {e}")

    @rank_zero_only
    def save(self):
        """保存所有日志数据"""
        # 强制刷新所有handlers
        for handler in self.logger.handlers:
            handler.flush()

    @rank_zero_only
    def finalize(self, status="completed"):
        """完成日志记录"""
        self.info(f"训练完成，状态: {status}")
        self.save()


class MetricsVisualizer:
    """
    简单的指标可视化工具，替代tensorboard功能
    """
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.metrics_history = {}

    def add_scalar(self, tag, value, step):
        """添加标量指标"""
        if tag not in self.metrics_history:
            self.metrics_history[tag] = {'steps': [], 'values': []}

        self.metrics_history[tag]['steps'].append(step)
        self.metrics_history[tag]['values'].append(value)

    def plot_metrics(self, save_plots=True):
        """绘制指标曲线"""
        try:
            import matplotlib.pyplot as plt

            if not self.metrics_history:
                return

            # 创建子图
            num_metrics = len(self.metrics_history)
            if num_metrics == 0:
                return

            fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 4 * num_metrics))
            if num_metrics == 1:
                axes = [axes]

            for i, (tag, data) in enumerate(self.metrics_history.items()):
                axes[i].plot(data['steps'], data['values'], 'b-', linewidth=2)
                axes[i].set_title(f'{tag}')
                axes[i].set_xlabel('Step')
                axes[i].set_ylabel('Value')
                axes[i].grid(True, alpha=0.3)

            plt.tight_layout()

            if save_plots:
                plot_file = os.path.join(self.save_dir, 'training_metrics.png')
                plt.savefig(plot_file, dpi=150, bbox_inches='tight')
                plt.close()
                return plot_file
            else:
                plt.show()

        except ImportError:
            print("matplotlib未安装，无法生成图表")
            return None
        except Exception as e:
            print(f"生成图表失败: {e}")
            return None

    def save_metrics(self):
        """保存指标数据到JSON文件"""
        try:
            metrics_file = os.path.join(self.save_dir, 'metrics_history.json')
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
            return metrics_file
        except Exception as e:
            print(f"保存指标数据失败: {e}")
            return None


def create_logger(save_dir="./logs", name="NanoDet", use_lightning_logger=True):
    """
    创建日志记录器的工厂函数

    Args:
        save_dir: 日志保存目录
        name: 日志记录器名称
        use_lightning_logger: 是否使用Lightning风格的日志记录器

    Returns:
        日志记录器实例
    """
    if use_lightning_logger:
        return NanoDetLightningLogger(save_dir=save_dir, name=name)
    else:
        return Logger(save_dir=save_dir, name=name)
