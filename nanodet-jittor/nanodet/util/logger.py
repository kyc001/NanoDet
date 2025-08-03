import logging
import os
import time

import numpy as np
# JITTOR MIGRATION: 导入 jittor 用于分布式环境判断
import jittor as jt
from termcolor import colored

from .path import mkdir

# JITTOR MIGRATION: 定义一个 rank_zero_only 装饰器，用于替换 PyTorch Lightning 的功能
def rank_zero_only(fn):
    """一个装饰器，确保函数只在 rank 0 的进程上执行。"""
    def wrapper(*args, **kwargs):
        # 在 Jittor 中，jt.rank 用于获取当前进程的排名
        if not (jt.world_size > 1 and jt.rank != 0):
            return fn(*args, **kwargs)
    return wrapper


class Logger:
    """一个通用的日志记录器，已适配 Jittor 环境。"""
    def __init__(self, save_dir="./", use_tensorboard=True):
        # JITTOR MIGRATION: 使用 jt.rank 获取当前进程的排名
        self.rank = jt.rank if jt.world_size > 1 else 0
        
        # mkdir 应该只在主进程上执行
        if self.rank == 0:
            mkdir(save_dir)
            
        fmt = (
            colored("[%(name)s]", "magenta", attrs=["bold"])
            + colored("[%(asctime)s]", "blue")
            + colored("%(levelname)s:", "green")
            + colored("%(message)s", "white")
        )
        
        # 日志文件只在主进程上创建
        log_file = os.path.join(save_dir, "logs.txt") if self.rank == 0 else os.devnull
        logging.basicConfig(
            level=logging.INFO,
            filename=log_file,
            filemode="w",
        )
        
        self.log_dir = os.path.join(save_dir, "logs")
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter(fmt, datefmt="%m-%d %H:%M:%S")
        console.setFormatter(formatter)
        logging.getLogger().addHandler(console)
        
        self.writer = None
        if use_tensorboard and self.rank == 0:
            try:
                # TensorBoard 的 SummaryWriter 是一个常用的独立工具
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                raise ImportError(
                    '请运行 "pip install tensorboard" 来安装 TensorBoard'
                ) from None
            logging.info(
                f"使用 Tensorboard，日志将保存在 {self.log_dir}"
            )
            self.writer = SummaryWriter(log_dir=self.log_dir)

    @rank_zero_only
    def log(self, string):
        logging.info(string)

    @rank_zero_only
    def scalar_summary(self, tag, phase, value, step):
        if self.writer:
            self.writer.add_scalars(tag, {phase: value}, step)


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
    一个为 Jittor 定制的日志记录器，其接口和行为模仿了原版的 NanoDetLightningLogger。
    """
    def __init__(self, save_dir="./", **kwargs):
        super().__init__()
        self._name = "NanoDet"
        self._version = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self._save_dir = os.path.join(save_dir, f"logs-{self._version}")

        # JITTOR MIGRATION: 直接使用 os.makedirs，不再需要 lightning_fabric
        if jt.rank == 0:
            os.makedirs(self._save_dir, exist_ok=True)
            
        self._init_logger()
        self._experiment = None
        self._kwargs = kwargs

    @property
    def name(self):
        return self._name

    @property
    def experiment(self):
        """
        实际的 TensorBoard SummaryWriter 对象。
        """
        if self._experiment is not None:
            return self._experiment

        if jt.rank != 0:
            return None

        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            raise ImportError(
                '请运行 "pip install tensorboard" 来安装 TensorBoard'
            ) from None

        self._experiment = SummaryWriter(log_dir=self._save_dir, **self._kwargs)
        return self._experiment

    @property
    def version(self):
        return self._version

    def _init_logger(self):
        """初始化 Python 的 logging 模块。"""
        self.logger = logging.getLogger(name=self.name)
        self.logger.setLevel(logging.INFO)

        # 避免在非主进程上重复添加 handlers
        if self.logger.hasHandlers() and jt.rank != 0:
             return

        # 文件处理器
        fh = logging.FileHandler(os.path.join(self._save_dir, "logs.txt"))
        fh.setLevel(logging.INFO)
        f_fmt = "[%(name)s][%(asctime)s]%(levelname)s: %(message)s"
        file_formatter = logging.Formatter(f_fmt, datefmt="%m-%d %H:%M:%S")
        fh.setFormatter(file_formatter)

        # 控制台处理器
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        c_fmt = (
            colored("[%(name)s]", "magenta", attrs=["bold"])
            + colored("[%(asctime)s]", "blue")
            + colored("%(levelname)s:", "green")
            + colored("%(message)s", "white")
        )
        console_formatter = logging.Formatter(c_fmt, datefmt="%m-%d %H:%M:%S")
        ch.setFormatter(console_formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    @rank_zero_only
    def info(self, string):
        self.logger.info(string)

    @rank_zero_only
    def log(self, string):
        self.logger.info(string)

    @rank_zero_only
    def dump_cfg(self, cfg_node):
        # 假设 cfg_node 有一个 dump 方法
        with open(os.path.join(self._save_dir, "train_cfg.yml"), "w") as f:
            cfg_node.dump(stream=f)

    @rank_zero_only
    def log_hyperparams(self, params):
        self.logger.info(f"hyperparams: {params}")

    @rank_zero_only
    def log_metrics(self, metrics, step):
        self.logger.info(f"Val_metrics: {metrics}")
        if self.experiment:
            for k, v in metrics.items():
                self.experiment.add_scalars("Val_metrics/" + k, {"Val": v}, step)

    # JITTOR MIGRATION: 新增 save 方法
    @rank_zero_only
    def save(self):
        """将所有待处理的日志事件写入磁盘。"""
        if self.experiment:
            self.experiment.flush()

    @rank_zero_only
    def finalize(self, status):
        if self.experiment:
            self.experiment.flush()
            self.experiment.close()
