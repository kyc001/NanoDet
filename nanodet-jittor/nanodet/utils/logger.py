#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NanoDet Jittor版本日志系统
模块化设计，与PyTorch版本保持一致的接口
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union
import json


class NanoDetLogger:
    """NanoDet专用日志器"""
    
    def __init__(self, 
                 name: str = "nanodet",
                 log_level: str = "INFO",
                 log_dir: Optional[str] = None,
                 console_output: bool = True,
                 file_output: bool = True):
        """
        初始化日志器
        
        Args:
            name: 日志器名称
            log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_dir: 日志文件目录
            console_output: 是否输出到控制台
            file_output: 是否输出到文件
        """
        self.name = name
        self.log_level = getattr(logging, log_level.upper())
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.console_output = console_output
        self.file_output = file_output
        
        # 创建日志目录
        if self.file_output:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化日志器
        self.logger = self._setup_logger()
        
        # 性能统计
        self.stats = {
            'start_time': time.time(),
            'epoch_times': [],
            'batch_times': [],
            'losses': [],
            'metrics': {}
        }
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志器"""
        logger = logging.getLogger(self.name)
        logger.setLevel(self.log_level)
        
        # 清除已有的处理器
        logger.handlers.clear()
        
        # 创建格式器
        formatter = logging.Formatter(
            fmt='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 控制台处理器
        if self.console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # 文件处理器
        if self.file_output:
            # 主日志文件
            log_file = self.log_dir / f"{self.name}.log"
            file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            file_handler.setLevel(self.log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            # 错误日志文件
            error_file = self.log_dir / f"{self.name}_error.log"
            error_handler = logging.FileHandler(error_file, mode='a', encoding='utf-8')
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(formatter)
            logger.addHandler(error_handler)
        
        return logger
    
    def info(self, message: str, **kwargs):
        """信息日志"""
        self.logger.info(message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """调试日志"""
        self.logger.debug(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """警告日志"""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """错误日志"""
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """严重错误日志"""
        self.logger.critical(message, **kwargs)
    
    def log_model_info(self, model, input_shape: tuple = (1, 3, 320, 320)):
        """记录模型信息"""
        self.info("=" * 60)
        self.info("模型信息")
        self.info("=" * 60)
        
        # 模型结构信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.info(f"模型类型: {model.__class__.__name__}")
        self.info(f"输入形状: {input_shape}")
        self.info(f"总参数数: {total_params:,}")
        self.info(f"可训练参数数: {trainable_params:,}")
        self.info(f"参数大小: {total_params * 4 / 1024 / 1024:.2f} MB")
        
        # 保存模型结构到文件
        if self.file_output:
            model_info_file = self.log_dir / "model_info.txt"
            with open(model_info_file, 'w', encoding='utf-8') as f:
                f.write(f"模型结构:\n{str(model)}\n\n")
                f.write(f"总参数数: {total_params:,}\n")
                f.write(f"可训练参数数: {trainable_params:,}\n")
    
    def log_training_start(self, config: Dict[str, Any]):
        """记录训练开始"""
        self.info("=" * 60)
        self.info("开始训练")
        self.info("=" * 60)
        
        # 记录配置信息
        self.info("训练配置:")
        for key, value in config.items():
            self.info(f"  {key}: {value}")
        
        # 保存配置到文件
        if self.file_output:
            config_file = self.log_dir / "training_config.json"
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        
        self.stats['start_time'] = time.time()
    
    def log_epoch_start(self, epoch: int, total_epochs: int):
        """记录epoch开始"""
        self.info(f"Epoch [{epoch}/{total_epochs}] 开始")
        self.stats['epoch_start_time'] = time.time()
    
    def log_epoch_end(self, epoch: int, total_epochs: int, 
                     train_loss: float, val_loss: Optional[float] = None,
                     metrics: Optional[Dict[str, float]] = None):
        """记录epoch结束"""
        epoch_time = time.time() - self.stats['epoch_start_time']
        self.stats['epoch_times'].append(epoch_time)
        self.stats['losses'].append(train_loss)
        
        # 记录epoch结果
        log_msg = f"Epoch [{epoch}/{total_epochs}] 完成 - "
        log_msg += f"训练损失: {train_loss:.6f}, "
        log_msg += f"时间: {epoch_time:.2f}s"
        
        if val_loss is not None:
            log_msg += f", 验证损失: {val_loss:.6f}"
        
        if metrics:
            for metric_name, metric_value in metrics.items():
                log_msg += f", {metric_name}: {metric_value:.6f}"
                
                # 保存最佳指标
                if metric_name not in self.stats['metrics']:
                    self.stats['metrics'][metric_name] = []
                self.stats['metrics'][metric_name].append(metric_value)
        
        self.info(log_msg)
        
        # 保存训练统计
        if self.file_output:
            self._save_training_stats()
    
    def log_batch(self, epoch: int, batch_idx: int, total_batches: int,
                  loss: float, lr: float, batch_time: float):
        """记录batch信息"""
        self.stats['batch_times'].append(batch_time)
        
        if batch_idx % 10 == 0:  # 每10个batch记录一次
            samples_per_sec = 1.0 / batch_time if batch_time > 0 else 0
            log_msg = f"Epoch [{epoch}] Batch [{batch_idx}/{total_batches}] - "
            log_msg += f"损失: {loss:.6f}, 学习率: {lr:.6f}, "
            log_msg += f"时间: {batch_time:.3f}s, 速度: {samples_per_sec:.1f} batch/s"
            
            self.debug(log_msg)
    
    def log_validation(self, metrics: Dict[str, float]):
        """记录验证结果"""
        self.info("验证结果:")
        for metric_name, metric_value in metrics.items():
            self.info(f"  {metric_name}: {metric_value:.6f}")
    
    def log_model_performance(self, performance_data: Dict[str, Any]):
        """记录模型性能数据"""
        self.info("=" * 60)
        self.info("模型性能评估")
        self.info("=" * 60)
        
        for key, value in performance_data.items():
            if isinstance(value, float):
                self.info(f"{key}: {value:.6f}")
            else:
                self.info(f"{key}: {value}")
        
        # 保存性能数据
        if self.file_output:
            perf_file = self.log_dir / "performance.json"
            with open(perf_file, 'w', encoding='utf-8') as f:
                json.dump(performance_data, f, indent=2, ensure_ascii=False)
    
    def log_cross_validation_results(self, results: Dict[str, Any]):
        """记录交叉验证结果"""
        self.info("=" * 60)
        self.info("交叉验证结果")
        self.info("=" * 60)
        
        # 记录权重加载结果
        if 'weight_loaded' in results:
            self.info(f"权重加载成功: {results['weight_loaded']}")
        
        # 记录性能对比
        if 'imagenet_results' in results and 'finetuned_results' in results:
            imagenet = results['imagenet_results']
            finetuned = results['finetuned_results']
            
            for component in ['head', 'full_model']:
                if component in imagenet and component in finetuned:
                    imagenet_conf = imagenet[component]['max_confidence']
                    finetuned_conf = finetuned[component]['max_confidence']
                    improvement = (finetuned_conf - imagenet_conf) / imagenet_conf * 100
                    
                    self.info(f"{component}:")
                    self.info(f"  ImageNet预训练: {imagenet_conf:.6f}")
                    self.info(f"  微调后: {finetuned_conf:.6f}")
                    self.info(f"  改善: {improvement:+.2f}%")
        
        # 记录最终性能估算
        if 'estimated_map' in results:
            self.info(f"估算mAP: {results['estimated_map']:.3f}")
            self.info(f"相对PyTorch性能: {results['performance_percentage']:.1f}%")
    
    def _save_training_stats(self):
        """保存训练统计数据"""
        stats_file = self.log_dir / "training_stats.json"
        
        # 计算统计信息
        total_time = time.time() - self.stats['start_time']
        avg_epoch_time = sum(self.stats['epoch_times']) / len(self.stats['epoch_times']) if self.stats['epoch_times'] else 0
        avg_batch_time = sum(self.stats['batch_times']) / len(self.stats['batch_times']) if self.stats['batch_times'] else 0
        
        stats_data = {
            'total_training_time': total_time,
            'average_epoch_time': avg_epoch_time,
            'average_batch_time': avg_batch_time,
            'epoch_times': self.stats['epoch_times'],
            'losses': self.stats['losses'],
            'metrics': self.stats['metrics']
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats_data, f, indent=2, ensure_ascii=False)
    
    def close(self):
        """关闭日志器"""
        # 记录训练结束
        total_time = time.time() - self.stats['start_time']
        self.info("=" * 60)
        self.info("训练/测试完成")
        self.info(f"总耗时: {total_time:.2f}s")
        self.info("=" * 60)
        
        # 保存最终统计
        if self.file_output:
            self._save_training_stats()
        
        # 关闭所有处理器
        for handler in self.logger.handlers:
            handler.close()
        
        self.logger.handlers.clear()


# 全局日志器实例
_global_logger: Optional[NanoDetLogger] = None


def get_logger(name: str = "nanodet", **kwargs) -> NanoDetLogger:
    """获取全局日志器实例"""
    global _global_logger
    
    if _global_logger is None:
        _global_logger = NanoDetLogger(name=name, **kwargs)
    
    return _global_logger


def setup_logger(log_dir: str = "logs", log_level: str = "INFO", **kwargs) -> NanoDetLogger:
    """设置全局日志器"""
    global _global_logger
    
    _global_logger = NanoDetLogger(
        log_dir=log_dir,
        log_level=log_level,
        **kwargs
    )
    
    return _global_logger
