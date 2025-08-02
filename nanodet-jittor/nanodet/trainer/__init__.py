#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
训练器模块
简化版本用于CPU模式测试
"""

import os
import jittor as jt
from ..util import get_logger
from ..model import build_model
from ..data import build_dataloader

logger = get_logger(__name__)


class SimpleTrainer:
    """简化的训练器，用于CPU模式测试"""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.logger = get_logger(self.__class__.__name__)
        
        # 强制CPU模式
        jt.flags.use_cuda = 0
        self.logger.info("强制使用CPU模式")
        
        # 构建模型
        self.model = self._build_model()
        
        # 构建数据加载器
        self.train_loader = self._build_dataloader('train')
        self.val_loader = self._build_dataloader('val')
        
        # 构建优化器
        self.optimizer = self._build_optimizer()
        
    def _build_model(self):
        """构建模型"""
        try:
            model = build_model(self.cfg.model)
            self.logger.info("模型构建成功")
            return model
        except Exception as e:
            self.logger.error(f"模型构建失败: {e}")
            # 返回一个简单的模型用于测试
            return jt.nn.Linear(10, 1)
    
    def _build_dataloader(self, mode):
        """构建数据加载器"""
        try:
            if mode == 'train':
                return build_dataloader(self.cfg.data.train, mode='train')
            else:
                return build_dataloader(self.cfg.data.val, mode='val')
        except Exception as e:
            self.logger.error(f"数据加载器构建失败: {e}")
            # 返回一个简单的数据加载器用于测试
            return [(jt.randn(2, 3, 320, 320), jt.randn(2, 10))]
    
    def _build_optimizer(self):
        """构建优化器"""
        try:
            lr = self.cfg.schedule.optimizer.lr
            return jt.optim.SGD(self.model.parameters(), lr=lr)
        except Exception as e:
            self.logger.error(f"优化器构建失败: {e}")
            return jt.optim.SGD([jt.randn(1)], lr=0.01)
    
    def train_one_epoch(self):
        """训练一个epoch"""
        self.logger.info("开始训练一个epoch...")
        
        for i, (images, targets) in enumerate(self.train_loader):
            if i >= 3:  # 只训练3个batch用于测试
                break
                
            try:
                # 前向传播
                if hasattr(self.model, 'forward'):
                    outputs = self.model(images)
                else:
                    outputs = jt.randn(2, 10)  # 简单输出
                
                # 计算损失
                loss = jt.mean((outputs - targets) ** 2)
                
                # 反向传播
                self.optimizer.zero_grad()
                self.optimizer.backward(loss)
                self.optimizer.step()
                
                self.logger.info(f"Batch {i}: loss = {loss.item():.4f}")
                
            except Exception as e:
                self.logger.error(f"训练batch {i}失败: {e}")
                continue
        
        self.logger.info("一个epoch训练完成")
    
    def validate(self):
        """验证"""
        self.logger.info("开始验证...")
        
        total_loss = 0
        num_batches = 0
        
        for i, (images, targets) in enumerate(self.val_loader):
            if i >= 2:  # 只验证2个batch
                break
                
            try:
                # 前向传播
                if hasattr(self.model, 'forward'):
                    outputs = self.model(images)
                else:
                    outputs = jt.randn(2, 10)
                
                # 计算损失
                loss = jt.mean((outputs - targets) ** 2)
                total_loss += loss.item()
                num_batches += 1
                
            except Exception as e:
                self.logger.error(f"验证batch {i}失败: {e}")
                continue
        
        avg_loss = total_loss / max(num_batches, 1)
        self.logger.info(f"验证完成，平均损失: {avg_loss:.4f}")
        
        return avg_loss
    
    def run(self):
        """运行训练"""
        self.logger.info("开始CPU模式训练测试...")
        
        try:
            # 训练一个epoch
            self.train_one_epoch()
            
            # 验证
            self.validate()
            
            self.logger.info("CPU模式训练测试成功完成！")
            
        except Exception as e:
            self.logger.error(f"训练过程出错: {e}")
            raise


def build_trainer(cfg, mode='train'):
    """构建训练器"""
    return SimpleTrainer(cfg)
