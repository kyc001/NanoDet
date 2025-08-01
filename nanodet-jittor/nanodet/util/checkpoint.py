#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
检查点管理模块
实现模型保存和加载功能
"""

import os
import pickle
import jittor as jt
from typing import Dict, Any, Optional
from .logger import get_logger

logger = get_logger(__name__)


def save_checkpoint(model, optimizer=None, epoch=None, metrics=None, save_path=None, **kwargs):
    """保存检查点"""
    if save_path is None:
        save_path = f"checkpoint_epoch_{epoch}.pkl"
    
    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 构建检查点数据
    checkpoint = {
        'model_state_dict': {},
        'epoch': epoch,
        'metrics': metrics or {},
        **kwargs
    }
    
    # 保存模型参数
    for name, param in model.named_parameters():
        checkpoint['model_state_dict'][name] = param.numpy()
    
    # 保存优化器状态（如果提供）
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    # 保存到文件
    with open(save_path, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    logger.info(f"Checkpoint saved to: {save_path}")
    
    if metrics:
        logger.info(f"Metrics: {metrics}")
    
    return save_path


def load_checkpoint(model, checkpoint_path, optimizer=None, strict=True):
    """加载检查点"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    
    # 加载检查点数据
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    # 加载模型参数
    model_state_dict = checkpoint.get('model_state_dict', {})
    
    loaded_count = 0
    total_count = 0
    missing_keys = []
    unexpected_keys = []
    
    # 获取模型参数
    model_params = {}
    for name, param in model.named_parameters():
        model_params[name] = param
    
    # 加载参数
    for name, value in model_state_dict.items():
        total_count += 1
        if name in model_params:
            try:
                model_params[name].assign(jt.array(value))
                loaded_count += 1
            except Exception as e:
                logger.warning(f"Failed to load parameter {name}: {e}")
                missing_keys.append(name)
        else:
            unexpected_keys.append(name)
    
    # 检查缺失的参数
    for name in model_params:
        if name not in model_state_dict:
            missing_keys.append(name)
    
    # 加载优化器状态（如果提供）
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("Optimizer state loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load optimizer state: {e}")
    
    # 报告加载结果
    logger.info(f"Loaded {loaded_count}/{total_count} parameters")
    
    if missing_keys and strict:
        logger.warning(f"Missing keys: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
    
    if unexpected_keys and strict:
        logger.warning(f"Unexpected keys: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
    
    # 返回其他信息
    result = {
        'epoch': checkpoint.get('epoch'),
        'metrics': checkpoint.get('metrics', {}),
        'loaded_count': loaded_count,
        'total_count': total_count,
        'missing_keys': missing_keys,
        'unexpected_keys': unexpected_keys
    }
    
    # 添加其他键值对
    for key, value in checkpoint.items():
        if key not in ['model_state_dict', 'optimizer_state_dict', 'epoch', 'metrics']:
            result[key] = value
    
    return result


def load_pytorch_checkpoint(model, pytorch_checkpoint_path, strict=True):
    """加载PyTorch检查点到Jittor模型"""
    import torch
    
    if not os.path.exists(pytorch_checkpoint_path):
        raise FileNotFoundError(f"PyTorch checkpoint not found: {pytorch_checkpoint_path}")
    
    logger.info(f"Loading PyTorch checkpoint from: {pytorch_checkpoint_path}")
    
    # 加载PyTorch检查点
    ckpt = torch.load(pytorch_checkpoint_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)
    
    # 获取Jittor模型参数
    jittor_state_dict = {}
    for name, param in model.named_parameters():
        jittor_state_dict[name] = param
    
    loaded_count = 0
    total_count = 0
    missing_keys = []
    unexpected_keys = []
    
    # 转换和加载参数
    for pytorch_name, pytorch_param in state_dict.items():
        jittor_name = pytorch_name
        
        # 移除'model.'前缀（如果存在）
        if jittor_name.startswith("model."):
            jittor_name = jittor_name[6:]
        
        # 跳过不需要的参数
        if "num_batches_tracked" in jittor_name or jittor_name.startswith("avg_"):
            continue
        
        if "distribution_project.project" in jittor_name:
            continue
        
        total_count += 1
        
        if jittor_name in jittor_state_dict:
            jittor_param = jittor_state_dict[jittor_name]
            
            try:
                if list(pytorch_param.shape) == list(jittor_param.shape):
                    jittor_param.assign(jt.array(pytorch_param.detach().numpy()))
                    loaded_count += 1
                elif "scale" in jittor_name and len(pytorch_param.shape) == 0 and list(jittor_param.shape) == [1]:
                    jittor_param.assign(jt.array([pytorch_param.detach().numpy()]))
                    loaded_count += 1
                else:
                    logger.warning(f"Shape mismatch for {jittor_name}: PyTorch{pytorch_param.shape} vs Jittor{jittor_param.shape}")
                    missing_keys.append(jittor_name)
            except Exception as e:
                logger.warning(f"Failed to load parameter {jittor_name}: {e}")
                missing_keys.append(jittor_name)
        else:
            unexpected_keys.append(jittor_name)
    
    # 检查缺失的参数
    for name in jittor_state_dict:
        pytorch_name = f"model.{name}"
        if pytorch_name not in state_dict and name not in [pn[6:] if pn.startswith("model.") else pn for pn in state_dict.keys()]:
            missing_keys.append(name)
    
    # 报告加载结果
    logger.info(f"Loaded {loaded_count}/{total_count} parameters from PyTorch checkpoint")
    
    if missing_keys and strict:
        logger.warning(f"Missing keys: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
    
    if unexpected_keys and strict:
        logger.warning(f"Unexpected keys: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
    
    return {
        'epoch': ckpt.get('epoch'),
        'metrics': ckpt.get('metrics', {}),
        'loaded_count': loaded_count,
        'total_count': total_count,
        'missing_keys': missing_keys,
        'unexpected_keys': unexpected_keys
    }


def get_latest_checkpoint(checkpoint_dir):
    """获取最新的检查点文件"""
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pkl')]
    
    if not checkpoint_files:
        return None
    
    # 按修改时间排序
    checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
    
    latest_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[0])
    logger.info(f"Found latest checkpoint: {latest_checkpoint}")
    
    return latest_checkpoint
