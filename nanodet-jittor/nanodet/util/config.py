#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
配置管理模块
实现与PyTorch版本一致的配置系统
"""

import os
import yaml
from typing import Dict, Any, Optional
from .logger import get_logger

logger = get_logger(__name__)


class Config:
    """配置类，支持字典式访问"""
    
    def __init__(self, config_dict: Dict[str, Any] = None):
        self._config = config_dict or {}
    
    def __getattr__(self, key: str) -> Any:
        if key in self._config:
            value = self._config[key]
            if isinstance(value, dict):
                return Config(value)
            return value
        raise AttributeError(f"Config has no attribute '{key}'")
    
    def __getitem__(self, key: str) -> Any:
        return self.__getattr__(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        self._config[key] = value
    
    def __contains__(self, key: str) -> bool:
        return key in self._config
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值，支持默认值"""
        try:
            return self.__getattr__(key)
        except AttributeError:
            return default
    
    def update(self, other: Dict[str, Any]) -> None:
        """更新配置"""
        self._config.update(other)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self._config.copy()

    def copy(self) -> 'Config':
        """复制配置对象"""
        return Config(self._config.copy())

    def pop(self, key: str, default=None) -> Any:
        """弹出并返回指定键的值"""
        return self._config.pop(key, default)

    def keys(self):
        """获取所有键"""
        return self._config.keys()

    def items(self):
        """获取所有键值对"""
        return self._config.items()

    def __repr__(self) -> str:
        return f"Config({self._config})"


def load_config(config_path: str) -> Config:
    """加载配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    logger.info(f"Loading config from: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config_dict = yaml.safe_load(f)
        elif config_path.endswith('.json'):
            import json
            config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path}")
    
    return Config(config_dict)


def save_config(config: Config, save_path: str) -> None:
    """保存配置文件"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        if save_path.endswith('.yaml') or save_path.endswith('.yml'):
            yaml.dump(config.to_dict(), f, default_flow_style=False, indent=2)
        elif save_path.endswith('.json'):
            import json
            json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported config format: {save_path}")
    
    logger.info(f"Config saved to: {save_path}")


def merge_configs(base_config: Config, override_config: Config) -> Config:
    """合并配置"""
    merged = Config(base_config.to_dict())
    merged.update(override_config.to_dict())
    return merged


# 默认配置
DEFAULT_CONFIG = Config({
    'model': {
        'name': 'NanoDetPlus',
        'backbone': {
            'name': 'ShuffleNetV2',
            'model_size': '1.0x',
            'out_stages': [2, 3, 4],
            'activation': 'LeakyReLU',
            'pretrain': True
        },
        'fpn': {
            'name': 'GhostPAN',
            'in_channels': [116, 232, 464],
            'out_channels': 96,
            'kernel_size': 5,
            'num_extra_level': 1,
            'use_depthwise': True,
            'activation': 'LeakyReLU'
        },
        'head': {
            'name': 'NanoDetPlusHead',
            'num_classes': 20,
            'input_channel': 96,
            'feat_channels': 96,
            'stacked_convs': 2,
            'kernel_size': 5,
            'strides': [8, 16, 32, 64],
            'activation': 'LeakyReLU',
            'reg_max': 7,
            'norm_cfg': {'type': 'BN'}
        }
    },
    'data': {
        'train': {
            'name': 'CocoDataset',
            'img_path': 'data/train',
            'ann_path': 'data/annotations/train.json',
            'input_size': [320, 320],
            'keep_ratio': True
        },
        'val': {
            'name': 'CocoDataset', 
            'img_path': 'data/val',
            'ann_path': 'data/annotations/val.json',
            'input_size': [320, 320],
            'keep_ratio': True
        }
    },
    'device': {
        'gpu_ids': [0],
        'workers_per_gpu': 4
    },
    'schedule': {
        'optimizer': {
            'name': 'SGD',
            'lr': 0.14,
            'momentum': 0.9,
            'weight_decay': 0.0001
        },
        'warmup': {
            'name': 'linear',
            'steps': 300,
            'ratio': 0.1
        },
        'total_epochs': 300,
        'lr_schedule': {
            'name': 'MultiStepLR',
            'milestones': [130, 180],
            'gamma': 0.1
        }
    },
    'evaluator': {
        'name': 'CocoDetectionEvaluator',
        'save_key': 'mAP'
    },
    'log': {
        'interval': 50
    },
    'save_dir': 'workspace'
})
