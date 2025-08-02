# Copyright 2021 RangiLyu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from datetime import datetime


class NanoDetLogger:
    """NanoDet Logger - 完全模仿PyTorch版本的日志格式"""
    
    def __init__(self, name="NanoDet", save_dir=None):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # 清除已有的handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # 创建formatter - 完全模仿PyTorch版本
        # [NanoDet][07-31 21:18:05]INFO:
        formatter = logging.Formatter(
            f'[{name}][%(asctime)s]%(levelname)s:%(message)s',
            datefmt='%m-%d %H:%M:%S'
        )
        
        # 控制台handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # 文件handler（如果指定了保存目录）
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(save_dir, 'logs.txt'))
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def info(self, message):
        """输出INFO级别日志"""
        self.logger.info(message)

    def warning(self, message):
        """输出WARNING级别日志"""
        self.logger.warning(message)

    def error(self, message):
        """输出ERROR级别日志"""
        self.logger.error(message)

    def debug(self, message):
        """输出DEBUG级别日志"""
        self.logger.debug(message)


# 全局logger实例
_loggers = {}


def get_logger(name="NanoDet", save_dir=None):
    """获取logger实例

    Args:
        name (str): Logger名称
        save_dir (str): 日志保存目录

    Returns:
        NanoDetLogger: Logger实例
    """
    global _loggers

    key = f"{name}_{save_dir}"
    if key not in _loggers:
        _loggers[key] = NanoDetLogger(name, save_dir)

    return _loggers[key]


def setup_logger(name="NanoDet", save_dir=None, level='INFO'):
    """设置logger（与get_logger功能相同，为了兼容性）

    Args:
        name (str): Logger名称
        save_dir (str): 日志保存目录
        level (str): 日志级别

    Returns:
        NanoDetLogger: Logger实例
    """
    logger = get_logger(name, save_dir)

    # 设置日志级别
    if level.upper() == 'DEBUG':
        logger.logger.setLevel(logging.DEBUG)
    elif level.upper() == 'INFO':
        logger.logger.setLevel(logging.INFO)
    elif level.upper() == 'WARNING':
        logger.logger.setLevel(logging.WARNING)
    elif level.upper() == 'ERROR':
        logger.logger.setLevel(logging.ERROR)

    return logger
