#!/usr/bin/env python3
"""
PyTorch到Jittor迁移工具
基于jittordet项目的工具，为nanodet-jittor提供完整的迁移支持
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import jittor as jt
import torch


class PyTorchToJittorMigrator:
    """PyTorch到Jittor迁移器"""
    
    def __init__(self, source_dir: str, target_dir: str):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.migration_log = []
        
    def log(self, message: str):
        """记录迁移日志"""
        print(f"[MIGRATION] {message}")
        self.migration_log.append(message)
    
    def convert_code_file(self, source_file: Path, target_file: Path) -> bool:
        """转换单个代码文件"""
        try:
            with open(source_file, 'r', encoding='utf-8') as f:
                pytorch_code = f.read()
            
            # 使用Jittor的转换器
            from jittor.utils.pytorch_converter import convert
            jittor_code = convert(pytorch_code)
            
            # 确保目标目录存在
            target_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(target_file, 'w', encoding='utf-8') as f:
                f.write(jittor_code)
            
            self.log(f"成功转换: {source_file} -> {target_file}")
            return True
            
        except Exception as e:
            self.log(f"转换失败 {source_file}: {str(e)}")
            return False
    
    def convert_checkpoint(self, source_checkpoint: str, target_checkpoint: str) -> bool:
        """转换模型检查点"""
        try:
            # 加载PyTorch检查点
            checkpoint = torch.load(source_checkpoint, map_location='cpu')
            
            # 转换检查点格式
            converted_checkpoint = self._convert_checkpoint_format(checkpoint)
            
            # 保存为Jittor格式
            jt.save(converted_checkpoint, target_checkpoint)
            
            self.log(f"成功转换检查点: {source_checkpoint} -> {target_checkpoint}")
            return True
            
        except Exception as e:
            self.log(f"检查点转换失败 {source_checkpoint}: {str(e)}")
            return False
    
    def _convert_checkpoint_format(self, checkpoint: Dict) -> Dict:
        """转换检查点格式"""
        if "pytorch-lightning_version" in checkpoint:
            # 移除PyTorch Lightning特定信息
            checkpoint.pop("pytorch-lightning_version", None)
            checkpoint.pop("lr_schedulers", None)
            
            # 处理state_dict
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
                new_state_dict = {}
                for key, value in state_dict.items():
                    # 移除"model."前缀
                    if key.startswith("model."):
                        new_key = key[6:]  # 移除"model."前缀
                    else:
                        new_key = key
                    new_state_dict[new_key] = value
                checkpoint["state_dict"] = new_state_dict
        
        return checkpoint
    
    def migrate_dataset_utils(self):
        """迁移数据集工具"""
        # 从jittordet复制数据集工具
        jittordet_utils = Path("../jittordet/jittordet/datasets")
        target_utils = self.target_dir / "nanodet" / "data"
        
        if jittordet_utils.exists():
            # 复制数据集基类
            base_file = jittordet_utils / "base.py"
            if base_file.exists():
                shutil.copy2(base_file, target_utils / "base.py")
                self.log("已复制数据集基类")
        
        # 复制转换工具
        transforms_dir = jittordet_utils / "transforms"
        if transforms_dir.exists():
            target_transforms = target_utils / "transforms"
            target_transforms.mkdir(exist_ok=True)
            
            for transform_file in transforms_dir.glob("*.py"):
                shutil.copy2(transform_file, target_transforms / transform_file.name)
                self.log(f"已复制转换工具: {transform_file.name}")
    
    def migrate_model_utils(self):
        """迁移模型工具"""
        # 从jittordet复制模型工具
        jittordet_models = Path("../jittordet/jittordet/models")
        target_models = self.target_dir / "nanodet" / "model"
        
        if jittordet_models.exists():
            # 复制通用层
            layers_dir = jittordet_models / "layers"
            if layers_dir.exists():
                target_layers = target_models / "layers"
                target_layers.mkdir(exist_ok=True)
                
                for layer_file in layers_dir.glob("*.py"):
                    shutil.copy2(layer_file, target_layers / layer_file.name)
                    self.log(f"已复制模型层: {layer_file.name}")
    
    def migrate_ops_utils(self):
        """迁移操作工具"""
        # 从jittordet复制操作工具
        jittordet_ops = Path("../jittordet/jittordet/ops")
        target_ops = self.target_dir / "nanodet" / "ops"
        
        if jittordet_ops.exists():
            target_ops.mkdir(exist_ok=True)
            
            for op_file in jittordet_ops.glob("*.py"):
                shutil.copy2(op_file, target_ops / op_file.name)
                self.log(f"已复制操作工具: {op_file.name}")
    
    def create_migration_report(self, output_file: str = "migration_report.txt"):
        """创建迁移报告"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("PyTorch到Jittor迁移报告\n")
            f.write("=" * 50 + "\n\n")
            
            for log_entry in self.migration_log:
                f.write(f"{log_entry}\n")
        
        self.log(f"迁移报告已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="PyTorch到Jittor迁移工具")
    parser.add_argument("--source", required=True, help="PyTorch项目目录")
    parser.add_argument("--target", required=True, help="Jittor项目目录")
    parser.add_argument("--checkpoint", help="要转换的检查点文件")
    parser.add_argument("--output-checkpoint", help="输出检查点文件")
    parser.add_argument("--migrate-utils", action="store_true", help="迁移工具函数")
    parser.add_argument("--report", default="migration_report.txt", help="迁移报告文件")
    
    args = parser.parse_args()
    
    # 创建迁移器
    migrator = PyTorchToJittorMigrator(args.source, args.target)
    
    # 转换检查点
    if args.checkpoint and args.output_checkpoint:
        migrator.convert_checkpoint(args.checkpoint, args.output_checkpoint)
    
    # 迁移工具函数
    if args.migrate_utils:
        migrator.migrate_dataset_utils()
        migrator.migrate_model_utils()
        migrator.migrate_ops_utils()
    
    # 生成报告
    migrator.create_migration_report(args.report)
    
    print("迁移完成！")


if __name__ == "__main__":
    main() 