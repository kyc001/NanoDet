#!/usr/bin/env python3
"""
从jittordet导入已实现的模块到nanodet-jittor
包括gfl_head、gfl_loss等已经实现好的模块
"""

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import List, Dict

# 添加jittordet到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "jittordet"))


def copy_file_with_import_fix(source_file: Path, target_file: Path, 
                             import_mapping: Dict[str, str] = None):
    """复制文件并修复导入路径"""
    if not source_file.exists():
        print(f"源文件不存在: {source_file}")
        return False
    
    # 确保目标目录存在
    target_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(source_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 修复导入路径
    if import_mapping:
        for old_import, new_import in import_mapping.items():
            content = content.replace(old_import, new_import)
    
    # 添加nanodet特定的导入修复
    content = content.replace("from jittordet.", "from nanodet.")
    content = content.replace("import jittordet.", "import nanodet.")
    
    with open(target_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"已复制并修复: {source_file} -> {target_file}")
    return True


def import_gfl_head():
    """导入GFL Head"""
    print("正在导入GFL Head...")
    
    # 源文件和目标文件路径
    source_dir = Path("../jittordet/jittordet/models/dense_heads")
    target_dir = Path("nanodet/model/head")
    
    # 需要复制的文件
    files_to_copy = [
        ("gfl_head.py", "gfl_head.py"),
        ("anchor_head.py", "anchor_head.py"),
        ("base_dense_head.py", "base_dense_head.py"),
    ]
    
    # 导入映射
    import_mapping = {
        "from jittordet.engine import MODELS, TASK_UTILS": "from nanodet.engine import MODELS, TASK_UTILS",
        "from jittordet.models.losses.gfocal_loss import DistributionFocalLoss": "from nanodet.model.loss.gfocal_loss import DistributionFocalLoss",
        "from jittordet.structures import InstanceData": "from nanodet.structures import InstanceData",
        "from jittordet.utils import multi_apply": "from nanodet.util import multi_apply",
    }
    
    for source_file, target_file in files_to_copy:
        source_path = source_dir / source_file
        target_path = target_dir / target_file
        copy_file_with_import_fix(source_path, target_path, import_mapping)


def import_gfl_loss():
    """导入GFL Loss"""
    print("正在导入GFL Loss...")
    
    # 源文件和目标文件路径
    source_dir = Path("../jittordet/jittordet/models/losses")
    target_dir = Path("nanodet/model/loss")
    
    # 需要复制的文件
    files_to_copy = [
        ("gfocal_loss.py", "gfocal_loss.py"),
        ("cross_entropy_loss.py", "cross_entropy_loss.py"),
        ("utils.py", "loss_utils.py"),
    ]
    
    # 导入映射
    import_mapping = {
        "from jittordet.engine import MODELS": "from nanodet.engine import MODELS",
        "from .cross_entropy_loss import": "from .cross_entropy_loss import",
        "from .utils import": "from .loss_utils import",
    }
    
    for source_file, target_file in files_to_copy:
        source_path = source_dir / source_file
        target_path = target_dir / target_file
        copy_file_with_import_fix(source_path, target_path, import_mapping)


def import_ops():
    """导入操作工具"""
    print("正在导入操作工具...")
    
    # 源文件和目标文件路径
    source_dir = Path("../jittordet/jittordet/ops")
    target_dir = Path("nanodet/ops")
    
    # 需要复制的文件
    files_to_copy = [
        ("bbox_transforms.py", "bbox_transforms.py"),
        ("bbox_overlaps.py", "bbox_overlaps.py"),
    ]
    
    # 导入映射
    import_mapping = {
        "import jittor as jt": "import jittor as jt",
        "import jittor.nn as nn": "import jittor.nn as nn",
    }
    
    for source_file, target_file in files_to_copy:
        source_path = source_dir / source_file
        target_path = target_dir / target_file
        copy_file_with_import_fix(source_path, target_path, import_mapping)


def import_utils():
    """导入工具函数"""
    print("正在导入工具函数...")
    
    # 源文件和目标文件路径
    source_dir = Path("../jittordet/jittordet/utils")
    target_dir = Path("nanodet/util")
    
    # 需要复制的文件
    files_to_copy = [
        ("bbox_transforms.py", "bbox_transforms_jittor.py"),
        ("bbox_overlaps.py", "bbox_overlaps_jittor.py"),
    ]
    
    # 导入映射
    import_mapping = {
        "import jittor as jt": "import jittor as jt",
        "import jittor.nn as nn": "import jittor.nn as nn",
    }
    
    for source_file, target_file in files_to_copy:
        source_path = source_dir / source_file
        target_path = target_dir / target_file
        copy_file_with_import_fix(source_path, target_path, import_mapping)


def create_init_files():
    """创建必要的__init__.py文件"""
    print("正在创建__init__.py文件...")
    
    # 需要创建__init__.py的目录
    dirs_to_init = [
        "nanodet/model/head",
        "nanodet/model/loss", 
        "nanodet/ops",
    ]
    
    for dir_path in dirs_to_init:
        init_file = Path(dir_path) / "__init__.py"
        if not init_file.exists():
            init_file.parent.mkdir(parents=True, exist_ok=True)
            init_file.touch()
            print(f"已创建: {init_file}")


def update_nanodet_imports():
    """更新nanodet中的导入语句"""
    print("正在更新nanodet中的导入语句...")
    
    # 需要更新的文件
    files_to_update = [
        "nanodet/model/arch/nanodet.py",
        "nanodet/model/head/gfl_head.py",
    ]
    
    for file_path in files_to_update:
        if Path(file_path).exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 更新导入语句
            content = content.replace(
                "from nanodet.model.head.gfl_head import GFLHead",
                "from nanodet.model.head.gfl_head import GFLHead"
            )
            content = content.replace(
                "from nanodet.model.loss.gfocal_loss import QualityFocalLoss, DistributionFocalLoss",
                "from nanodet.model.loss.gfocal_loss import QualityFocalLoss, DistributionFocalLoss"
            )
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"已更新: {file_path}")


def main():
    parser = argparse.ArgumentParser(description="从jittordet导入已实现的模块")
    parser.add_argument("--all", action="store_true", help="导入所有模块")
    parser.add_argument("--gfl-head", action="store_true", help="导入GFL Head")
    parser.add_argument("--gfl-loss", action="store_true", help="导入GFL Loss")
    parser.add_argument("--ops", action="store_true", help="导入操作工具")
    parser.add_argument("--utils", action="store_true", help="导入工具函数")
    
    args = parser.parse_args()
    
    if args.all or args.gfl_head:
        import_gfl_head()
    
    if args.all or args.gfl_loss:
        import_gfl_loss()
    
    if args.all or args.ops:
        import_ops()
    
    if args.all or args.utils:
        import_utils()
    
    create_init_files()
    update_nanodet_imports()
    
    print("导入完成！")
    print("\n使用说明:")
    print("1. 现在可以直接使用 jittordet 中已经实现好的模块")
    print("2. 例如: from nanodet.model.head.gfl_head import GFLHead")
    print("3. 例如: from nanodet.model.loss.gfocal_loss import QualityFocalLoss")


if __name__ == "__main__":
    main() 