#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
列出 JittorDet 包的所有组件
"""

import os
import sys
from pathlib import Path

def list_package_structure(package_path, indent=0):
    """列出包的结构"""
    package_dir = Path(package_path)
    
    if not package_dir.exists():
        print(f"路径不存在: {package_path}")
        return
    
    for item in sorted(package_dir.iterdir()):
        if item.name.startswith('.') or item.name in ['__pycache__', '.git']:
            continue
            
        prefix = "  " * indent
        if item.is_dir():
            print(f"{prefix}📁 {item.name}/")
            list_package_structure(item, indent + 1)
        else:
            print(f"{prefix}📄 {item.name}")

def list_python_files(package_path):
    """列出所有 Python 文件"""
    package_dir = Path(package_path)
    python_files = []
    
    for py_file in package_dir.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        rel_path = py_file.relative_to(package_dir)
        python_files.append(str(rel_path))
    
    return sorted(python_files)

def main():
    """主函数"""
    jittordet_path = "/home/kyc/project/nanodet/jittordet/jittordet"
    
    print("=" * 80)
    print("JITTORDET 包结构")
    print("=" * 80)
    
    print("\n📁 目录结构:")
    list_package_structure(jittordet_path)
    
    print("\n" + "=" * 80)
    print("📄 Python 文件列表:")
    print("=" * 80)
    
    python_files = list_python_files(jittordet_path)
    for i, file_path in enumerate(python_files, 1):
        print(f"{i:3d}. {file_path}")
    
    print(f"\n总计: {len(python_files)} 个 Python 文件")

if __name__ == "__main__":
    main() 