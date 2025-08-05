#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JittorDet 包组件分析脚本
输出 jittordet 包的所有组件和模块
"""

import os
import sys
import importlib
import inspect
from pathlib import Path

def get_module_info(module_path, module_name):
    """获取模块信息"""
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        return None

def analyze_package(package_path, package_name="jittordet"):
    """分析包的组件"""
    components = {
        "modules": [],
        "classes": [],
        "functions": [],
        "constants": []
    }
    
    package_dir = Path(package_path)
    
    # 遍历所有 Python 文件
    for py_file in package_dir.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
            
        # 计算相对路径作为模块名
        rel_path = py_file.relative_to(package_dir)
        module_name = str(rel_path).replace("/", ".").replace("\\", ".").replace(".py", "")
        full_module_name = f"{package_name}.{module_name}" if module_name else package_name
        
        try:
            # 尝试导入模块
            module = get_module_info(str(py_file), full_module_name)
            if module is None:
                continue
                
            # 分析模块内容
            for name, obj in inspect.getmembers(module):
                if name.startswith('_'):
                    continue
                    
                if inspect.isclass(obj):
                    components["classes"].append({
                        "name": name,
                        "module": full_module_name,
                        "file": str(py_file)
                    })
                elif inspect.isfunction(obj):
                    components["functions"].append({
                        "name": name,
                        "module": full_module_name,
                        "file": str(py_file)
                    })
                elif not inspect.ismodule(obj):
                    components["constants"].append({
                        "name": name,
                        "module": full_module_name,
                        "file": str(py_file)
                    })
                    
            components["modules"].append({
                "name": full_module_name,
                "file": str(py_file)
            })
            
        except Exception as e:
            print(f"无法分析模块 {full_module_name}: {e}")
    
    return components

def print_components(components):
    """打印组件信息"""
    print("=" * 80)
    print("JITTORDET 包组件分析报告")
    print("=" * 80)
    
    print(f"\n📦 模块总数: {len(components['modules'])}")
    print(f"🏗️  类总数: {len(components['classes'])}")
    print(f"🔧 函数总数: {len(components['functions'])}")
    print(f"📊 常量总数: {len(components['constants'])}")
    
    print("\n" + "=" * 80)
    print("📦 模块列表:")
    print("=" * 80)
    for module in components["modules"]:
        print(f"  • {module['name']}")
        print(f"    文件: {module['file']}")
    
    print("\n" + "=" * 80)
    print("🏗️  类列表:")
    print("=" * 80)
    for cls in components["classes"]:
        print(f"  • {cls['name']}")
        print(f"    模块: {cls['module']}")
        print(f"    文件: {cls['file']}")
    
    print("\n" + "=" * 80)
    print("🔧 函数列表:")
    print("=" * 80)
    for func in components["functions"]:
        print(f"  • {func['name']}")
        print(f"    模块: {func['module']}")
        print(f"    文件: {func['file']}")
    
    print("\n" + "=" * 80)
    print("📊 常量列表:")
    print("=" * 80)
    for const in components["constants"]:
        print(f"  • {const['name']}")
        print(f"    模块: {const['module']}")
        print(f"    文件: {const['file']}")

def main():
    """主函数"""
    # 设置路径
    jittordet_path = "/home/kyc/project/nanodet/jittordet/jittordet"
    
    if not os.path.exists(jittordet_path):
        print(f"错误: 找不到 jittordet 路径: {jittordet_path}")
        return
    
    print("正在分析 jittordet 包...")
    components = analyze_package(jittordet_path)
    print_components(components)

if __name__ == "__main__":
    main() 