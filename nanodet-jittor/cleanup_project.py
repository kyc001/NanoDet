#!/usr/bin/env python3
"""
🧹 NanoDet-Jittor 项目清理脚本
删除调试文件、临时文件和重复文件，保留核心功能
"""

import os
import shutil
import glob

def cleanup_project():
    print("🧹 开始清理 NanoDet-Jittor 项目")
    print("=" * 50)
    
    # 要删除的调试和临时文件
    files_to_remove = [
        # 调试脚本
        "debug_*.py",
        "test_*.py", 
        "simple_*.py",
        "complete_train.py",
        "cpu_training.py",
        "final_train.py",
        "robust_train.py",
        "start_training.py",
        "train_with_logs.py",
        "success_summary.py",
        "check_model_size.py",
        "compare_params.py",
        
        # 错误日志
        "*.log",
        
        # 临时配置文件
        "config/*test*.yml",
        "config/*final*.yml",
    ]
    
    # 要删除的目录
    dirs_to_remove = [
        # 缓存目录
        "__pycache__",
        "nanodet/__pycache__",
        "jittordet/__pycache__",
        "tools/__pycache__",
        
        # 过多的日志目录（保留最新的几个）
        # 这个需要特殊处理
    ]
    
    # 要保留的重要文件
    important_files = [
        "final_training.py",  # 最终训练脚本
        "test_basic_jittor.py",  # 基础测试脚本
        "debug_nanodet_loss.py",  # 核心调试脚本
        "convert_pretrained_weights.py",  # 权重转换脚本
        "minimal_train.py",  # 最小训练脚本
    ]
    
    removed_count = 0
    
    # 删除匹配的文件
    for pattern in files_to_remove:
        for file_path in glob.glob(pattern):
            if os.path.basename(file_path) not in important_files:
                try:
                    os.remove(file_path)
                    print(f"🗑️ 删除文件: {file_path}")
                    removed_count += 1
                except Exception as e:
                    print(f"❌ 删除失败 {file_path}: {e}")
    
    # 删除缓存目录
    for root, dirs, files in os.walk("."):
        for dir_name in dirs:
            if dir_name == "__pycache__":
                dir_path = os.path.join(root, dir_name)
                try:
                    shutil.rmtree(dir_path)
                    print(f"🗑️ 删除目录: {dir_path}")
                    removed_count += 1
                except Exception as e:
                    print(f"❌ 删除失败 {dir_path}: {e}")
    
    # 清理过多的日志目录（保留最新的5个）
    log_dirs = glob.glob("workspace/*/logs-*")
    if len(log_dirs) > 5:
        # 按时间排序，删除旧的
        log_dirs.sort()
        for old_log in log_dirs[:-5]:
            try:
                shutil.rmtree(old_log)
                print(f"🗑️ 删除旧日志: {old_log}")
                removed_count += 1
            except Exception as e:
                print(f"❌ 删除失败 {old_log}: {e}")
    
    # 清理tools目录中的重复修复脚本
    tools_to_remove = [
        "tools/fix_depthwise_conv.py",
        "tools/fix_depthwise_conv_jittordet.py", 
        "tools/fix_jittor_depthwise_conv.py",
        "tools/fix_jittor_depthwise_conv_v2.py",
        "tools/fix_jittor_depthwise_conv_v3.py",
        # 保留 fix_jittor_depthwise_conv_final.py
        
        "tools/test_*.py",
        "tools/train_*.py",  # 除了 train.py
    ]
    
    important_tools = [
        "tools/train.py",
        "tools/fix_jittor_depthwise_conv_final.py",
        "tools/convert_pytorch_weights.py",
        "tools/inference.py",
    ]
    
    for tool_file in tools_to_remove:
        if os.path.exists(tool_file) and os.path.basename(tool_file) not in [os.path.basename(f) for f in important_tools]:
            try:
                os.remove(tool_file)
                print(f"🗑️ 删除工具: {tool_file}")
                removed_count += 1
            except Exception as e:
                print(f"❌ 删除失败 {tool_file}: {e}")
    
    print(f"\n✅ 清理完成！")
    print(f"📊 总共删除了 {removed_count} 个文件/目录")
    
    # 显示保留的重要文件
    print(f"\n📋 保留的重要文件:")
    important_files_full = [
        "final_training.py",
        "test_basic_jittor.py", 
        "debug_nanodet_loss.py",
        "convert_pretrained_weights.py",
        "minimal_train.py",
        "tools/train.py",
        "tools/fix_jittor_depthwise_conv_final.py",
        "tools/convert_pytorch_weights.py",
        "tools/inference.py",
        "config/nanodet-plus-m_320_voc_bs64_50epochs.yml",
        "workspace/nanodet-plus-m_320_voc_bs64_50epochs/training_completed.txt",
    ]
    
    for important_file in important_files_full:
        if os.path.exists(important_file):
            print(f"   ✅ {important_file}")
        else:
            print(f"   ❌ {important_file} (不存在)")

if __name__ == "__main__":
    cleanup_project()
