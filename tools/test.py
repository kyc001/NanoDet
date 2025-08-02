#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NanoDet Jittor Test Script
使用PyTorch评估工具评估Jittor模型，确保评估方法完全一致
与PyTorch版本保持完全一致的接口
"""

import os
import sys
import argparse
import subprocess
import json
import re
from pathlib import Path

# 添加路径
sys.path.append('/home/kyc/project/nanodet/nanodet-jittor')
from nanodet.util import get_logger


def parse_args():
    """解析命令行参数 - 与PyTorch版本一致"""
    parser = argparse.ArgumentParser(description='NanoDet Jittor Test')
    parser.add_argument('--config', required=True, help='test config file path')
    parser.add_argument('--model', required=True, help='model checkpoint path (PyTorch format)')
    parser.add_argument('--task', default='val', choices=['val', 'test'], help='evaluation task')
    parser.add_argument('--save_result', help='save evaluation results to file')
    args = parser.parse_args()
    return args


def call_pytorch_evaluation(checkpoint_path, config_path, task='val'):
    """调用PyTorch版本的评估工具"""
    logger = get_logger('NanoDet')
    logger.info("Calling PyTorch evaluation tools...")
    
    # 检查文件是否存在
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint file not found: {checkpoint_path}")
        return None
    
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        return None
    
    # 构建评估命令
    pytorch_root = "/home/kyc/project/nanodet/nanodet-pytorch"
    
    cmd = [
        "/home/kyc/miniconda3/envs/nano/bin/python",
        f"{pytorch_root}/tools/test.py",
        "--config", config_path,
        "--model", checkpoint_path,
        "--task", task
    ]
    
    logger.info(f"Evaluation command: {' '.join(cmd)}")
    logger.info(f"Working directory: {pytorch_root}")
    
    try:
        # 运行评估
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10分钟超时
            cwd=pytorch_root
        )
        
        if result.returncode == 0:
            logger.info("PyTorch evaluation completed successfully")
            
            # 解析评估结果
            map_results = parse_evaluation_output(result.stdout)
            
            # 显示结果
            logger.info("Evaluation results:")
            for metric, value in map_results.items():
                logger.info(f"  {metric}: {value:.4f}")
            
            return map_results
            
        else:
            logger.error("PyTorch evaluation failed")
            logger.error(f"STDOUT: {result.stdout}")
            logger.error(f"STDERR: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        logger.error("PyTorch evaluation timeout")
        return None
    except Exception as e:
        logger.error(f"PyTorch evaluation exception: {e}")
        return None


def parse_evaluation_output(output_text):
    """解析评估输出，提取mAP结果"""
    logger = get_logger('NanoDet')
    
    map_results = {}
    
    # 分割输出为行
    lines = output_text.split('\n')
    
    for line in lines:
        line = line.strip()
        
        # 查找mAP相关的行
        if 'Average Precision' in line:
            # 示例: Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.275
            if 'IoU=0.50:0.95' in line and 'area=   all' in line:
                match = re.search(r'= ([\d.]+)', line)
                if match:
                    map_results['mAP'] = float(match.group(1))
                    logger.info(f"Extracted mAP@0.5:0.95: {map_results['mAP']}")
            
            # 示例: Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.483
            elif 'IoU=0.50' in line and 'area=   all' in line and '0.50:0.95' not in line:
                match = re.search(r'= ([\d.]+)', line)
                if match:
                    map_results['mAP_50'] = float(match.group(1))
                    logger.info(f"Extracted mAP@0.5: {map_results['mAP_50']}")
            
            # 示例: Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.xxx
            elif 'IoU=0.75' in line and 'area=   all' in line:
                match = re.search(r'= ([\d.]+)', line)
                if match:
                    map_results['mAP_75'] = float(match.group(1))
                    logger.info(f"Extracted mAP@0.75: {map_results['mAP_75']}")
        
        elif 'Average Recall' in line:
            # 示例: Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.xxx
            if 'IoU=0.50:0.95' in line and 'maxDets=100' in line:
                match = re.search(r'= ([\d.]+)', line)
                if match:
                    map_results['AR_100'] = float(match.group(1))
                    logger.info(f"Extracted AR@100: {map_results['AR_100']}")
    
    # 如果没有找到标准格式，尝试其他格式
    if not map_results:
        logger.warning("Standard mAP format not found, trying alternative parsing...")
        
        for line in lines:
            # 查找包含数字的行
            if 'mAP' in line.lower() or 'map' in line.lower():
                # 尝试提取数字
                numbers = re.findall(r'[\d.]+', line)
                if numbers:
                    try:
                        value = float(numbers[-1])  # 取最后一个数字
                        if 0 <= value <= 1:  # mAP应该在0-1之间
                            map_results['mAP'] = value
                            logger.info(f"Alternative method extracted mAP: {value}")
                            break
                    except:
                        continue
    
    if not map_results:
        logger.warning("Unable to parse evaluation results")
        # 返回原始输出的最后几行用于调试
        logger.info("Last 10 lines of evaluation output:")
        for line in lines[-10:]:
            if line.strip():
                logger.info(f"  {line}")
    
    return map_results


def save_results(results, save_path):
    """保存评估结果"""
    logger = get_logger('NanoDet')
    
    if results:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Evaluation results saved to: {save_path}")
    else:
        logger.warning("No evaluation results to save")


def main():
    """主函数"""
    args = parse_args()
    
    logger = get_logger('NanoDet')
    logger.info("Starting NanoDet Jittor model evaluation using PyTorch tools")
    logger.info(f"Checkpoint: {args.model}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Task: {args.task}")
    
    # 调用评估
    results = call_pytorch_evaluation(
        args.model,
        args.config,
        args.task
    )
    
    if results:
        logger.info("Evaluation completed successfully")
        
        # 保存结果
        if args.save_result:
            save_results(results, args.save_result)
        
        # 显示最终结果
        print("\n" + "="*60)
        print("📊 Final Evaluation Results:")
        print("="*60)
        
        for metric, value in results.items():
            print(f"  {metric}: {value:.4f}")
        
        # 与PyTorch基准对比
        pytorch_map = 0.275  # PyTorch基准mAP
        if 'mAP' in results:
            jittor_map = results['mAP']
            relative_performance = jittor_map / pytorch_map * 100
            
            print(f"\nComparison with PyTorch baseline:")
            print(f"  PyTorch mAP: {pytorch_map:.4f}")
            print(f"  Jittor mAP:  {jittor_map:.4f}")
            print(f"  Relative:    {relative_performance:.1f}%")
            
            if relative_performance >= 95:
                print(f"  🎯 Excellent! Jittor achieves 95%+ of PyTorch performance")
            elif relative_performance >= 90:
                print(f"  ✅ Good! Jittor achieves 90%+ of PyTorch performance")
            elif relative_performance >= 80:
                print(f"  ⚠️ Acceptable! Jittor achieves 80%+ of PyTorch performance")
            else:
                print(f"  ❌ Needs optimization! Jittor performance < 80% of PyTorch")
        
        print("="*60)
        
    else:
        logger.error("Evaluation failed")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
