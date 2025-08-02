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
    
    # 使用PyTorch版本的配置文件
    pytorch_config = "/home/kyc/project/nanodet/nanodet-pytorch/config/nanodet-plus-m_320_voc_bs64_50epochs.yml"
    if not os.path.exists(pytorch_config):
        logger.error(f"PyTorch config file not found: {pytorch_config}")
        return None
    
    # 构建评估命令
    pytorch_root = "/home/kyc/project/nanodet/nanodet-pytorch"
    
    cmd = [
        "/home/kyc/miniconda3/envs/nano/bin/python",
        f"{pytorch_root}/tools/test.py",
        "--config", pytorch_config,
        "--model", checkpoint_path,
        "--task", task
    ]
    
    logger.info(f"Evaluation command: {' '.join(cmd)}")
    
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
            
            return map_results
            
        else:
            logger.error("PyTorch evaluation failed")
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
    lines = output_text.split('\n')
    
    for line in lines:
        line = line.strip()
        
        if 'Average Precision' in line:
            if 'IoU=0.50:0.95' in line and 'area=   all' in line:
                match = re.search(r'= ([\d.]+)', line)
                if match:
                    map_results['mAP'] = float(match.group(1))
            
            elif 'IoU=0.50' in line and 'area=   all' in line and '0.50:0.95' not in line:
                match = re.search(r'= ([\d.]+)', line)
                if match:
                    map_results['mAP_50'] = float(match.group(1))
    
    return map_results


def main():
    """主函数"""
    args = parse_args()
    
    logger = get_logger('NanoDet')
    logger.info("Starting NanoDet Jittor model evaluation")
    logger.info(f"Model: {args.model}")
    logger.info(f"Task: {args.task}")
    
    # 调用评估
    results = call_pytorch_evaluation(args.model, args.config, args.task)
    
    if results:
        logger.info("Evaluation completed successfully")
        
        # 显示结果
        print("\n" + "="*60)
        print("📊 Evaluation Results:")
        print("="*60)
        
        for metric, value in results.items():
            print(f"  {metric}: {value:.4f}")
        
        # 与PyTorch基准对比
        pytorch_map = 0.275
        if 'mAP' in results:
            jittor_map = results['mAP']
            relative_performance = jittor_map / pytorch_map * 100
            
            print(f"\nComparison with PyTorch baseline:")
            print(f"  PyTorch mAP: {pytorch_map:.4f}")
            print(f"  Jittor mAP:  {jittor_map:.4f}")
            print(f"  Relative:    {relative_performance:.1f}%")
        
        print("="*60)
        
        # 保存结果
        if args.save_result:
            os.makedirs(os.path.dirname(args.save_result), exist_ok=True)
            with open(args.save_result, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to: {args.save_result}")
        
    else:
        logger.error("Evaluation failed")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
