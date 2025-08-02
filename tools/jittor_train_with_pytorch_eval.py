#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Jittor训练 + PyTorch评估工具
使用Jittor进行训练，但使用PyTorch版本的评估工具进行mAP计算
确保评估方法完全一致
"""

import os
import sys
import json
import subprocess
import time
import torch
import jittor as jt
import numpy as np
from pathlib import Path

# 添加路径
sys.path.append('/home/kyc/project/nanodet/nanodet-jittor')
from nanodet.util import get_logger, load_config, save_checkpoint, load_pytorch_checkpoint
from nanodet.model.arch.nanodet_plus import NanoDetPlus


def create_jittor_model(config):
    """创建Jittor模型"""
    logger = get_logger('JittorTraining')
    logger.info("创建Jittor模型...")
    
    model = NanoDetPlus(
        config.model.arch.backbone,
        config.model.arch.fpn,
        config.model.arch.aux_head,
        config.model.arch.head
    )
    
    logger.info("模型创建成功")
    return model


def convert_jittor_to_pytorch_checkpoint(jittor_model, save_path):
    """将Jittor模型转换为PyTorch格式的检查点"""
    logger = get_logger('JittorTraining')
    logger.info(f"转换Jittor模型为PyTorch格式: {save_path}")
    
    # 获取Jittor模型参数
    jittor_state_dict = {}
    for name, param in jittor_model.named_parameters():
        jittor_state_dict[name] = param.numpy()
    
    # 转换为PyTorch格式
    pytorch_state_dict = {}
    for name, param_np in jittor_state_dict.items():
        pytorch_name = f"model.{name}"  # 添加'model.'前缀
        pytorch_state_dict[pytorch_name] = torch.tensor(param_np)
    
    # 保存为PyTorch检查点格式
    checkpoint = {
        'state_dict': pytorch_state_dict,
        'epoch': 0,
        'optimizer': None,
        'lr_scheduler': None,
        'best_map': 0.0
    }
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(checkpoint, save_path)
    
    logger.info(f"PyTorch格式检查点已保存: {save_path}")
    return save_path


def call_pytorch_evaluation(pytorch_checkpoint_path, config_path):
    """调用PyTorch版本的评估工具"""
    logger = get_logger('JittorTraining')
    logger.info("调用PyTorch版本的评估工具...")
    
    # 构建评估命令
    pytorch_root = "/home/kyc/project/nanodet/nanodet-pytorch"
    pytorch_config = f"{pytorch_root}/config/nanodet-plus-m_320_voc_bs64_50epochs.yml"
    
    cmd = [
        "/home/kyc/miniconda3/envs/nano/bin/python",
        f"{pytorch_root}/tools/test.py",
        "--config", pytorch_config,
        "--model", pytorch_checkpoint_path,
        "--task", "val"
    ]
    
    logger.info(f"评估命令: {' '.join(cmd)}")
    
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
            logger.info("PyTorch评估成功完成")
            
            # 解析mAP结果
            output_lines = result.stdout.split('\n')
            map_results = {}
            
            for line in output_lines:
                if 'Average Precision' in line and 'IoU=0.50:0.95' in line:
                    # 提取mAP@0.5:0.95
                    parts = line.split('=')
                    if len(parts) >= 2:
                        try:
                            map_value = float(parts[-1].strip())
                            map_results['mAP'] = map_value
                        except:
                            pass
                elif 'Average Precision' in line and 'IoU=0.50' in line:
                    # 提取mAP@0.5
                    parts = line.split('=')
                    if len(parts) >= 2:
                        try:
                            map_value = float(parts[-1].strip())
                            map_results['mAP_50'] = map_value
                        except:
                            pass
            
            logger.info(f"评估结果: {map_results}")
            return map_results
            
        else:
            logger.error(f"PyTorch评估失败: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        logger.error("PyTorch评估超时")
        return None
    except Exception as e:
        logger.error(f"PyTorch评估异常: {e}")
        return None


def train_jittor_model(config_path, num_epochs=10):
    """训练Jittor模型"""
    logger = get_logger('JittorTraining')
    logger.info(f"开始Jittor训练: {num_epochs}轮")
    
    # 加载配置
    config = load_config(config_path)
    
    # 创建模型
    model = create_jittor_model(config)
    
    # 设置Jittor
    jt.flags.use_cuda = 1 if jt.has_cuda else 0
    logger.info(f"使用设备: {'CUDA' if jt.flags.use_cuda else 'CPU'}")
    
    # 创建优化器
    optimizer = jt.optim.AdamW(
        model.parameters(),
        lr=config.schedule.optimizer.lr,
        weight_decay=config.schedule.optimizer.weight_decay
    )
    
    # 创建工作目录
    workspace = config.save_dir
    os.makedirs(workspace, exist_ok=True)
    
    # 训练循环
    model.train()
    
    for epoch in range(1, num_epochs + 1):
        logger.info(f"开始第 {epoch}/{num_epochs} 轮训练")
        
        # 模拟训练过程（这里需要实现真实的数据加载和训练循环）
        # 为了演示，我们先创建一个简化的训练循环
        
        epoch_loss = 0.0
        num_batches = 108  # 与PyTorch版本一致
        
        for batch_idx in range(num_batches):
            # 创建模拟数据
            batch_size = 64
            input_data = np.random.randn(batch_size, 3, 320, 320).astype(np.float32)
            jittor_input = jt.array(input_data)
            
            # 前向传播
            output = model(jittor_input)
            
            # 计算简化损失
            loss = jt.mean(output ** 2) * 0.001  # 简化的损失函数
            
            # 反向传播
            optimizer.zero_grad()
            optimizer.backward(loss)
            optimizer.step()
            
            epoch_loss += float(loss.numpy())
            
            if (batch_idx + 1) % 20 == 0:
                logger.info(f"  Epoch {epoch}, Batch {batch_idx+1}/{num_batches}, Loss: {float(loss.numpy()):.6f}")
        
        avg_loss = epoch_loss / num_batches
        logger.info(f"第 {epoch} 轮完成，平均损失: {avg_loss:.6f}")
        
        # 每5轮进行一次评估
        if epoch % 5 == 0:
            logger.info(f"第 {epoch} 轮评估...")
            
            # 保存Jittor检查点
            jittor_checkpoint_path = f"{workspace}/jittor_epoch_{epoch}.pkl"
            save_checkpoint(
                model,
                optimizer=optimizer,
                epoch=epoch,
                metrics={'loss': avg_loss},
                save_path=jittor_checkpoint_path
            )
            
            # 转换为PyTorch格式
            pytorch_checkpoint_path = f"{workspace}/pytorch_format_epoch_{epoch}.ckpt"
            convert_jittor_to_pytorch_checkpoint(model, pytorch_checkpoint_path)
            
            # 调用PyTorch评估
            map_results = call_pytorch_evaluation(pytorch_checkpoint_path, config_path)
            
            if map_results:
                logger.info(f"第 {epoch} 轮 mAP 结果:")
                for metric, value in map_results.items():
                    logger.info(f"  {metric}: {value:.4f}")
                
                # 保存评估结果
                results_file = f"{workspace}/eval_results_epoch_{epoch}.json"
                with open(results_file, 'w') as f:
                    json.dump({
                        'epoch': epoch,
                        'loss': avg_loss,
                        'map_results': map_results
                    }, f, indent=2)
            else:
                logger.warning(f"第 {epoch} 轮评估失败")
    
    logger.info("Jittor训练完成")
    return workspace


def main():
    """主函数"""
    print("🚀 开始Jittor训练 + PyTorch评估")
    print("=" * 60)
    
    try:
        # 配置文件路径
        config_path = "config/nanodet-plus-m_320_voc_jittor.yml"
        
        if not os.path.exists(config_path):
            print(f"❌ 配置文件不存在: {config_path}")
            return
        
        # 开始训练
        workspace = train_jittor_model(config_path, num_epochs=10)
        
        print(f"\n🎯 训练完成！")
        print(f"工作目录: {workspace}")
        print(f"检查点和评估结果已保存")
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
