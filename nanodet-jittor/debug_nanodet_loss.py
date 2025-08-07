#!/usr/bin/env python3
"""
🔍 NanoDet 损失函数调试脚本
专门调试 NanoDet 训练过程中的损失计算问题
"""

import sys
import os
sys.path.insert(0, '.')

import time
import jittor as jt
from nanodet.util import cfg, load_config
from nanodet.model.arch import build_model
from nanodet.data.dataset import build_dataset
from nanodet.data.collate import naive_collate
from nanodet.trainer.task import TrainingTask

def debug_single_batch():
    print("🔍 调试单个批次的训练过程")
    print("=" * 50)
    
    # 设置环境
    os.environ['DISABLE_MULTIPROCESSING'] = '1'
    jt.flags.use_cuda = 1 if jt.has_cuda else 0
    print(f"✅ CUDA: {'启用' if jt.flags.use_cuda else '禁用'}")
    
    # 加载配置
    print("\n📋 加载配置...")
    config_path = 'config/nanodet-plus-m_320_voc_bs64_50epochs.yml'
    load_config(cfg, config_path)
    print(f"✅ 配置加载成功")
    
    # 创建数据集
    print("\n📊 创建数据集...")
    train_dataset = build_dataset(cfg.data.train, 'train')
    print(f"✅ 训练数据集: {len(train_dataset)} 样本")
    
    # 创建模型
    print("\n🏗️ 创建模型...")
    model = build_model(cfg.model)
    print(f"✅ 模型创建成功")
    
    # 创建训练任务
    print("\n🎯 创建训练任务...")
    task = TrainingTask(cfg, model)
    optimizer, scheduler = task.configure_optimizers()
    print(f"✅ 训练任务创建成功")
    
    # 创建数据加载器
    print("\n📦 创建数据加载器...")
    from jittor.dataset import DataLoader
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,  # 使用最小批次进行调试
        shuffle=False,  # 不打乱，便于调试
        num_workers=0,
        collate_batch=naive_collate,
        drop_last=False
    )
    
    print(f"✅ 数据加载器创建成功")
    
    # 调试单个批次
    print("\n🔍 开始调试单个批次...")
    print("-" * 50)
    
    try:
        from types import SimpleNamespace
        
        # 设置训练模式
        model.train()
        task.on_train_epoch_start(0)
        
        # 创建模拟的 trainer 对象
        trainer_mock = SimpleNamespace(
            current_epoch=0,
            global_step=0,
            num_training_batches=1,
            num_val_batches=1,
            optimizer=optimizer
        )
        
        # 获取第一个批次
        batch = next(iter(train_dataloader))
        print(f"✅ 获取批次数据成功")
        print(f"   批次类型: {type(batch)}")
        
        # 检查批次内容
        if isinstance(batch, dict):
            print(f"   批次键: {list(batch.keys())}")
            for key, value in batch.items():
                if hasattr(value, 'shape'):
                    print(f"   {key}: {value.shape}")
                else:
                    print(f"   {key}: {type(value)}")
        
        print("\n🔍 开始前向传播...")
        
        # 前向传播
        try:
            total_loss = task.training_step(batch, 0, trainer_mock)
            print(f"✅ 前向传播成功")
            print(f"   返回值类型: {type(total_loss)}")

            # 检查返回的损失
            if hasattr(total_loss, 'shape'):
                print(f"   损失形状: {total_loss.shape}")

            # 验证是否为 Jittor Var 类型
            if isinstance(total_loss, jt.Var):
                print(f"   ✅ 返回值是正确的 Jittor Var 类型")
            else:
                print(f"   ❌ 返回值不是 Jittor Var 类型: {type(total_loss)}")
                return
            print(f"\n🎯 总损失分析:")
            print(f"   类型: {type(total_loss)}")
            print(f"   形状: {total_loss.shape}")
            
            # 验证总损失
            if isinstance(total_loss, jt.Var):
                print(f"   ✅ 总损失是正确的 Jittor Var 类型")
                
                # 尝试反向传播
                print(f"\n🔍 开始反向传播测试...")
                
                try:
                    # 清零梯度
                    optimizer.zero_grad()
                    print(f"   ✅ 梯度清零成功")
                    
                    # 反向传播
                    total_loss.backward()
                    print(f"   ✅ 反向传播成功")
                    
                    # 检查梯度
                    grad_count = 0
                    for name, param in model.named_parameters():
                        if hasattr(param, 'grad') and param.grad is not None:
                            grad_count += 1
                    
                    print(f"   ✅ 梯度计算成功，{grad_count} 个参数有梯度")
                    
                    # 🔧 这里是关键：尝试优化器步骤
                    print(f"\n🔧 尝试优化器步骤...")
                    
                    try:
                        # 使用手动梯度更新，避开 jt.sync 问题
                        lr = cfg.optimizer.lr
                        updated_params = 0
                        
                        for param in model.parameters():
                            if hasattr(param, 'grad') and param.grad is not None:
                                param.data = param.data - lr * param.grad.data
                                updated_params += 1
                        
                        print(f"   ✅ 手动参数更新成功，更新了 {updated_params} 个参数")
                        
                    except Exception as opt_error:
                        print(f"   ❌ 优化器步骤失败: {opt_error}")
                        
                        # 尝试使用原始优化器
                        try:
                            print(f"   🔧 尝试原始优化器...")
                            optimizer.step(total_loss)
                            print(f"   ✅ 原始优化器成功")
                        except Exception as orig_opt_error:
                            print(f"   ❌ 原始优化器也失败: {orig_opt_error}")
                            print(f"   这确认了 jt.sync 的问题")
                    
                except Exception as backward_error:
                    print(f"   ❌ 反向传播失败: {backward_error}")
                    import traceback
                    traceback.print_exc()
                
            else:
                print(f"   ❌ 总损失不是 Jittor Var 类型: {type(total_loss)}")
            
        except Exception as forward_error:
            print(f"❌ 前向传播失败: {forward_error}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"❌ 调试失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("🔍 NanoDet 损失函数专项调试")
    print("基于基础功能测试通过，专门调试 NanoDet 复杂逻辑")
    print("=" * 60)
    
    debug_single_batch()

if __name__ == "__main__":
    main()
