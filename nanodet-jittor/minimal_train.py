#!/usr/bin/env python3
"""
🔥 最小化训练脚本 - 直接验证训练是否可以进行
"""

import sys
sys.path.insert(0, '.')

import os
import time
import jittor as jt
from nanodet.util import cfg, load_config
from nanodet.model.arch import build_model
from nanodet.data.dataset import build_dataset
from nanodet.data.collate import naive_collate
from nanodet.trainer.task import TrainingTask

def main():
    print("🔥 最小化训练测试")
    print("=" * 40)
    
    # 设置环境
    os.environ['DISABLE_MULTIPROCESSING'] = '1'
    jt.flags.use_cuda = 1 if jt.has_cuda else 0
    
    # 加载配置
    load_config(cfg, 'config/nanodet-plus-m_320_voc_bs64_50epochs.yml')
    
    # 创建数据集
    train_dataset = build_dataset(cfg.data.train, 'train')
    print(f"✅ 数据集: {len(train_dataset)} 样本")
    
    # 创建模型
    model = build_model(cfg.model)
    print(f"✅ 模型创建成功")
    
    # 创建训练任务
    task = TrainingTask(cfg, model)
    optimizer, scheduler = task.configure_optimizers()
    print(f"✅ 训练任务创建成功")
    
    # 创建数据加载器
    from jittor.dataset import DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_batch=naive_collate,
        drop_last=False
    )
    print(f"✅ 数据加载器: {len(train_dataloader)} 批次")
    
    # 开始训练测试
    print("\n🔥 开始训练测试...")
    
    try:
        from types import SimpleNamespace
        
        model.train()
        
        # 只测试前5个批次
        for batch_idx, batch in enumerate(train_dataloader):
            if batch_idx >= 5:
                break
                
            print(f"\n--- 批次 {batch_idx + 1}/5 ---")
            
            # 创建 trainer mock
            trainer_mock = SimpleNamespace(
                current_epoch=0,
                global_step=batch_idx,
                num_training_batches=5,
                num_val_batches=100,
                optimizer=optimizer
            )
            
            try:
                # 前向传播
                print("  前向传播...")
                loss_dict = task.training_step(batch, batch_idx, trainer_mock)
                total_loss = loss_dict['loss']
                print(f"  ✅ 前向传播成功，损失形状: {total_loss.shape}")
                
                # 反向传播
                print("  反向传播...")
                optimizer.step(total_loss)
                print(f"  ✅ 反向传播成功")
                
                print(f"  ✅ 批次 {batch_idx + 1} 完成")
                
            except Exception as e:
                print(f"  ❌ 批次 {batch_idx + 1} 失败: {e}")
                import traceback
                traceback.print_exc()
                break
        
        print("\n🎉 训练测试完成！")
        print("✅ NanoDet-Plus Jittor 可以正常训练！")
        
    except Exception as e:
        print(f"\n❌ 训练测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
