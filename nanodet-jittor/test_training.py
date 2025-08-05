#!/usr/bin/env python3
"""
简单的训练测试脚本
测试 NanoDet-Plus 的训练是否能正常进行
"""

import os
import sys
import time

# 将项目根目录添加到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import jittor as jt
from nanodet.trainer import TrainingTask
from nanodet.util import load_config, Logger

def test_training():
    """测试训练流程"""
    print("🚀 开始测试 NanoDet-Plus 训练...")
    
    # 设置 Jittor
    jt.flags.use_cuda = 1
    
    # 加载配置
    config_path = "config/nanodet-plus-m_320_voc_bs64_50epochs.yml"
    cfg = load_config(config_path, {})
    
    # 创建日志记录器
    logger = Logger(-1, use_tensorboard=False)
    
    # 构建训练器
    trainer = TrainingTask(cfg, logger)
    
    print("✅ 训练器创建成功")
    
    # 测试一个训练步骤
    try:
        print("🔄 开始测试训练步骤...")
        
        # 运行几个训练步骤
        for i in range(3):
            print(f"  步骤 {i+1}/3...")
            trainer.run_step()
            print(f"  ✅ 步骤 {i+1} 完成")
            
        print("🎉 训练测试成功！")
        return True
        
    except Exception as e:
        print(f"❌ 训练测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_training()
    if success:
        print("\n🎯 总结：")
        print("✅ DepthwiseConv 修复成功")
        print("✅ 标签问题修复成功") 
        print("✅ 训练流程正常")
        print("✅ NanoDet-Plus Jittor 版本可以正常训练！")
        sys.exit(0)
    else:
        print("\n❌ 测试失败，需要进一步调试")
        sys.exit(1)
