#!/usr/bin/env python3
"""
简单的CPU模式测试脚本
验证Jittor在CPU模式下的基本功能
"""

import os
import sys
import logging
import jittor as jt
from pathlib import Path

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def test_basic_cpu_operations():
    """测试基本的CPU操作"""
    logger = setup_logging()
    
    # 强制使用CPU
    jt.flags.use_cuda = 0
    logger.info("强制设置为CPU模式")
    logger.info(f"Jittor CUDA available: {jt.has_cuda}")
    logger.info(f"Jittor using CUDA: {jt.flags.use_cuda}")
    
    try:
        # 测试基本张量操作
        logger.info("测试基本张量操作...")
        
        # 创建张量
        x = jt.randn(4, 3, 32, 32)
        y = jt.randn(4, 10)
        logger.info(f"创建张量 x: {x.shape}, y: {y.shape}")
        
        # 测试卷积操作
        logger.info("测试卷积操作...")
        conv = jt.nn.Conv2d(3, 16, 3, padding=1)
        conv_out = conv(x)
        logger.info(f"卷积输出: {conv_out.shape}")
        
        # 测试池化操作
        logger.info("测试池化操作...")
        pool = jt.nn.AdaptiveAvgPool2d((1, 1))
        pool_out = pool(conv_out)
        logger.info(f"池化输出: {pool_out.shape}")
        
        # 测试全连接层
        logger.info("测试全连接层...")
        fc = jt.nn.Linear(16, 10)
        pool_flat = pool_out.view(pool_out.shape[0], -1)
        fc_out = fc(pool_flat)
        logger.info(f"全连接输出: {fc_out.shape}")
        
        # 测试损失计算
        logger.info("测试损失计算...")
        loss_fn = jt.nn.MSELoss()
        loss = loss_fn(fc_out, y)
        logger.info(f"损失值: {loss.item():.4f}")
        
        # 测试反向传播
        logger.info("测试反向传播...")
        optimizer = jt.optim.SGD([conv.weight, conv.bias, fc.weight, fc.bias], lr=0.01)
        optimizer.zero_grad()
        optimizer.backward(loss)
        optimizer.step()
        logger.info("反向传播完成")
        
        # 测试简单的训练循环
        logger.info("测试简单训练循环...")
        for i in range(3):
            # 前向传播
            conv_out = conv(x)
            pool_out = pool(conv_out)
            pool_flat = pool_out.view(pool_out.shape[0], -1)
            fc_out = fc(pool_flat)
            
            # 计算损失
            loss = loss_fn(fc_out, y)
            
            # 反向传播
            optimizer.zero_grad()
            optimizer.backward(loss)
            optimizer.step()
            
            logger.info(f"Epoch {i+1}: loss = {loss.item():.4f}")
        
        logger.info("✅ CPU模式基本操作测试成功！")
        return True
        
    except Exception as e:
        logger.error(f"❌ CPU模式测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_model_creation():
    """测试模型创建"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("测试简单模型创建...")
        
        # 创建一个简单的CNN模型
        class SimpleCNN(jt.nn.Module):
            def __init__(self):
                super(SimpleCNN, self).__init__()
                self.conv1 = jt.nn.Conv2d(3, 16, 3, padding=1)
                self.conv2 = jt.nn.Conv2d(16, 32, 3, padding=1)
                self.pool = jt.nn.AdaptiveAvgPool2d((1, 1))
                self.fc = jt.nn.Linear(32, 10)
                self.relu = jt.nn.ReLU()
            
            def execute(self, x):
                x = self.relu(self.conv1(x))
                x = self.relu(self.conv2(x))
                x = self.pool(x)
                x = x.view(x.shape[0], -1)
                x = self.fc(x)
                return x
        
        model = SimpleCNN()
        logger.info("模型创建成功")
        
        # 测试模型前向传播
        x = jt.randn(2, 3, 32, 32)
        output = model(x)
        logger.info(f"模型输出: {output.shape}")
        
        # 测试模型训练
        target = jt.randn(2, 10)
        loss_fn = jt.nn.MSELoss()
        optimizer = jt.optim.SGD(model.parameters(), lr=0.01)
        
        for i in range(3):
            output = model(x)
            loss = loss_fn(output, target)
            
            optimizer.zero_grad()
            optimizer.backward(loss)
            optimizer.step()
            
            logger.info(f"训练步骤 {i+1}: loss = {loss.item():.4f}")
        
        logger.info("✅ 模型创建和训练测试成功！")
        return True
        
    except Exception as e:
        logger.error(f"❌ 模型测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    logger = setup_logging()
    
    logger.info("开始CPU模式综合测试...")
    
    # 测试基本操作
    basic_success = test_basic_cpu_operations()
    
    # 测试模型创建
    model_success = test_model_creation()
    
    if basic_success and model_success:
        logger.info("🎉 所有CPU模式测试通过！")
        print("✅ CPU模式验证成功！Jittor可以在CPU模式下正常工作。")
        return True
    else:
        logger.error("❌ 部分测试失败")
        print("❌ CPU模式验证失败！")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
