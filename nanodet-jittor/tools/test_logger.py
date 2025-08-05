#!/usr/bin/env python3
"""
测试重构后的logger功能
验证日志系统是否正常工作
"""

import os
import sys
import time
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_basic_logger():
    """测试基础Logger类"""
    print("=" * 50)
    print("测试基础Logger类")
    print("=" * 50)
    
    from nanodet.util.logger import Logger
    
    # 创建测试目录
    test_dir = "./test_logs/basic_logger"
    
    # 创建logger
    logger = Logger(save_dir=test_dir, name="TestLogger")
    
    # 测试各种日志级别
    logger.info("这是一条信息日志")
    logger.warning("这是一条警告日志")
    logger.error("这是一条错误日志")
    
    # 测试指标记录
    for step in range(5):
        metrics = {
            'loss': 10.0 - step * 2,
            'accuracy': 0.5 + step * 0.1,
            'lr': 0.01 - step * 0.001
        }
        logger.log_metrics(metrics, step)
        time.sleep(0.1)
    
    print("✅ 基础Logger测试完成")
    return True

def test_lightning_logger():
    """测试NanoDetLightningLogger类"""
    print("=" * 50)
    print("测试NanoDetLightningLogger类")
    print("=" * 50)
    
    from nanodet.util.logger import NanoDetLightningLogger
    
    # 创建测试目录
    test_dir = "./test_logs/lightning_logger"
    
    # 创建logger
    logger = NanoDetLightningLogger(save_dir=test_dir, name="NanoDetTest")
    
    # 测试基本日志功能
    logger.info("开始测试NanoDetLightningLogger")
    logger.warning("这是一个警告")
    
    # 测试超参数记录
    hyperparams = {
        'learning_rate': 0.01,
        'batch_size': 32,
        'epochs': 100,
        'model': 'nanodet-plus-m'
    }
    logger.log_hyperparams(hyperparams)
    
    # 测试配置文件保存
    class MockConfig:
        def __init__(self):
            self.model = "nanodet-plus-m"
            self.lr = 0.01
            
        def dump(self, stream):
            stream.write(f"model: {self.model}\n")
            stream.write(f"lr: {self.lr}\n")
    
    mock_cfg = MockConfig()
    logger.dump_cfg(mock_cfg)
    
    # 测试验证指标记录
    for epoch in range(3):
        val_metrics = {
            'val_loss': 5.0 - epoch * 1.5,
            'val_mAP': 0.3 + epoch * 0.15,
            'val_mAP_50': 0.4 + epoch * 0.12
        }
        logger.log_metrics(val_metrics, epoch)
        time.sleep(0.1)
    
    # 测试完成
    logger.finalize("success")
    
    print("✅ NanoDetLightningLogger测试完成")
    return True

def test_metrics_visualizer():
    """测试MetricsVisualizer类"""
    print("=" * 50)
    print("测试MetricsVisualizer类")
    print("=" * 50)
    
    from nanodet.util.logger import MetricsVisualizer
    
    # 创建测试目录
    test_dir = "./test_logs/visualizer"
    os.makedirs(test_dir, exist_ok=True)
    
    # 创建可视化器
    visualizer = MetricsVisualizer(save_dir=test_dir)
    
    # 添加一些模拟指标
    for step in range(20):
        # 模拟训练损失下降
        train_loss = 10.0 * (0.9 ** step) + 0.1
        val_loss = 8.0 * (0.85 ** step) + 0.2
        
        # 模拟mAP上升
        map_score = 0.8 * (1 - 0.9 ** step)
        
        visualizer.add_scalar('train_loss', train_loss, step)
        visualizer.add_scalar('val_loss', val_loss, step)
        visualizer.add_scalar('mAP', map_score, step)
    
    # 保存指标数据
    metrics_file = visualizer.save_metrics()
    print(f"指标数据已保存到: {metrics_file}")
    
    # 尝试生成图表
    plot_file = visualizer.plot_metrics(save_plots=True)
    if plot_file:
        print(f"训练曲线已保存到: {plot_file}")
    else:
        print("图表生成失败（可能是matplotlib未安装）")
    
    print("✅ MetricsVisualizer测试完成")
    return True

def test_logger_factory():
    """测试logger工厂函数"""
    print("=" * 50)
    print("测试logger工厂函数")
    print("=" * 50)
    
    from nanodet.util.logger import create_logger
    
    # 测试创建基础logger
    basic_logger = create_logger(
        save_dir="./test_logs/factory_basic",
        name="FactoryBasic",
        use_lightning_logger=False
    )
    basic_logger.info("基础logger创建成功")
    
    # 测试创建Lightning logger
    lightning_logger = create_logger(
        save_dir="./test_logs/factory_lightning",
        name="FactoryLightning",
        use_lightning_logger=True
    )
    lightning_logger.info("Lightning logger创建成功")
    
    print("✅ Logger工厂函数测试完成")
    return True

def test_file_outputs():
    """测试文件输出"""
    print("=" * 50)
    print("检查生成的文件")
    print("=" * 50)
    
    test_dirs = [
        "./test_logs/basic_logger",
        "./test_logs/lightning_logger",
        "./test_logs/visualizer",
        "./test_logs/factory_basic",
        "./test_logs/factory_lightning"
    ]
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            print(f"\n📁 {test_dir}:")
            for file in os.listdir(test_dir):
                file_path = os.path.join(test_dir, file)
                if os.path.isfile(file_path):
                    size = os.path.getsize(file_path)
                    print(f"  📄 {file} ({size} bytes)")
        else:
            print(f"❌ 目录不存在: {test_dir}")
    
    return True

def main():
    """主测试函数"""
    print("🚀 开始测试重构后的logger功能...")
    
    tests = [
        ("基础Logger", test_basic_logger),
        ("NanoDetLightningLogger", test_lightning_logger),
        ("MetricsVisualizer", test_metrics_visualizer),
        ("Logger工厂函数", test_logger_factory),
        ("文件输出检查", test_file_outputs)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            print(f"\n🧪 运行测试: {test_name}")
            success = test_func()
            results.append((test_name, success))
            if success:
                print(f"✅ {test_name} 测试通过")
            else:
                print(f"❌ {test_name} 测试失败")
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
            results.append((test_name, False))
    
    # 总结结果
    print("\n" + "=" * 50)
    print("测试结果总结")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\n总计: {passed}/{total} 个测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！Logger重构成功！")
        return True
    else:
        print("⚠️ 部分测试失败，需要检查问题")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
