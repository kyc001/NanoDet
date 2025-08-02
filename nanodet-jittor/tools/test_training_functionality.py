#!/usr/bin/env python3
"""
验证NanoDet训练功能
使用手动构建的模型进行训练验证
"""

import os
import sys
import logging
import jittor as jt
from pathlib import Path
import time

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def create_model():
    """创建NanoDetPlus模型"""
    from nanodet.model import build_model
    
    model_cfg = {
        'name': 'NanoDetPlus',
        'backbone': {
            'name': 'ShuffleNetV2',
            'model_size': '1.0x',
            'out_stages': [2, 3, 4],
            'activation': 'LeakyReLU',
            'pretrain': True
        },
        'fpn': {
            'name': 'GhostPAN',
            'in_channels': [116, 232, 464],
            'out_channels': 96,
            'kernel_size': 5,
            'num_extra_level': 1,
            'use_depthwise': True,
            'activation': 'LeakyReLU'
        },
        'head': {
            'name': 'NanoDetPlusHead',
            'num_classes': 20,
            'input_channel': 96,
            'feat_channels': 96,
            'stacked_convs': 2,
            'kernel_size': 5,
            'strides': [8, 16, 32, 64],
            'activation': 'LeakyReLU',
            'reg_max': 7,
            'norm_cfg': {'type': 'BN'},
            'loss': {
                'loss_qfl': {
                    'name': 'QualityFocalLoss',
                    'use_sigmoid': True,
                    'beta': 2.0,
                    'loss_weight': 1.0
                },
                'loss_dfl': {
                    'name': 'DistributionFocalLoss',
                    'loss_weight': 0.25
                },
                'loss_bbox': {
                    'name': 'GIoULoss',
                    'loss_weight': 2.0
                }
            }
        },
        'aux_head': {
            'name': 'SimpleConvHead',
            'num_classes': 20,
            'input_channel': 192,
            'feat_channels': 192,
            'stacked_convs': 4,
            'strides': [8, 16, 32, 64],
            'activation': 'LeakyReLU',
            'norm_cfg': {'type': 'BN'}
        },
        'detach_epoch': 10
    }
    
    return build_model(model_cfg)

def create_dummy_data(batch_size=2):
    """创建模拟训练数据"""
    # 创建图像数据
    images = jt.randn(batch_size, 3, 320, 320)
    
    # 创建模拟的GT数据
    gt_meta = {
        'img': images,
        'gt_bboxes': [jt.randn(5, 4) * 100 + 50 for _ in range(batch_size)],  # 5个bbox per image
        'gt_labels': [jt.randint(0, 20, (5,)) for _ in range(batch_size)],  # 5个标签 per image
        'img_info': [
            {'height': 320, 'width': 320, 'id': i} for i in range(batch_size)
        ]
    }
    
    return gt_meta

def test_training_loop():
    """测试训练循环"""
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("测试训练循环")
    logger.info("=" * 50)
    
    try:
        # 创建模型
        logger.info("创建模型...")
        model = create_model()
        model.train()
        logger.info("✅ 模型创建成功")
        
        # 创建优化器
        logger.info("创建优化器...")
        optimizer = jt.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
        logger.info("✅ 优化器创建成功")
        
        # 训练循环
        logger.info("开始训练循环...")
        num_epochs = 3
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # 创建一个batch的数据
            gt_meta = create_dummy_data(batch_size=2)
            
            start_time = time.time()
            
            # 前向传播
            head_out, loss, loss_states = model.forward_train(gt_meta)
            
            # 反向传播
            optimizer.zero_grad()
            optimizer.backward(loss)
            optimizer.step()
            
            end_time = time.time()
            
            logger.info(f"  损失: {loss.item():.4f}")
            logger.info(f"  时间: {end_time - start_time:.2f}s")
            
            # 打印详细的损失信息
            if loss_states:
                for key, value in loss_states.items():
                    if hasattr(value, 'item'):
                        logger.info(f"  {key}: {value.item():.4f}")
                    else:
                        logger.info(f"  {key}: {value}")
        
        logger.info("✅ 训练循环测试成功")
        return True
        
    except Exception as e:
        logger.error(f"❌ 训练循环测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_inference():
    """测试推理功能"""
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("测试推理功能")
    logger.info("=" * 50)
    
    try:
        # 创建模型
        logger.info("创建模型...")
        model = create_model()
        model.eval()
        logger.info("✅ 模型创建成功")
        
        # 测试推理
        logger.info("测试推理...")
        batch_size = 4
        images = jt.randn(batch_size, 3, 320, 320)
        
        start_time = time.time()
        
        with jt.no_grad():
            outputs = model(images)
        
        end_time = time.time()
        
        logger.info(f"输入: {images.shape}")
        logger.info(f"输出: {outputs.shape}")
        logger.info(f"推理时间: {end_time - start_time:.2f}s")
        logger.info(f"平均每张图片: {(end_time - start_time) / batch_size * 1000:.1f}ms")
        
        logger.info("✅ 推理测试成功")
        return True
        
    except Exception as e:
        logger.error(f"❌ 推理测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_model_save_load():
    """测试模型保存和加载"""
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("测试模型保存和加载")
    logger.info("=" * 50)
    
    try:
        # 创建模型
        logger.info("创建模型...")
        model = create_model()
        logger.info("✅ 模型创建成功")
        
        # 保存模型
        save_path = project_root / "work_dirs" / "test_model.pkl"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"保存模型到: {save_path}")
        jt.save(model.state_dict(), str(save_path))
        logger.info("✅ 模型保存成功")
        
        # 创建新模型并加载权重
        logger.info("创建新模型并加载权重...")
        new_model = create_model()
        state_dict = jt.load(str(save_path))
        new_model.load_state_dict(state_dict)
        logger.info("✅ 模型加载成功")
        
        # 验证加载的模型
        logger.info("验证加载的模型...")
        new_model.eval()
        images = jt.randn(1, 3, 320, 320)
        
        with jt.no_grad():
            outputs = new_model(images)
        
        logger.info(f"加载模型输出: {outputs.shape}")
        logger.info("✅ 模型保存加载测试成功")
        
        # 清理文件
        save_path.unlink()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 模型保存加载测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    # 强制使用CPU
    jt.flags.use_cuda = 0
    
    logger = setup_logging()
    logger.info("开始NanoDet训练功能验证...")
    logger.info(f"Jittor CUDA available: {jt.has_cuda}")
    logger.info(f"Jittor using CUDA: {jt.flags.use_cuda}")
    
    # 测试推理功能
    inference_success = test_inference()
    if not inference_success:
        logger.error("❌ 推理测试失败")
        return False
    
    # 测试训练循环
    training_success = test_training_loop()
    if not training_success:
        logger.error("❌ 训练测试失败")
        return False
    
    # 测试模型保存加载
    save_load_success = test_model_save_load()
    if not save_load_success:
        logger.error("❌ 模型保存加载测试失败")
        return False
    
    logger.info("🎉 所有训练功能测试通过！")
    logger.info("✅ NanoDet Jittor版本核心功能验证成功！")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
