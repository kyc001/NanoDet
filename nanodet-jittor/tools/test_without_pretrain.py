#!/usr/bin/env python3
"""
测试不使用预训练权重的模型构建
验证是否是预训练权重加载导致的段错误
"""

import os
import sys
import logging
import jittor as jt
from pathlib import Path
import traceback

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

def test_backbone_without_pretrain():
    """测试不使用预训练权重的Backbone"""
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("测试不使用预训练权重的Backbone")
    logger.info("=" * 50)
    
    try:
        from nanodet.model.backbone import build_backbone
        
        # 禁用预训练权重
        backbone_cfg = {
            'name': 'ShuffleNetV2',
            'model_size': '1.0x',
            'out_stages': [2, 3, 4],
            'activation': 'LeakyReLU',
            'pretrain': False  # 关键：禁用预训练
        }
        
        logger.info(f"构建Backbone（无预训练）: {backbone_cfg}")
        backbone = build_backbone(backbone_cfg)
        logger.info("✅ Backbone构建成功（无预训练）")
        
        # 测试前向传播
        x = jt.randn(1, 3, 320, 320)
        outputs = backbone(x)
        logger.info(f"前向传播成功，输出: {[out.shape for out in outputs]}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Backbone测试失败: {e}")
        traceback.print_exc()
        return False

def test_full_model_without_pretrain():
    """测试完整模型（无预训练）"""
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("测试完整模型（无预训练）")
    logger.info("=" * 50)
    
    try:
        from nanodet.model import build_model
        
        # 手动构建配置，禁用预训练
        model_cfg = {
            'name': 'NanoDetPlus',
            'backbone': {
                'name': 'ShuffleNetV2',
                'model_size': '1.0x',
                'out_stages': [2, 3, 4],
                'activation': 'LeakyReLU',
                'pretrain': False  # 关键：禁用预训练
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
        
        logger.info("构建完整模型（无预训练）...")
        model = build_model(model_cfg)
        logger.info("✅ 完整模型构建成功（无预训练）")
        
        # 测试推理
        x = jt.randn(1, 3, 320, 320)
        model.eval()
        with jt.no_grad():
            outputs = model(x)
        logger.info(f"推理成功，输出: {outputs.shape}")
        
        return model
        
    except Exception as e:
        logger.error(f"❌ 完整模型测试失败: {e}")
        traceback.print_exc()
        return None

def test_config_file_without_pretrain():
    """测试修改配置文件禁用预训练"""
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("测试配置文件（禁用预训练）")
    logger.info("=" * 50)
    
    try:
        from nanodet.util.config import load_config
        from nanodet.model import build_model
        
        # 加载配置文件
        config_path = project_root / "config" / "nanodet-plus-m_320_voc.yml"
        cfg = load_config(str(config_path))
        logger.info("✅ 配置文件加载成功")
        
        # 修改配置，禁用预训练
        cfg.model.arch.backbone.pretrain = False
        logger.info("修改配置：禁用预训练权重")
        
        # 构建模型
        logger.info("构建模型（配置文件，无预训练）...")
        model = build_model(cfg.model)
        logger.info("✅ 配置文件模型构建成功（无预训练）")
        
        # 测试推理
        x = jt.randn(1, 3, 320, 320)
        model.eval()
        with jt.no_grad():
            outputs = model(x)
        logger.info(f"推理成功，输出: {outputs.shape}")
        
        return model
        
    except Exception as e:
        logger.error(f"❌ 配置文件模型测试失败: {e}")
        traceback.print_exc()
        return None

def test_training_without_pretrain():
    """测试训练（无预训练）"""
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("测试训练（无预训练）")
    logger.info("=" * 50)
    
    try:
        # 使用手动构建的模型
        model = test_full_model_without_pretrain()
        if model is None:
            return False
        
        model.train()
        
        # 创建优化器
        optimizer = jt.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
        
        # 创建模拟数据
        batch_size = 4
        images = jt.randn(batch_size, 3, 320, 320)
        gt_meta = {
            'img': images,
            'gt_bboxes': [jt.randn(3, 4) * 100 + 50 for _ in range(batch_size)],
            'gt_labels': [jt.randint(0, 20, (3,)) for _ in range(batch_size)],
            'img_info': [
                {'height': 320, 'width': 320, 'id': i} for i in range(batch_size)
            ]
        }
        
        # 训练几个步骤
        logger.info("开始训练测试...")
        for step in range(3):
            # 前向传播
            head_out, loss, loss_states = model.forward_train(gt_meta)
            
            # 反向传播
            optimizer.zero_grad()
            optimizer.backward(loss)
            optimizer.step()
            
            logger.info(f"Step {step + 1}: loss = {loss.item():.4f}")
            
            if loss_states:
                for key, value in loss_states.items():
                    if hasattr(value, 'item'):
                        logger.info(f"  {key}: {value.item():.4f}")
        
        logger.info("✅ 训练测试成功（无预训练）")
        return True
        
    except Exception as e:
        logger.error(f"❌ 训练测试失败: {e}")
        traceback.print_exc()
        return False

def main():
    """主函数"""
    # 强制CPU模式
    jt.flags.use_cuda = 0
    
    logger = setup_logging()
    logger.info("开始测试无预训练权重的模型...")
    
    # 测试1: Backbone
    backbone_ok = test_backbone_without_pretrain()
    if not backbone_ok:
        logger.error("Backbone测试失败")
        return False
    
    # 测试2: 完整模型
    model_ok = test_full_model_without_pretrain()
    if model_ok is None:
        logger.error("完整模型测试失败")
        return False
    
    # 测试3: 配置文件模型
    config_model_ok = test_config_file_without_pretrain()
    if config_model_ok is None:
        logger.error("配置文件模型测试失败")
        return False
    
    # 测试4: 训练
    training_ok = test_training_without_pretrain()
    if not training_ok:
        logger.error("训练测试失败")
        return False
    
    logger.info("🎉 所有无预训练测试通过！")
    logger.info("✅ 确认问题出在预训练权重加载上")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
