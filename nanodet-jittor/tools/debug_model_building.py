#!/usr/bin/env python3
"""
调试模型构建过程
逐步定位段错误的具体位置
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
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def test_config_loading():
    """测试配置加载"""
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("测试配置加载")
    logger.info("=" * 50)
    
    try:
        from nanodet.util.config import load_config
        
        config_path = project_root / "config" / "nanodet-plus-m_320_voc.yml"
        logger.info(f"加载配置: {config_path}")
        
        cfg = load_config(str(config_path))
        logger.info("✅ 配置加载成功")
        
        # 检查模型配置
        logger.info(f"模型配置键: {list(cfg.model.keys()) if hasattr(cfg.model, 'keys') else 'N/A'}")
        
        return cfg
        
    except Exception as e:
        logger.error(f"❌ 配置加载失败: {e}")
        traceback.print_exc()
        return None

def test_model_components_separately(cfg):
    """分别测试模型各个组件"""
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("分别测试模型组件")
    logger.info("=" * 50)
    
    try:
        # 测试backbone
        logger.info("测试Backbone构建...")
        from nanodet.model.backbone import build_backbone
        
        backbone_cfg = cfg.model.arch.backbone
        logger.info(f"Backbone配置: {backbone_cfg}")
        
        backbone = build_backbone(backbone_cfg)
        logger.info("✅ Backbone构建成功")
        
        # 测试FPN
        logger.info("测试FPN构建...")
        from nanodet.model.fpn import build_fpn
        
        fpn_cfg = cfg.model.arch.fpn
        logger.info(f"FPN配置: {fpn_cfg}")
        
        fpn = build_fpn(fpn_cfg)
        logger.info("✅ FPN构建成功")
        
        # 测试Head
        logger.info("测试Head构建...")
        from nanodet.model.head import build_head
        
        head_cfg = cfg.model.arch.head
        logger.info(f"Head配置: {head_cfg}")
        
        head = build_head(head_cfg)
        logger.info("✅ Head构建成功")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 组件构建失败: {e}")
        traceback.print_exc()
        return False

def test_build_model_function(cfg):
    """测试build_model函数"""
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("测试build_model函数")
    logger.info("=" * 50)
    
    try:
        from nanodet.model import build_model
        
        logger.info("调用build_model...")
        logger.info(f"模型配置: {cfg.model}")
        
        # 这里可能会段错误
        model = build_model(cfg.model)
        
        logger.info("✅ build_model成功")
        return model
        
    except Exception as e:
        logger.error(f"❌ build_model失败: {e}")
        traceback.print_exc()
        return None

def test_manual_model_creation():
    """手动创建模型（不依赖配置文件）"""
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("手动创建模型")
    logger.info("=" * 50)
    
    try:
        from nanodet.model import build_model
        
        # 手动构建配置
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
        
        logger.info("手动构建模型...")
        model = build_model(model_cfg)
        logger.info("✅ 手动模型构建成功")
        
        return model
        
    except Exception as e:
        logger.error(f"❌ 手动模型构建失败: {e}")
        traceback.print_exc()
        return None

def compare_configs(cfg):
    """比较配置文件和手动配置的差异"""
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("比较配置差异")
    logger.info("=" * 50)
    
    try:
        # 打印配置文件的模型结构
        logger.info("配置文件模型结构:")
        logger.info(f"  模型名称: {cfg.model.arch.name if hasattr(cfg.model.arch, 'name') else 'N/A'}")
        logger.info(f"  Backbone: {cfg.model.arch.backbone.name if hasattr(cfg.model.arch.backbone, 'name') else 'N/A'}")
        logger.info(f"  FPN: {cfg.model.arch.fpn.name if hasattr(cfg.model.arch.fpn, 'name') else 'N/A'}")
        logger.info(f"  Head: {cfg.model.arch.head.name if hasattr(cfg.model.arch.head, 'name') else 'N/A'}")
        
        # 检查是否有特殊配置
        if hasattr(cfg.model.arch, 'aux_head'):
            logger.info(f"  AuxHead: {cfg.model.arch.aux_head.name if hasattr(cfg.model.arch.aux_head, 'name') else 'N/A'}")
        
        if hasattr(cfg.model.arch, 'detach_epoch'):
            logger.info(f"  DetachEpoch: {cfg.model.arch.detach_epoch}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 配置比较失败: {e}")
        traceback.print_exc()
        return False

def main():
    """主函数"""
    # 强制CPU模式
    jt.flags.use_cuda = 0
    
    logger = setup_logging()
    logger.info("开始模型构建调试...")
    
    # 步骤1: 测试配置加载
    cfg = test_config_loading()
    if cfg is None:
        logger.error("配置加载失败，停止调试")
        return False
    
    # 步骤2: 比较配置
    compare_configs(cfg)
    
    # 步骤3: 分别测试组件
    components_ok = test_model_components_separately(cfg)
    if not components_ok:
        logger.error("组件测试失败，停止调试")
        return False
    
    # 步骤4: 测试手动模型创建
    manual_model = test_manual_model_creation()
    if manual_model is None:
        logger.error("手动模型创建失败")
        return False
    
    # 步骤5: 测试配置文件模型创建
    logger.info("准备测试配置文件模型创建...")
    logger.info("如果这里段错误，说明配置文件中有特殊参数导致问题")
    
    config_model = test_build_model_function(cfg)
    if config_model is None:
        logger.error("配置文件模型创建失败")
        return False
    
    logger.info("🎉 所有测试通过！")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
