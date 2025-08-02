#!/usr/bin/env python3
"""
逐步验证NanoDet各个组件
严格对照PyTorch版本进行测试
"""

import os
import sys
import logging
import jittor as jt
from pathlib import Path

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

def test_backbone_component():
    """测试Backbone组件"""
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("测试Backbone组件")
    logger.info("=" * 50)
    
    try:
        from nanodet.model.backbone import build_backbone
        
        # 严格按照配置文件的backbone配置
        backbone_cfg = {
            'name': 'ShuffleNetV2',
            'model_size': '1.0x',
            'out_stages': [2, 3, 4],
            'activation': 'LeakyReLU',
            'pretrain': True
        }
        
        logger.info(f"构建Backbone: {backbone_cfg}")
        backbone = build_backbone(backbone_cfg)
        logger.info("✅ Backbone构建成功")
        
        # 测试前向传播
        x = jt.randn(1, 3, 320, 320)
        logger.info(f"输入张量: {x.shape}")
        
        with jt.no_grad():
            outputs = backbone(x)
        
        logger.info(f"Backbone输出层数: {len(outputs)}")
        for i, output in enumerate(outputs):
            logger.info(f"  输出{i}: {output.shape}")
        
        # 验证输出通道数是否符合预期
        expected_channels = [116, 232, 464]  # ShuffleNetV2 1.0x的输出通道
        if len(outputs) == len(expected_channels):
            for i, (output, expected) in enumerate(zip(outputs, expected_channels)):
                if output.shape[1] == expected:
                    logger.info(f"  ✅ 输出{i}通道数正确: {expected}")
                else:
                    logger.error(f"  ❌ 输出{i}通道数错误: 期望{expected}, 实际{output.shape[1]}")
                    return False
        else:
            logger.error(f"❌ 输出层数错误: 期望{len(expected_channels)}, 实际{len(outputs)}")
            return False
        
        logger.info("✅ Backbone组件测试通过")
        return True
        
    except Exception as e:
        logger.error(f"❌ Backbone测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_fpn_component():
    """测试FPN组件"""
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("测试FPN组件")
    logger.info("=" * 50)
    
    try:
        from nanodet.model.backbone import build_backbone
        from nanodet.model.fpn import build_fpn
        
        # 先构建backbone
        backbone_cfg = {
            'name': 'ShuffleNetV2',
            'model_size': '1.0x',
            'out_stages': [2, 3, 4],
            'activation': 'LeakyReLU',
            'pretrain': True
        }
        backbone = build_backbone(backbone_cfg)
        
        # 严格按照配置文件的FPN配置
        fpn_cfg = {
            'name': 'GhostPAN',
            'in_channels': [116, 232, 464],
            'out_channels': 96,
            'kernel_size': 5,
            'num_extra_level': 1,
            'use_depthwise': True,
            'activation': 'LeakyReLU'
        }
        
        logger.info(f"构建FPN: {fpn_cfg}")
        fpn = build_fpn(fpn_cfg)
        logger.info("✅ FPN构建成功")
        
        # 测试前向传播
        x = jt.randn(1, 3, 320, 320)
        logger.info(f"输入张量: {x.shape}")
        
        with jt.no_grad():
            backbone_outputs = backbone(x)
            fpn_outputs = fpn(backbone_outputs)
        
        logger.info(f"FPN输出层数: {len(fpn_outputs)}")
        for i, output in enumerate(fpn_outputs):
            logger.info(f"  输出{i}: {output.shape}")
        
        # 验证输出通道数是否符合预期
        expected_out_channels = 96
        expected_num_levels = 4  # 3个backbone输出 + 1个extra level
        
        if len(fpn_outputs) == expected_num_levels:
            logger.info(f"✅ FPN输出层数正确: {expected_num_levels}")
            for i, output in enumerate(fpn_outputs):
                if output.shape[1] == expected_out_channels:
                    logger.info(f"  ✅ 输出{i}通道数正确: {expected_out_channels}")
                else:
                    logger.error(f"  ❌ 输出{i}通道数错误: 期望{expected_out_channels}, 实际{output.shape[1]}")
                    return False
        else:
            logger.error(f"❌ FPN输出层数错误: 期望{expected_num_levels}, 实际{len(fpn_outputs)}")
            return False
        
        logger.info("✅ FPN组件测试通过")
        return True
        
    except Exception as e:
        logger.error(f"❌ FPN测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_head_component():
    """测试Head组件"""
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("测试Head组件")
    logger.info("=" * 50)
    
    try:
        from nanodet.model.backbone import build_backbone
        from nanodet.model.fpn import build_fpn
        from nanodet.model.head import build_head
        
        # 构建backbone和FPN
        backbone_cfg = {
            'name': 'ShuffleNetV2',
            'model_size': '1.0x',
            'out_stages': [2, 3, 4],
            'activation': 'LeakyReLU',
            'pretrain': True
        }
        backbone = build_backbone(backbone_cfg)
        
        fpn_cfg = {
            'name': 'GhostPAN',
            'in_channels': [116, 232, 464],
            'out_channels': 96,
            'kernel_size': 5,
            'num_extra_level': 1,
            'use_depthwise': True,
            'activation': 'LeakyReLU'
        }
        fpn = build_fpn(fpn_cfg)
        
        # 严格按照配置文件的Head配置
        head_cfg = {
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
        }
        
        logger.info(f"构建Head: {head_cfg}")
        head = build_head(head_cfg)
        logger.info("✅ Head构建成功")
        
        # 测试前向传播
        x = jt.randn(1, 3, 320, 320)
        logger.info(f"输入张量: {x.shape}")
        
        with jt.no_grad():
            backbone_outputs = backbone(x)
            fpn_outputs = fpn(backbone_outputs)
            head_outputs = head(fpn_outputs)
        
        logger.info(f"Head输出: {len(head_outputs)} 个张量")
        for i, output in enumerate(head_outputs):
            logger.info(f"  输出{i}: {output.shape}")
        
        logger.info("✅ Head组件测试通过")
        return True
        
    except Exception as e:
        logger.error(f"❌ Head测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    # 强制使用CPU
    jt.flags.use_cuda = 0
    
    logger = setup_logging()
    logger.info("开始逐步验证NanoDet组件...")
    logger.info(f"Jittor CUDA available: {jt.has_cuda}")
    logger.info(f"Jittor using CUDA: {jt.flags.use_cuda}")
    
    # 逐步测试各个组件
    backbone_success = test_backbone_component()
    if not backbone_success:
        logger.error("❌ Backbone测试失败，停止后续测试")
        return False
    
    fpn_success = test_fpn_component()
    if not fpn_success:
        logger.error("❌ FPN测试失败，停止后续测试")
        return False
    
    head_success = test_head_component()
    if not head_success:
        logger.error("❌ Head测试失败")
        return False
    
    logger.info("🎉 所有组件测试通过！")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
