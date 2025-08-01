#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
模块化使用示例
展示如何使用NanoDet Jittor版本的模块化系统
"""

import os
import sys
import jittor as jt

# 添加路径
sys.path.append('/home/kyc/project/nanodet/nanodet-jittor')

# 模块化导入 - 就像PyTorch版本一样
from nanodet.util import (
    get_logger, setup_logger,
    Config, load_config, DEFAULT_CONFIG,
    load_pytorch_checkpoint, save_checkpoint,
    COCOEvaluator, SimpleEvaluator
)
from nanodet.model.arch.nanodet_plus import NanoDetPlus


def main():
    """主函数 - 展示模块化使用"""
    print("🚀 NanoDet Jittor 模块化使用示例")
    print("=" * 60)
    
    # 1. 设置日志系统
    logger = setup_logger('nanodet_example', level='INFO')
    logger.info("开始模块化使用示例")
    
    # 2. 配置管理
    logger.info("📋 配置管理示例")
    
    # 使用默认配置
    config = DEFAULT_CONFIG
    logger.info(f"默认配置加载成功，模型名称: {config.model.name}")
    
    # 修改配置
    config.model.backbone.pretrain = True
    config.data.train.input_size = [416, 416]
    logger.info(f"配置修改成功，输入尺寸: {config.data.train.input_size}")
    
    # 3. 模型创建
    logger.info("🔧 模型创建示例")
    
    model = NanoDetPlus(
        config.model.backbone,
        config.model.fpn,
        {'name': 'SimpleConvHead', 'num_classes': 20, 'input_channel': 192, 'feat_channels': 192, 'stacked_convs': 4, 'strides': [8, 16, 32, 64], 'activation': 'LeakyReLU', 'reg_max': 7},
        config.model.head
    )
    
    logger.info("模型创建成功")
    
    # 4. 检查点管理
    logger.info("💾 检查点管理示例")
    
    # 加载PyTorch检查点
    pytorch_checkpoint = "/home/kyc/project/nanodet/nanodet-pytorch/workspace/nanodet-plus-m_320_voc_bs64/model_best/model_best.ckpt"
    
    if os.path.exists(pytorch_checkpoint):
        result = load_pytorch_checkpoint(model, pytorch_checkpoint)
        logger.info(f"PyTorch检查点加载成功: {result['loaded_count']}/{result['total_count']} 参数")
        
        if result['epoch']:
            logger.info(f"检查点来自第 {result['epoch']} 轮")
        
        if result['metrics']:
            logger.info(f"检查点指标: {result['metrics']}")
    else:
        logger.warning("PyTorch检查点不存在，跳过加载")
    
    # 保存Jittor检查点
    save_path = "examples/example_checkpoint.pkl"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    save_checkpoint(
        model,
        epoch=1,
        metrics={'mAP': 0.275, 'mAP_50': 0.483},
        save_path=save_path,
        config=config.to_dict()
    )
    
    logger.info(f"Jittor检查点保存成功: {save_path}")
    
    # 5. 模型推理示例
    logger.info("🔍 模型推理示例")
    
    model.eval()
    
    # 创建测试输入
    import numpy as np
    np.random.seed(42)
    input_data = np.random.randn(1, 3, 320, 320).astype(np.float32)
    jittor_input = jt.array(input_data)
    
    with jt.no_grad():
        output = model(jittor_input)
        
        # 分析输出
        cls_preds = output[:, :, :20]
        cls_scores = jt.sigmoid(cls_preds)
        
        max_conf = float(cls_scores.max().numpy())
        mean_conf = float(cls_scores.mean().numpy())
        
        logger.info(f"推理成功 - 最高置信度: {max_conf:.6f}, 平均置信度: {mean_conf:.6f}")
        
        # 统计检测数量
        for threshold in [0.01, 0.05, 0.1]:
            max_scores = jt.max(cls_scores, dim=2)[0]
            valid_detections = int((max_scores > threshold).sum().numpy())
            logger.info(f"阈值 {threshold}: {valid_detections} 个检测")
    
    # 6. 评估器示例
    logger.info("📊 评估器示例")
    
    # 简单评估器
    simple_evaluator = SimpleEvaluator(num_classes=20)
    
    # 模拟一些检测结果和真值
    detections = [
        {'category_id': 1, 'bbox': [10, 10, 50, 50], 'score': 0.9},
        {'category_id': 2, 'bbox': [60, 60, 40, 40], 'score': 0.8}
    ]
    
    ground_truths = [
        {'category_id': 1, 'bbox': [12, 12, 48, 48]},
        {'category_id': 2, 'bbox': [65, 65, 35, 35]}
    ]
    
    simple_evaluator.add_result(1, detections, ground_truths)
    metrics = simple_evaluator.evaluate()
    
    logger.info(f"简单评估结果 - mAP: {metrics['mAP']:.4f}")
    
    # COCO评估器（如果可用）
    try:
        ann_file = "/home/kyc/project/nanodet/nanodet-pytorch/data/annotations/voc_val.json"
        if os.path.exists(ann_file):
            coco_evaluator = COCOEvaluator(ann_file)
            coco_evaluator.add_result(1, detections)
            logger.info("COCO评估器创建成功")
        else:
            logger.info("COCO标注文件不存在，跳过COCO评估器示例")
    except Exception as e:
        logger.warning(f"COCO评估器不可用: {e}")
    
    # 7. 配置保存示例
    logger.info("💾 配置保存示例")
    
    config_save_path = "examples/example_config.yaml"
    from nanodet.util import save_config
    save_config(config, config_save_path)
    
    logger.info(f"配置保存成功: {config_save_path}")
    
    # 8. 总结
    logger.info("✅ 模块化使用示例完成")
    
    print(f"\n🎯 模块化系统特性:")
    print(f"  ✅ 统一的日志系统")
    print(f"  ✅ 灵活的配置管理")
    print(f"  ✅ 完整的检查点管理")
    print(f"  ✅ PyTorch兼容的权重加载")
    print(f"  ✅ 标准化的评估系统")
    print(f"  ✅ 模块化导入支持")
    
    print(f"\n📦 使用方式:")
    print(f"```python")
    print(f"from nanodet.util import get_logger, Config, load_pytorch_checkpoint")
    print(f"from nanodet.model.arch.nanodet_plus import NanoDetPlus")
    print(f"")
    print(f"# 创建模型")
    print(f"model = NanoDetPlus(...)")
    print(f"")
    print(f"# 加载权重")
    print(f"load_pytorch_checkpoint(model, 'checkpoint.ckpt')")
    print(f"")
    print(f"# 设置日志")
    print(f"logger = get_logger('my_app')")
    print(f"```")


if __name__ == '__main__':
    main()
