#!/usr/bin/env python3
"""
CPU模式训练验证脚本
测试Jittor在CPU模式下的训练功能
"""

import os
import sys
import logging
import jittor as jt
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from nanodet.util import get_logger, load_config, Config
from nanodet.trainer import build_trainer

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def test_cpu_training():
    """测试CPU模式训练"""
    logger = setup_logging()
    
    # 强制使用CPU
    jt.flags.use_cuda = 0
    logger.info("强制设置为CPU模式")
    logger.info(f"Jittor CUDA available: {jt.has_cuda}")
    logger.info(f"Jittor using CUDA: {jt.flags.use_cuda}")
    
    # 配置文件路径
    config_path = project_root / "config" / "nanodet-plus-m_320_voc.yml"
    
    if not config_path.exists():
        logger.error(f"配置文件不存在: {config_path}")
        return False
    
    try:
        # 加载配置
        logger.info(f"加载配置文件: {config_path}")
        cfg = load_config(str(config_path))

        # 强制设置为CPU模式
        cfg.device.gpu_ids = []
        cfg.device.workers_per_gpu = 1
        cfg.device.batchsize_per_gpu = 2  # 非常小的batch size
        
        # 设置训练参数
        cfg.schedule.total_epochs = 1  # 只训练1个epoch
        cfg.schedule.val_intervals = 1
        cfg.schedule.save_checkpoint_epochs = 1
        
        # 设置数据路径
        data_root = project_root / "data"
        if not data_root.exists():
            logger.error(f"数据目录不存在: {data_root}")
            return False

        cfg.data.train.ann_file = str(data_root / "annotations" / "voc_train.json")
        cfg.data.train.img_path = str(data_root / "VOCdevkit" / "VOC2007" / "JPEGImages")
        cfg.data.val.ann_file = str(data_root / "annotations" / "voc_val.json")
        cfg.data.val.img_path = str(data_root / "VOCdevkit" / "VOC2007" / "JPEGImages")
        
        # 检查数据文件
        if not Path(cfg.data.train.ann_file).exists():
            logger.error(f"训练标注文件不存在: {cfg.data.train.ann_file}")
            return False
            
        # 设置工作目录
        work_dir = project_root / "work_dirs" / "cpu_test"
        work_dir.mkdir(parents=True, exist_ok=True)
        cfg.save_dir = str(work_dir)
        
        logger.info("配置加载完成，开始构建训练器...")
        
        # 构建训练器
        trainer = build_trainer(cfg, "train")
        
        logger.info("训练器构建成功，开始训练...")
        
        # 开始训练
        trainer.run()
        
        logger.info("CPU模式训练测试完成！")
        return True
        
    except Exception as e:
        logger.error(f"CPU训练测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_cpu_training()
    if success:
        print("✅ CPU模式训练测试成功！")
        sys.exit(0)
    else:
        print("❌ CPU模式训练测试失败！")
        sys.exit(1)
