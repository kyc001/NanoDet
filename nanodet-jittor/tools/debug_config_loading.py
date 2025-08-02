#!/usr/bin/env python3
"""
调试配置文件加载问题
逐步定位段错误的具体原因
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

def test_yaml_loading():
    """测试YAML文件加载"""
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("测试YAML文件加载")
    logger.info("=" * 50)
    
    try:
        import yaml
        
        config_path = project_root / "config" / "nanodet-plus-m_320_voc.yml"
        logger.info(f"配置文件路径: {config_path}")
        logger.info(f"文件存在: {config_path.exists()}")
        
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                content = f.read()
                logger.info(f"文件大小: {len(content)} 字符")
                logger.info("文件前100字符:")
                logger.info(content[:100])
        
        # 尝试加载YAML
        logger.info("尝试加载YAML...")
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        logger.info("✅ YAML加载成功")
        logger.info(f"配置键: {list(config_dict.keys())}")
        
        return config_dict
        
    except Exception as e:
        logger.error(f"❌ YAML加载失败: {e}")
        traceback.print_exc()
        return None

def test_config_class():
    """测试Config类"""
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("测试Config类")
    logger.info("=" * 50)
    
    try:
        from nanodet.util.config import Config
        
        # 测试简单字典
        simple_dict = {'a': 1, 'b': {'c': 2}}
        logger.info("测试简单字典...")
        config = Config(simple_dict)
        logger.info(f"✅ Config创建成功: {config.a}, {config.b.c}")
        
        # 测试复制
        logger.info("测试复制...")
        config_copy = config.copy()
        logger.info(f"✅ Config复制成功: {config_copy.a}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Config类测试失败: {e}")
        traceback.print_exc()
        return False

def test_load_config_function():
    """测试load_config函数"""
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("测试load_config函数")
    logger.info("=" * 50)
    
    try:
        from nanodet.util.config import load_config
        
        config_path = project_root / "config" / "nanodet-plus-m_320_voc.yml"
        logger.info(f"调用load_config: {config_path}")
        
        # 这里可能会段错误
        cfg = load_config(str(config_path))
        
        logger.info("✅ load_config成功")
        logger.info(f"配置类型: {type(cfg)}")
        logger.info(f"配置键: {list(cfg.keys()) if hasattr(cfg, 'keys') else 'N/A'}")
        
        return cfg
        
    except Exception as e:
        logger.error(f"❌ load_config失败: {e}")
        traceback.print_exc()
        return None

def test_step_by_step():
    """逐步测试"""
    logger = setup_logging()
    
    # 强制CPU模式避免GPU相关问题
    jt.flags.use_cuda = 0
    logger.info("设置为CPU模式")
    
    # 步骤1: 测试YAML加载
    config_dict = test_yaml_loading()
    if config_dict is None:
        logger.error("YAML加载失败，停止测试")
        return False
    
    # 步骤2: 测试Config类
    config_class_ok = test_config_class()
    if not config_class_ok:
        logger.error("Config类测试失败，停止测试")
        return False
    
    # 步骤3: 测试load_config函数
    logger.info("准备测试load_config函数...")
    logger.info("如果这里段错误，说明问题在load_config函数内部")
    
    cfg = test_load_config_function()
    if cfg is None:
        logger.error("load_config函数失败")
        return False
    
    logger.info("🎉 所有测试通过！")
    return True

if __name__ == "__main__":
    success = test_step_by_step()
    sys.exit(0 if success else 1)
