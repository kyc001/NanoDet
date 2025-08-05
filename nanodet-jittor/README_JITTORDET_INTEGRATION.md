# JittorDet集成指南

本指南说明如何在nanodet-jittor中使用jittordet中已经实现好的模块，如gfl_head、gfl_loss等。

## 概述

jittordet是一个基于Jittor的目标检测框架，已经实现了许多常用的模块。我们可以直接使用这些模块，避免重复实现。

## 可用的模块

### 1. GFL Head (广义焦点损失头部)
- **位置**: `jittordet/jittordet/models/dense_heads/gfl_head.py`
- **功能**: 实现广义焦点损失的检测头部
- **特点**: 支持质量焦点损失(QFL)和分布焦点损失(DFL)

### 2. GFL Loss (广义焦点损失)
- **位置**: `jittordet/jittordet/models/losses/gfocal_loss.py`
- **功能**: 实现质量焦点损失和分布焦点损失
- **包含**:
  - `QualityFocalLoss`: 质量焦点损失
  - `DistributionFocalLoss`: 分布焦点损失

### 3. IoU Loss (IoU损失)
- **位置**: `jittordet/jittordet/models/losses/iou_loss.py`
- **功能**: 实现各种IoU损失函数
- **包含**: `GIoULoss`, `DIoULoss`, `CIoULoss`等

### 4. 边界框操作工具
- **位置**: `jittordet/jittordet/ops/`
- **功能**: 边界框转换、重叠计算等
- **包含**:
  - `delta2bbox`: 边界框转换
  - `bbox2distance`: 边界框距离计算
  - `bbox_overlaps`: 边界框重叠计算

## 使用方法

### 方法1: 使用适配层 (推荐)

我们创建了一个适配层，可以自动选择使用jittordet或nanodet-jittor的实现：

```python
from nanodet.model.head.jittordet_adapter import (
    GFLHead, QualityFocalLoss, DistributionFocalLoss, GIoULoss
)

# 创建GFL Head
head = GFLHead(
    num_classes=80,
    in_channels=256,
    stacked_convs=4,
    reg_max=16
)

# 创建损失函数
qfl = QualityFocalLoss(use_sigmoid=True, beta=2.0, loss_weight=1.0)
dfl = DistributionFocalLoss(loss_weight=0.25)
iou_loss = GIoULoss(loss_weight=2.0)
```

### 方法2: 直接导入

如果jittordet可用，也可以直接导入：

```python
import sys
from pathlib import Path

# 添加jittordet到Python路径
jittordet_path = Path("path/to/jittordet")
sys.path.insert(0, str(jittordet_path))

# 直接导入jittordet模块
from jittordet.models.dense_heads.gfl_head import GFLHead
from jittordet.models.losses.gfocal_loss import QualityFocalLoss, DistributionFocalLoss
```

## 工具脚本

### 1. 导入脚本
```bash
# 导入所有模块
python tools/import_from_jittordet.py --all

# 只导入特定模块
python tools/import_from_jittordet.py --gfl-head --gfl-loss
```

### 2. 测试脚本
```bash
# 测试适配层
python tools/test_jittordet_adapter.py
```

### 3. 使用示例
```bash
# 查看使用示例
python tools/use_jittordet_modules.py
```

## 项目结构

```
nanodet-jittor/
├── nanodet/
│   └── model/
│       └── head/
│           └── jittordet_adapter.py  # 适配层
├── tools/
│   ├── import_from_jittordet.py      # 导入脚本
│   ├── test_jittordet_adapter.py     # 测试脚本
│   └── use_jittordet_modules.py      # 使用示例
└── README_JITTORDET_INTEGRATION.md   # 本文档
```

## 优势

1. **避免重复实现**: 直接使用jittordet中已经实现好的模块
2. **自动回退**: 如果jittordet不可用，会自动使用nanodet-jittor的实现
3. **统一接口**: 提供统一的接口，无需修改现有代码
4. **易于维护**: 减少代码重复，降低维护成本

## 注意事项

1. **依赖关系**: 确保jittordet项目在正确的位置
2. **版本兼容**: 注意jittordet和nanodet-jittor的版本兼容性
3. **导入路径**: 可能需要调整Python路径以正确导入模块
4. **测试验证**: 使用前请运行测试脚本验证功能

## 故障排除

### 问题1: 模块导入失败
**解决方案**: 检查jittordet项目路径是否正确

### 问题2: 版本不兼容
**解决方案**: 使用适配层，它会自动处理兼容性问题

### 问题3: 功能不完整
**解决方案**: 检查jittordet中是否有完整的实现，必要时补充缺失的功能

## 贡献

欢迎贡献代码和文档！请遵循以下步骤：

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 创建Pull Request

## 许可证

本项目遵循Apache 2.0许可证。 