# NanoDet-Jittor 使用指南

## 🚀 快速开始 (5分钟)

### 1. 环境检查
```bash
# 确保在正确的conda环境中
conda activate nano

# 检查Jittor和CUDA
python -c "import jittor as jt; print('Jittor:', jt.__version__); jt.flags.use_cuda=1; print('CUDA:', jt.flags.use_cuda)"
```

### 2. 基础功能测试
```bash
cd nanodet-jittor
python test_basic_functionality.py
```

**期望输出**: `🎉 All tests passed!`

### 3. 准备VOC数据集

#### 选项A: 创建小规模测试数据集 (推荐新手)
```bash
# 如果你有完整的VOC数据集
python tools/create_mini_voc_dataset.py \
    --src-dir /path/to/VOCdevkit \
    --dst-dir data/VOC_mini \
    --train-samples 100 \
    --val-samples 50
```

#### 选项B: 下载完整VOC数据集
```bash
# 自动下载和准备 (需要网络连接)
python tools/download_voc_dataset.py \
    --data-dir data \
    --download \
    --convert \
    --verify
```

### 4. 验证数据集
```bash
python test_voc_dataset.py
```

## 📊 当前功能状态

### ✅ 可以使用的功能

#### 1. ShuffleNetV2 Backbone
```python
import jittor as jt
from nanodet.model.backbone import build_backbone

# 创建backbone
cfg = {
    'name': 'ShuffleNetV2',
    'model_size': '1.0x',  # 支持 0.5x, 1.0x, 1.5x, 2.0x
    'out_stages': [2, 3, 4],
    'activation': 'ReLU'
}

backbone = build_backbone(cfg)

# 测试推理
x = jt.randn(1, 3, 320, 320)
outputs = backbone(x)
print([o.shape for o in outputs])
# 输出: [[1,116,40,40,], [1,232,20,20,], [1,464,10,10,]]
```

#### 2. 卷积模块
```python
from nanodet.model.module.conv import ConvModule, DepthwiseConvModule

# 标准卷积模块
conv = ConvModule(
    in_channels=3,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=1,
    norm_cfg=dict(type='BN'),
    activation='ReLU'
)

# 深度可分离卷积
dw_conv = DepthwiseConvModule(
    in_channels=64,
    out_channels=128,
    kernel_size=3,
    stride=1,
    padding=1
)
```

#### 3. 激活函数
```python
from nanodet.model.module.activation import Swish, HardSwish, act_layers

# 使用自定义激活函数
swish = Swish()
hard_swish = HardSwish()

# 或从字典获取
relu = act_layers['ReLU']()
leaky_relu = act_layers['LeakyReLU']()
```

#### 4. VOC数据集工具
```bash
# 下载VOC数据集
python tools/download_voc_dataset.py --download

# 创建小规模数据集
python tools/create_mini_voc_dataset.py --train-samples 50

# 验证数据集
python test_voc_dataset.py
```

### 🚧 开发中的功能

#### 1. FPN网络 (即将完成)
- GhostPAN特征金字塔
- 多尺度特征融合

#### 2. 检测头 (即将完成)
- NanoDetPlusHead主检测头
- SimpleConvHead辅助检测头
- 动态软标签分配器

#### 3. 损失函数 (即将完成)
- QualityFocalLoss
- DistributionFocalLoss  
- GIoULoss

### ❌ 待开发的功能

#### 1. 训练框架
- 数据加载器
- 训练循环
- 模型保存/加载

#### 2. 评估系统
- mAP计算
- 推理脚本

## 🔧 配置文件说明

### VOC数据集配置 (`config/nanodet-plus-m_320_voc.yml`)

```yaml
# 模型配置
model:
  arch:
    name: NanoDetPlus
    backbone:
      name: ShuffleNetV2
      model_size: 1.0x
    head:
      num_classes: 20  # VOC有20个类别

# 数据配置  
data:
  train:
    name: VOCDataset
    data_dir: data/VOCdevkit
  val:
    name: VOCDataset
    data_dir: data/VOCdevkit

# 训练配置
device:
  batchsize_per_gpu: 32  # 适合RTX4060 8GB
  precision: 16          # 混合精度训练
```

## 📈 性能基准

### 当前性能 (ShuffleNetV2 1.0x)
```
参数量: 0.79M
推理速度: 1-3ms (RTX4060)
支持batch size: 1-32
输出通道: [116, 232, 464]
```

### 内存使用 (RTX4060 8GB)
```
Batch Size  推理时间  显存占用
    1       1.33ms   ~1GB
    8       2.24ms   ~3GB  
   16       2.28ms   ~5GB
   32       2.45ms   ~7GB  ← 推荐最大值
```

## 🐛 常见问题

### 1. ImportError: No module named 'jittor'
```bash
# 检查环境
conda activate nano
pip install jittor
```

### 2. CUDA out of memory
```bash
# 减少batch size
# 在配置文件中修改: batchsize_per_gpu: 16
```

### 3. VOC数据集下载失败
```bash
# 使用手动下载
# 1. 从官网下载VOC2007和VOC2012
# 2. 解压到data/VOCdevkit/
# 3. 运行转换脚本
python tools/download_voc_dataset.py --convert --verify
```

### 4. 可视化失败
```bash
# 安装opencv
pip install opencv-python
```

## 📝 开发计划

### 本周目标
- [ ] 完成GhostPAN FPN实现
- [ ] 完成NanoDetPlusHead实现
- [ ] 实现基础训练框架

### 下周目标  
- [ ] 完整端到端训练
- [ ] 与PyTorch版本对齐验证
- [ ] 性能优化

## 🎯 使用建议

### 新手用户
1. 先运行基础测试: `python test_basic_functionality.py`
2. 创建小规模数据集: `python tools/create_mini_voc_dataset.py`
3. 验证数据集: `python test_voc_dataset.py`
4. 等待训练功能完成

### 高级用户
1. 可以开始实现FPN和Head模块
2. 参考PyTorch版本的实现
3. 贡献代码到项目

### 研究人员
1. 关注模型架构的Jittor适配
2. 验证数值精度和性能对齐
3. 进行消融实验

---

**最后更新**: 2025-01-31  
**项目状态**: 基础组件完成，数据准备就绪  
**完成度**: 35%
