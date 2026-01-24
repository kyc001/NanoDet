# NanoDet-Plus PyTorch ↔ Jittor 对齐项目

本项目将 NanoDet-Plus 目标检测模型从 PyTorch 迁移到 Jittor 框架，并通过完整的训练和推理实验验证两个框架的一致性。

## 目录

- [项目概述](#项目概述)
- [结果速览](#结果速览)
- [环境配置](#环境配置)
- [数据准备](#数据准备)
- [项目结构](#项目结构)
- [训练脚本](#训练脚本)
- [测试脚本](#测试脚本)
- [权重转换](#权重转换)
- [实验结果](#实验结果)
- [Loss 曲线](#loss-曲线)
- [性能对比](#性能对比)
- [推理可视化对比](#推理可视化对比)
- [实验日志](#实验日志)
- [结论](#结论)
- [已知问题](#已知问题)
- [参考](#参考)

---

## 项目概述

- **模型**: NanoDet-Plus-m (轻量级目标检测模型)
- **Backbone**: ShuffleNetV2 1.0x
- **FPN**: GhostPAN
- **Head**: NanoDetPlusHead
- **数据集**: VOC2007 (20类, 5011训练集, 4952测试集)
- **输入尺寸**: 320 x 320

---

## 结果速览

- **最终 mAP**: PyTorch 0.3315 vs Jittor 0.3194（-3.7%）
- **推理速度**: Jittor 114.5 FPS vs PyTorch 109.0 FPS（+5.0%）
- **训练耗时**: PyTorch ~50 分钟，Jittor ~69 分钟
- **权重转换**: PT→JT mAP 0.3311（-0.12%），JT→PT mAP 0.3210（+0.5%）
- **完整报告**: `full_experiment_report.md`

---

## 环境配置

### 硬件环境
- **GPU**: NVIDIA GeForce RTX 3090 (24GB)
- **CPU**: Intel Xeon Platinum (128 cores)

### 软件环境

```bash
# 创建 conda 环境
conda create -n nanojittor python=3.8 -y
conda activate nanojittor

# 安装 PyTorch
pip install torch torchvision pytorch_lightning

# 安装 Jittor
pip install jittor

# 安装其他依赖
pip install pycocotools opencv-python matplotlib tabulate pyyaml
```

### 环境信息
| 软件 | 版本 |
|------|------|
| Python | 3.8.20 |
| PyTorch | 2.x (via pytorch_lightning) |
| Jittor | 1.3.10.0 |
| CUDA | 11.4.120 |

---

## 数据准备

### 1. 下载 VOC2007 数据集

```bash
# 下载 VOC2007 训练验证集
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
tar -xvf VOCtrainval_06-Nov-2007.tar -C data/

# 下载 VOC2007 测试集
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
tar -xvf VOCtest_06-Nov-2007.tar -C data/
```

### 2. 数据目录结构

```
data/VOCdevkit/VOC2007/
├── Annotations/          # XML 标注文件
├── ImageSets/Main/       # 数据集划分
│   ├── trainval.txt      # 训练验证集 (5011张)
│   ├── test.txt          # 测试集 (4952张)
│   ├── trainval_small50.txt  # 小样本过拟合集 (50张)
│   └── test_small50.txt      # 小样本测试集 (50张)
└── JPEGImages/           # 图片文件
```

---

## 项目结构

```
NanoDet/
├── README.md                           # 本文件
├── PROJECT_HANDOVER.md                 # 项目交接文档
├── full_experiment_report.md           # 完整实验报告
│
├── nanodet-pytorch/                    # PyTorch 版本
│   ├── config/
│   │   ├── nanodet-plus-m_320_voc.yml              # 全量训练配置
│   │   └── nanodet-plus-m_320_voc_small50_overfit.yml  # 小样本配置
│   ├── nanodet/                        # 模型代码
│   │   ├── model/                      # 模型定义
│   │   ├── data/                       # 数据加载
│   │   ├── trainer/                    # 训练逻辑
│   │   └── util/                       # 工具函数
│   └── tools/
│       ├── train.py                    # 训练脚本
│       └── test.py                     # 测试脚本
│
├── nanodet-jittor/                     # Jittor 版本
│   ├── config/
│   │   ├── nanodet-plus-m_320_voc.yml              # 全量训练配置
│   │   └── nanodet-plus-m_320_voc_small50_overfit.yml  # 小样本配置
│   ├── nanodet/                        # 模型代码 (与PyTorch对齐)
│   └── tools/
│       ├── train.py                    # 训练脚本
│       └── test.py                     # 测试脚本
│
├── tools/                              # 公共工具脚本
│   ├── convert_pt_to_jittor.py         # PyTorch → Jittor 权重转换
│   ├── convert_jittor_to_pt.py         # Jittor → PyTorch 权重转换
│   ├── compare_pt_jt_models.py         # 权重对比工具
│   ├── benchmark_fps.py                # FPS 性能测试
│   ├── plot_loss_curves.py             # Loss 曲线绘制
│   └── roundtrip_pt_jt_check.py        # 往返一致性检查
│
├── workspace/                          # 训练输出目录
│   ├── pytorch_full_train.log          # PyTorch 全量训练日志
│   ├── jittor_full_train.log           # Jittor 全量训练日志
│   ├── pytorch_small50_train.log       # PyTorch 小样本训练日志
│   ├── jittor_small50_train.log        # Jittor 小样本训练日志
│   ├── pytorch_fps.log                 # PyTorch FPS 测试日志
│   ├── jittor_fps.log                  # Jittor FPS 测试日志
│   ├── pt2jt_converted.pkl             # PT→JT 转换模型
│   ├── jt2pt_converted.pth             # JT→PT 转换模型
│   ├── pt2jt_inference.log             # PT→JT 转换推理日志
│   ├── jt2pt_inference.log             # JT→PT 转换推理日志
│   ├── figures/                        # 可视化图表
│   │   ├── detection_comparison/       # 推理可视化对比图
│   │   │   ├── comparison_*.jpg
│   │   │   └── detection_comparison_grid.jpg
│   │   ├── loss_curves.png             # Loss 曲线对比图
│   │   ├── map_curves.png              # mAP 曲线对比图
│   │   └── fps_comparison.png          # FPS 对比图
│   └── nanodet-plus-m_320_voc/         # 模型检查点
│       ├── model_best.ckpt             # Jittor 最佳模型
│       ├── model_last.ckpt             # Jittor 最后模型
│       ├── logs-*/                     # TensorBoard 日志
│       └── NanoDet/*/checkpoints/      # PyTorch 检查点
│
└── data/                               # 数据集目录
    └── VOCdevkit/VOC2007/
```

---

## 训练脚本

### PyTorch 训练

```bash
# 全量训练 (50 epochs)
python nanodet-pytorch/tools/train.py nanodet-pytorch/config/nanodet-plus-m_320_voc.yml

# 小样本过拟合 (20 epochs)
python nanodet-pytorch/tools/train.py nanodet-pytorch/config/nanodet-plus-m_320_voc_small50_overfit.yml
```

### Jittor 训练

```bash
# 全量训练 (50 epochs)
python nanodet-jittor/tools/train.py nanodet-jittor/config/nanodet-plus-m_320_voc.yml

# 小样本过拟合 (20 epochs)
python nanodet-jittor/tools/train.py nanodet-jittor/config/nanodet-plus-m_320_voc_small50_overfit.yml
```

### 训练参数

| 参数 | 值 |
|------|-----|
| Batch Size | 64 |
| Optimizer | AdamW |
| Learning Rate | 0.001 |
| Weight Decay | 0.05 |
| Total Epochs | 50 |
| Warmup Steps | 300 |
| LR Schedule | MultiStepLR [30, 45] |
| Grad Clip | 35 |
| Precision | FP32 |
| Val Interval | 10 epochs |
| EMA Decay | 0.9998 |

---

## 测试脚本

### PyTorch 测试

```bash
python nanodet-pytorch/tools/test.py \
  --config nanodet-pytorch/config/nanodet-plus-m_320_voc.yml \
  --model workspace/nanodet-plus-m_320_voc/NanoDet/*/checkpoints/epoch=49-step=3900.ckpt \
  --task val
```

### Jittor 测试

```bash
python nanodet-jittor/tools/test.py \
  --config nanodet-jittor/config/nanodet-plus-m_320_voc.yml \
  --model workspace/nanodet-plus-m_320_voc/model_best.ckpt \
  --task val
```

---

## 权重转换

### PyTorch → Jittor

```bash
python tools/convert_pt_to_jittor.py \
  --config nanodet-jittor/config/nanodet-plus-m_320_voc.yml \
  --pt_ckpt workspace/nanodet-plus-m_320_voc/NanoDet/*/checkpoints/epoch=49-step=3900.ckpt \
  --jt_ckpt workspace/pt2jt_converted.pkl
```

### Jittor → PyTorch

```bash
python tools/convert_jittor_to_pt.py \
  --jt_ckpt workspace/nanodet-plus-m_320_voc/model_best.ckpt \
  --pt_ckpt workspace/jt2pt_converted.pth
```

### 权重转换验证

| 转换方向 | 原始 mAP | 转换后推理 mAP | 精度损失 |
|----------|----------|----------------|----------|
| PT → JT | 0.3315 | 0.3311 | -0.12% |
| JT → PT | 0.3194 | 0.3210 | +0.5% |

**结论**: 双向权重转换脚本有效，转换后精度损失 < 1%

---

## 实验结果

### 训练结果对比

| 指标 | PyTorch | Jittor | 差异 |
|------|---------|--------|------|
| **最终 mAP** | 0.3315 | 0.3194 | -3.7% |
| **AP@0.50** | 0.5470 | 0.5310 | -2.9% |
| **AP@0.75** | 0.3418 | 0.3250 | -4.9% |
| **训练时间** | 50 分钟 | 69 分钟 | +38% |
| **显存占用** | 7.5 GB | ~17.5 GB | +133% |

### 各 Epoch mAP 对比

| Epoch | PyTorch mAP | Jittor mAP | PyTorch AP@50 | Jittor AP@50 |
|-------|-------------|------------|---------------|--------------|
| 10 | 0.2071 | 0.1739 | 0.3800 | 0.3270 |
| 20 | 0.2700 | 0.2469 | 0.4738 | 0.4360 |
| 30 | 0.2726 | 0.2736 | 0.4730 | 0.4730 |
| 40 | 0.3284 | 0.3173 | 0.5435 | 0.5270 |
| **50** | **0.3315** | **0.3194** | **0.5470** | **0.5310** |

---

## Loss 曲线

### Loss 曲线对比

![Loss Curves](workspace/figures/loss_curves.png)

### mAP 曲线对比

![mAP Curves](workspace/figures/map_curves.png)

---

## 性能对比

### FPS 测试结果

![FPS Comparison](workspace/figures/fps_comparison.png)

| 框架 | 平均推理时间 | FPS | 测试图片数 |
|------|-------------|-----|-----------|
| **PyTorch** | 9.18 ± 1.16 ms | **109.0** | 200 |
| **Jittor** | 8.73 ± 0.38 ms | **114.5** | 200 |

**测试环境**: RTX 3090, 320x320 输入, batch_size=1

### FPS 测试脚本

```bash
# PyTorch FPS 测试
python tools/benchmark_fps.py \
  --framework pytorch \
  --config nanodet-pytorch/config/nanodet-plus-m_320_voc.yml \
  --model workspace/nanodet-plus-m_320_voc/NanoDet/*/checkpoints/epoch=49-step=3900.ckpt \
  --num_images 200 --warmup 20

# Jittor FPS 测试
python tools/benchmark_fps.py \
  --framework jittor \
  --config nanodet-jittor/config/nanodet-plus-m_320_voc.yml \
  --model workspace/nanodet-plus-m_320_voc/model_best.ckpt \
  --num_images 200 --warmup 20
```

---

## 推理可视化对比

### 检测效果对比

对同一张图片分别使用 PyTorch 和 Jittor 模型进行推理，展示检测框与 GT 框对比：

- **绿色虚线框**: Ground Truth (GT) 标注
- **彩色实线框**: 模型检测结果

#### 示例 1：人物与狗检测

![Detection Comparison 1](workspace/figures/detection_comparison/comparison_000001.jpg)

#### 示例 2：多人检测

![Detection Comparison 2](workspace/figures/detection_comparison/comparison_004000.jpg)

#### 示例 3：鸟与盆栽检测

![Detection Comparison 3](workspace/figures/detection_comparison/comparison_008937.jpg)

#### 汇总网格（10 张）

![Detection Comparison Grid](workspace/figures/detection_comparison/detection_comparison_grid.jpg)

### 可视化脚本

```bash
python tools/visualize_comparison.py \
  --pt_config nanodet-pytorch/config/nanodet-plus-m_320_voc.yml \
  --jt_config nanodet-jittor/config/nanodet-plus-m_320_voc.yml \
  --pt_model workspace/nanodet-plus-m_320_voc/NanoDet/*/checkpoints/epoch=49-step=3900.ckpt \
  --jt_model workspace/nanodet-plus-m_320_voc/model_best.ckpt \
  --image_dir data/VOCdevkit/VOC2007/JPEGImages \
  --image_list data/VOCdevkit/VOC2007/ImageSets/Main/test.txt \
  --annotation_dir data/VOCdevkit/VOC2007/Annotations \
  --num_images 10 --score_thresh 0.35
```

### 可视化结论

- 两个框架的检测框位置高度一致
- 置信度存在轻微差异（5-15%），属于正常范围
- Jittor 版本可作为 PyTorch 版本的有效替代

---

## 实验日志

### 训练日志

| 框架 | 日志文件 | 说明 |
|------|---------|------|
| PyTorch | `workspace/pytorch_full_train.log` | 50 epoch 全量训练 |
| Jittor | `workspace/jittor_full_train.log` | 50 epoch 全量训练 |
| PyTorch | `workspace/pytorch_small50_train.log` | 小样本过拟合 |
| Jittor | `workspace/jittor_small50_train.log` | 小样本过拟合 |

### 推理日志

| 实验 | 日志文件 | 说明 |
|------|---------|------|
| PyTorch FPS | `workspace/pytorch_fps.log` | FPS 性能测试 |
| Jittor FPS | `workspace/jittor_fps.log` | FPS 性能测试 |
| PT→JT 推理 | `workspace/pt2jt_inference.log` | 转换后推理验证 |
| JT→PT 推理 | `workspace/jt2pt_inference.log` | 转换后推理验证 |

### 绘制曲线

```bash
python tools/plot_loss_curves.py \
  --pt_log workspace/pytorch_full_train.log \
  --jt_log workspace/jittor_full_train.log \
  --output_dir workspace/figures
```

---

## 结论

1. **训练对齐**: PyTorch 和 Jittor 框架训练结果对齐良好，最终 mAP 差异约 3.7%，在可接受范围内。

2. **推理性能**: Jittor 推理速度略快于 PyTorch (114.5 FPS vs 109.0 FPS)，约快 5%。

3. **权重转换**: 双向权重转换脚本有效，转换后精度损失 < 1%。

4. **收敛趋势**: 两个框架的 Loss 下降趋势基本一致，说明训练过程对齐良好。

5. **Epoch 30 交叉点**: 在 Epoch 30 附近，Jittor mAP 一度与 PyTorch 持平 (0.2736 vs 0.2726)。

---

## 已知问题

1. **Jittor AMP 不稳定**: 开启混合精度后出现 `cudnn_conv ... best_algo_idx!=-1` 错误，建议使用 FP32。

2. **显存占用差异**: Jittor 显存占用约为 PyTorch 的 2 倍，大模型训练时需注意。

---

## 参考

- [NanoDet 官方仓库](https://github.com/RangiLyu/nanodet)
- [Jittor 官方文档](https://cg.cs.tsinghua.edu.cn/jittor/)
- [PyTorch 官方文档](https://pytorch.org/docs/)

---

*最后更新: 2026-01-24*
