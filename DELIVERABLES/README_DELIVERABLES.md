# 项目交付清单（Jittor 版 NanoDet-Plus）

本目录聚合参评所需的最小必要材料：图片、脚本、日志与命令清单。

## 目录结构
- images/
  - sample_dets/: 批量推理可视化结果（若未生成，请先运行脚本）
  - curves.png: 训练损失与 mAP 曲线
- scripts/
  - run_full_val.sh: 全量验证（固化指标）
  - run_tiny20_overfit.sh: 20 张 overfit 快速验证链路
  - plot_from_log.sh: 从训练日志解析并绘制曲线
  - vis_batch.sh: 批量可视化 val 样本
- logs/
  - full_val_final.txt: 全量验证日志
  - tiny20_overfit.txt: tiny20 训练日志

## 一键命令
1) 全量验证
    bash scripts/run_full_val.sh

2) 20 张 overfit（3 epoch）
    bash scripts/run_tiny20_overfit.sh

3) 解析日志并绘图
    bash scripts/plot_from_log.sh

4) 批量可视化（保存到 images/sample_dets）
    bash scripts/vis_batch.sh

## 指标快照（来自 full_val）
- mAP=0.3476, AP50=0.563（VOC val=1494, 输入 320）

## 注意
- 运行前确保激活环境：`conda activate nano`
- 数据路径参考 config：data/VOCdevkit/VOC2007/JPEGImages 与 data/annotations/*.json
- 如需复现实验日志与图，先执行 run_* 再执行 plot 和 vis

