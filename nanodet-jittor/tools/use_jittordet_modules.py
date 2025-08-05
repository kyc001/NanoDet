#!/usr/bin/env python3
"""
使用jittordet中已实现的模块的示例
展示如何直接导入和使用jittordet中的gfl_head、gfl_loss等模块
"""

import sys
from pathlib import Path

# 添加jittordet到Python路径
jittordet_path = Path(__file__).parent.parent.parent / "jittordet"
sys.path.insert(0, str(jittordet_path))

# 现在可以直接导入jittordet中的模块
try:
    from jittordet.models.dense_heads.gfl_head import GFLHead
    from jittordet.models.losses.gfocal_loss import QualityFocalLoss, DistributionFocalLoss
    from jittordet.models.losses.iou_loss import GIoULoss
    from jittordet.ops.bbox_transforms import delta2bbox, bbox2distance
    from jittordet.utils.bbox_overlaps import bbox_overlaps
    
    print("✅ 成功导入jittordet模块!")
    print("可用的模块:")
    print("- GFLHead: 广义焦点损失头部")
    print("- QualityFocalLoss: 质量焦点损失")
    print("- DistributionFocalLoss: 分布焦点损失")
    print("- GIoULoss: 广义IoU损失")
    print("- delta2bbox: 边界框转换函数")
    print("- bbox_overlaps: 边界框重叠计算")
    
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请确保jittordet项目在正确的位置")


def example_usage():
    """使用示例"""
    print("\n=== 使用示例 ===")
    
    # 示例1: 创建GFL Head
    print("1. 创建GFL Head:")
    print("""
    head = GFLHead(
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        reg_max=16
    )
    """)
    
    # 示例2: 创建损失函数
    print("2. 创建损失函数:")
    print("""
    qfl = QualityFocalLoss(
        use_sigmoid=True,
        beta=2.0,
        loss_weight=1.0
    )
    
    dfl = DistributionFocalLoss(
        loss_weight=0.25
    )
    
    iou_loss = GIoULoss(
        loss_weight=2.0
    )
    """)
    
    # 示例3: 使用边界框操作
    print("3. 使用边界框操作:")
    print("""
    # 转换边界框
    bboxes = delta2bbox(rois, deltas, max_shape=(800, 800))
    
    # 计算边界框重叠
    overlaps = bbox_overlaps(bboxes1, bboxes2)
    """)


def create_nanodet_integration():
    """创建nanodet集成示例"""
    print("\n=== nanodet集成示例 ===")
    
    integration_code = '''
# 在nanodet中使用jittordet模块的示例

import jittor as jt
from jittordet.models.dense_heads.gfl_head import GFLHead
from jittordet.models.losses.gfocal_loss import QualityFocalLoss, DistributionFocalLoss

class NanoDetWithJittorDet(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()
        
        # 使用jittordet的GFL Head
        self.gfl_head = GFLHead(
            num_classes=num_classes,
            in_channels=256,
            stacked_convs=4,
            reg_max=16
        )
        
        # 使用jittordet的损失函数
        self.qfl = QualityFocalLoss(
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0
        )
        
        self.dfl = DistributionFocalLoss(
            loss_weight=0.25
        )
    
    def execute(self, x):
        # 前向传播
        cls_scores, bbox_preds = self.gfl_head(x)
        return cls_scores, bbox_preds
    
    def loss(self, cls_scores, bbox_preds, targets):
        # 计算损失
        loss_cls = self.qfl(cls_scores, targets)
        loss_bbox = self.dfl(bbox_preds, targets)
        return loss_cls + loss_bbox
'''
    
    print(integration_code)


if __name__ == "__main__":
    example_usage()
    create_nanodet_integration()
    
    print("\n=== 建议 ===")
    print("1. 直接使用jittordet中的模块，避免重复实现")
    print("2. 在nanodet-jittor中创建适配层，统一接口")
    print("3. 利用jittordet的完整实现，专注于nanodet特有的功能") 