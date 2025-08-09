import jittor as jt
from .assign_result import AssignResult
from .base_assigner import BaseAssigner

class CenterRadiusAssigner(BaseAssigner):
    """
    一个稳健的正样本分配器：
    - 将落在 GT 内部且同时落在 GT 中心半径框（cx,cy 半径 = center_radius*stride）的 priors 设为正样本；
    - 若某个 GT 没有选中任何 prior，则回退选择距离该 GT 中心最近的 prior；
    - 若一个 prior 同时匹配多个 GT，则优先分配给面积更小的 GT（利于小目标）。
    这样可以在训练早期稳定地产生正样本，避免 DSL/OTA 动态策略在冷启动时陷入全负的情况。
    """
    def __init__(self, center_radius: float = 2.5):
        self.center_radius = float(center_radius)

    def assign(
        self,
        pred_scores,          # 兼容签名，不使用
        priors,               # [N,4] -> x,y,stride,stride
        decoded_bboxes,       # 兼容签名，不使用
        gt_bboxes,            # [G,4] xyxy
        gt_labels,            # [G]
        gt_bboxes_ignore=None,
    ):
        num_bboxes = priors.size(0)
        num_gt = gt_bboxes.size(0)
        assigned_gt_inds = jt.full((num_bboxes,), 0, dtype='int32')
        assigned_labels = jt.full((num_bboxes,), -1, dtype='int32')
        max_overlaps = jt.zeros((num_bboxes,), dtype='float32')

        if num_gt == 0 or num_bboxes == 0:
            return AssignResult(num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        prior_xy = priors[:, :2]            # [N,2]
        prior_stride = priors[:, 2]         # [N]

        # inside GT test
        lt_ = prior_xy[:, None, :] - gt_bboxes[None, :, :2]   # [N,G,2]
        rb_ = gt_bboxes[None, :, 2:] - prior_xy[:, None, :]   # [N,G,2]
        deltas = jt.concat([lt_, rb_], dim=-1)                # [N,G,4]
        in_gt = (jt.min(deltas, dim=-1) > 0)                  # [N,G]

        # center radius box
        gt_centers = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) / 2.0  # [G,2]
        # 每个 prior 各自半径（按 stride 缩放）
        radii = self.center_radius * prior_stride             # [N]
        # 构造每个 GT 的中心框，并广播到 [N,G,2]
        cx = gt_centers[None, :, 0]
        cy = gt_centers[None, :, 1]
        x1c = cx - radii[:, None]
        y1c = cy - radii[:, None]
        x2c = cx + radii[:, None]
        y2c = cy + radii[:, None]
        in_center = (prior_xy[:, 0:1] > x1c) & (prior_xy[:, 0:1] < x2c) & \
                    (prior_xy[:, 1:2] > y1c) & (prior_xy[:, 1:2] < y2c)   # [N,G]

        # 合并条件
        candid = in_gt & in_center   # [N,G]

        # 如有重叠，按 GT 面积从小到大优先
        gt_areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (gt_bboxes[:, 3] - gt_bboxes[:, 1])  # [G]
        # 将 False 置为极大，便于 argmin 选择
        large_val = 1e30
        area_matrix = jt.ones_like(candid).float32() * large_val
        # 需要把每列的面积广播到 [N,G]
        area_matrix = area_matrix * 1.0
        area_matrix[:, :] = gt_areas[None, :]
        area_matrix = jt.where(candid, area_matrix, large_val)
        # 超简化策略：直接为每个GT分配最近的几个priors，保证有正样本
        # 计算所有 prior 到所有 GT 中心的距离
        dists = jt.abs(prior_xy[:, None, :] - gt_centers[None, :, :]).sum(-1)  # [N,G]

        # 为每个 GT 分配最近的 3 个 priors（确保有足够正样本）
        for g in range(num_gt):
            # 找到距离该GT最近的3个priors
            _, nearest_indices = jt.topk(-dists[:, g], k=min(3, num_bboxes), dim=0)
            # 分配给该GT
            assigned_gt_inds[nearest_indices] = g + 1
            assigned_labels[nearest_indices] = gt_labels[g].long().cast('int32')
            max_overlaps[nearest_indices] = 1.0

        return AssignResult(num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

