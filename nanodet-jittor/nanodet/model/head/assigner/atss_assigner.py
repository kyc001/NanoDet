from ...loss.iou_loss import bbox_overlaps
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


## [Jittor 迁移] ##
# 导入 jittor 相关的模块
import jittor as jt


class ATSSAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox. (Jittor Version)"""

    def __init__(self, topk, ignore_iof_thr=-1):
        self.topk = topk
        self.ignore_iof_thr = ignore_iof_thr

    def assign(
        self, bboxes, num_level_bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None
    ):
        INF = 100000000
        bboxes = bboxes[:, :4]
        ## [Jittor 迁移] ##
        # .size(0) -> .shape[0]
        num_gt, num_bboxes = gt_bboxes.shape[0], bboxes.shape[0]

        # compute iou between all bbox and gt
        overlaps = bbox_overlaps(bboxes, gt_bboxes)

        # assign 0 by default
        ## [Jittor 迁移] ##
        # .new_full(...) -> jt.full(...)
        assigned_gt_inds = jt.full((num_bboxes,), 0, dtype=jt.int64)

        if num_gt == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            ## [Jittor 迁移] ##
            # .new_zeros(...) -> jt.zeros(...)
            max_overlaps = jt.zeros((num_bboxes,))
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0 # Jittor 支持此操作
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = jt.full((num_bboxes,), -1, dtype=jt.int64)
            return AssignResult(
                num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels
            )

        # compute center distance between all bbox and gt
        gt_cx = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
        gt_cy = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0
        ## [Jittor 迁移] ##
        # torch.stack -> jt.stack
        gt_points = jt.stack((gt_cx, gt_cy), dim=1)

        bboxes_cx = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
        bboxes_cy = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
        bboxes_points = jt.stack((bboxes_cx, bboxes_cy), dim=1)

        # 广播和数学运算在 Jittor 中保持不变
        distances = (
            (bboxes_points[:, None, :] - gt_points[None, :, :]).pow(2).sum(-1).sqrt()
        )

        if (
            self.ignore_iof_thr > 0
            and gt_bboxes_ignore is not None
            and gt_bboxes_ignore.numel() > 0
            and bboxes.numel() > 0
        ):
            ignore_overlaps = bbox_overlaps(bboxes, gt_bboxes_ignore, mode="iof")
            ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            ignore_idxs = ignore_max_overlaps > self.ignore_iof_thr
            distances[ignore_idxs, :] = INF
            assigned_gt_inds[ignore_idxs] = -1

        # Selecting candidates based on the center distance
        candidate_idxs = []
        start_idx = 0
        for level, bboxes_per_level in enumerate(num_level_bboxes):
            end_idx = start_idx + bboxes_per_level
            distances_per_level = distances[start_idx:end_idx, :]
            selectable_k = min(self.topk, bboxes_per_level)
            # .topk() 在 Jittor 中用法相同
            _, topk_idxs_per_level = distances_per_level.topk(
                selectable_k, dim=0, largest=False
            )
            candidate_idxs.append(topk_idxs_per_level + start_idx)
            start_idx = end_idx
        ## [Jittor 迁移] ##
        # torch.cat -> jt.concat
        candidate_idxs = jt.concat(candidate_idxs, dim=0)

        # get corresponding iou for the these candidates
        ## [Jittor 迁移] ##
        # torch.arange -> jt.arange
        candidate_overlaps = overlaps[candidate_idxs, jt.arange(num_gt)]
        # .mean() 和 .std() 在 Jittor 中用法相同
        overlaps_mean_per_gt = candidate_overlaps.mean(0)
        overlaps_std_per_gt = candidate_overlaps.std(0)
        overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt

        is_pos = candidate_overlaps >= overlaps_thr_per_gt[None, :]

        # limit the positive sample's center in gt
        for gt_idx in range(num_gt):
            candidate_idxs[:, gt_idx] += gt_idx * num_bboxes
        
        ## [Jittor 迁移] ##
        # .contiguous() 在 Jittor 中通常不是必需的，可以移除
        ep_bboxes_cx = bboxes_cx.view(1, -1).expand(num_gt, num_bboxes).reshape(-1)
        ep_bboxes_cy = bboxes_cy.view(1, -1).expand(num_gt, num_bboxes).reshape(-1)
        candidate_idxs = candidate_idxs.view(-1)

        l_ = ep_bboxes_cx[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 0]
        t_ = ep_bboxes_cy[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 1]
        r_ = gt_bboxes[:, 2] - ep_bboxes_cx[candidate_idxs].view(-1, num_gt)
        b_ = gt_bboxes[:, 3] - ep_bboxes_cy[candidate_idxs].view(-1, num_gt)
        is_in_gts = jt.stack([l_, t_, r_, b_], dim=1).min(dim=1)[0] > 0.01
        is_pos = is_pos & is_in_gts

        # if an anchor box is assigned to multiple gts...
        ## [Jittor 迁移] ##
        # torch.full_like -> jt.full_like
        overlaps_inf = jt.full_like(overlaps, -INF).t().reshape(-1)
        index = candidate_idxs.view(-1)[is_pos.view(-1)]
        overlaps_inf[index] = overlaps.t().reshape(-1)[index]
        overlaps_inf = overlaps_inf.view(num_gt, -1).t()

        max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1)
        assigned_gt_inds[max_overlaps != -INF] = (
            argmax_overlaps[max_overlaps != -INF] + 1
        )

        if gt_labels is not None:
            assigned_labels = jt.full((num_bboxes,), -1, dtype=jt.int64)
            ## [Jittor 迁移] ##
            # torch.nonzero(...) -> jt.nonzero(...)[0]
            pos_inds = jt.nonzero(assigned_gt_inds > 0)[0]
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None
        return AssignResult(
            num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels
        )