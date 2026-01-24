import jittor as jt
import jittor.nn as nn
import jittor as jt
import jittor.nn as F
# import jittor.distributed as dist  # 不需要分布式

from ...loss.iou_loss import bbox_overlaps
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


class DynamicSoftLabelAssigner(BaseAssigner):
    def __init__(self, topk=13, iou_factor=3.0, ignore_iof_thr=-1):
        self.topk = topk
        self.iou_factor = iou_factor
        self.ignore_iof_thr = ignore_iof_thr

    def assign(
        self,
        pred_scores,
        priors,
        decoded_bboxes,
        gt_bboxes,
        gt_labels,
        gt_bboxes_ignore=None,
    ):
        def _bce_with_logits(logits, targets):
            # Jittor 的 BCE 接口不支持 reduction="none"，这里实现逐元素 BCE
            # stable: max(x,0) - x*target + log(1+exp(-abs(x)))
            max_val = nn.relu(logits)
            return max_val - logits * targets + jt.log(1 + jt.exp(-jt.abs(logits)))

        INF = 100000000
        num_gt = gt_bboxes.size(0)
        num_bboxes = decoded_bboxes.size(0)

        # assign 0 by default
        assigned_gt_inds = jt.full((num_bboxes,), 0, dtype='int32')

        prior_center = priors[:, :2]
        lt_ = prior_center[:, None] - gt_bboxes[:, :2]
        rb_ = gt_bboxes[:, 2:] - prior_center[:, None]

        deltas = jt.cat([lt_, rb_], dim=-1)
        is_in_gts = jt.min(deltas, dim=-1) > 0
        valid_mask = jt.sum(is_in_gts, dim=1) > 0

        valid_indices = jt.nonzero(valid_mask).squeeze(-1)
        if valid_indices.ndim == 0:
            valid_indices = valid_indices.unsqueeze(0)
        valid_decoded_bbox = decoded_bboxes[valid_indices]
        valid_pred_scores = pred_scores[valid_indices]
        num_valid = valid_decoded_bbox.size(0)

        if num_gt == 0 or num_bboxes == 0 or num_valid == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = jt.zeros((num_bboxes,), dtype='float32')
            if num_gt == 0:
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = jt.full((num_bboxes,), -1, dtype='int32')
            return AssignResult(num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        pairwise_ious = bbox_overlaps(valid_decoded_bbox, gt_bboxes)
        iou_cost = -jt.log(pairwise_ious + 1e-7)

        gt_onehot_label = (
            jt.nn.one_hot(gt_labels.long(), pred_scores.shape[-1])
            .float()
            .unsqueeze(0)
            .repeat(num_valid, 1, 1)
        )
        valid_pred_scores = valid_pred_scores.unsqueeze(1).repeat(1, num_gt, 1)

        soft_label = gt_onehot_label * pairwise_ious.unsqueeze(-1)
        scale_factor = soft_label - jt.sigmoid(valid_pred_scores)

        cls_cost = _bce_with_logits(valid_pred_scores, soft_label) * jt.pow(
            jt.abs(scale_factor), 2.0
        )
        cls_cost = cls_cost.sum(dim=-1)

        cost_matrix = cls_cost + iou_cost * self.iou_factor

        matched_pred_ious, matched_gt_inds = self.dynamic_k_matching(
            cost_matrix, pairwise_ious, num_gt, valid_mask
        )

        fg_indices = jt.nonzero(valid_mask).squeeze(-1)
        if fg_indices.ndim == 0:
            fg_indices = fg_indices.unsqueeze(0)

        assigned_labels = jt.full((num_bboxes,), -1, dtype='int32')
        max_overlaps = jt.full((num_bboxes,), -INF, dtype='float32')

        if fg_indices.numel() > 0:
            assigned_gt_inds[fg_indices] = matched_gt_inds + 1
            assigned_labels[fg_indices] = gt_labels[matched_gt_inds].long()
            max_overlaps[fg_indices] = matched_pred_ious

        if (
            self.ignore_iof_thr > 0
            and gt_bboxes_ignore is not None
            and gt_bboxes_ignore.numel() > 0
            and num_bboxes > 0
        ):
            ignore_overlaps = bbox_overlaps(
                valid_decoded_bbox, gt_bboxes_ignore, mode="iof"
            )
            ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            ignore_idxs = ignore_max_overlaps > self.ignore_iof_thr
            if ignore_idxs.numel() > 0:
                ignore_indices = valid_indices[ignore_idxs]
                assigned_gt_inds[ignore_indices] = -1

        return AssignResult(num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

    def dynamic_k_matching(self, cost, pairwise_ious, num_gt, valid_mask):
        matching_matrix = jt.zeros_like(cost)
        candidate_topk = min(self.topk, pairwise_ious.size(0))
        topk_ious, _ = jt.topk(pairwise_ious, k=candidate_topk, dim=0)
        dynamic_ks = jt.clamp(topk_ious.sum(0).int32(), min_v=1)

        for gt_idx in range(num_gt):
            k = int(dynamic_ks[gt_idx])
            _, pos_idx = jt.topk(-cost[:, gt_idx], k=k, dim=0)
            matching_matrix[pos_idx, gt_idx] = 1.0

        prior_match_gt_mask = matching_matrix.sum(1) > 1
        pm_indices = jt.nonzero(prior_match_gt_mask).squeeze(-1)
        if pm_indices.ndim == 0:
            pm_indices = pm_indices.unsqueeze(0)
        if pm_indices.numel() > 0:
            cost_pm = cost[pm_indices]
            cost_argmin, _ = jt.argmax(-cost_pm, dim=1)  # argmin via max(-cost)
            matching_matrix[pm_indices] = 0.0
            for i in range(pm_indices.size(0)):
                matching_matrix[pm_indices[i], cost_argmin[i]] = 1.0

        fg_mask_inboxes = matching_matrix.sum(1) > 0.0
        valid_indices = jt.nonzero(valid_mask).squeeze(-1)
        if valid_indices.ndim == 0:
            valid_indices = valid_indices.unsqueeze(0)
        if valid_indices.numel() > 0:
            valid_mask[valid_indices] = fg_mask_inboxes

        fg_indices = jt.nonzero(fg_mask_inboxes).squeeze(-1)
        if fg_indices.ndim == 0:
            fg_indices = fg_indices.unsqueeze(0)
        if fg_indices.numel() == 0:
            return jt.array([], dtype=jt.float32), jt.array([], dtype=jt.int32)

        matching_matrix_fg = matching_matrix[fg_indices]
        matched_gt_inds, _ = jt.argmax(matching_matrix_fg, dim=1)
        matched_pred_ious = (matching_matrix * pairwise_ious).sum(1)[fg_indices]
        return matched_pred_ious, matched_gt_inds
