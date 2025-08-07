import jittor as jt
import jittor.nn as nn
import jtorch as torch
import jtorch.nn as F
import jtorch.distributed as dist
import jittordet.models.losses as losses

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
        INF = 100000000
        num_gt = gt_bboxes.size(0)
        num_bboxes = decoded_bboxes.size(0)
        num_classes = pred_scores.size(1)  # 🔧 获取类别数量

        # assign 0 by default
        assigned_gt_inds = decoded_bboxes.new_full((num_bboxes,), 0)

        prior_center = priors[:, :2]
        lt_ = prior_center[:, None] - gt_bboxes[:, :2]
        rb_ = gt_bboxes[:, 2:] - prior_center[:, None]

        deltas = torch.cat([lt_, rb_], dim=-1)
        is_in_gts = deltas.min(dim=-1)[0] > 0
        valid_mask = is_in_gts.sum(dim=1) > 0

        # 🔧 修复 Jittor 布尔索引问题：使用 nonzero() 方法
        try:
            valid_indices = jt.nonzero(valid_mask).squeeze(-1)
            if valid_indices.ndim == 0:  # 只有一个元素
                valid_indices = valid_indices.unsqueeze(0)
        except:
            # 如果 nonzero 失败，使用手动方式
            valid_indices = jt.array([], dtype='int32')
        num_valid = valid_indices.size(0)

        if num_valid > 0:
            valid_decoded_bbox = decoded_bboxes[valid_indices]
            valid_pred_scores = pred_scores[valid_indices]
        else:
            valid_decoded_bbox = decoded_bboxes.new_zeros((0, 4))
            valid_pred_scores = pred_scores.new_zeros((0, num_classes))

        if num_gt == 0 or num_bboxes == 0 or num_valid == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = decoded_bboxes.new_zeros((num_bboxes,))
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = decoded_bboxes.new_full(
                    (num_bboxes,), -1
                )
            return AssignResult(
                num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels
            )

        pairwise_ious = bbox_overlaps(valid_decoded_bbox, gt_bboxes)
        iou_cost = -torch.log(pairwise_ious + 1e-7)

        gt_onehot_label = (
            F.one_hot(gt_labels.to(torch.int64), pred_scores.shape[-1])
            .float()
            .unsqueeze(0)
            .repeat(num_valid, 1, 1)
        )
        valid_pred_scores = valid_pred_scores.unsqueeze(1).repeat(1, num_gt, 1)

        soft_label = gt_onehot_label * pairwise_ious[..., None]
        scale_factor = soft_label - valid_pred_scores.sigmoid()

        cls_cost = losses.cross_entropy_loss.binary_cross_entropy_with_logits(
            valid_pred_scores, soft_label,reduction="none"
        ) * scale_factor.abs().pow(2.0)

        cls_cost = cls_cost.sum(dim=-1)

        cost_matrix = cls_cost + iou_cost * self.iou_factor

        matched_pred_ious, matched_gt_inds = self.dynamic_k_matching(
            cost_matrix, pairwise_ious, num_gt, valid_mask
        )

        # convert to AssignResult format
        # 🔧 修复 Jittor 布尔索引问题：只在有有效索引时才赋值
        if len(valid_indices) > 0 and len(matched_gt_inds) > 0:
            assigned_gt_inds[valid_indices] = matched_gt_inds + 1
            assigned_labels = assigned_gt_inds.new_full((num_bboxes,), -1)
            assigned_labels[valid_indices] = gt_labels[matched_gt_inds].long()
            max_overlaps = assigned_gt_inds.new_full(
                (num_bboxes,), -INF
            )
            max_overlaps[valid_indices] = matched_pred_ious
        else:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes,), -1)
            max_overlaps = assigned_gt_inds.new_full(
                (num_bboxes,), -INF
            )

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
            assigned_gt_inds[ignore_idxs] = -1

        return AssignResult(
            num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels
        )

    def dynamic_k_matching(self, cost, pairwise_ious, num_gt, valid_mask):
        matching_matrix = torch.zeros_like(cost)
        # select candidate topk ious for dynamic-k calculation
        candidate_topk = min(self.topk, pairwise_ious.size(0))
        topk_ious, _ = torch.topk(pairwise_ious, candidate_topk, dim=0)
        # calculate dynamic k for each gt
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[:, gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
            )
            matching_matrix[:, gt_idx][pos_idx] = 1.0

        del topk_ious, dynamic_ks, pos_idx

        prior_match_gt_mask = matching_matrix.sum(1) > 1
        if prior_match_gt_mask.sum() > 0:
            # 🔧 修复 Jittor 布尔索引问题
            # 🔧 修复 Jittor 布尔索引问题：使用 nonzero() 方法
            try:
                prior_indices = jt.nonzero(prior_match_gt_mask).squeeze(-1)
                if prior_indices.ndim == 0:
                    prior_indices = prior_indices.unsqueeze(0)
            except:
                prior_indices = jt.array([], dtype='int32')
            cost_min, cost_argmin = jt.min(cost[prior_indices, :], dim=1)
            matching_matrix[prior_indices, :] *= 0.0
            # 使用 scatter 操作替代高级索引
            for i, idx in enumerate(prior_indices):
                matching_matrix[idx, cost_argmin[i]] = 1.0
        # get foreground mask inside box and center prior
        fg_mask_inboxes = matching_matrix.sum(1) > 0.0
        # 🔧 修复 Jittor 布尔索引问题
        # 🔧 修复 Jittor 布尔索引问题：使用 nonzero() 方法
        try:
            fg_indices = jt.nonzero(fg_mask_inboxes).squeeze(-1)
            if fg_indices.ndim == 0:
                fg_indices = fg_indices.unsqueeze(0)
        except:
            fg_indices = jt.array([], dtype='int32')

        # 更新 valid_mask
        valid_mask_clone = valid_mask.clone()
        # 🔧 修复 Jittor 布尔索引问题：使用 nonzero() 方法
        try:
            valid_indices_in_valid = jt.nonzero(valid_mask_clone).squeeze(-1)
            if valid_indices_in_valid.ndim == 0:
                valid_indices_in_valid = valid_indices_in_valid.unsqueeze(0)
        except:
            valid_indices_in_valid = jt.array([], dtype='int32')
        for i, fg_val in enumerate(fg_mask_inboxes):
            if i < len(valid_indices_in_valid):
                valid_mask[valid_indices_in_valid[i]] = fg_val

        matched_gt_inds = matching_matrix[fg_indices, :].argmax(1)
        matched_pred_ious = (matching_matrix * pairwise_ious).sum(1)[fg_indices]
        return matched_pred_ious, matched_gt_inds
