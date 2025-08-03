from ...loss.iou_loss import bbox_overlaps
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


## [Jittor 迁移] ##
# 导入 jittor 相关的模块
import jittor as jt


class DynamicSoftLabelAssigner(BaseAssigner):
    """Computes matching between predictions and ground truth with
    dynamic soft label assignment. (Jittor Version)
    """

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
        ## [Jittor 迁移] ##
        # .size(0) -> .shape[0]
        num_gt = gt_bboxes.shape[0]
        num_bboxes = decoded_bboxes.shape[0]

        # assign 0 by default
        ## [Jittor 迁移] ##
        # .new_full(...) -> jt.full(...)
        assigned_gt_inds = jt.full((num_bboxes,), 0, dtype=jt.int64)

        prior_center = priors[:, :2]
        lt_ = prior_center[:, None] - gt_bboxes[:, :2]
        rb_ = gt_bboxes[:, 2:] - prior_center[:, None]

        ## [Jittor 迁移] ##
        # torch.cat -> jt.concat
        deltas = jt.concat([lt_, rb_], dim=-1)
        # .min(...).values -> .min(...)[0]
        is_in_gts = deltas.min(dim=-1)[0] > 0
        # Jittor Fix: 处理 is_in_gts 可能为一维向量的情况
        if is_in_gts.ndim > 1:
            # 如果是二维或更高维，正常在 dim=1 上求和
            valid_mask = is_in_gts.sum(dim=1) > 0
        else:
            # 如果是一维（表示只有一个真实标注框），直接比较
            valid_mask = is_in_gts > 0

        valid_decoded_bbox = decoded_bboxes[valid_mask]
        valid_pred_scores = pred_scores[valid_mask]
        num_valid = valid_decoded_bbox.shape[0]

        if num_gt == 0 or num_bboxes == 0 or num_valid == 0:
            max_overlaps = jt.zeros((num_bboxes,)) # .new_zeros -> jt.zeros
            if num_gt == 0:
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = jt.full((num_bboxes,), -1, dtype=jt.int64)
            return AssignResult(
                num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels
            )

        pairwise_ious = bbox_overlaps(valid_decoded_bbox, gt_bboxes)
        ## [Jittor 迁移] ##
        # torch.log(x) -> x.log()
        iou_cost = -(pairwise_ious + 1e-7).log()
        
        ## [Jittor 迁移] ##
        # F.one_hot -> jt.nn.one_hot
        # .to(torch.int64) -> .int64()
        # .float() -> .float32()
        gt_onehot_label = (
            jt.nn.one_hot(gt_labels.int64(), num_classes=pred_scores.shape[-1])
            .float32()
            .unsqueeze(0)
            .repeat(num_valid, 1, 1)
        )
        valid_pred_scores = valid_pred_scores.unsqueeze(1).repeat(1, num_gt, 1)

        soft_label = gt_onehot_label * pairwise_ious[..., None]
        scale_factor = soft_label - valid_pred_scores.sigmoid()
        
        ## [Jittor 迁移] ##
        # F.binary_cross_entropy_with_logits -> jt.nn.binary_cross_entropy_with_logits
        cls_cost = jt.nn.binary_cross_entropy_with_logits(
            valid_pred_scores, soft_label, reduction="none"
        ) * scale_factor.abs().pow(2.0)
        cls_cost = cls_cost.sum(dim=-1)

        cost_matrix = cls_cost + iou_cost * self.iou_factor

        matched_pred_ious, matched_gt_inds = self.dynamic_k_matching(
            cost_matrix, pairwise_ious, num_gt, valid_mask
        )

        # convert to AssignResult format
        assigned_gt_inds[valid_mask] = matched_gt_inds + 1
        assigned_labels = jt.full((num_bboxes,), -1, dtype=jt.int64)
        ## [Jittor 迁移] ##
        # .long() -> .int64()
        assigned_labels[valid_mask] = gt_labels[matched_gt_inds].int64()
        max_overlaps = jt.full((num_bboxes,), -INF, dtype=jt.float32)
        max_overlaps[valid_mask] = matched_pred_ious

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
        ## [Jittor 迁移] ##
        # torch.zeros_like -> jt.zeros_like
        matching_matrix = jt.zeros_like(cost)
        candidate_topk = min(self.topk, pairwise_ious.shape[0])
        ## [Jittor 迁移] ##
        # torch.topk -> jt.topk
        topk_ious, _ = jt.topk(pairwise_ious, candidate_topk, dim=0)
        ## [Jittor 迁移] ##
        # torch.clamp(..., min=x) -> jt.clamp(..., min_v=x)
        dynamic_ks = jt.clamp(topk_ious.sum(0).int(), min_v=1)
        
        for gt_idx in range(num_gt):
            _, pos_idx = jt.topk(
                cost[:, gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
            )
            matching_matrix[:, gt_idx][pos_idx] = 1.0

        del topk_ious, dynamic_ks, pos_idx

        prior_match_gt_mask = matching_matrix.sum(1) > 1
        if prior_match_gt_mask.sum() > 0:
            ## [Jittor 迁移] ##
            # torch.min -> jt.min
            cost_min, cost_argmin = jt.min(cost[prior_match_gt_mask, :], dim=1)
            matching_matrix[prior_match_gt_mask, :] *= 0.0
            matching_matrix[prior_match_gt_mask, cost_argmin] = 1.0
        
        fg_mask_inboxes = matching_matrix.sum(1) > 0.0
        ## [Jittor 迁移] ##
        # .clone() 在 Jittor 中用法相同
        valid_mask[valid_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[fg_mask_inboxes, :].argmax(1)
        matched_pred_ious = (matching_matrix * pairwise_ious).sum(1)[fg_mask_inboxes]
        return matched_pred_ious, matched_gt_inds