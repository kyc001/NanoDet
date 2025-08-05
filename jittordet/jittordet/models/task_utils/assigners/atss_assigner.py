# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import List, Optional

import jittor as jt

from jittordet.engine import TASK_UTILS
from jittordet.structures import InstanceData
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


def bbox_center_distance(bboxes, priors):
    """Compute the center distance between bboxes and priors.

    Args:
        bboxes (Tensor): Shape (n, 4) for , "xyxy" format.
        priors (Tensor): Shape (n, 4) for priors, "xyxy" format.

    Returns:
        Tensor: Center distances between bboxes and priors.
    """
    bbox_cx = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
    bbox_cy = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
    bbox_points = jt.stack((bbox_cx, bbox_cy), dim=1)

    priors_cx = (priors[:, 0] + priors[:, 2]) / 2.0
    priors_cy = (priors[:, 1] + priors[:, 3]) / 2.0
    priors_points = jt.stack((priors_cx, priors_cy), dim=1)

    distances = (priors_points[:, None, :] -
                 bbox_points[None, :, :]).pow(2).sum(-1).sqrt()

    return distances


@TASK_UTILS.register_module()
class ATSSAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each prior.

    Each proposals will be assigned with `0` or a positive integer
    indicating the ground truth index.

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    If ``alpha`` is not None, it means that the dynamic cost
    ATSSAssigner is adopted, which is currently only used in the DDOD.

    Args:
        topk (int): number of priors selected in each level
        alpha (float, optional): param of cost rate for each proposal only
            in DDOD. Defaults to None.
        iou_calculator (:obj:`ConfigDict` or dict): Config dict for iou
            calculator. Defaults to ``dict(type='BboxOverlaps2D')``
        ignore_iof_thr (float): IoF threshold for ignoring bboxes (if
            `gt_bboxes_ignore` is specified). Negative values mean not
            ignoring any bboxes. Defaults to -1.
    """

    def __init__(self,
                 topk: int,
                 alpha: Optional[float] = None,
                 iou_calculator=dict(type='BboxOverlaps2D'),
                 ignore_iof_thr: float = -1) -> None:
        self.topk = topk
        self.alpha = alpha
        self.iou_calculator = TASK_UTILS.build(iou_calculator)
        self.ignore_iof_thr = ignore_iof_thr

    # https://github.com/sfzhang15/ATSS/blob/master/atss_core/modeling/rpn/atss/loss.py
    def assign(
            self,
            pred_instances: InstanceData,
            num_level_priors: List[int],
            gt_instances: InstanceData,
            gt_instances_ignore: Optional[InstanceData] = None
    ) -> AssignResult:
        gt_bboxes = gt_instances.bboxes
        priors = pred_instances.priors
        gt_labels = gt_instances.labels
        if gt_instances_ignore is not None:
            gt_bboxes_ignore = gt_instances_ignore.bboxes
        else:
            gt_bboxes_ignore = None

        INF = 100000000
        priors = priors[:, :4]
        num_gt, num_priors = gt_bboxes.size(0), priors.size(0)

        message = 'Invalid alpha parameter because cls_scores or ' \
                  'bbox_preds are None. If you want to use the ' \
                  'cost-based ATSSAssigner,  please set cls_scores, ' \
                  'bbox_preds and self.alpha at the same time. '

        # compute iou between all bbox and gt
        if self.alpha is None:
            # ATSSAssigner
            overlaps = self.iou_calculator(priors, gt_bboxes)
            if ('scores' in pred_instances or 'bboxes' in pred_instances):
                warnings.warn(message)

        else:
            # Dynamic cost ATSSAssigner in DDOD
            assert ('scores' in pred_instances
                    and 'bboxes' in pred_instances), message
            cls_scores = pred_instances.scores
            bbox_preds = pred_instances.bboxes

            # compute cls cost for bbox and GT
            cls_cost = jt.sigmoid(cls_scores[:, gt_labels])

            # compute iou between all bbox and gt
            overlaps = self.iou_calculator(bbox_preds, gt_bboxes)

            # make sure that we are in element-wise multiplication
            assert cls_cost.shape == overlaps.shape

            # overlaps is actually a cost matrix
            overlaps = cls_cost**(1 - self.alpha) * overlaps**self.alpha

        # assign 0 by default
        assigned_gt_inds = jt.full((num_priors, ), 0, dtype=jt.int64)

        if num_gt == 0 or num_priors == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = jt.zeros((num_priors, ), dtype=overlaps.dtype)
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            assigned_labels = jt.full((num_priors, ), -1, dtype=jt.int64)
            return AssignResult(
                num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        # compute center distance between all bbox and gt
        distances = bbox_center_distance(gt_bboxes, priors)

        if (self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None
                and gt_bboxes_ignore.numel() > 0 and priors.numel() > 0):
            ignore_overlaps = self.iou_calculator(
                priors, gt_bboxes_ignore, mode='iof')
            ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            ignore_idxs = ignore_max_overlaps > self.ignore_iof_thr
            distances[ignore_idxs, :] = INF
            assigned_gt_inds[ignore_idxs] = -1

        # Selecting candidates based on the center distance
        candidate_idxs = []
        start_idx = 0
        for level, priors_per_level in enumerate(num_level_priors):
            # on each pyramid level, for each gt,
            # select k bbox whose center are closest to the gt center
            end_idx = start_idx + priors_per_level
            distances_per_level = distances[start_idx:end_idx, :]
            selectable_k = min(self.topk, priors_per_level)
            _, topk_idxs_per_level = distances_per_level.topk(
                selectable_k, dim=0, largest=False)
            candidate_idxs.append(topk_idxs_per_level + start_idx)
            start_idx = end_idx
        candidate_idxs = jt.concat(candidate_idxs, dim=0)

        # get corresponding iou for the these candidates, and compute the
        # mean and std, set mean + std as the iou threshold
        candidate_overlaps = overlaps[candidate_idxs, jt.arange(num_gt)]
        overlaps_mean_per_gt = candidate_overlaps.mean(0)
        overlaps_std_per_gt = [
            candidate_overlaps[:, i].std() for i in range(num_gt)
        ]
        overlaps_std_per_gt = jt.concat(overlaps_std_per_gt, dim=0)
        overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt

        is_pos = candidate_overlaps >= overlaps_thr_per_gt[None, :]

        # limit the positive sample's center in gt
        for gt_idx in range(num_gt):
            candidate_idxs[:, gt_idx] += gt_idx * num_priors
        priors_cx = (priors[:, 0] + priors[:, 2]) / 2.0
        priors_cy = (priors[:, 1] + priors[:, 3]) / 2.0
        ep_priors_cx = priors_cx.view(1, -1).expand(
            num_gt, num_priors).contiguous().view(-1)
        ep_priors_cy = priors_cy.view(1, -1).expand(
            num_gt, num_priors).contiguous().view(-1)
        candidate_idxs = candidate_idxs.view(-1)

        # calculate the left, top, right, bottom distance between positive
        # prior center and gt side
        l_ = ep_priors_cx[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 0]
        t_ = ep_priors_cy[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 1]
        r_ = gt_bboxes[:, 2] - ep_priors_cx[candidate_idxs].view(-1, num_gt)
        b_ = gt_bboxes[:, 3] - ep_priors_cy[candidate_idxs].view(-1, num_gt)
        is_in_gts = jt.stack([l_, t_, r_, b_], dim=1).min(dim=1)[0] > 0.01

        is_pos = is_pos & is_in_gts

        # if an anchor box is assigned to multiple gts,
        # the one with the highest IoU will be selected.
        overlaps_inf = jt.full_like(overlaps, -INF).t().contiguous().view(-1)
        index = candidate_idxs.view(-1)[is_pos.view(-1)]
        overlaps_inf[index] = overlaps.t().contiguous().view(-1)[index]
        overlaps_inf = overlaps_inf.view(num_gt, -1).t()

        argmax_overlaps, max_overlaps = jt.argmax(overlaps_inf, dim=1)
        assigned_gt_inds[
            max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF] + 1

        assigned_labels = jt.full((num_priors, ), -1)
        pos_inds = jt.nonzero(assigned_gt_inds > 0).squeeze(dim=1)
        if pos_inds.numel() > 0:
            assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] -
                                                  1]
        return AssignResult(
            num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)
