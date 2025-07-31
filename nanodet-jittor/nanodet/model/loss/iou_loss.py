# Copyright 2021 RangiLyu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jittor as jt
from jittor import nn

from .utils import weighted_loss


def bbox_overlaps(bboxes1, bboxes2, mode="iou", is_aligned=False, eps=1e-6):
    """Calculate overlap between two set of bboxes.

    If ``is_aligned `` is ``False``, then calculate the overlaps between each
    bbox of bboxes1 and bboxes2, otherwise the overlaps between each aligned
    pair of bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (B, m, 4) in <x1, y1, x2, y2> format or empty.
        bboxes2 (Tensor): shape (B, n, 4) in <x1, y1, x2, y2> format or empty.
            B indicates the batch dim, in shape (B1, B2, ..., Bn).
            If ``is_aligned `` is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union) or "iof" (intersection over
            foreground) or "giou" (generalized intersection over union).
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.
        eps (float, optional): A value added to the denominator for numerical
            stability. Default 1e-6.

    Returns:
        Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)
    """

    assert mode in ["iou", "iof", "giou"], f"Unsupported mode {mode}"
    # Either the boxes are empty or the length of boxes's last dimenstion is 4
    assert bboxes1.size(-1) == 4 or bboxes1.size(0) == 0
    assert bboxes2.size(-1) == 4 or bboxes2.size(0) == 0

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return jt.zeros(batch_shape + (rows,), dtype=bboxes1.dtype)
        else:
            return jt.zeros(batch_shape + (rows, cols), dtype=bboxes1.dtype)

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])

    if is_aligned:
        lt = jt.maximum(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
        rb = jt.minimum(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]

        wh = jt.clamp(rb - lt, min_v=0)  # [B, rows, 2]
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ["iou", "giou"]:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == "giou":
            enclosed_lt = jt.minimum(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = jt.maximum(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = jt.maximum(
            bboxes1[..., :, None, :2], bboxes2[..., None, :, :2]
        )  # [B, rows, cols, 2]
        rb = jt.minimum(
            bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:]
        )  # [B, rows, cols, 2]

        wh = jt.clamp(rb - lt, min_v=0)  # [B, rows, cols, 2]
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ["iou", "giou"]:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]
        if mode == "giou":
            enclosed_lt = jt.minimum(
                bboxes1[..., :, None, :2], bboxes2[..., None, :, :2]
            )
            enclosed_rb = jt.maximum(
                bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:]
            )

    eps_tensor = jt.array([eps], dtype=union.dtype)
    union = jt.maximum(union, eps_tensor)
    ious = overlap / union
    if mode in ["iou", "iof"]:
        return ious
    # calculate gious
    enclose_wh = jt.clamp(enclosed_rb - enclosed_lt, min_v=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = jt.maximum(enclose_area, eps_tensor)
    gious = ious - (enclose_area - union) / enclose_area
    return gious


@weighted_loss
def iou_loss(pred, target, eps=1e-6):
    """IoU loss.

    Computing the IoU loss between a set of predicted bboxes and target bboxes.
    The loss is calculated as negative log of IoU.

    Args:
        pred (jt.Var): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (jt.Var): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).

    Return:
        jt.Var: Loss tensor.
    """
    ious = bbox_overlaps(pred, target, is_aligned=True).clamp(min=eps)
    loss = -ious.log()
    return loss


@weighted_loss
def giou_loss(pred, target, eps=1e-7):
    r"""`Generalized Intersection over Union: A Metric and A Loss for Bounding
    Box Regression <https://arxiv.org/abs/1902.09630>`_.

    Args:
        pred (jt.Var): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (jt.Var): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).

    Return:
        jt.Var: Loss tensor.
    """
    gious = bbox_overlaps(pred, target, mode="giou", is_aligned=True, eps=eps)
    loss = 1 - gious
    return loss


class IoULoss(nn.Module):
    """IoULoss.

    Args:
        eps (float): Eps to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self, eps=1e-6, reduction="mean", loss_weight=1.0):
        super(IoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def execute(
        self,
        pred,
        target,
        weight=None,
        avg_factor=None,
        reduction_override=None,
        **kwargs,
    ):
        if weight is not None and not jt.any(weight > 0):
            if pred.ndim == weight.ndim + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        loss = self.loss_weight * iou_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs,
        )
        return loss


class GIoULoss(nn.Module):
    """GIoULoss.

    Args:
        eps (float): Eps to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self, eps=1e-6, reduction="mean", loss_weight=1.0):
        super(GIoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def execute(
        self,
        pred,
        target,
        weight=None,
        avg_factor=None,
        reduction_override=None,
        **kwargs,
    ):
        if weight is not None and not jt.any(weight > 0):
            if pred.ndim == weight.ndim + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        loss = self.loss_weight * giou_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs,
        )
        return loss
