import math
import jittor as jt
from jittor import nn
from .utils import weighted_loss


def bbox_overlaps(bboxes1, bboxes2, mode="iou", is_aligned=False, eps=1e-6):
    """計算兩組邊界框之間的重疊度 (Jittor 版本)。"""
    assert mode in ["iou", "iof", "giou"], f"Unsupported mode {mode}"
    assert bboxes1.shape[-1] == 4 or bboxes1.shape[0] == 0
    assert bboxes2.shape[-1] == 4 or bboxes2.shape[0] == 0
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.shape[-2]
    cols = bboxes2.shape[-2]
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            # [遷移] .new() -> jt.empty()
            return jt.empty(batch_shape + (rows,))
        else:
            return jt.empty(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])

    if is_aligned:
        # [遷移] torch.max/min -> jt.maximum/minimum
        lt = jt.maximum(bboxes1[..., :2], bboxes2[..., :2])
        rb = jt.minimum(bboxes1[..., 2:], bboxes2[..., 2:])
        wh = (rb - lt).clamp(min_v=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ["iou", "giou"]:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == "giou":
            enclosed_lt = jt.minimum(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = jt.maximum(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = jt.maximum(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])
        rb = jt.minimum(bboxes1[..., :, None, 2:], bboxes2[..., None, :, :2])
        wh = (rb - lt).clamp(min_v=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ["iou", "giou"]:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]
        if mode == "giou":
            enclosed_lt = jt.minimum(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])
            enclosed_rb = jt.maximum(bboxes1[..., :, None, 2:], bboxes2[..., None, :, :2])

    # [遷移] .new_tensor([eps]) -> jt.array([eps])
    eps_var = jt.array([eps]).cast(union.dtype)
    union = jt.maximum(union, eps_var)
    ious = overlap / union
    if mode in ["iou", "iof"]:
        return ious
    
    # GIoU 計算
    enclose_wh = (enclosed_rb - enclosed_lt).clamp(min_v=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = jt.maximum(enclose_area, eps_var)
    gious = ious - (enclose_area - union) / enclose_area
    return gious

@weighted_loss
def iou_loss(pred, target, eps=1e-6):
    """IoU loss (Jittor 版本)。"""
    ious = bbox_overlaps(pred, target, is_aligned=True).clamp(min_v=eps)
    # [遷移] .log() 是 Jittor 的寫法
    loss = -ious.log()
    return loss

@weighted_loss
def bounded_iou_loss(pred, target, beta=0.2, eps=1e-3):
    """BIoULoss (Jittor 版本)。"""
    pred_ctrx = (pred[:, 0] + pred[:, 2]) * 0.5
    pred_ctry = (pred[:, 1] + pred[:, 3]) * 0.5
    pred_w = pred[:, 2] - pred[:, 0]
    pred_h = pred[:, 3] - pred[:, 1]
    # [遷移] @torch.no_grad() -> @jt.no_grad()
    with jt.no_grad():
        target_ctrx = (target[:, 0] + target[:, 2]) * 0.5
        target_ctry = (target[:, 1] + target[:, 3]) * 0.5
        target_w = target[:, 2] - target[:, 0]
        target_h = target[:, 3] - target[:, 1]

    dx = target_ctrx - pred_ctrx
    dy = target_ctry - pred_ctry

    loss_dx = 1 - jt.maximum(
        (target_w - 2 * dx.abs()) / (target_w + 2 * dx.abs() + eps),
        jt.zeros_like(dx))
    loss_dy = 1 - jt.maximum(
        (target_h - 2 * dy.abs()) / (target_h + 2 * dy.abs() + eps),
        jt.zeros_like(dy))
    loss_dw = 1 - jt.minimum(target_w / (pred_w + eps), pred_w / (target_w + eps))
    loss_dh = 1 - jt.minimum(target_h / (pred_h + eps), pred_h / (target_h + eps))
    loss_comb = jt.stack([loss_dx, loss_dy, loss_dw, loss_dh], dim=-1).view(loss_dx.shape[0], -1)
    
    # [遷移] torch.where -> jt.where
    loss = jt.where(
        loss_comb < beta, 0.5 * loss_comb * loss_comb / beta, loss_comb - 0.5 * beta
    ).sum(dim=-1)
    return loss

@weighted_loss
def giou_loss(pred, target, eps=1e-7):
    """GIoU loss (Jittor 版本)。"""
    gious = bbox_overlaps(pred, target, mode="giou", is_aligned=True, eps=eps)
    loss = 1 - gious
    return loss

@weighted_loss
def diou_loss(pred, target, eps=1e-7):
    """DIoU loss (Jittor 版本)。"""
    lt = jt.maximum(pred[:, :2], target[:, :2])
    rb = jt.minimum(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp(min_v=0)
    overlap = wh[:, 0] * wh[:, 1]
    ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = ap + ag - overlap + eps
    ious = overlap / union
    enclose_x1y1 = jt.minimum(pred[:, :2], target[:, :2])
    enclose_x2y2 = jt.maximum(pred[:, 2:], target[:, 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min_v=0)
    cw = enclose_wh[:, 0]
    ch = enclose_wh[:, 1]
    c2 = cw**2 + ch**2 + eps
    b1_x1, b1_y1, b1_x2, b1_y2 = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = target[:, 0], target[:, 1], target[:, 2], target[:, 3]
    left = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4
    right = ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
    rho2 = left + right
    dious = ious - rho2 / c2
    loss = 1 - dious
    return loss

@weighted_loss
def ciou_loss(pred, target, eps=1e-7):
    """CIoU loss (Jittor 版本)。"""
    lt = jt.maximum(pred[:, :2], target[:, :2])
    rb = jt.minimum(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp(min_v=0)
    overlap = wh[:, 0] * wh[:, 1]
    ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = ap + ag - overlap + eps
    ious = overlap / union
    enclose_x1y1 = jt.minimum(pred[:, :2], target[:, :2])
    enclose_x2y2 = jt.maximum(pred[:, 2:], target[:, 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min_v=0)
    cw = enclose_wh[:, 0]
    ch = enclose_wh[:, 1]
    c2 = cw**2 + ch**2 + eps
    b1_x1, b1_y1, b1_x2, b1_y2 = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = target[:, 0], target[:, 1], target[:, 2], target[:, 3]
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    left = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4
    right = ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
    rho2 = left + right
    factor = 4 / math.pi**2
    # [遷移] torch.pow(torch.atan(...)) -> jt.atan(...).pow(2)
    v = factor * (jt.atan(w2 / h2) - jt.atan(w1 / h1)).pow(2)
    cious = ious - (rho2 / c2 + v**2 / (1 - ious + v))
    loss = 1 - cious
    return loss

class IoULoss(nn.Module):
    """IoULoss (Jittor 版本)。"""
    def __init__(self, eps=1e-6, reduction="mean", loss_weight=1.0):
        super(IoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    # [遷移] forward -> execute
    def execute(self, pred, target, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        # [遷移] torch.any -> jt.any
        if (weight is not None) and (not jt.any(weight > 0)) and (reduction != "none"):
            if pred.ndim == weight.ndim + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()
        loss = self.loss_weight * iou_loss(
            pred, target, weight, eps=self.eps, reduction=reduction,
            avg_factor=avg_factor, **kwargs)
        return loss

class BoundedIoULoss(nn.Module):
    """BoundedIoULoss (Jittor 版本)。"""
    def __init__(self, beta=0.2, eps=1e-3, reduction="mean", loss_weight=1.0):
        super(BoundedIoULoss, self).__init__()
        self.beta = beta
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def execute(self, pred, target, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        if weight is not None and not jt.any(weight > 0):
            if pred.ndim == weight.ndim + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        loss = self.loss_weight * bounded_iou_loss(
            pred, target, weight, beta=self.beta, eps=self.eps,
            reduction=reduction, avg_factor=avg_factor, **kwargs)
        return loss

class GIoULoss(nn.Module):
    """GIoULoss (Jittor 版本)。"""
    def __init__(self, eps=1e-6, reduction="mean", loss_weight=1.0):
        super(GIoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def execute(self, pred, target, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        if weight is not None and not jt.any(weight > 0):
            if pred.ndim == weight.ndim + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        loss = self.loss_weight * giou_loss(
            pred, target, weight, eps=self.eps, reduction=reduction,
            avg_factor=avg_factor, **kwargs)
        return loss

class DIoULoss(nn.Module):
    """DIoULoss (Jittor 版本)。"""
    def __init__(self, eps=1e-6, reduction="mean", loss_weight=1.0):
        super(DIoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def execute(self, pred, target, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        if weight is not None and not jt.any(weight > 0):
            if pred.ndim == weight.ndim + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        loss = self.loss_weight * diou_loss(
            pred, target, weight, eps=self.eps, reduction=reduction,
            avg_factor=avg_factor, **kwargs)
        return loss

class CIoULoss(nn.Module):
    """CIoULoss (Jittor 版本)。"""
    def __init__(self, eps=1e-6, reduction="mean", loss_weight=1.0):
        super(CIoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def execute(self, pred, target, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        if weight is not None and not jt.any(weight > 0):
            if pred.ndim == weight.ndim + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        loss = self.loss_weight * ciou_loss(
            pred, target, weight, eps=self.eps, reduction=reduction,
            avg_factor=avg_factor, **kwargs)
        return loss