import jittor as jt
import jittor.nn as nn

from .utils import weighted_loss


def _bce_with_logits(logits, targets):
    # Jittor 的 BCE 接口不支持 reduction="none"，这里实现逐元素 BCE
    # stable: max(x,0) - x*target + log(1+exp(-abs(x)))
    max_val = nn.relu(logits)
    return max_val - logits * targets + jt.log(1 + jt.exp(-jt.abs(logits)))


@weighted_loss
def quality_focal_loss(pred, target, beta=2.0):
    """
    Quality Focal Loss (Jittor 版本), 与 PyTorch 实现保持一致。
    pred: [N, C] logits
    target: (label, score)
    """
    assert (
        len(target) == 2
    ), "target for QFL must be (label, score)"
    label, score = target

    pred_sigmoid = jt.sigmoid(pred)
    scale_factor = pred_sigmoid
    zerolabel = jt.zeros_like(pred)
    loss = _bce_with_logits(pred, zerolabel) * jt.pow(scale_factor, beta)

    bg_class_ind = pred.shape[1]
    pos = jt.nonzero((label >= 0) & (label < bg_class_ind))
    if len(pos.shape) > 1:
        pos = pos.squeeze(1)
    if pos.numel() > 0:
        pos_label = label[pos].long()
        scale_factor = score[pos] - pred_sigmoid[pos, pos_label]
        pos_loss = _bce_with_logits(pred[pos, pos_label], score[pos]) * jt.pow(
            jt.abs(scale_factor), beta
        )
        loss[pos, pos_label] = pos_loss

    loss = loss.sum(dim=1)
    return loss


@weighted_loss
def distribution_focal_loss(pred, label):
    """
    Distribution Focal Loss (Jittor 版本), 与 PyTorch 实现保持一致。
    pred: [N, n+1] logits
    label: [N] float in [0, n]
    """
    dis_left = label.floor().int32()
    dis_right = dis_left + 1
    weight_left = dis_right.float32() - label
    weight_right = label - dis_left.float32()

    # 使用 log_softmax 计算交叉熵，避免依赖外部接口
    log_prob = nn.log_softmax(pred, dim=1)
    n = pred.shape[0]
    idx = jt.arange(n)
    left_loss = -log_prob[idx, dis_left]
    right_loss = -log_prob[idx, dis_right]
    loss = left_loss * weight_left + right_loss * weight_right
    return loss


class QualityFocalLoss(nn.Module):
    def __init__(self, use_sigmoid=True, beta=2.0, reduction="mean", loss_weight=1.0):
        super(QualityFocalLoss, self).__init__()
        assert use_sigmoid is True, "Only sigmoid in QFL supported now."
        self.use_sigmoid = use_sigmoid
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def execute(
        self, pred, target, weight=None, avg_factor=None, reduction_override=None
    ):
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        if self.use_sigmoid:
            loss_cls = self.loss_weight * quality_focal_loss(
                pred,
                target,
                weight,
                beta=self.beta,
                reduction=reduction,
                avg_factor=avg_factor,
            )
        else:
            raise NotImplementedError
        return loss_cls


class DistributionFocalLoss(nn.Module):
    def __init__(self, reduction="mean", loss_weight=1.0):
        super(DistributionFocalLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def execute(
        self, pred, target, weight=None, avg_factor=None, reduction_override=None
    ):
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        loss_cls = self.loss_weight * distribution_focal_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor
        )
        return loss_cls
