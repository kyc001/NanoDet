import jittor as jt
from jittor import nn
from .utils import weighted_loss


@weighted_loss
def quality_focal_loss(pred, target, beta=2.0):
    """Quality Focal Loss (QFL) 的 Jittor 版本。"""
    assert len(target) == 2, "QFL 的 target 必須是包含類別和品質標籤的元組"
    label, score = target

    pred_sigmoid = pred.sigmoid()
    scale_factor = pred_sigmoid
    # [遷移] .new_zeros -> jt.zeros_like
    zerolabel = jt.zeros_like(pred)
    # [遷移] F.binary_cross_entropy_with_logits -> jt.nn.binary_cross_entropy_with_logits
    # Jittor 中此函數不接受 reduction 參數，預設行為就是 'none'
    loss = jt.nn.binary_cross_entropy_with_logits(
        pred, zerolabel
    ) * scale_factor.pow(beta)

    # [遷移] .size(1) -> .shape[1]
    bg_class_ind = pred.shape[1]
    # [遷移] torch.nonzero(...) -> jt.nonzero(...), 並處理返回形狀
    pos_search = ((label >= 0) & (label < bg_class_ind)).nonzero()
    if pos_search.numel() == 0:
        return loss.sum(dim=1, keepdims=False)
    
    pos = pos_search[:, 0]
    # [遷移] .long() -> .int64()
    pos_label = label[pos].int64()
    
    scale_factor = score[pos] - pred_sigmoid[pos, pos_label]
    loss[pos, pos_label] = jt.nn.binary_cross_entropy_with_logits(
        pred[pos, pos_label], score[pos]
    ) * scale_factor.abs().pow(beta)

    loss = loss.sum(dim=1, keepdims=False)
    return loss


@weighted_loss
def distribution_focal_loss(pred, label):
    """Distribution Focal Loss (DFL) 的 Jittor 版本。"""
    # [遷移] .long() -> .int64(), .float() -> .float32()
    dis_left = label.int64()
    dis_right = dis_left + 1
    weight_left = dis_right.float32() - label
    weight_right = label - dis_left.float32()
    
    # [遷移] F.cross_entropy -> jt.nn.cross_entropy_loss
    loss = (
        jt.nn.cross_entropy_loss(pred, dis_left, reduction="none") * weight_left
        + jt.nn.cross_entropy_loss(pred, dis_right, reduction="none") * weight_right
    )
    return loss


class QualityFocalLoss(nn.Module):
    """Quality Focal Loss (QFL) 類別的 Jittor 版本。"""
    def __init__(self, use_sigmoid=True, beta=2.0, reduction="mean", loss_weight=1.0):
        super(QualityFocalLoss, self).__init__()
        assert use_sigmoid is True, "目前 QFL 只支持 sigmoid"
        self.use_sigmoid = use_sigmoid
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    # [遷移] forward -> execute
    def execute(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
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
    """Distribution Focal Loss (DFL) 類別的 Jittor 版本。"""
    def __init__(self, reduction="mean", loss_weight=1.0):
        super(DistributionFocalLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    # [遷移] forward -> execute
    def execute(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        loss_cls = self.loss_weight * distribution_focal_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor
        )
        return loss_cls
