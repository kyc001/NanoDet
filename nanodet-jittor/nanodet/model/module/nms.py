import jittor as jt


def batched_nms(boxes, scores, idxs, nms_cfg, class_agnostic=False):
    """Batched NMS (Jittor Version)."""
    nms_cfg_ = nms_cfg.copy()
    class_agnostic = nms_cfg_.pop("class_agnostic", class_agnostic)
    if class_agnostic:
        boxes_for_nms = boxes
    else:
        max_coordinate = boxes.max()
        offsets = idxs.cast(boxes.dtype) * (max_coordinate + 1)
        boxes_for_nms = boxes + offsets[:, None]

    nms_op_name = nms_cfg_.pop("type", "nms")
    assert nms_op_name == "nms", "Only vanilla NMS is supported in this placeholder"

    # Jittor's nms signature: jt.nms(dets, iou_thr), where dets = [x1,y1,x2,y2,score]
    iou_threshold = nms_cfg_.pop("iou_threshold", 0.5)

    dets_for_nms = jt.concat([boxes_for_nms, scores[:, None]], -1)
    keep = jt.nms(dets_for_nms, iou_threshold)

    boxes = boxes[keep]
    scores = scores[keep]

    return jt.concat([boxes, scores[:, None]], -1), keep

def multiclass_nms(
    multi_bboxes, multi_scores, score_thr, nms_cfg, max_num=-1, score_factors=None
):
    """Multiclass NMS (Jittor Version)."""
    num_classes = multi_scores.shape[1] - 1
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.reshape(multi_scores.shape[0], -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(multi_scores.shape[0], num_classes, 4)
    scores = multi_scores[:, :-1]

    valid_mask = scores > score_thr

    # [遷移] jt.masked_select -> Jittor's boolean indexing
    bboxes = bboxes[valid_mask.unsqueeze(-1).expand_as(bboxes)].reshape(-1, 4)
    if score_factors is not None:
        scores = scores * score_factors[:, None]
    scores = scores[valid_mask]
    # [遷移] .nonzero(as_tuple=False) -> .nonzero()
    labels = valid_mask.nonzero()[:, 1]

    if bboxes.numel() == 0:
        # [遷移] .new_zeros -> jt.zeros
        bboxes = jt.zeros((0, 5), dtype=multi_bboxes.dtype)
        labels = jt.zeros((0,), dtype='int64')
        # [遷移] ONNX export logic is removed
        return bboxes, labels

    dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)

    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]

    return dets, labels[keep]

# Override with a safer per-class NMS implementation to avoid boolean indexing pitfalls in Jittor
def multiclass_nms(
    multi_bboxes, multi_scores, score_thr, nms_cfg, max_num=-1, score_factors=None
):
    """Multiclass NMS (per-class loop to ensure correct indexing on Jittor).
    Args:
        multi_bboxes: (N, 4) or (N, num_classes*4)
        multi_scores: (N, num_classes+1), last column is background
    Returns:
        dets: (M, 5) [x1,y1,x2,y2,score], labels: (M,)
    """
    num_classes = multi_scores.shape[1] - 1  # last column is background
    scores_all = multi_scores[:, :num_classes]
    if multi_bboxes.shape[1] > 4:
        # (N, num_classes*4) -> (N, num_classes, 4)
        bboxes_all = multi_bboxes.reshape(multi_scores.shape[0], -1, 4)
    else:
        # shared boxes for all classes
        bboxes_all = multi_bboxes[:, None].expand(multi_scores.shape[0], num_classes, 4)

    dets_list = []
    labels_list = []
    for cls_id in range(num_classes):
        cls_scores = scores_all[:, cls_id]
        valid = cls_scores > score_thr
        if valid.sum().item() == 0:
            continue
        boxes = bboxes_all[:, cls_id, :]
        boxes = boxes[valid]
        scores = cls_scores[valid]
        if score_factors is not None:
            scores = scores * score_factors[valid]
        # run NMS for this class (class-agnostic boxes)
        idxs = jt.zeros((scores.shape[0],), dtype='int32') + cls_id
        dets_cls, keep = batched_nms(boxes, scores, idxs, nms_cfg, class_agnostic=True)
        dets_list.append(dets_cls)
        labels_list.append(jt.ones((dets_cls.shape[0],), dtype='int64') * cls_id)

    if len(dets_list) == 0:
        return jt.zeros((0, 5), dtype=multi_bboxes.dtype), jt.zeros((0,), dtype='int64')

    dets = jt.concat(dets_list, dim=0)
    labels = jt.concat(labels_list, dim=0)

    # sort by score desc (handle Jittor argsort possibly returning (idx, sorted))
    order_pack = jt.argsort(dets[:, 4], descending=True)
    order = order_pack[0] if isinstance(order_pack, (tuple, list)) else order_pack
    # ensure integer dtype
    if getattr(order, 'dtype', None) != 'int32':
        try:
            order = order.int32()
        except Exception:
            pass
    dets = dets[order]
    labels = labels[order]

    if max_num > 0 and dets.shape[0] > max_num:
        dets = dets[:max_num]
        labels = labels[:max_num]

    return dets, labels
