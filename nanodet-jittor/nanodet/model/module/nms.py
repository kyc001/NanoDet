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
    
    # Jittor's nms takes iou_threshold
    iou_threshold = nms_cfg_.pop("iou_threshold", 0.5)

    # Jittor's nms is simpler and doesn't have a split_thr logic internally
    # It returns indices directly.
    keep = jt.nms(boxes_for_nms, scores, iou_threshold)
    
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

    # [遷移] torch.masked_select -> Jittor's boolean indexing
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