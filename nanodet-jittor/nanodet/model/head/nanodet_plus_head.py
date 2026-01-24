import math

import cv2
import numpy as np
import jittor as jt
import jittor.nn as nn

# 🔧 直接使用 JittorDet 的成熟实现，不再使用本地复制
from nanodet.util import  multi_apply, overlay_bbox_cv
# 使用本地兼容实现，避免修改标准库 jittordet 并修复 clamp_ 参数不兼容
from nanodet.util.box_transform import distance2bbox, bbox2distance
from ..loss.gfocal_loss import DistributionFocalLoss, QualityFocalLoss
from ..loss.iou_loss import GIoULoss
from ..module.init_weights import normal_init
# 使用 JittorDet 自带的 NMS，避免本地实现差异
from ..module.nms import multiclass_nms

# NOTE: 为了对齐 PyTorch 版本训练行为，这里不再使用分离 head + Scale。
# 替换外部 Integral，使用本地 DistributionProject 实现以严格对齐 PyTorch 逻辑
# from jittordet.models.dense_heads.gfl_head import Integral
class DistributionProject(nn.Module):
    def __init__(self, reg_max: int):
        super().__init__()
        self.reg_max = reg_max
        # 使用与 PyTorch Integral 完全一致的投影向量
        self.register_buffer("project", jt.arange(0, reg_max + 1, dtype=jt.float32))

    def execute(self, reg_logits: jt.Var) -> jt.Var:
        # 与 PyTorch Integral 完全对齐：在每个方向的 (reg_max+1) 维上做 softmax 再投影
        shape = reg_logits.shape  # [..., 4*(m+1)]
        x = reg_logits.float32().reshape(*shape[:-1], 4, self.reg_max + 1)
        x = nn.softmax(x, dim=-1)
        proj = self.project.float32()  # [m+1]
        x = (x * proj).sum(dim=-1)  # [..., 4]
        return x.reshape(*shape[:-1], 4).float32()
def reduce_mean(tensor):
    return tensor
from .assigner.dsl_assigner import DynamicSoftLabelAssigner
from .assigner.center_radius_assigner import CenterRadiusAssigner
from ...data.transform.warp import warp_boxes
from ..module.conv import ConvModule, DepthwiseConvModule



class NanoDetPlusHead(nn.Module):


    def __init__(
        self,
        num_classes,
        loss,
        input_channel,
        feat_channels=96,
        stacked_convs=2,
        kernel_size=5,
        strides=[8, 16, 32],
        conv_type="DWConv",
        norm_cfg=dict(type="BN"),
        reg_max=7,
        activation="LeakyReLU",
        assigner_cfg=dict(topk=13),
        share_cls_reg_tower=False,
        **kwargs
    ):
        super(NanoDetPlusHead, self).__init__()
        self.num_classes = num_classes
        # 兼容旧配置参数（当前实现与 PyTorch 对齐，不使用分离 head）
        self.share_cls_reg_tower = share_cls_reg_tower
        self.in_channels = input_channel
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.kernel_size = kernel_size
        self.strides = strides
        self.reg_max = reg_max
        self.activation = activation
        self.ConvModule = ConvModule if conv_type == "Conv" else DepthwiseConvModule

        self.loss_cfg = loss
        self.norm_cfg = norm_cfg

        # 按配置选择分配器，默认 DSL；可切 CenterRadiusAssigner 保证有正样本
        assigner_type = assigner_cfg.get('type', 'DSL') if isinstance(assigner_cfg, dict) else 'DSL'
        if assigner_type == 'CenterRadius':
            self.assigner = CenterRadiusAssigner(center_radius=assigner_cfg.get('center_radius', 2.5))
        else:
            self.assigner = DynamicSoftLabelAssigner(**{k:v for k,v in assigner_cfg.items() if k!='type'})
        self.distribution_project = DistributionProject(self.reg_max)

        self.loss_qfl = QualityFocalLoss(
            beta=self.loss_cfg.loss_qfl.beta,
            loss_weight=self.loss_cfg.loss_qfl.loss_weight,
        )
        self.loss_dfl = DistributionFocalLoss(
            loss_weight=self.loss_cfg.loss_dfl.loss_weight
        )
        self.loss_bbox = GIoULoss(loss_weight=self.loss_cfg.loss_bbox.loss_weight)
        self._init_layers()
        self.init_weights()

    def _init_layers(self):
        # 分类分支 conv 塔
        self.cls_convs = nn.ModuleList()
        for _ in self.strides:
            self.cls_convs.append(self._buid_not_shared_head())
        # 逐层输出头：与 PyTorch 对齐，单头输出 cls + reg
        self.gfl_cls = nn.ModuleList(
            [
                nn.Conv2d(
                    self.feat_channels,
                    self.num_classes + 4 * (self.reg_max + 1),
                    1,
                    padding=0,
                )
                for _ in self.strides
            ]
        )

    def _buid_not_shared_head(self):
        cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            cls_convs.append(
                self.ConvModule(
                    chn,
                    self.feat_channels,
                    self.kernel_size,
                    stride=1,
                    padding=self.kernel_size // 2,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None,
                    activation=self.activation,
                )
            )
        return cls_convs

    def init_weights(self):
        # 初始化分类分支 conv 塔
        for m in list(self.cls_convs.modules()):
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        # init cls head with confidence = 0.01
        bias_cls = -4.595
        for i in range(len(self.strides)):
            normal_init(self.gfl_cls[i], std=0.01, bias=bias_cls)
        print("Finish initialize NanoDet-Plus Head.")

    def execute(self, feats):
        # 与 strides 个数保持一致
        if isinstance(feats, (list, tuple)) and len(feats) > len(self.strides):
            feats = feats[:len(self.strides)]
        outputs = []
        for feat, cls_convs, gfl_cls in zip(feats, self.cls_convs, self.gfl_cls):
            for conv in cls_convs:
                feat = conv(feat)
            output = gfl_cls(feat)
            outputs.append(output.flatten(start_dim=2))
        outputs = jt.cat(outputs, dim=2).permute(0, 2, 1)
        return outputs

    def loss(self, preds, gt_meta, aux_preds=None):

        # Jittor 无 .device 概念，直接忽略 device 参数
        device = None
        batch_size = preds.shape[0]
        gt_bboxes = gt_meta["gt_bboxes"]
        gt_labels = gt_meta["gt_labels"]

        gt_bboxes_ignore = gt_meta.get("gt_bboxes_ignore", None)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(batch_size)]

        input_height, input_width = gt_meta["img"].shape[2:]
        featmap_sizes = [
            (int(math.ceil(input_height / stride)), int(math.ceil(input_width / stride)))
            for stride in self.strides
        ]
        # get grid cells of one image
        mlvl_center_priors = [
            self.get_single_level_center_priors(
                batch_size,
                featmap_sizes[i],
                stride,
                dtype=jt.float32,
                device=None,
            )
            for i, stride in enumerate(self.strides)
        ]
        # 🔧 修复：使用 jt.cat 而不是 jt.cat
        center_priors = jt.cat(mlvl_center_priors, dim=1)

        cls_preds, reg_preds = preds.split(
            [self.num_classes, 4 * (self.reg_max + 1)], dim=-1
        )
        


        # 与 PyTorch 对齐：直接投影后乘以 stride
        dis_preds = self.distribution_project(reg_preds) * center_priors[..., 2, None]
        decoded_bboxes = distance2bbox(center_priors[..., :2], dis_preds)

        if aux_preds is not None:
            # use auxiliary head to assign
            aux_cls_preds, aux_reg_preds = aux_preds.split(
                [self.num_classes, 4 * (self.reg_max + 1)], dim=-1
            )
            aux_dis_preds = self.distribution_project(aux_reg_preds) * center_priors[..., 2, None]
            aux_decoded_bboxes = distance2bbox(center_priors[..., :2], aux_dis_preds)
            # 与 PyTorch 对齐：assigner 使用 detach
            batch_assign_res = multi_apply(
                self.target_assign_single_img,
                aux_cls_preds.detach(),
                center_priors,
                aux_decoded_bboxes.detach(),
                gt_bboxes,
                gt_labels,
                gt_bboxes_ignore,
            )
        else:
            # use self prediction to assign
            batch_assign_res = multi_apply(
                self.target_assign_single_img,
                cls_preds.detach(),
                center_priors,
                decoded_bboxes.detach(),
                gt_bboxes,
                gt_labels,
                gt_bboxes_ignore,
            )

        loss, loss_states = self._get_loss_from_assign(
            cls_preds, reg_preds, decoded_bboxes, batch_assign_res
        )

        if aux_preds is not None:
            aux_loss, aux_loss_states = self._get_loss_from_assign(
                aux_cls_preds, aux_reg_preds, aux_decoded_bboxes, batch_assign_res
            )
            loss = loss + aux_loss
            for k, v in aux_loss_states.items():
                loss_states["aux_" + k] = v
        return loss, loss_states

    def _get_loss_from_assign(self, cls_preds, reg_preds, decoded_bboxes, assign):
        (
            labels,
            label_scores,
            label_weights,
            bbox_targets,
            dist_targets,
            num_pos,
        ) = assign
        # 与 PyTorch 对齐：使用总正样本数作为 avg_factor
        num_total_samples = jt.clamp(
            reduce_mean(jt.array(float(sum(num_pos)))), min_v=1.0
        )

        labels = jt.cat(labels, dim=0)
        label_scores = jt.cat(label_scores, dim=0)
        label_weights = jt.cat(label_weights, dim=0)
        bbox_targets = jt.cat(bbox_targets, dim=0)
        cls_preds = cls_preds.reshape(-1, self.num_classes)
        reg_preds = reg_preds.reshape(-1, 4 * (self.reg_max + 1))
        decoded_bboxes = decoded_bboxes.reshape(-1, 4)

        loss_qfl = self.loss_qfl(
            cls_preds,
            (labels, label_scores),
            weight=label_weights,
            avg_factor=num_total_samples,
        )

        pos_inds = jt.nonzero(
            (labels >= 0) & (labels < self.num_classes)
        ).squeeze(-1)

        if len(pos_inds) > 0:
            # Jittor: max(dim=1) 直接返回最大值，不是 (values, indices) 元组
            weight_targets = cls_preds[pos_inds].detach().sigmoid().max(dim=1)
            bbox_avg_factor = jt.clamp(reduce_mean(weight_targets.sum()), min_v=1.0)

            loss_bbox = self.loss_bbox(
                decoded_bboxes[pos_inds],
                bbox_targets[pos_inds],
                weight=weight_targets,
                avg_factor=bbox_avg_factor,
            )

            dist_targets = jt.cat(dist_targets, dim=0)
            loss_dfl = self.loss_dfl(
                reg_preds[pos_inds].reshape(-1, self.reg_max + 1),
                dist_targets[pos_inds].reshape(-1),
                weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
                avg_factor=4.0 * bbox_avg_factor,
            )
        else:
            # 负样本分支，设置损失为0
            loss_bbox = reg_preds.sum() * 0
            loss_dfl = reg_preds.sum() * 0

        loss = loss_qfl + loss_bbox + loss_dfl
        loss_states = dict(loss_qfl=loss_qfl, loss_bbox=loss_bbox, loss_dfl=loss_dfl)
        return loss, loss_states

    @jt.no_grad()
    def target_assign_single_img(
        self,
        cls_preds,
        center_priors,
        decoded_bboxes,
        gt_bboxes,
        gt_labels,
        gt_bboxes_ignore=None,
    ):

        # 统一 GT 输入为 ndarray 形状 [num_gt,4] / [num_gt]
        import numpy as np
        if isinstance(gt_bboxes, list):
            # 常见情形：每张图传入 [ndarray(N,4)] 的单元素列表
            gt_bboxes = gt_bboxes[0] if len(gt_bboxes) == 1 else np.array(gt_bboxes)
        if isinstance(gt_labels, list):
            gt_labels = gt_labels[0] if len(gt_labels) == 1 else np.array(gt_labels)
        if gt_bboxes_ignore is not None and isinstance(gt_bboxes_ignore, list):
            gt_bboxes_ignore = gt_bboxes_ignore[0] if len(gt_bboxes_ignore) == 1 else np.array(gt_bboxes_ignore)

        # 🔧 转为 Jittor 张量
        gt_bboxes = jt.array(gt_bboxes)
        gt_labels = jt.array(gt_labels)
        gt_bboxes = gt_bboxes.cast(decoded_bboxes.dtype)

        if gt_bboxes_ignore is not None:
            gt_bboxes_ignore = jt.array(gt_bboxes_ignore)
            gt_bboxes_ignore = gt_bboxes_ignore.cast(decoded_bboxes.dtype)

        assign_result = self.assigner.assign(
            cls_preds,
            center_priors,
            decoded_bboxes,
            gt_bboxes,
            gt_labels,
            gt_bboxes_ignore,
        )
        pos_inds, neg_inds, pos_gt_bboxes, pos_assigned_gt_inds = self.sample(
            assign_result, gt_bboxes
        )

        num_priors = center_priors.size(0)
        bbox_targets = jt.zeros_like(center_priors)
        dist_targets = jt.zeros_like(center_priors)
        labels = jt.full((num_priors,), self.num_classes, dtype=jt.int64)
        label_weights = jt.zeros((num_priors,), dtype=jt.float32)
        label_scores = jt.zeros_like(labels).float32()

        num_pos_per_img = int(pos_inds.size(0))
        pos_ious = assign_result.max_overlaps[pos_inds]

        if len(pos_inds) > 0:
            bbox_targets[pos_inds, :] = pos_gt_bboxes
            dist_targets[pos_inds, :] = (
                bbox2distance(center_priors[pos_inds, :2], pos_gt_bboxes)
                / center_priors[pos_inds, None, 2]
            )
            dist_targets = dist_targets.clamp(min_v=0, max_v=self.reg_max - 0.1)
            labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
            label_scores[pos_inds] = pos_ious
            label_weights[pos_inds] = 1.0
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        return (
            labels,
            label_scores,
            label_weights,
            bbox_targets,
            dist_targets,
            num_pos_per_img,
        )

    def sample(self, assign_result, gt_bboxes):
        # 显式转换 dtype 以适配 jittor unique 行为
        gt_inds_i32 = assign_result.gt_inds.cast('int32')
        pos_inds = jt.nonzero(gt_inds_i32 > 0).squeeze(-1)
        pos_inds = pos_inds.unique() if pos_inds.numel() > 0 else pos_inds
        neg_inds = jt.nonzero(gt_inds_i32 == 0).squeeze(-1)
        neg_inds = neg_inds.unique() if neg_inds.numel() > 0 else neg_inds
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1

        if gt_bboxes.numel() == 0:
            # hack for index error case
            assert pos_assigned_gt_inds.numel() == 0
            pos_gt_bboxes = jt.zeros_like(gt_bboxes).view(-1, 4)  # 修复：使用 jt.zeros_like 替代 jt.empty_like
        else:
            if len(gt_bboxes.shape) < 2:
                gt_bboxes = gt_bboxes.view(-1, 4)
            pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds, :]
        return pos_inds, neg_inds, pos_gt_bboxes, pos_assigned_gt_inds

    def post_process(self, preds, meta):

        cls_scores, bbox_preds = preds.split(
            [self.num_classes, 4 * (self.reg_max + 1)], dim=-1
        )
        result_list = self.get_bboxes(cls_scores, bbox_preds, meta)
        det_results = {}
        # normalize meta fields to python types and numpy arrays
        warp_mats = meta.get("warp_matrix", None)
        if warp_mats is None:
            warp_matrixes = [np.eye(3, dtype=np.float32)] * len(result_list)
        else:
            warp_matrixes = warp_mats if isinstance(warp_mats, list) else [warp_mats]
        img_heights = list(meta["img_info"].get("height", []))
        img_widths = list(meta["img_info"].get("width", []))
        img_ids = list(meta["img_info"].get("id", []))
        # ensure same length
        n_items = len(result_list)
        if len(warp_matrixes) != n_items:
            warp_matrixes = (warp_matrixes * n_items)[:n_items]
        if len(img_heights) != n_items:
            img_heights = (img_heights * n_items)[:n_items]
        if len(img_widths) != n_items:
            img_widths = (img_widths * n_items)[:n_items]
        if len(img_ids) != n_items:
            img_ids = (img_ids * n_items)[:n_items]

        for idx, (result, img_width, img_height, img_id, warp_matrix) in enumerate(
            zip(result_list, img_widths, img_heights, img_ids, warp_matrixes)
        ):
            det_result = {}
            det_bboxes, det_labels = result
            det_bboxes = det_bboxes.detach().cpu().numpy()
            # restore to original image coords with robust warp handling
            try:
                # robustly convert warp_matrix to numpy 3x3
                if isinstance(warp_matrix, jt.Var):
                    W = warp_matrix.numpy()
                else:
                    W = np.array(warp_matrix)
                W = W.astype(np.float64)
                # handle batched warp_matrix of shape (B,3,3)
                if W.ndim == 3:
                    if W.shape[0] == 1:
                        W = W[0]
                    elif W.shape[0] > idx:
                        W = W[idx]
                    else:
                        raise ValueError(f"unexpected warp_matrix shape: {W.shape}")
                if W.shape == (2, 3):
                    # upgrade to 3x3
                    W = np.vstack([W, [0.0, 0.0, 1.0]])
                elif W.shape != (3, 3):
                    raise ValueError(f"unexpected warp_matrix shape: {W.shape}")
                invW = np.linalg.inv(W)
                det_bboxes[:, :4] = warp_boxes(
                    det_bboxes[:, :4], invW, int(img_width), int(img_height)
                )
            except Exception as e:
                # fallback: skip warp if malformed matrix
                print(f"[warn] warp_boxes failed for img_id={img_id}: {e}. Use input-scale boxes.")
                pass
            classes = det_labels.detach().cpu().numpy()
            for i in range(self.num_classes):
                inds = classes == i
                if np.any(inds):
                    merged = np.concatenate(
                        [
                            det_bboxes[inds, :4].astype(np.float32),
                            det_bboxes[inds, 4:5].astype(np.float32),
                        ],
                        axis=1,
                    )
                    det_result[i] = merged.tolist()
                else:
                    det_result[i] = []
            # ensure python int for key
            if hasattr(img_id, 'item'):
                img_id = int(img_id.item())
            det_results[int(img_id)] = det_result
        return det_results

    def show_result(
        self, img, dets, class_names, score_thres=0.3, show=True, save_path=None
    ):
        result = overlay_bbox_cv(img, dets, class_names, score_thresh=score_thres)
        # 保存可视化结果
        if save_path:
            try:
                import os
                os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
                ok = cv2.imwrite(save_path, result)
                if not ok:
                    print(f"[warn] cv2.imwrite failed for: {save_path}")
            except Exception as e:
                print(f"[warn] save visualization failed: {e}")
        if show:
            cv2.imshow("det", result)
        return result

    def get_bboxes(self, cls_preds, reg_preds, img_metas):

        # Jittor 的 Var 无 .device 属性，保持占位
        b = cls_preds.shape[0]
        input_height, input_width = img_metas["img"].shape[2:]
        input_shape = (input_height, input_width)

        # 优先使用 forward 记录的实际特征图尺寸，避免 ceil 推断造成的偏差
        if hasattr(self, '_last_featmap_sizes') and len(getattr(self, '_last_featmap_sizes')) == len(self.strides):
            featmap_sizes = list(self._last_featmap_sizes)
        else:
            featmap_sizes = [
                (int(math.ceil(input_height / stride)), int(math.ceil(input_width / stride)))
                for stride in self.strides
            ]
        # get grid cells of one image
        mlvl_center_priors = [
            self.get_single_level_center_priors(
                b,
                featmap_sizes[i],
                stride,
                dtype=jt.float32,
                device=None,
            )
            for i, stride in enumerate(self.strides)
        ]
        center_priors = jt.cat(mlvl_center_priors, dim=1)
        dis_preds = self.distribution_project(reg_preds).float32() * center_priors[..., 2, None].float32()
        bboxes = distance2bbox(center_priors[..., :2], dis_preds, max_shape=input_shape)
        # 分类分数需做 sigmoid，execute 阶段未对 cls 做激活
        scores = cls_preds.sigmoid()
        result_list = []
        # 动态读取评估阈值与最大检测数
        try:
            from nanodet.util import cfg as _cfg
            score_thr = float(getattr(_cfg, 'eval_score_thr', 0.05))
            iou_thr = float(getattr(_cfg, 'eval_iou_thr', 0.6))
            max_det = int(getattr(_cfg, 'eval_max_det', 100))
        except Exception:
            score_thr, iou_thr, max_det = 0.05, 0.6, 100
        for i in range(b):
            # 按 mmdet 接口约定，需在 scores 末尾补一列背景类得分
            score, bbox = scores[i], bboxes[i]
            padding = jt.zeros((score.shape[0], 1), dtype=score.dtype)
            score = jt.concat([score, padding], dim=1)
            results = multiclass_nms(
                bbox,
                score,
                score_thr,
                dict(type="nms", iou_threshold=iou_thr),
                max_det,
            )
            result_list.append(results)
        return result_list

    def get_single_level_center_priors(
        self, batch_size, featmap_size, stride, dtype, device
    ):

        h, w = featmap_size
        # 与 PyTorch 版本对齐：不加 0.5 偏移，直接 i*stride（在我们的数据上更优）
        x_range = (jt.arange(w, dtype=jt.float32) * stride).float32()
        y_range = (jt.arange(h, dtype=jt.float32) * stride).float32()
        # Jittor meshgrid 默认就是 'ij' indexing，与 PyTorch 一致
        y, x = jt.meshgrid(y_range, x_range)
        y = y.flatten().float32()
        x = x.flatten().float32()
        strides = jt.full((x.shape[0],), stride, dtype=jt.float32)
        priors = jt.stack([x, y, strides, strides], dim=-1).float32()
        return priors.unsqueeze(0).repeat(batch_size, 1, 1)

    def _forward_onnx(self, feats):
        outputs = []
        for feat, cls_convs, gfl_cls in zip(
            feats,
            self.cls_convs,
            self.gfl_cls,
        ):
            for conv in cls_convs:
                feat = conv(feat)
            output = gfl_cls(feat)
            cls_pred, reg_pred = output.split(
                [self.num_classes, 4 * (self.reg_max + 1)], dim=1
            )
            cls_pred = cls_pred.sigmoid()
            out = jt.cat([cls_pred, reg_pred], dim=1)
            outputs.append(out.flatten(start_dim=2))
        return jt.cat(outputs, dim=2).permute(0, 2, 1)
