import math

import cv2
import numpy as np
import jittor as jt
import jittor.nn as nn

# 🔧 直接使用 JittorDet 的成熟实现，不再使用本地复制
from nanodet.util import  multi_apply, overlay_bbox_cv
# 使用本地兼容实现，避免修改标准库 jittordet 并修复 clamp_ 参数不兼容
from nanodet.util.box_transform import distance2bbox, bbox2distance
from jittordet.models.losses import DistributionFocalLoss, QualityFocalLoss  # 🔧 直接使用 JittorDet
from jittordet.models.losses.iou_loss import GIoULoss
from jittordet.models.utils.initialize import normal_init
# Use project-local NMS wrapper to avoid API mismatch
from ..module.nms import multiclass_nms
# 替换外部 Integral，使用本地 DistributionProject 实现以严格对齐 PyTorch 逻辑
# from jittordet.models.dense_heads.gfl_head import Integral
class DistributionProject(nn.Module):
    def __init__(self, reg_max: int):
        super().__init__()
        self.reg_max = reg_max
        # [m+1] 投影向量 0..m
        self.register_buffer("project", jt.arange(0, reg_max + 1, dtype=jt.float32))

    def execute(self, reg_logits: jt.Var) -> jt.Var:
        # 完全模拟 PyTorch Integral 的实现：F.softmax + F.linear
        shape = reg_logits.shape
        # 强制 float32 精度并添加数值稳定性
        reg_logits = reg_logits.float32()
        x = reg_logits.reshape(*shape[:-1], 4, self.reg_max + 1)
        # 数值稳定的 softmax：减去最大值避免溢出
        x_max = x.max(dim=-1, keepdims=True)[0]
        x_stable = x - x_max
        x = nn.softmax(x_stable, dim=-1)
        # 使用 matmul 模拟 F.linear，与 PT 的 F.linear(x, self.project) 完全一致
        proj = self.project.float32()
        # [..., 4, m+1] @ [m+1] -> [..., 4]
        x = jt.matmul(x, proj)
        return x.reshape(*shape[:-1], 4).float32()
from jittordet.utils import reduce_mean  # 🔧 使用 JittorDet 的 reduce_mean
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
        **kwargs
    ):
        super(NanoDetPlusHead, self).__init__()
        self.num_classes = num_classes
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
        self.cls_convs = nn.ModuleList()
        for _ in self.strides:
            cls_convs = self._buid_not_shared_head()
            self.cls_convs.append(cls_convs)

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
        for m in self.cls_convs.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        # init cls head with confidence = 0.01
        bias_cls = -4.595
        for i in range(len(self.strides)):
            normal_init(self.gfl_cls[i], std=0.01, bias=bias_cls)
        print("Finish initialize NanoDet-Plus Head.")

    def execute(self, feats):
        # 只消费与 strides 对应的前 N 层
        if isinstance(feats, (list, tuple)) and len(feats) > len(self.strides):
            feats = feats[:len(self.strides)]
        outputs = []
        for feat, cls_convs, gfl_cls in zip(
            feats,
            self.cls_convs,
            self.gfl_cls,
        ):
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
        


        # nanodet/model/head/nanodet_plus_head.py (修改后的版本)

        # 1. 先从 center_priors 获取 batch_size 和 num_priors，这样代码更具通用性
        batch_size = center_priors.shape[0]
        num_priors = center_priors.shape[1]

        # 2. 正常计算 dis_preds，其形状为 [136000, 4]
        dis_preds = self.distribution_project(reg_preds)

        # 3. (关键步骤) 将 dis_preds 的形状从 [136000, 4] 调整为 [64, 2125, 4]
        dis_preds = dis_preds.reshape(batch_size, num_priors, 4)

        # 4. 现在可以安全地执行乘法了，强制 float32 精度
        # [64, 2125, 4] * [64, 2125, 1] -> 广播后 -> [64, 2125, 4] * [64, 2125, 4]
        dis_preds = dis_preds.float32() * center_priors[..., 2, None].float32()
        decoded_bboxes = distance2bbox(center_priors[..., :2], dis_preds)

        if aux_preds is not None:
            # use auxiliary head to assign
            aux_cls_preds, aux_reg_preds = aux_preds.split(
                [self.num_classes, 4 * (self.reg_max + 1)], dim=-1
            )
            # 1. 先计算分布预测，得到一个被拍平的 [total_priors, 4] 形状的张量
            _aux_dis_preds = self.distribution_project(aux_reg_preds)
            
            # 2. 从 center_priors 获取 batch_size 和 num_priors
            batch_size = center_priors.shape[0]
            num_priors = center_priors.shape[1]

            # 3. (关键!) 将 _aux_dis_preds 的形状恢复成 [batch_size, num_priors, 4]
            _aux_dis_preds = _aux_dis_preds.reshape(batch_size, num_priors, 4)

            # 4. 现在可以安全地进行广播乘法了
            aux_dis_preds = _aux_dis_preds * center_priors[..., 2, None]
            aux_decoded_bboxes = distance2bbox(center_priors[..., :2], aux_dis_preds)
            # 🔧 修复：避免 detach() 断开计算图，保持梯度连接
            batch_assign_res = multi_apply(
                self.target_assign_single_img,
                aux_cls_preds,  # 移除 .detach()
                center_priors,
                aux_decoded_bboxes,  # 移除 .detach()
                gt_bboxes,
                gt_labels,
                gt_bboxes_ignore,
            )
        else:
            # use self prediction to assign
            # 🔧 修复：避免 detach() 断开计算图，保持梯度连接
            batch_assign_res = multi_apply(
                self.target_assign_single_img,
                cls_preds,  # 移除 .detach()
                center_priors,
                decoded_bboxes,  # 移除 .detach()
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
        # 🔧 修复计算图断裂：正确计算平均因子
        try:
            # 计算所有图像的正样本总数
            total_pos = sum(num_pos)  # 这是一个 Python int
            num_total_samples = jt.clamp(jt.array(float(max(1, total_pos))), min_v=1.0)
            # 正样本总数计算完成
        except Exception as e:
            # avg_factor 计算失败，使用默认值
            num_total_samples = jt.array(1.0)

        # 🔧 修复关键 bug：使用 jt.cat 拼接所有目标
        labels = jt.cat(labels, dim=0)
        label_scores = jt.cat(label_scores, dim=0)
        label_weights = jt.cat(label_weights, dim=0)
        bbox_targets = jt.cat(bbox_targets, dim=0)
        dist_targets = jt.cat(dist_targets, dim=0)  # 🔧 添加 dist_targets 拼接
        cls_preds = cls_preds.reshape(-1, self.num_classes)
        reg_preds = reg_preds.reshape(-1, 4 * (self.reg_max + 1))
        decoded_bboxes = decoded_bboxes.reshape(-1, 4)
        # 🔧 使用 JittorDet 标准方法处理正样本索引
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0) & (labels < bg_class_ind)).nonzero().squeeze(1)

        # 🔧 QFL 损失调试信息 (已清理)

        # 🔧 使用 JittorDet 的标准损失计算方法
        loss_qfl = self.loss_qfl(
            cls_preds,
            (labels, label_scores),
            weight=label_weights,
            avg_factor=num_total_samples,
        )

        # 超简化损失计算：直接用正样本数作为平均因子
        if len(pos_inds) > 0:
            num_pos = len(pos_inds)

            # 简化bbox损失：直接用正样本数平均
            loss_bbox = self.loss_bbox(
                decoded_bboxes[pos_inds],
                bbox_targets[pos_inds]
            ) / max(num_pos, 1)

            # 简化DFL损失：直接用正样本数平均
            loss_dfl = self.loss_dfl(
                reg_preds[pos_inds].reshape(-1, self.reg_max + 1),
                dist_targets[pos_inds].reshape(-1)
            ) / max(num_pos * 4, 1)
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

        # 🔧 修复：确保返回 Python int，避免 .item() 调用
        num_pos_per_img = int(pos_inds.size(0))
        pos_ious = assign_result.max_overlaps[pos_inds].clamp(min_v=0.0, max_v=1.0)

        if len(pos_inds) > 0:
            bbox_targets[pos_inds, :] = pos_gt_bboxes
            # 🔧 修复：添加 max_dis 参数
            dist_targets[pos_inds, :] = (
                bbox2distance(center_priors[pos_inds, :2], pos_gt_bboxes, max_dis=self.reg_max)
                / center_priors[pos_inds, None, 2]
            )
            # 🔧 修复：Jittor clamp 参数名不同
            dist_targets = dist_targets.clamp(min_v=0, max_v=self.reg_max - 0.1)
            labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
            label_scores[pos_inds] = pos_ious
            label_weights[pos_inds] = 1.0
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # 仅前两次调用输出轻量调试，便于确认正样本数量
        if not hasattr(self, "_dbg_seen"):
            self._dbg_seen = 0
        if self._dbg_seen < 2:
            try:
                mean_iou = float(pos_ious.mean()) if len(pos_inds) > 0 else -1.0
            except Exception:
                mean_iou = -1.0
            print(f"[AssignDebug] num_pos={num_pos_per_img}, pos_inds_shape={tuple(pos_inds.shape)}, pos_iou_mean={mean_iou:.4f}, gt_num={int(gt_bboxes.shape[0])}, priors={int(num_priors)}")
            self._dbg_seen += 1

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
        warp_matrixes = (
            meta["warp_matrix"]
            if isinstance(meta["warp_matrix"], list)
            else meta["warp_matrix"]
        )
        img_heights = (
            meta["img_info"]["height"]
        )
        img_widths = (
            meta["img_info"]["width"]
        )
        img_ids = (
            meta["img_info"].get("id", 0)
        )

        for result, img_width, img_height, img_id, warp_matrix in zip(
            result_list, img_widths, img_heights, img_ids, warp_matrixes
        ):
            det_result = {}
            det_bboxes, det_labels = result
            det_bboxes = det_bboxes.detach().cpu().numpy()
            det_bboxes[:, :4] = warp_boxes(
                det_bboxes[:, :4], np.linalg.inv(warp_matrix), img_width, img_height
            )
            classes = det_labels.detach().cpu().numpy()
            for i in range(self.num_classes):
                inds = classes == i
                det_result[i] = np.concatenate(
                    [
                        det_bboxes[inds, :4].astype(np.float32),
                        det_bboxes[inds, 4:5].astype(np.float32),
                    ],
                    axis=1,
                ).tolist()
            det_results[img_id] = det_result
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

        # 与 PyTorch 保持一致（注意其实现中对 w 的写法为 ceil(w)/stride）：
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
        scores = cls_preds.sigmoid()
        result_list = []
        for i in range(b):
            # add a dummy background class at the end of all labels
            # same with mmdetection2.0
            score, bbox = scores[i], bboxes[i]
            padding = jt.zeros((score.shape[0], 1), dtype=score.dtype)
            score = jt.concat([score, padding], dim=1)
            # 使用项目内封装的 NMS（需要 nms_cfg 参数）
            results = multiclass_nms(
                bbox,
                score,
                0.05,
                dict(type="nms", iou_threshold=0.6),
                100,
            )
            result_list.append(results)
        return result_list

    def get_single_level_center_priors(
        self, batch_size, featmap_size, stride, dtype, device
    ):

        h, w = featmap_size
        # 🔧 强制 float32 精度，确保与 PyTorch 完全一致的坐标生成
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
