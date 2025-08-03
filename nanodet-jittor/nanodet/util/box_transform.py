# JITTOR MIGRATION: 导入 jittor 库
import jittor as jt


def distance2bbox(points, distance, max_shape=None):
    """将距离预测解码为边界框。

    Args:
        points (jt.Var): 形状为 (n, 2) 的张量, [x, y]。
        distance (jt.Var): 从给定点到4个边界（左、上、右、下）的距离。
        max_shape (tuple): 图像的形状。

    Returns:
        jt.Var: 解码后的边界框。
    """
    x1 = points[..., 0] - distance[..., 0]
    y1 = points[..., 1] - distance[..., 1]
    x2 = points[..., 0] + distance[..., 2]
    y2 = points[..., 1] + distance[..., 3]
    if max_shape is not None:
        # JITTOR MIGRATION: .clamp 的参数从 (min, max) 改为 (min_v, max_v)
        x1 = x1.clamp(min_v=0, max_v=max_shape[1])
        y1 = y1.clamp(min_v=0, max_v=max_shape[0])
        x2 = x2.clamp(min_v=0, max_v=max_shape[1])
        y2 = y2.clamp(min_v=0, max_v=max_shape[0])
    # JITTOR MIGRATION: torch.stack -> jt.stack
    return jt.stack([x1, y1, x2, y2], -1)


def bbox2distance(points, bbox, max_dis=None, eps=0.1):
    """根据距离解码边界框。

    Args:
        points (jt.Var): 形状为 (n, 2) 的张量, [x, y]。
        bbox (jt.Var): 形状为 (n, 4) 的张量, "xyxy" 格式。
        max_dis (float): 距离的上限。
        eps (float): 一个小值，确保目标值 < max_dis, 而不是 <=。

    Returns:
        jt.Var: 解码后的距离。
    """
    left = points[:, 0] - bbox[:, 0]
    top = points[:, 1] - bbox[:, 1]
    right = bbox[:, 2] - points[:, 0]
    bottom = bbox[:, 3] - points[:, 1]
    if max_dis is not None:
        # JITTOR MIGRATION: .clamp 的参数从 (min, max) 改为 (min_v, max_v)
        left = left.clamp(min_v=0, max_v=max_dis - eps)
        top = top.clamp(min_v=0, max_v=max_dis - eps)
        right = right.clamp(min_v=0, max_v=max_dis - eps)
        bottom = bottom.clamp(min_v=0, max_v=max_dis - eps)
    # JITTOR MIGRATION: torch.stack -> jt.stack
    return jt.stack([left, top, right, bottom], -1)
