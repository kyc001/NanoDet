# JITTOR MIGRATION by Google LLC.
import pickle
# JITTOR MIGRATION: 导入 jittor 和 numpy
import jittor as jt
import numpy as np


def list_scatter(input_list, chunk_sizes):
    """一个辅助函数，用于按块大小切分列表。"""
    ret = []
    current_pos = 0
    for size in chunk_sizes:
        ret.append(input_list[current_pos : current_pos + size])
        current_pos += size
    return tuple(ret)


def scatter(inputs, target_gpus, dim=0, chunk_sizes=None):
    """
    JITTOR MIGRATION:
    在 Jittor 中，模型和数据的分发（scatter）由 `jt.DataParallel` 自动处理。
    此函数被简化，以处理非 Jittor 变量（如配置字典或列表）的分发，
    但不处理模型参数或张量。

    将 Python 对象复制到目标设备列表。
    """
    num_gpus = len(target_gpus)

    def scatter_map(obj):
        # JITTOR MIGRATION: 移除了对 Variable 和 Tensor 的处理
        if isinstance(obj, jt.Var):
            # 警告：不应手动分发 Jittor 变量
            print("Warning: Manually scattering a jt.Var is not recommended. Use jt.DataParallel.")
            # 简单的广播式复制
            return [obj for _ in range(num_gpus)]
        if isinstance(obj, list):
            if chunk_sizes:
                return list_scatter(obj, chunk_sizes)
            else:
                # 默认平均切分
                chunk_size = (len(obj) + num_gpus - 1) // num_gpus
                return [obj[i * chunk_size : (i + 1) * chunk_size] for i in range(num_gpus)]
        if isinstance(obj, tuple):
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, dict):
            return list(map(type(obj), zip(*map(scatter_map, obj.items()))))
        # 对于其他类型的对象，直接复制
        return [obj for _ in target_gpus]

    return scatter_map(inputs)


def scatter_kwargs(inputs, kwargs, target_gpus, dim=0, chunk_sizes=None):
    """
    JITTOR MIGRATION:
    分发位置参数和关键字参数。
    """
    inputs = scatter(inputs, target_gpus, dim, chunk_sizes) if inputs else []
    kwargs = scatter(kwargs, target_gpus, dim, chunk_sizes) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs


def gather_results_jittor(result_part):
    """
    JITTOR MIGRATION:
    使用 Jittor 的分布式通信（jt.mpi）从所有进程收集结果。
    """
    # JITTOR MIGRATION: 使用 jt.world_size 和 jt.rank
    if jt.world_size == 1:
        return result_part
        
    rank = jt.rank
    world_size = jt.world_size

    # 使用 pickle 将结果部分序列化为字节
    pickled_data = pickle.dumps(result_part)
    
    # JITTOR MIGRATION: 将字节数据转换为 uint8 的 Jittor 变量
    part_tensor = jt.array(np.frombuffer(pickled_data, dtype=np.uint8))

    # 收集所有进程的数据部分的大小
    shape_tensor = jt.array([part_tensor.shape[0]])
    
    # JITTOR MIGRATION: 使用 jt.all_gather 收集所有形状
    shape_list_vars = jt.all_gather(shape_tensor)
    shape_list = [s.item() for s in shape_list_vars]

    # 找到最大长度并进行填充
    shape_max = max(shape_list)
    part_send = jt.zeros(shape_max, dtype=jt.uint8)
    part_send[: part_tensor.shape[0]] = part_tensor

    # JITTOR MIGRATION: 使用 jt.all_gather 收集所有数据
    # all_gather 返回一个包含所有进程数据的列表
    part_recv_list = jt.all_gather(part_send)

    if rank == 0:
        all_res = {}
        for i in range(world_size):
            # JITTOR MIGRATION: 从 Jittor 变量中获取数据并反序列化
            shape = shape_list[i]
            received_bytes = part_recv_list[i][:shape].numpy().tobytes()
            all_res.update(pickle.loads(received_bytes))
        return all_res
    else:
        return None # 只有主进程返回完整结果
