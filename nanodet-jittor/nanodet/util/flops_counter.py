import sys
from functools import partial

import numpy as np
# JITTOR MIGRATION: 导入 jittor 库
import jittor as jt
import jittor.nn as nn


def get_model_complexity_info(
    model,
    input_shape,
    print_per_layer_stat=True,
    as_strings=True,
    input_constructor=None,
    flush=False,
    ost=sys.stdout,
):
    """获取模型的复杂度信息。
    此方法可以计算具有相应输入形状的模型的 FLOPs 和参数计数。
    它还可以打印模型中每个层的复杂度信息。

    Args:
        model (nn.Module): 用于复杂度计算的模型。
        input_shape (tuple): 用于计算的输入形状。
        print_per_layer_stat (bool): 是否打印模型中每个层的复杂度信息。默认为 True。
        as_strings (bool): 以字符串形式输出 FLOPs 和参数计数。默认为 True。
        input_constructor (None | callable): 如果指定，它将采用一个生成输入的可调用方法。
            否则，它将生成一个具有输入形状的随机张量来计算 FLOPs。默认为 None。
        flush (bool): 与 :func:`print` 中的相同。默认为 False。
        ost (stream): 与 :func:`print` 中的 `file` 参数相同。默认为 sys.stdout。

    Returns:
        tuple[float | str]: 如果 `as_strings` 设置为 True，它将以字符串格式返回 FLOPs 和参数计数。
            否则，它将以浮点数格式返回。
    """
    assert type(input_shape) is tuple
    assert len(input_shape) >= 1
    assert isinstance(model, nn.Module)
    flops_model = add_flops_counting_methods(model)
    flops_model.eval()
    flops_model.start_flops_count()
    if input_constructor:
        input = input_constructor(input_shape)
        _ = flops_model(**input)
    else:
        # JITTOR MIGRATION: 调整创建输入张量的逻辑
        try:
            params = list(flops_model.parameters())
            if params:
                dtype = params[0].dtype
            else:
                # 对于没有参数的模型，使用默认的 float32
                dtype = jt.float32
            batch = jt.empty((1, *input_shape), dtype=dtype)
        except Exception:
            # 捕获其他可能的错误
            batch = jt.empty((1, *input_shape))

        _ = flops_model(batch)

    flops_count, params_count = flops_model.compute_average_flops_cost()
    if print_per_layer_stat:
        print_model_with_flops(
            flops_model, flops_count, params_count, ost=ost, flush=flush
        )
    flops_model.stop_flops_count()

    if as_strings:
        return flops_to_string(flops_count), params_to_string(params_count)

    return flops_count, params_count


def flops_to_string(flops, units="GFLOPs", precision=2):
    """将 FLOPs 数转换为字符串。"""
    if units is None:
        if flops // 10**9 > 0:
            return str(round(flops / 10.0**9, precision)) + " GFLOPs"
        elif flops // 10**6 > 0:
            return str(round(flops / 10.0**6, precision)) + " MFLOPs"
        elif flops // 10**3 > 0:
            return str(round(flops / 10.0**3, precision)) + " KFLOPs"
        else:
            return str(flops) + " FLOPs"
    else:
        if units == "GFLOPs":
            return str(round(flops / 10.0**9, precision)) + " " + units
        elif units == "MFLOPs":
            return str(round(flops / 10.0**6, precision)) + " " + units
        elif units == "KFLOPs":
            return str(round(flops / 10.0**3, precision)) + " " + units
        else:
            return str(flops) + " FLOPs"


def params_to_string(num_params, units=None, precision=2):
    """将参数数量转换为字符串。"""
    if units is None:
        if num_params // 10**6 > 0:
            return str(round(num_params / 10**6, precision)) + " M"
        elif num_params // 10**3:
            return str(round(num_params / 10**3, precision)) + " k"
        else:
            return str(num_params)
    else:
        if units == "M":
            return str(round(num_params / 10.0**6, precision)) + " " + units
        elif units == "K":
            return str(round(num_params / 10.0**3, precision)) + " " + units
        else:
            return str(num_params)


def print_model_with_flops(
    model,
    total_flops,
    total_params,
    units="GFLOPs",
    precision=3,
    ost=sys.stdout,
    flush=False,
):
    """打印一个带有每层 FLOPs 的模型。"""

    def accumulate_params(self):
        if is_supported_instance(self):
            return self.__params__
        else:
            sum = 0
            for m in self.children():
                sum += m.accumulate_params()
            return sum

    def accumulate_flops(self):
        if is_supported_instance(self):
            return self.__flops__ / model.__batch_counter__
        else:
            sum = 0
            for m in self.children():
                sum += m.accumulate_flops()
            return sum

    def flops_repr(self):
        accumulated_num_params = self.accumulate_params()
        accumulated_flops_cost = self.accumulate_flops()
        return ", ".join(
            [
                params_to_string(
                    accumulated_num_params, units="M", precision=precision
                ),
                "{:.3%} Params".format(accumulated_num_params / total_params),
                flops_to_string(
                    accumulated_flops_cost, units=units, precision=precision
                ),
                "{:.3%} FLOPs".format(accumulated_flops_cost / total_flops),
                self.original_extra_repr(),
            ]
        )

    def add_extra_repr(m):
        m.accumulate_flops = accumulate_flops.__get__(m)
        m.accumulate_params = accumulate_params.__get__(m)
        flops_extra_repr = flops_repr.__get__(m)
        if m.extra_repr != flops_extra_repr:
            m.original_extra_repr = m.extra_repr
            m.extra_repr = flops_extra_repr
            assert m.extra_repr != m.original_extra_repr

    def del_extra_repr(m):
        if hasattr(m, "original_extra_repr"):
            m.extra_repr = m.original_extra_repr
            del m.original_extra_repr
        if hasattr(m, "accumulate_flops"):
            del m.accumulate_flops

    model.apply(add_extra_repr)
    print(model, file=ost, flush=flush)
    model.apply(del_extra_repr)


def get_model_parameters_number(model):
    """计算模型的参数数量。"""
    # JITTOR MIGRATION: p.requires_grad 在 Jittor 中是 .requires_grad
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params


def add_flops_counting_methods(net_main_module):
    net_main_module.start_flops_count = start_flops_count.__get__(net_main_module)
    net_main_module.stop_flops_count = stop_flops_count.__get__(net_main_module)
    net_main_module.reset_flops_count = reset_flops_count.__get__(net_main_module)
    net_main_module.compute_average_flops_cost = compute_average_flops_cost.__get__(
        net_main_module
    )
    net_main_module.reset_flops_count()
    return net_main_module


def compute_average_flops_cost(self):
    """计算平均 FLOPs 成本。"""
    batches_count = self.__batch_counter__
    flops_sum = 0
    for module in self.modules():
        if is_supported_instance(module):
            flops_sum += module.__flops__
    params_sum = get_model_parameters_number(self)
    return flops_sum / batches_count, params_sum


def start_flops_count(self):
    """激活平均 FLOPs 消耗的计算。"""
    add_batch_counter_hook_function(self)

    def add_flops_counter_hook_function(module):
        if is_supported_instance(module):
            if hasattr(module, "__flops_handle__"):
                return
            else:
                # JITTOR MIGRATION: register_forward_hook 在 Jittor 中用法相同
                handle = module.register_forward_hook(MODULES_MAPPING[type(module)])
            module.__flops_handle__ = handle

    self.apply(partial(add_flops_counter_hook_function))


def stop_flops_count(self):
    """停止计算平均 FLOPs 消耗。"""
    remove_batch_counter_hook_function(self)
    self.apply(remove_flops_counter_hook_function)


def reset_flops_count(self):
    """重置已计算的统计数据。"""
    add_batch_counter_variables_or_reset(self)
    self.apply(add_flops_counter_variable_or_reset)


# ---- 内部函数 ----
def empty_flops_counter_hook(module, input, output):
    module.__flops__ += 0


def upsample_flops_counter_hook(module, input, output):
    output_size = output
    batch_size = output_size.shape[0]
    output_elements_count = batch_size
    for val in output_size.shape[1:]:
        output_elements_count *= val
    module.__flops__ += int(output_elements_count)


def relu_flops_counter_hook(module, input, output):
    active_elements_count = output.numel()
    module.__flops__ += int(active_elements_count)


def linear_flops_counter_hook(module, input, output):
    input = input[0]
    output_last_dim = output.shape[-1]
    module.__flops__ += int(np.prod(input.shape) * output_last_dim)


def pool_flops_counter_hook(module, input, output):
    input = input[0]
    module.__flops__ += int(np.prod(input.shape))


def bn_flops_counter_hook(module, input, output):
    input = input[0]
    batch_flops = np.prod(input.shape)
    if module.affine:
        batch_flops *= 2
    module.__flops__ += int(batch_flops)


def deconv_flops_counter_hook(conv_module, input, output):
    input = input[0]
    batch_size = input.shape[0]
    input_height, input_width = input.shape[2:]
    kernel_height, kernel_width = conv_module.kernel_size
    in_channels = conv_module.in_channels
    out_channels = conv_module.out_channels
    groups = conv_module.groups
    filters_per_channel = out_channels // groups
    conv_per_position_flops = (
        kernel_height * kernel_width * in_channels * filters_per_channel
    )
    active_elements_count = batch_size * input_height * input_width
    overall_conv_flops = conv_per_position_flops * active_elements_count
    bias_flops = 0
    if conv_module.bias is not None:
        output_height, output_width = output.shape[2:]
        bias_flops = out_channels * batch_size * output_height * output_height
    overall_flops = overall_conv_flops + bias_flops
    conv_module.__flops__ += int(overall_flops)


def conv_flops_counter_hook(conv_module, input, output):
    input = input[0]
    batch_size = input.shape[0]
    output_dims = list(output.shape[2:])
    kernel_dims = list(conv_module.kernel_size)
    in_channels = conv_module.in_channels
    out_channels = conv_module.out_channels
    groups = conv_module.groups
    filters_per_channel = out_channels // groups
    conv_per_position_flops = (
        int(np.prod(kernel_dims)) * in_channels * filters_per_channel
    )
    active_elements_count = batch_size * int(np.prod(output_dims))
    overall_conv_flops = conv_per_position_flops * active_elements_count
    bias_flops = 0
    if conv_module.bias is not None:
        bias_flops = out_channels * active_elements_count
    overall_flops = overall_conv_flops + bias_flops
    conv_module.__flops__ += int(overall_flops)


def batch_counter_hook(module, input, output):
    batch_size = 1
    if len(input) > 0:
        input = input[0]
        batch_size = len(input)
    else:
        print(
            "Warning! No positional inputs found for a module, "
            "assuming batch size is 1."
        )
    module.__batch_counter__ += batch_size


def add_batch_counter_variables_or_reset(module):
    module.__batch_counter__ = 0


def add_batch_counter_hook_function(module):
    if hasattr(module, "__batch_counter_handle__"):
        return
    handle = module.register_forward_hook(batch_counter_hook)
    module.__batch_counter_handle__ = handle


def remove_batch_counter_hook_function(module):
    if hasattr(module, "__batch_counter_handle__"):
        module.__batch_counter_handle__.remove()
        del module.__batch_counter_handle__


def add_flops_counter_variable_or_reset(module):
    if is_supported_instance(module):
        if hasattr(module, "__flops__") or hasattr(module, "__params__"):
            print(
                "Warning: variables __flops__ or __params__ are already "
                "defined for the module"
                + type(module).__name__
                + " ptflops can affect your code!"
            )
        module.__flops__ = 0
        module.__params__ = get_model_parameters_number(module)


def is_supported_instance(module):
    if type(module) in MODULES_MAPPING:
        return True
    return False


def remove_flops_counter_hook_function(module):
    if is_supported_instance(module):
        if hasattr(module, "__flops_handle__"):
            module.__flops_handle__.remove()
            del module.__flops_handle__


# JITTOR MIGRATION: 将 nn.Module 替换为 jt.nn.Module
MODULES_MAPPING = {
    # convolutions
    nn.Conv1d: conv_flops_counter_hook,
    nn.Conv2d: conv_flops_counter_hook,
    nn.Conv3d: conv_flops_counter_hook,
    # activations
    nn.ReLU: relu_flops_counter_hook,
    nn.PReLU: relu_flops_counter_hook,
    nn.ELU: relu_flops_counter_hook,
    nn.LeakyReLU: relu_flops_counter_hook,
    nn.ReLU6: relu_flops_counter_hook,
    # poolings
    nn.Pool: pool_flops_counter_hook, # Jittor 的池化层基类
    # BNs
    nn.BatchNorm: bn_flops_counter_hook, # Jittor 的BN层基类
    # FC
    nn.Linear: linear_flops_counter_hook,
    # Upscale
    nn.Upsample: upsample_flops_counter_hook,
    # Deconvolution
    nn.ConvTranspose2d: deconv_flops_counter_hook,
}
