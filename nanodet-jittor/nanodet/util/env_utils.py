# JITTOR MIGRATION: 导入所需的标准库
import os
import platform
import warnings

# JITTOR MIGRATION: 将 torch.multiprocessing 替换为 Python 内置的 multiprocessing
import multiprocessing as mp


def set_multi_processing(
    mp_start_method: str = "fork", opencv_num_threads: int = 0, distributed: bool = True
) -> None:
    """设置多进程相关的环境。
    
    此函数从 PyTorch 环境迁移而来，适用于 Jittor，因为它依赖于
    Python 标准的多进程和环境变量设置。

    Args:
        mp_start_method (str): 设置启动子进程的方法。默认为 'fork'。
        opencv_num_threads (int): OpenCV 的线程数。默认为 0。
        distributed (bool): 是否为分布式环境。默认为 True。
    """
    # 设置多进程启动方法为 `fork` 以加速训练
    if platform.system() != "Windows":
        current_method = mp.get_start_method(allow_none=True)
        if current_method is not None and current_method != mp_start_method:
            warnings.warn(
                f"多进程启动方法 `{mp_start_method}` 与之前的设置 "
                f"`{current_method}` 不同。将强制设置为 `{mp_start_method}`。 "
                "您可以通过修改配置中的 `mp_start_method` 来更改此行为。"
            )
        mp.set_start_method(mp_start_method, force=True)

    try:
        import cv2

        # 禁用 OpenCV 多线程以避免系统过载
        cv2.setNumThreads(opencv_num_threads)
    except ImportError:
        pass

    # 设置 OMP 线程数
    if "OMP_NUM_THREADS" not in os.environ and distributed:
        omp_num_threads = 1
        warnings.warn(
            "默认将每个进程的 OMP_NUM_THREADS 环境变量设置为 "
            f"{omp_num_threads}，以避免系统过载。请根据需要进一步调整该变量 "
            "以获得最佳性能。"
        )
        os.environ["OMP_NUM_THREADS"] = str(omp_num_threads)

    # 设置 MKL 线程数
    if "MKL_NUM_THREADS" not in os.environ and distributed:
        mkl_num_threads = 1
        warnings.warn(
            "默认将每个进程的 MKL_NUM_THREADS 环境变量设置为 "
            f"{mkl_num_threads}，以避免系统过载。请根据需要进一步调整该变量 "
            "以获得最佳性能。"
        )
        os.environ["MKL_NUM_THREADS"] = str(mkl_num_threads)
