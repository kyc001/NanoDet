import os
import jittor as jt

def rank_filter(func):
    def func_filter(*args, **kwargs):
        # Robustly determine local rank from Jittor or environment
        try:
            local_rank = int(getattr(jt, 'rank', 0))
        except Exception:
            local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        if local_rank < 1:
            return func(*args, **kwargs)
        else:
            return None
    return func_filter
