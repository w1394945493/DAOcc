"""DAOcc 的 MMCV 1.x / PyTorch 2.1+ 并行接口兼容层。

MMCV 的 Scatter 用整数 GPU 编号获取后台拷贝流，而新版 PyTorch 的
私有函数 _get_stream 接受 torch.device。两者直接组合会在首次 CPU 到
GPU 的数据分发时抛出 ``'int' object has no attribute 'type'``。
此模块由 mmdet3d/__init__.py 在导入时启用，无需修改 site-packages。
由于依赖私有接口，未来升级 PyTorch/MMCV 后仍需重新验证。
"""

from functools import wraps

import torch
from packaging.version import Version


def patch_mmcv_scatter():
    """为 MMCV 的流查询转换设备参数；重复调用不会叠加包装。"""
    import mmcv
    from mmcv.parallel import _functions

    # 仅处理旧 MMCV 与新 PyTorch 的组合，保留旧环境的原始行为。
    if Version(mmcv.__version__).major != 1:
        return
    if Version(torch.__version__.split('+')[0]) < Version('2.1'):
        return

    # 保存 MMCV 当前引用的函数；标记用于避免重复导入时多次包装。
    original = _functions._get_stream
    if getattr(original, '_daocc_device_compat', False):
        return

    @wraps(original)
    def get_stream(device):
        # 非负整数是 GPU 编号；-1 是 MMCV 的 CPU 哨兵。
        # 已经是 torch.device 的参数原样传递，流缓存仍由 PyTorch 管理。
        if isinstance(device, int):
            device = torch.device('cpu' if device == -1 else 'cuda',
                                  None if device == -1 else device)
        return original(device)

    get_stream._daocc_device_compat = True
    # 只替换 MMCV 模块内的引用，不修改 PyTorch 自身的全局函数。
    _functions._get_stream = get_stream
