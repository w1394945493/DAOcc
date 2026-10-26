from ._compat import patch_mmcv_scatter

# 在模型/数据分发前启用 MMCV 1.x 与 PyTorch 2.1 的设备参数兼容处理。
patch_mmcv_scatter()
