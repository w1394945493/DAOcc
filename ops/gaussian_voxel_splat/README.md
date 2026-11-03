# Gaussian voxel splat CUDA 算子

该算子把稀疏 3D Gaussian 聚合到规则三维体素，直接输出
`[B=1,C=128,Z,Y,X]`。它本身尚未生成最终 BEV；调用方随后把高度 `Z`
折叠到通道维，才得到 DAOcc 使用的 `[B,C*Z,X,Y]` BEV 特征。

实现沿用 GaussianFormer `model/head/localagg` 的工程框架：先计算每个
Gaussian 的轴对齐体素包围盒，再用 CUB 生成、排序“体素-Gaussian”关系，
最后每个体素只遍历覆盖它的候选 Gaussian。这里没有原样复制 localagg 的
计算语义，而是适配了 DAOcc 当前实现：

- 特征通道为 128，而不是固定的 18 类语义；
- 支持 XYZ 三轴不同的尺度和旋转椭球；
- 权重为 `opacity * exp(-0.5 * MahalanobisDistance)`；
- 支持 `normalized` 加权平均及 `sum` 加权和；
- 返回完整高度体素，BEV 高度折叠仍由 Python 调用层完成；
- 提供中心、尺度/旋转所构成的精度矩阵、不透明度及特征的反向传播。

编译示例：

```bash
cd /home/wangyushen/forks/CUDA-DAOcc/DAOcc/ops/gaussian_voxel_splat
CC=/home/wangyushen/miniforge3/envs/daocc-cu121/bin/x86_64-conda-linux-gnu-gcc \
CXX=/home/wangyushen/miniforge3/envs/daocc-cu121/bin/x86_64-conda-linux-gnu-g++ \
CONDA_PREFIX=/home/wangyushen/miniforge3/envs/daocc-cu121 \
CUDA_HOME=/usr/local/cuda-12.2 MAX_JOBS=4 \
/home/wangyushen/miniforge3/envs/daocc-cu121/bin/python \
setup.py build_ext --inplace
```

数值测试：

```bash
cd /home/wangyushen/forks/CUDA-DAOcc/DAOcc
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD \
/home/wangyushen/miniforge3/envs/daocc-cu121/bin/python \
ops/gaussian_voxel_splat/test_equivalence.py
```

`tools/build_initial_gaussian_anchors.py` 默认使用 CUDA 实现；传入
`--splat-implementation pytorch` 可切回保留的原 PyTorch 实现进行对照。
