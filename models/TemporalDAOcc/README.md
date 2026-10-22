# TemporalDAOcc 第一版

在单帧 DAOcc 的 decoder.neck 与 Occupancy Head 之间增加上一帧 BEV 对齐融合。
检测头继续使用当前帧原始特征，原有多 sweep 点云不变。原模型文件没有修改。

## 文件与职责

- `dataset.py`：按 scene_token、时间戳查找上一帧，经过原 pipeline 独立处理两帧；不依赖 batch 顺序，兼容 CBGS。
- `fusion.py`：反向坐标采样、覆盖范围掩码，以及 off/fixed/gate 三种融合模式。
- `model.py`：继承原 BEVFusion；历史分支 eval + no_grad，当前帧正常训练；提供显式流式缓存接口。
- `run.py`：注册模型/数据集，完整加载 base_config，再应用时序配置，调用原训练/测试入口。

## 运行

在 CUDA-DAOcc 根目录，使用 daocc 环境。`--daocc-root` 可指向已有编译扩展的外部 DAOcc；时序代码始终来自本目录。需要 infos 含 scene_token、timestamp、ego2global 与 lidar2ego 的旋转和平移。

```bash
# 从已训练的单帧权重初始化；门控参数为新增参数，需要进一步训练。
python DAOcc/models/TemporalDAOcc/run.py train \
  DAOcc/configs/temporal_daocc/temporal_daocc.yaml \
  --daocc-root /home/wys/wangyushen/forks/DAOcc \
  --run-dir runs/temporal_daocc \
  --load_from /实际路径/epoch_6_ema.pth

# 成对测试：每个样本独立计算当前帧和上一帧，可复用原 test.py 的参数。
python DAOcc/models/TemporalDAOcc/run.py test \
  DAOcc/configs/temporal_daocc/temporal_daocc.yaml \
  /实际路径/时序模型.pth \
  --daocc-root /home/wys/wangyushen/forks/DAOcc

# 在线缓存测试：按场景和时间顺序遍历验证集，保存原格式预测列表。
python DAOcc/models/TemporalDAOcc/run.py stream \
  DAOcc/configs/temporal_daocc/temporal_daocc.yaml \
  --daocc-root /home/wys/wangyushen/forks/DAOcc \
  --checkpoint /实际路径/时序模型.pth \
  --out results/temporal_predictions.pkl

# CPU 核心单元测试
python DAOcc/models/TemporalDAOcc/test_temporal.py
```

配置 `model.temporal.mode: off` 可关闭融合，`fixed` 使用 alpha，`gate` 使用可学习空间门控。对照实验仍应使用相同权重和输入。

## 几何与缓存约定

原始 BEV 空间轴为 X、Y，采样时转换为 Y、X；使用体素中心和 align_corners=False。
变换为 `A_prev @ inverse(G_prev) @ G_cur @ inverse(A_cur)`，G 是 LiDAR 到世界坐标的变换，A 是数据增强变换，因此两帧独立增强也能对齐。采用 z=0 的二维 BEV 近似，不能补偿动态物体运动或精确恢复俯仰引起的高度差。

缓存原始当前 BEV，而不是融合结果。首帧、不同场景、时间倒退/重复、超过 max_time_gap 和超出历史覆盖范围的区域回退当前帧。定位重置由调用者把 state 设为 None。流式接口 batch size=1，缓存保留 GPU Tensor；不要把随机抽样测速顺序当作在线时序顺序。

## 当前范围

本版提供 PyTorch 成对训练、成对测试和流式推理。未改造现有 TensorRT 导出/C++ 推理程序；原来的 .plan 仍是单帧模型，不能直接产生本变体结果。后续部署应显式输入历史 BEV、采样网格与掩码，另外导出融合模块并管理 GPU 缓存，再测量新增延迟。

CPU 单元测试不等于真实数据训练验证；完整训练、精度提升及 TensorRT 导出兼容性需要后续实验验证。
