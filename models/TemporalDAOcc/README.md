# TemporalDAOcc 第一版

在单帧 DAOcc 的 decoder.neck 与 Occupancy Head 之间增加上一帧 BEV 对齐融合。
检测头继续使用当前帧原始特征，原有多 sweep 点云不变。原模型文件没有修改。

## 文件与职责

```text
DAOcc/
├── models/TemporalDAOcc/     # 模型及特征融合
│   ├── model.py
│   └── fusion.py
├── datasets/
│   └── temporal_daocc.py    # 历史帧选择与数据加载
└── scripts/
    ├── run_temporal.py      # 训练、测试、流式推理
    └── test_temporal.py     # CPU 检查与 BEV 可视化
```

- `datasets/temporal_daocc.py`：参照 BEVDet4D，用原始 data_infos 索引减去配置偏移量，跨场景时使用当前帧占位；不依赖 batch 顺序，兼容 CBGS。
- `fusion.py`：反向坐标采样、覆盖范围掩码，以及 off/fixed/gate 三种融合模式。
- `model.py`：继承原 BEVFusion；历史分支 eval + no_grad，当前帧正常训练；提供显式流式缓存接口。
- `scripts/run_temporal.py`：注册模型/数据集，完整加载 base_config，再应用时序配置，调用原训练/测试入口。

## 运行

在 CUDA-DAOcc 根目录，使用 daocc 环境。`--daocc-root` 可指向已有编译扩展的外部 DAOcc；时序代码始终来自本目录。需要 infos 含 scene_token、timestamp、ego2global 与 lidar2ego 的旋转和平移。

```bash
# 从已训练的单帧权重初始化；门控参数为新增参数，需要进一步训练。
python DAOcc/scripts/run_temporal.py train \
  DAOcc/configs/temporal_daocc/temporal_daocc.yaml \
  --daocc-root /home/wys/wangyushen/forks/DAOcc \
  --run-dir runs/temporal_daocc \
  --load_from /实际路径/epoch_6_ema.pth

# 成对测试：每个样本独立计算当前帧和上一帧，可复用原 test.py 的参数。
python DAOcc/scripts/run_temporal.py test \
  DAOcc/configs/temporal_daocc/temporal_daocc.yaml \
  /实际路径/时序模型.pth \
  --daocc-root /home/wys/wangyushen/forks/DAOcc

# 在线缓存测试：按场景和时间顺序遍历验证集，保存原格式预测列表。
python DAOcc/scripts/run_temporal.py stream \
  DAOcc/configs/temporal_daocc/temporal_daocc.yaml \
  --daocc-root /home/wys/wangyushen/forks/DAOcc \
  --checkpoint /实际路径/时序模型.pth \
  --out results/temporal_predictions.pkl

# CPU 核心单元测试
python DAOcc/scripts/test_temporal.py
```

配置 `model.temporal.mode: off` 可关闭融合，`fixed` 使用 alpha，`gate` 使用可学习空间门控。对照实验仍应使用相同权重和输入。

## BEVDet4D 风格历史数据加载

配置顶层 `multi_adj_frame_id_cfg: [1, 2, 1]` 表示 `range(1,2,1)`，即前1帧。
`[1,9,1]` 表示前1～8帧；`[1,6,2]` 表示前1、3、5帧。这里的帧数是当前 info 列表的索引偏移，不是秒数，也不是 sweeps 数。

每个候选为 `max(index-offset, 0)`，跨场景则用当前样本占位，与 BEVDet4D 的 get_adj_info 一致。
不会在候选跨场景时继续搜索同场景更早的帧；列表前端的截断可能使多个槽位重复引用第0帧。
`data.*.max_time_gap: null` 表示不额外过滤时间间隔；可设置秒数启用额外保护。
与原 BEVDet4D 的不同在于保留 `history_valid`，让当前帧占位不会被 DAOCC 当作有效历史参与融合。

单历史输出字典，兼容现有模型；多历史输出固定长度字典列表，并附带对应位姿、时间戳、偏移与有效性。
数据仍分别经过原 DAOcc pipeline，并未照搬 BEVDet 的图像打包和共享增强实现。

**多历史数据加载已经支持，多历史融合网络尚未实现。** 默认单历史可继续训练；配置多个偏移时，当前模型会明确报错，避免只用第一帧却误以为融合了全部历史。
可视化使用 `DAOcc/scripts/test_temporal.py --visualize ...` 默认展示所有配置历史，按真实时间从早到晚组织为多行三列：历史原始 BEV、对齐后的历史 BEV、当前 BEV 参照。所有子图共用色标。无效历史放在末尾，明确标注 unavailable，不伪装成当前帧。
仍可加 `--history-slot 0` 只查看第一个偏移。JSON 的 histories 列表记录每行的偏移、token、时间差、有效性和变换矩阵。`--num-samples` 是当前帧组数，不是历史数量。
在线缓存接口仍只支持前1帧，不能用多历史配置运行 stream。

## 几何与缓存约定

原始 BEV 空间轴为 X、Y，采样时转换为 Y、X；使用体素中心和 align_corners=False。
变换为 `A_prev @ inverse(G_prev) @ G_cur @ inverse(A_cur)`，G 是 LiDAR 到世界坐标的变换，A 是数据增强变换，因此两帧独立增强也能对齐。采用 z=0 的二维 BEV 近似，不能补偿动态物体运动或精确恢复俯仰引起的高度差。

默认 EMA 模式缓存融合后的记忆 BEV；fixed/gate/off 模式仍缓存原始当前 BEV。首帧、不同场景、时间倒退/重复、超过 max_time_gap 和超出历史覆盖范围的区域回退当前帧。定位重置由调用者把 state 设为 None。流式接口 batch size=1，缓存保留 GPU Tensor；不要把随机抽样测速顺序当作在线时序顺序。

## 无可学习参数的 EMA

默认 `model.temporal.mode: ema`，`alpha: 0.5`。更新为 `M_t = (1-alpha)*F_t + alpha*warp(M_{t-1})`，alpha 是历史权重。该融合模块不创建可学习参数，首帧记忆为当前 BEV；旧记忆失效的区域重新初始化为当前 BEV。

使用 `scripts/run_temporal.py stream ...` 才是持续递归的 EMA。流式入口自动使用单记忆模式，配置中的多历史偏移不影响它；仍按场景、时间排序运行并检查时间间隔。成对输入只有当前/历史两帧，执行的是一次 EMA 更新，不会继承上个 batch 的记忆；当前仍不支持多历史列表直接送入预测模型。

原 DAOcc 权重可用于试验，因为 EMA 没有新增参数。但特征分布会变化，预测精度是否改善需评估。Occupancy Head 使用 EMA 特征，Object Head 仍使用当前原始特征。`test_temporal.py --visualize` 仍是原始历史对齐图，不展示递归 EMA 结果。

## 当前范围

### 统一评估流式 EMA

在 CUDA-DAOcc 根目录运行（使用原始 PyTorch 权重即可）：

```bash
python qat/evaluate-model.py \
  --config DAOcc/configs/temporal_daocc/temporal_daocc.yaml \
  --daocc-root /home/wys/wangyushen/forks/DAOcc \
  --backend pytorch --precision fp32 --inference-mode stream \
  --ckpt /实际路径/epoch_6_ema.pth \
  --split val --workers 2 \
  --result-dir results/evaluation_ema
```

当前配置必须为 `mode: ema`。评估入口只加载当前帧，忽略离线多历史偏移，依次调用 forward_stream；首帧、场景变化、异常时间间隔清空记忆。预测放回原数据集顺序后统一评估检测、占据 IoU/mIoU 和 Ray-IoU。

同一配置、权重和精度下，将 `--inference-mode stream` 改为 `single`，并换一个结果目录，即可得到关闭时序的单帧对照。可选 `--out-dir` 按场景保存每帧预测。JSON 记录实际推理模式、EMA 系数、时间阈值、遍历顺序与重置次数，CSV 表格标明 EMA 模式。TensorRT 暂不支持 stream。

本版提供 PyTorch 成对训练、成对测试和流式推理。未改造现有 TensorRT 导出/C++ 推理程序；原来的 .plan 仍是单帧模型，不能直接产生本变体结果。后续部署应显式输入历史 BEV、采样网格与掩码，另外导出融合模块并管理 GPU 缓存，再测量新增延迟。

CPU 单元测试不等于真实数据训练验证；完整训练、精度提升及 TensorRT 导出兼容性需要后续实验验证。
