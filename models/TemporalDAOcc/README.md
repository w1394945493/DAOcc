# TemporalDAOcc：可学习多帧时序融合

当前帧和历史关键帧使用相同的单帧数据 pipeline（包括各自的 LiDAR sweeps）。历史分支使用共享权重，在 eval/no_grad 下计算到 Decoder，当前分支正常训练。历史 BEV 根据位姿及增强矩阵对齐到当前帧；无效历史位置置零。

固定 N 个历史槽位与当前 BEV 拼接为 `[B,(N+1)C,X,Y]`，通过 `1×1 Conv + BN + ReLU + 3×3 Conv + BN + ReLU` 得到 `[B,C,X,Y]`。OCC 和 3D 检测头共享此结果。已移除 fixed/ema/gate 加权方式；首帧也经过融合卷积。

## 配置

使用 `configs/temporal_daocc/temporal_daocc.yaml`。`base_config` 指向完整单帧配置，当前保留已有 custom 基础配置及其数据路径；运行前按实际环境调整该字段或基础数据路径。原始 DAOcc 文件没有修改。

`multi_adj_frame_id_cfg: [1,2,1]` 选择前1个关键帧；`[1,4,1]` 选择前1、2、3帧。`num_history` 自动计算。数据集始终返回历史列表。改历史数量会改变卷积输入通道，不能直接恢复不同历史数量的时序 checkpoint。`max_time_gap: null` 不额外裁剪历史时间窗口；可在模型和数据配置中设相同秒数上限。

历史按底层 data_infos 索引选择，再验证场景和时间，不依赖训练 batch 顺序或 CBGS 重复索引。无效历史使用占位输入，融合前置零。首次训练可加载单帧权重初始化主干，新增融合卷积需训练；不要使用旧加权时序模型直接推理期待有效结果。

## 独立入口

在 DAOcc 根目录执行，配置覆盖参数与原入口一致：

```bash
# 单卡训练；首次建议 FP32 验证。
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD python tools/temporal_daocc.py train \
  configs/temporal_daocc/temporal_daocc.yaml \
  --run-dir /tmp/temporal_daocc_train --fp16 None

# 多卡训练：自动复用 dist_train.py，不需要 mpi4py。
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=$PWD python -m torch.distributed.run \
  --standalone --nproc_per_node=2 tools/temporal_daocc.py train \
  configs/temporal_daocc/temporal_daocc.yaml \
  --run-dir /tmp/temporal_daocc_train --fp16 None

# 普通评估：重新计算各历史帧；可用 torchrun 多卡。
PYTHONPATH=$PWD python tools/temporal_daocc.py test \
  configs/temporal_daocc/temporal_daocc.yaml /path/to/temporal_checkpoint.pth \
  --eval bbox

# 缓存推理：单进程单卡，按完整场景连续处理；输出放回数据集原索引。
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD python tools/temporal_daocc.py stream \
  configs/temporal_daocc/temporal_daocc.yaml \
  --checkpoint /path/to/temporal_checkpoint.pth \
  --out /tmp/temporal_predictions.pkl --eval bbox
```

train 的 `--load_from` 用于初始化，`--resume_from` 用于相同结构恢复。stream 的 `--eval` 可省略，仅保存预测。

## 缓存语义

训练、普通评估和缓存推理均使用相同历史槽位的原始 BEV。缓存按样本 token 查找，不递归保存融合结果；最多保留最大历史偏移数量的特征。场景切换、重复时间或倒序会重置状态。有效历史缺失时显式报错，因此 stream 不能任意跳帧，应按 dataset.stream_indices 连续处理。

缓存等价要求权重固定、eval 模式，以及各帧输入预处理确定一致。二维 warp 使用 z=0 平面，补偿自车运动，不显式补偿动态物体运动。

旧 scripts/run_temporal.py 及可视化脚本未修改；请使用新 tools/temporal_daocc.py 入口，旧脚本的 EMA 或单历史假设不再适用。
