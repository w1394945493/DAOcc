# DAOcc + VAD 风格矢量地图头

本目录为可选扩展，所有实现均为新增文件，不修改原单帧 DAOcc、注册文件、训练或测试脚本。显式导入 `mmdet3d.vad_map` 才注册。独立入口自动注册并复用原工具。

## 模型和监督

原相机/LiDAR -> Fuser -> Decoder 的最终 `[B,512,X,Y]` BEV 同时供原检测头、原 OCC 头和新增 `VADVectorMapHead` 使用。派生类通过临时 forward hook 获取 Decoder 输出，主干只执行一次，原任务逻辑完整复用；地图损失也向共享主干反向传播。

地图分支现为纯卷积头：共享3×3卷积后，分别输出类别中心热力图 `[B,K,X,Y]`、分类别中心偏移 `[B,K,2,X,Y]`、分类别折线点偏移 `[B,K,P,2,X,Y]`。不含 Query、Attention、Transformer 解码器，也不使用 Hungarian 匹配。保留 VADVectorMapHead 注册名仅用于入口兼容，VAD 指输出的矢量地图格式。默认共享通道128，每条线20点，最多保留100条预测。

GT折线包围框中心作为代表点，不受点序起点和方向影响。若同类中心冲突，按固定顺序在半径2格范围内找空位，同时保证正例不会被3×3局部峰值抑制互相删除。中心偏移不限制范围，恢复真实代表点；无空位时不覆盖其他GT，而记录 `stats/vector_map/num_dropped`。`num_relocated` 记录挪位数量，`num_assigned` 记录成功分配数量。不同类别回归独立，可共享格点。中心不是保证落在线上的几何点。

推理取类别热力图局部峰值和全局top-k，在对应格点读取中心偏移及折线点偏移，恢复米制折线。输出点不强制裁剪，避免把异常预测人为修正成正确边界。新卷积头与旧Transformer头权重不兼容，需要重新训练地图分支；可继续加载原单帧DAOcc权重初始化主干。

默认三类：divider（road_divider 与 lane_divider合并）、ped_crossing（人行横道轮廓）、boundary（道路/车道面并集边界）。这是二维折线实例，不预测车道中心线拓扑或三维高度。默认地图范围 LiDAR x∈[-30,30]m、y∈[-15,15]m；原 BEV 仍为108×108m。

Dataset 根据 nuScenes 地图、样本位姿提取 GT。原 pipeline 执行完毕后，将 LiDAR 平面增强同步应用于地图，再裁剪、等距采样。全局位置和 yaw 定义平面地图，与 VAD 的二维提取约定相同；不支持带俯仰/侧倾的非平面BDA。地图文件仅用作训练/评估监督，网络推理不读取地图。

道路边界先从完整多边形并集提取，再裁剪，避免窗口边框成为虚假 GT。使用空间索引缓存地图几何；首次加载某区域需要时间及 CPU 内存，后续查询复用索引。GT只在训练样本中追加，不改变原多sweep点云处理。地图监督不使用 OCC 的 mask_camera，而采用完整局部地图范围。

监督分配由GT代表点直接确定。损失包括CenterNet式高斯热力图损失、格点中心偏移L1、按地图长宽归一化的点偏移L1及米制方向余弦损失。只在同一GT的正反序/闭环等价序中选择回归误差最小者，不做跨实例集合匹配。没有额外的解码层辅助损失。地图分支保持FP32计算。

## 文件

- `model.py`：DAOccVADMap 派生模型。
- `head.py`：卷积地图头、格点监督分配、损失及解码。
- `geometry.py`：地图 GT 提取、增强、等价点序。
- `dataset.py`：数据集扩展与联合评估。
- `evaluation.py`：独立 Chamfer AP（0.5/1.0/1.5m阈值，0～1）。
- `configs/daocc_vad_map.yaml`：单帧原 DAOcc 叠加配置。
- `run.py`：独立注册、配置、训练/测试入口。
- `check.py`：无需 GPU/数据的 CPU 自检。

## 训练

在 DAOcc 根目录执行，已安装该项目的环境中：

```bash
conda activate daocc-cu121
cd /home/wangyushen/forks/CUDA-DAOcc/DAOcc

# 单卡：配置默认FP32，原有检测/OCC监督保留。
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD python -m mmdet3d.vad_map.run train \
  mmdet3d/vad_map/configs/daocc_vad_map.yaml \
  --run-dir /home/wangyushen/Data/Outputs/daocc/vad_map \
  --data.samples_per_gpu 1 --data.workers_per_gpu 2

# 多卡：torchrun 自动转交原 tools/dist_train.py。
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=$PWD python -m torch.distributed.run \
  --standalone --nproc_per_node=2 --module mmdet3d.vad_map.run train \
  mmdet3d/vad_map/configs/daocc_vad_map.yaml \
  --run-dir /home/wangyushen/Data/Outputs/daocc/vad_map \
  --data.samples_per_gpu 1 --data.workers_per_gpu 2
```

可追加 `--load_from /path/to/single_frame.pth` 初始化共享主干。新增地图头参数在旧权重中不存在，需要训练；首次迁移不要用 resume_from。恢复同结构训练用 `--resume_from /path/to/checkpoint.pth`。

配置显式继承 `daocc_occ3d_nus_w_mask_inin.yaml`，保留其中数据路径和原LiDAR实现。若要使用另一个完整单帧基础配置，在入口增加：

```bash
--base-config configs/nuscenes/occ3d/daocc_occ3d_nus_w_mask_inin_spconv2_voxel.yaml
```

更换基础配置必须保持最终 BEV 通道、物理范围与地图头配置一致。当前示例沿用训练集外层CBGSDataset包装，若基础配置无此包装需同步调整 data.train 的层级。

仅预测车道分隔线可修改新配置顶层 `vector_map_classes: [divider]`；类别数和监督标签自动跟随。若需要多区域地图数据，确认 dataset_root 下 maps/expansion 对应 JSON 均存在；缺少 location/map_location 时可从 nuScenes sample/scene/log 表查询。

## 推理和评估

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD python -m mmdet3d.vad_map.run test \
  mmdet3d/vad_map/configs/daocc_vad_map.yaml /path/to/trained_checkpoint.pth \
  --eval bbox
```

地图头输出每样本 `vector_map={points:[K,20,2],scores:[K],labels:[K]}`，坐标已撤销LiDAR增强，统一为米制未增强LiDAR平面坐标。评估保留原检测/OCC指标，自动追加 vector_map/*。`score_threshold` 会影响低置信度召回，可在评估时设得更低。评估器为本扩展实现，采用等距100点的对称Chamfer距离和贪心实例匹配；不宣称与VAD官方mAP逐位一致。地图GT为空的类别AP为NaN，不计入mAP。

原 tools/test.py 的逐帧保存逻辑未更改；若其保存器只选择原任务字段，不保证保存新 vector_map 字段。模型返回值与联合评估均包含矢量结果。

## 自检与配置预览

```bash
PYTHONPATH=$PWD python -m mmdet3d.vad_map.check
PYTHONPATH=$PWD python -m mmdet3d.vad_map.run config \
  mmdet3d/vad_map/configs/daocc_vad_map.yaml --save-config /tmp/daocc_vad_map.yaml
```

不要将叠加配置直接交给原 tools/train.py，否则不会解析 base_config 或注册新增类型。原文件完全不变，关闭扩展时继续使用原入口和原配置。
