
(10.30) 继续探究lidar 历史帧聚合 作为输入 → 体素化 → lidar backbone处理流程

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD \
/home/wangyushen/miniforge3/envs/daocc-cu121/bin/python \
tools/visualize_spconv2_lidar_pipeline.py \
configs/nuscenes/occ3d/daocc_occ3d_nus_w_mask_inin_spconv2_voxel_fixed_s128.yaml \
--gpu 0 \
--num-samples 20 \
--warmup-iters 3 \
--history-sweeps 9 \
--checkpoint /home/wangyushen/Data/Weights/daocc/occ3d_nus_w_mask/epoch_6.pth \
--visualize \
--output-dir /home/wangyushen/Data/Outputs/daocc/outputs/spconv2_lidar_visualization

lidar分支各部分耗时情况：当前 LiDAR 分支的主要开销集中在稀疏 backbone，占比约 88%。耗时约16.366ms，总耗时约18.576ms

| 阶段 | 平均耗时 | P50 | P90 | 占总耗时 |
|---|---:|---:|---:|---:|
| spconv2 体素化 | 1.304 ms | 1.295 ms | 1.438 ms | 7.0% |
| 体素内点特征均值聚合 | 0.123 ms | 0.121 ms | 0.133 ms | 0.7% |
| 构造稀疏张量 | 0.241 ms | 0.237 ms | 0.279 ms | 1.3% |
| spconv2 稀疏 backbone | 16.366 ms | 15.996 ms | 18.976 ms | 88.1% |
| 稠密化与高度维折叠 | 0.542 ms | 0.502 ms | 0.605 ms | 2.9% |
| **LiDAR 分支合计** | **18.576 ms** | **18.212 ms** | **21.204 ms** | **100%** |