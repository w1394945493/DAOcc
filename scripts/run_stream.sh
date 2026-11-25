


#*---------------------------------------------#
#* (11.24) 1-3D检测实例引导 + 流式训练，decoder前融合；GaussianBEVTransform固定尺度、旋转、位置
#! 报nan
conda activate daocc-cu121
cd /home/wangyushen/forks/CUDA-DAOcc/DAOcc
NUM_FRAMES=28130
BATCH_SIZE=4
NUM_WORKERS=4
GPUS=2
EPOCHS=24
MAX_ITERS=$((NUM_FRAMES * EPOCHS / (BATCH_SIZE * GPUS)))
INTERVAL=$(((MAX_ITERS + EPOCHS - 1) / EPOCHS))
CUDA_VISIBLE_DEVICES=5,6 torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="$GPUS" \
  tools/dist_train.py \
  configs/nuscenes/occ3d/daocc_occ3d_nus_w_mask_stream_instance_gaussian_bevtransform.yaml \
  --launcher pytorch \
  --run-dir /home/wangyushen/Data/Outputs/daocc/train/stream_pre_cbgs_faug_skip10_rev10_instance_gaussianbev \
  --model.encoders.camera.backbone.init_cfg.checkpoint /home/wangyushen/Data/Weights/daocc/pretrained/htc_r50_backbone.pth \
  --model.encoders.camera.vtransform.learn_gaussian_geometry True \
  --model.heads.object.multi_point_features False \
  --model.heads.object.learnable_sampling False \
  --model.heads.occ.instance_fusion.edge_matching False \
  --model.heads.occ.instance_fusion.local_edges False \
  --model.heads.object.roi_features False \
  --streaming.history_fusion.mode refine \
  --data.train.dataset.sequence_cbgs True \
  --data.train.dataset.skip_empty_gt True \
  --data.train.dataset.keep_consistent_seq_aug False \
  --data.train.dataset.skip_prob 0.1 \
  --data.train.dataset.sequence_flip_prob 0.1 \
  --streaming.history_fusion.enabled True \
  --streaming.bev_warp.enabled True \
  --streaming.fuse_before_decoder True \
  --data.samples_per_gpu "$BATCH_SIZE" \
  --data.workers_per_gpu "$NUM_WORKERS" \
  --runner.max_iters "$MAX_ITERS" \
  --evaluation.interval "$INTERVAL" \
  --checkpoint_config.interval "$INTERVAL" \
  --log_config.interval 50 \
  --optimizer.lr 2.0e-4 \
  --checkpoint_config.max_keep_ckpts 1


#*
#* 单卡训练命令
conda activate daocc-cu121
cd /home/wangyushen/forks/CUDA-DAOcc/DAOcc
CUDA_VISIBLE_DEVICES=5 \
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
PYTHONWARNINGS=ignore \
python tools/dist_train.py \
  configs/nuscenes/occ3d/daocc_occ3d_nus_w_mask_stream_instance_gaussian_bevtransform.yaml \
  --launcher none \
  --run-dir /home/wangyushen/Data/Outputs/daocc/debug/stream_instance_gaussianbev_single_gpu \
  --model.encoders.camera.vtransform.learn_gaussian_geometry True \
  --data.train.dataset.sequence_cbgs True \
  --data.train.dataset.keep_consistent_seq_aug False \
  --data.train.dataset.skip_prob 0.1 \
  --data.train.dataset.sequence_flip_prob 0.1 \
  --data.samples_per_gpu 1 \
  --data.workers_per_gpu 0 \
  --runner.max_iters 500 \
  --log_config.interval 10 \
  --evaluation.interval 500 \
  --checkpoint_config.interval 500 \

#* 异常记录
conda activate daocc-cu121
cd /home/wangyushen/forks/CUDA-DAOcc/DAOcc

NUM_FRAMES=28130
BATCH_SIZE=4
GPUS=1
EPOCHS=24
MAX_ITERS=$((NUM_FRAMES * EPOCHS / (BATCH_SIZE * GPUS)))

CUDA_VISIBLE_DEVICES=6 torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=1 \
  tools/dist_train_stream_nan_debug.py \
  configs/nuscenes/occ3d/daocc_occ3d_nus_w_mask_stream_instance_gaussian_bevtransform.yaml \
  --launcher pytorch \
  --run-dir /home/wangyushen/Data/Outputs/daocc/debug/stream_instance_gaussianbev_nan_2 \
  --model.encoders.camera.backbone.init_cfg.checkpoint /home/wangyushen/Data/Weights/daocc/pretrained/htc_r50_backbone.pth \
  --model.encoders.camera.vtransform.learn_gaussian_geometry True \
  --model.heads.object.multi_point_features False \
  --model.heads.object.learnable_sampling False \
  --model.heads.occ.instance_fusion.edge_matching False \
  --model.heads.occ.instance_fusion.local_edges False \
  --model.heads.object.roi_features False \
  --streaming.history_fusion.mode refine \
  --data.train.dataset.sequence_cbgs True \
  --data.train.dataset.skip_empty_gt True \
  --data.train.dataset.keep_consistent_seq_aug False \
  --data.train.dataset.skip_prob 0.1 \
  --data.train.dataset.sequence_flip_prob 0.1 \
  --streaming.history_fusion.enabled True \
  --streaming.bev_warp.enabled True \
  --streaming.fuse_before_decoder True \
  --data.samples_per_gpu "$BATCH_SIZE" \
  --data.workers_per_gpu 4 \
  --runner.max_iters "$MAX_ITERS" \
  --log_config.interval 50 \
  --optimizer.lr 2.0e-4 \
  --nan_debug.enabled True \
  --nan_debug.out_dir /home/wangyushen/Data/Outputs/daocc/debug/stream_instance_gaussianbev_nan_2/records \
  --nan_debug.save_batch True \
  --nan_debug.rolling_checkpoint_interval 25 \
  --nan_debug.rolling_checkpoint_keep 2 \
  --nan_debug.activation_modules "['encoders.camera.backbone','encoders.camera.neck','encoders.camera.vtransform.predictor','encoders.camera.vtransform','history_fusion','decoder.backbone','heads.object','heads.occ']" \
  --nan_debug.activation_history_size 20 \
  --nan_debug.stop_after_consecutive_nan 2



#* ------------------------------------------------------------------#
#* 1-对照实验 stream_instance_bevfusion
# --model.heads.object.instance_selection.train_filter gt_quality \
conda activate daocc-cu121
cd /home/wangyushen/forks/CUDA-DAOcc/DAOcc
NUM_FRAMES=28130
BATCH_SIZE=4
NUM_WORKERS=4
GPUS=2
EPOCHS=24
MAX_ITERS=$((NUM_FRAMES * EPOCHS / (BATCH_SIZE * GPUS)))
INTERVAL=$(((MAX_ITERS + EPOCHS - 1) / EPOCHS))
CUDA_VISIBLE_DEVICES=2,3 torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="$GPUS" \
  tools/dist_train.py \
  configs/nuscenes/occ3d/daocc_occ3d_nus_w_mask_stream_instance_bevfusion.yaml \
  --launcher pytorch \
  --run-dir /home/wangyushen/Data/Outputs/daocc/train/stream_pre_cbgs_faug_skip10_rev10_instance_grad35 \
  --model.encoders.camera.backbone.init_cfg.checkpoint /home/wangyushen/Data/Weights/daocc/pretrained/htc_r50_backbone.pth \
  --model.heads.object.multi_point_features False \
  --model.heads.object.learnable_sampling False \
  --model.heads.occ.instance_fusion.edge_matching False \
  --model.heads.occ.instance_fusion.local_edges False \
  --model.heads.object.roi_features False \
  --streaming.history_fusion.mode refine \
  --data.train.dataset.sequence_cbgs True \
  --data.train.dataset.skip_empty_gt True \
  --data.train.dataset.keep_consistent_seq_aug False \
  --data.train.dataset.skip_prob 0.1 \
  --data.train.dataset.sequence_flip_prob 0.1 \
  --streaming.history_fusion.enabled True \
  --streaming.bev_warp.enabled True \
  --streaming.fuse_before_decoder True \
  --data.samples_per_gpu "$BATCH_SIZE" \
  --data.workers_per_gpu "$NUM_WORKERS" \
  --runner.max_iters "$MAX_ITERS" \
  --evaluation.interval "$INTERVAL" \
  --checkpoint_config.interval "$INTERVAL" \
  --log_config.interval 50 \
  --optimizer.lr 2.0e-4 \
  --checkpoint_config.max_keep_ckpts 1 \


#* (11.25) 检测框时序运动补偿可视化
# 按2×N布局显示连续视频帧：
# 第一行：GT框 + 当前检测框 + 上一帧检测框的自车运动补偿结果；
# 第二行：GT框 + 当前检测框 + 上一帧检测框经自车运动补偿和(vx, vy)速度外推后的结果。
conda activate daocc-cu121
cd /home/wangyushen/forks/CUDA-DAOcc/DAOcc
CUDA_VISIBLE_DEVICES=6 python tools/visualize_detection_velocity_compensation.py \
  --scene-indices 0 1 2 \
  --start-frame 0 \
  --num-frames 6 \
  --score-threshold 0.3 \
  --gpu 0 \
  --output-dir /home/wangyushen/Data/Outputs/daocc/outputs/detection_velocity_compensation

