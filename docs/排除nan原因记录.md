
<!-- 训练命令： -->
conda activate daocc-cu121
cd /home/wangyushen/forks/CUDA-DAOcc/DAOcc

GPUS=4,5,6,7 \
BATCH_SIZE=4 \
WORKERS=4 \
MAX_EPOCHS=6 \
EVAL_BEFORE_RESUME=1 \
NAN_CONSECUTIVE=10 \
NAN_SAVE_BATCH=1 \
CONFIG=/home/wangyushen/forks/CUDA-DAOcc/DAOcc/configs/nuscenes/occ3d/daocc_occ3d_nus_w_mask_inin_spconv2_voxel_fixed_s128_nan_debug.yaml \
RUN_DIR=/home/wangyushen/Data/Outputs/daocc/w_mask/train_spconv2_voxel_fp16_fixed_s128_nan_debug_2 \
PYTHON=python \
bash scripts/train_occ3d_nan_debug.sh

<!-- 第一个epoch训练中，出现NaN，经分析，并非损失溢出，最早记录到异常的层是：
module.decoder.neck.conv.1.running_mean
module.decoder.neck.conv.1.running_var
对应/home/wangyushen/forks/CUDA-DAOcc/DAOcc/mmdet3d/models/backbones/second.py

可能的一些解决措施：
1. 完善中间激活统计和异常前滚动checkpoint。
2. 先测试仅将 FPN_LSS 强制FP32。
3. 若仍溢出，再将 CustomResNet + FPN_LSS 强制FP32。
4. 根据激活范数结果判断是否为捷径增加BN。
5. 最后比较原FP32、局部FP32混合精度、增加捷径归一化三种方案的稳定性和mIoU。 -->