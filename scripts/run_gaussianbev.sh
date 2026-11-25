

#* 11.24 分层可视化高斯：
conda activate daocc-cu121
cd /home/wangyushen/forks/CUDA-DAOcc/DAOcc
python tools/visualize_gaussian_height_layers.py \
    --config configs/nuscenes/occ3d/daocc_occ3d_nus_w_mask_inin_spconv2_voxel_gaussian_bev_iter.yaml \
    --checkpoint /home/wangyushen/Data/Outputs/daocc/train/daocc/w_mask/train_spconv2_voxel_gaussian_bev_learn_geometry_iter_fp16_htc/latest.pth \
    --start-index 10 \
    --num-samples 5 \
    --output-dir /home/wangyushen/Data/Outputs/daocc/outputs/gaussian_height_layers \
    --model.encoders.camera.vtransform.learn_gaussian_geometry true \