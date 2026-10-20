
conda activate daocc

# 准备构建及界面依赖，保留当前 NumPy。
python -m pip install \
  "numpy==1.22.0" "vtk==9.2.6" \
  wheel traits traitsui pyface apptools envisage PyQt5

# 使用当前环境的 VTK 构建，不让隔离环境另装新版。
python -m pip install \
  --no-build-isolation --no-cache-dir \
  "mayavi==4.8.2" "vtk==9.2.6" "numpy==1.22.0"


python /home/wys/wangyushen/forks/DAOcc/tools/create_data.py \
  nuscenes \
  --root-path /home/wys/wangyushen/data/Datasets/nuscenes/v1.0-mini \
  --version v1.0-mini \
  --out-dir /home/wys/wangyushen/data/Datasets/nuscenes/anns/daocc \
  --extra-tag nuscenes \
  --occ3d-root /home/wys/wangyushen/data/Datasets/nuscenes/gts

#* 评估-可视化
#*================================================================================#
#* with mask
cd /home/wys/wangyushen/forks/DAOcc
python /home/wys/wangyushen/forks/DAOcc/tools/test.py \
  /home/wys/wangyushen/forks/DAOcc/configs/nuscenes/occ3d/daocc_occ3d_nus_w_mask_custom.yaml \
  /home/wys/wangyushen/data/Weights/daocc/occ3d_nus_w_mask/epoch_6.pth \
  --out-dir /home/wys/wangyushen/data/Outputs/daocc/results/w_mask/predictions \

# * 可视化：直接可视化此前的预测结果
cd /home/wys/wangyushen/forks/DAOcc
# 可视化3D检测结果，导出occ npy文件
python tools/visualize.py \
  configs/nuscenes/occ3d/daocc_occ3d_nus_w_mask_custom.yaml \
  --mode pred \
  --split val \
  --pred-dir /home/wys/wangyushen/data/Outputs/daocc/results/w_mask/predictions \
  --out-dir /home/wys/wangyushen/data/Outputs/daocc/results/w_mask/visualizations \
  --bbox-thickness 2 \
  --data.workers_per_gpu 2

cd /home/wys/wangyushen/forks/DAOcc

python tools/vis_occ3d.py \
  /home/wys/wangyushen/data/Outputs/daocc/results/w_mask/visualizations/scene-0103/occ/1533151603547590-3e8750f331d7499e9b5123e9eb70f2e2.npy

python tools/vis_occ3d.py \
  /home/wys/wangyushen/data/Outputs/daocc/results/w_mask/visualizations/scene-0103/occ/1533151603547590-3e8750f331d7499e9b5123e9eb70f2e2_gt.npy

# 仅可视化相机掩码内结果
python tools/vis_occ3d.py \
  /home/wys/wangyushen/data/Outputs/daocc/results/w_mask/visualizations/scene-0103/occ/1533151603547590-3e8750f331d7499e9b5123e9eb70f2e2.npy \
  --camera-mask \
  /home/wys/wangyushen/data/Datasets/nuscenes/gts/scene-0103/3e8750f331d7499e9b5123e9eb70f2e2/labels.npz


#*================================================================================#
# * without mask
python /home/wys/wangyushen/forks/DAOcc/tools/test.py \
  /home/wys/wangyushen/forks/DAOcc/configs/nuscenes/occ3d/daocc_occ3d_nus_wo_mask_custom.yaml \
  /home/wys/wangyushen/data/Weights/daocc/occ3d_nus_wo_mask/epoch_6.pth \
  --out-dir /home/wys/wangyushen/data/Outputs/daocc/results/wo_mask/predictions

python tools/visualize.py \
  /home/wys/wangyushen/forks/DAOcc/configs/nuscenes/occ3d/daocc_occ3d_nus_wo_mask_custom.yaml \
  --mode pred \
  --split val \
  --pred-dir /home/wys/wangyushen/data/Outputs/daocc/results/wo_mask/predictions \
  --out-dir /home/wys/wangyushen/data/Outputs/daocc/results/wo_mask/visualizations \
  --bbox-thickness 2 \
  --data.workers_per_gpu 2

python tools/vis_occ3d.py \
  /home/wys/wangyushen/data/Outputs/daocc/results/wo_mask/visualizations/scene-0103/occ/1533151603547590-3e8750f331d7499e9b5123e9eb70f2e2.npy

python tools/vis_occ3d.py \
  /home/wys/wangyushen/data/Outputs/daocc/results/wo_mask/visualizations/scene-0103/occ/1533151603547590-3e8750f331d7499e9b5123e9eb70f2e2_gt.npy
