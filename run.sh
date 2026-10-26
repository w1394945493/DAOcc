
conda activate daocc

# 准备构建及界面依赖，保留当前 NumPy。
python -m pip install \
  "numpy==1.22.0" "vtk==9.2.6" \
  wheel traits traitsui pyface apptools envisage PyQt5

# 使用当前环境的 VTK 构建，不让隔离环境另装新版。
python -m pip install \
  --no-build-isolation --no-cache-dir \
  "mayavi==4.8.2" "vtk==9.2.6" "numpy==1.22.0"


python tools/create_data.py \
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
python tools/test.py \
  configs/nuscenes/occ3d/daocc_occ3d_nus_w_mask_custom.yaml \
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
python tools/test.py \
  configs/nuscenes/occ3d/daocc_occ3d_nus_wo_mask_custom.yaml \
  /home/wys/wangyushen/data/Weights/daocc/occ3d_nus_wo_mask/epoch_6.pth \
  --out-dir /home/wys/wangyushen/data/Outputs/daocc_cu121/results/wo_mask/predictions

python tools/visualize.py \
  configs/nuscenes/occ3d/daocc_occ3d_nus_wo_mask_custom.yaml \
  --mode pred \
  --split val \
  --pred-dir /home/wys/wangyushen/data/Outputs/daocc_cu121/results/wo_mask/predictions \
  --out-dir /home/wys/wangyushen/data/Outputs/daocc_cu121/results/wo_mask/visualizations \
  --bbox-thickness 2 \
  --data.workers_per_gpu 2

python tools/vis_occ3d.py \
  /home/wys/wangyushen/data/Outputs/daocc_cu121/results/wo_mask/visualizations/scene-0103/occ/1533151603547590-3e8750f331d7499e9b5123e9eb70f2e2.npy

python tools/vis_occ3d.py \
  /home/wys/wangyushen/data/Outputs/daocc_cu121/results/wo_mask/visualizations/scene-0103/occ/1533151603547590-3e8750f331d7499e9b5123e9eb70f2e2_gt.npy

# *======================================================================================================#
# *(10.26) DAOcc：CUDA 12.1 环境配置与验证
# 本文件是分段执行的命令记录，请按步骤复制到已初始化 Conda 的 Bash 中执行。
# 不要直接 bash run.sh：上半部分还包含旧环境、数据处理和可视化命令。
# 目标组合：Python 3.8 / torch 2.1.2+cu121 / torchvision 0.16.2+cu121
#           mmcv-full 1.7.2 / mmdet 2.28.2 / numpy 1.23.5 / numba 0.58.1。
# 安装成功不等于推理验证完成；每一步检查通过后再继续。

# 1. 创建独立环境；已存在 daocc-cu121 时跳过 create，直接 activate。
conda create -n daocc-cu121 python=3.8 pip -y
conda activate daocc-cu121
cd /home/wys/wangyushen/forks/CUDA-DAOcc/DAOcc
# 使用含兼容补丁的 cu121 分支，检查输出应为 cu121。
git branch --show-current

# 2. 选择 CUDA 编译工具链。nvcc 的版本与 torch.version.cuda 是两项检查。
# 已配置永久激活钩子时，conda activate 会自动执行以下设置，无需重复追加 PATH：
# $CONDA_PREFIX/etc/conda/activate.d/cuda121.sh
# 对应 deactivate.d/cuda121.sh 在退出时恢复旧 CUDA_HOME 和路径。
# 尚未配置钩子时，手动执行以下三行（仅影响当前终端）：
export CUDA_HOME=/usr/local/cuda-12.1
export PATH="$CUDA_HOME/bin:$PATH"
hash -r

# 本终端专用于 DAOcc：隔离 ROS Python 3.10 路径及用户级包，避免混入 Python 3.8。
unset PYTHONPATH
export PYTHONNOUSERSITE=1
python --version
which python
echo "$CUDA_HOME"
which nvcc
nvcc -V
nvidia-smi

# 3. 安装基础数值库和 PyTorch。
# 每条命令前的 PIP_CONFIG_FILE=/dev/null、PIP_EXTRA_INDEX_URL= 仅对该命令生效：
# 忽略 pip 配置文件并清空附加源，避免 NVIDIA 附加源的 TLS 报错。
# 环境变量 PIP_INDEX_URL 仍可能生效；需要固定主源时显式指定 --index-url。
# NumPy 先固定 1.23.5，减少旧依赖兼容风险，不在此流程升级到 NumPy 2。
PIP_CONFIG_FILE=/dev/null PIP_EXTRA_INDEX_URL= \
python -m pip install numpy==1.23.5 \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple

# PyTorch 使用官方 cu121 源，不能沿用原来的 cu113 包或 MMCV 下载地址。
PIP_CONFIG_FILE=/dev/null PIP_EXTRA_INDEX_URL= \
python -m pip install torch==2.1.2 torchvision==0.16.2 \
  --index-url https://download.pytorch.org/whl/cu121

# 检查实际 GPU 运算，不能仅凭 nvcc -V 判断 PyTorch 是否可用。
python - <<'PYTORCH_CHECK'
import torch
import torchvision
import numpy as np
print("torch / torchvision / CUDA / numpy:",
      torch.__version__, torchvision.__version__, torch.version.cuda, np.__version__)
assert torch.version.cuda == "12.1"
assert torch.cuda.is_available(), "GPU 不可用，请检查驱动和设备访问"
x = torch.randn(128, 128, device="cuda")
y = x @ x
torch.cuda.synchronize()
print("GPU 运算通过:", torch.cuda.get_device_name(0), y.shape)
PYTORCH_CHECK

# 4. 先安装构建工具，再单独编译 MMCV。
# MMCV_WITH_OPS=1 启用扩展；--no-binary 强制源码构建；MAX_JOBS 控制内存占用。
# --no-build-isolation 使用当前环境的 torch/CUDA，不另建依赖版本不同的构建环境。
# 不替换成 MMCV 2.x / MMDetection 3.x，项目仍依赖旧 runner、parallel 和注册器接口。
PIP_CONFIG_FILE=/dev/null PIP_EXTRA_INDEX_URL= \
python -m pip install wheel packaging ninja==1.11.1.1 yapf==0.40.0

PIP_CONFIG_FILE=/dev/null PIP_EXTRA_INDEX_URL= \
MMCV_WITH_OPS=1 MAX_JOBS=4 python -m pip install -v \
  --no-build-isolation --no-binary=mmcv-full mmcv-full==1.7.2

# 验证 MMCV 扩展不仅可导入，而且能执行 CUDA NMS。
python - <<'MMCV_CHECK'
import torch
from mmcv.ops import nms
boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.]], device="cuda")
scores = torch.tensor([0.9, 0.8], device="cuda")
_, keep = nms(boxes, scores, 0.5)
torch.cuda.synchronize()
assert keep.tolist() == [0]
print("MMCV CUDA NMS 通过")
MMCV_CHECK

# 5. 安装本分支更新后的依赖，不能使用原 cu113 分支的 requirements。
# trimesh 被项目可视化模块导入，测试入口也可能间接需要它。
PIP_CONFIG_FILE=/dev/null PIP_EXTRA_INDEX_URL= \
python -m pip install -r requirements.txt

PIP_CONFIG_FILE=/dev/null PIP_EXTRA_INDEX_URL= \
python -m pip install trimesh

# 6. 可选：Mayavi 占用网格可视化（tools/vis_occ3d.py）。仅推理可跳过此步骤。
# 曾在 mayavi 4.8.2 生成 TVTK 类时段错误；当时 pip 选择了 vtk 9.5.2。
# 以下固定 vtk 9.2.6 是候选规避方案，安装后仍需检查 TVTK 和实际图形窗口。
# 必须先安装 VTK，再关闭构建隔离，否则隔离环境可能重新选择较新 VTK。
PIP_CONFIG_FILE=/dev/null PIP_EXTRA_INDEX_URL= \
python -m pip install \
  "numpy==1.23.5" "vtk==9.2.6" \
  "traits==6.4.3" "traitsui==8.0.0" "pyface==8.0.0" \
  apptools envisage pygments packaging wheel "PyQt5==5.15.10"

PIP_CONFIG_FILE=/dev/null PIP_EXTRA_INDEX_URL= \
python -m pip install \
  --no-build-isolation --no-cache-dir \
  "mayavi==4.8.2" "vtk==9.2.6" "numpy==1.23.5"

python - <<'TVTK_CHECK'
import mayavi
import vtk
from tvtk.api import tvtk
print("Mayavi / VTK:", mayavi.__version__, vtk.vtkVersion.GetVTKVersion())
sphere = tvtk.SphereSource()
sphere.update()
assert sphere.output.number_of_points > 0
print("TVTK 数据管线通过；窗口渲染还需可用的显示/OpenGL 环境")
TVTK_CHECK

# 7. 编译安装 DAOcc；这一步依赖 cu121 分支的源码适配：
# - setup.py 使用 C++17，满足 PyTorch 2.1 的头文件要求；
# - 旧 THC 头文件改用 ATen CUDAContext，未使用的 THCState 声明已注释；
# - 内置 spconv 注册覆盖 MMCV 同名项，保证使用项目自己的稀疏张量实现；
# - mmdet3d/_compat.py 将 MMCV 的整数设备编号转换成 torch.device；
# - Numba 导入和 NumPy 旧类型别名已适配。
# 不复用旧 torch 环境编译的扩展；同一源码目录的 .so 会被两个环境共享，
# 需要保留旧环境可运行状态时，应先备份构建产物或使用独立源码目录。
# 原命令保留为注释（与下一条二选一，不需要重复安装）：
# python -m pip install -v --no-build-isolation -e .
# --no-deps：依赖已安装，避免本步再次解析；-e：Python 修改即时生效。
# C++/CUDA 源码变化仍需重编译。关闭 pip 自身更新检查可减少无关版本/TLS 提示。
PIP_CONFIG_FILE=/dev/null PIP_EXTRA_INDEX_URL= MAX_JOBS=1 \
python -m pip --disable-pip-version-check install \
  -v --no-build-isolation --no-deps -e .

# 失败时保存完整日志，再提取最早的编译器错误，而不是只看结尾 RuntimeError：
# MAX_JOBS=1 python -m pip --disable-pip-version-check install \
#   -v --no-build-isolation --no-deps -e . > /tmp/daocc-cu121-build.log 2>&1
# grep -nE -m 10 -C 5 'error:|fatal error:|FAILED:|Killed|unsupported' /tmp/daocc-cu121-build.log

# 8. 检查依赖和测试入口；通过后再运行实际配置/权重的推理命令。
# --help 只验证导入与参数解析，不代表模型前向和完整数据集测试已通过。
python -m pip check
python tools/test.py --help
# 后续运行时请使用当前 CUDA-DAOcc/DAOcc 目录，上半部分部分旧命令指向 forks/DAOcc。
# 推荐将 CUDA 12.1 的预测结果写入 data/Outputs/daocc_cu121，便于与旧环境比较。
