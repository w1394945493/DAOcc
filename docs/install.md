# CUDA 12.1 installation (migration in progress)

Target: Python 3.8, CUDA Toolkit 12.1, PyTorch 2.1.2+cu121 and
Torchvision 0.16.2+cu121. MMCV-full 1.7.2 / MMDetection 2.28.2 are a
candidate pairing, **not an end-to-end validated DAOcc environment**.
Changing dependencies alone does not finish the source migration.

## 1. Create the environment and select CUDA

Skip creation if `daocc-cu121` already exists.

```bash
conda create -n daocc-cu121 python=3.8 pip -y
conda activate daocc-cu121
export CUDA_HOME=/usr/local/cuda-12.1
export PATH="$CUDA_HOME/bin:$PATH"
hash -r
python --version
which python
nvcc -V
nvidia-smi
```

Set CUDA after activating the environment, including in new terminals.
Use a CUDA 12.1-compatible NVIDIA driver. Keep the original `daocc`
environment for comparison. If `PYTHONPATH` exposes unrelated environments
(for example ROS Python 3.10 packages), use a clean shell without those paths.

## 2. Install and verify PyTorch

NumPy 1.23.5 retains the `np.bool` / `np.int` aliases used by this source.
Use this pin instead of 1.24.4 until those call sites have been migrated.

```bash
python -m pip install numpy==1.23.5
python -m pip install torch==2.1.2 torchvision==0.16.2 \
  --index-url https://download.pytorch.org/whl/cu121
python - <<'PYTHON'
import torch
import torchvision
print(torch.__version__, torchvision.__version__, torch.version.cuda)
assert torch.version.cuda == "12.1"
assert torch.cuda.is_available()
x = torch.randn(128, 128, device="cuda")
y = x @ x
torch.cuda.synchronize()
print(torch.cuda.get_device_name(0), y.shape)
PYTHON
```

## 3. Build MMCV separately, then install ordinary dependencies

Use the existing checkout on the `cu121` branch:

```bash
cd /home/wys/wangyushen/forks/CUDA-DAOcc/DAOcc
python -m pip install wheel packaging ninja==1.11.1.1 yapf==0.40.0
MMCV_WITH_OPS=1 MAX_JOBS=4 python -m pip install -v \
  --no-build-isolation --no-binary=mmcv-full mmcv-full==1.7.2
```

A working C++ compiler is required (the local GCC 11 is the initial target).
Build against the selected PyTorch and CUDA rather than reusing cu113 wheels.
MMCV 1.7.2 may itself need source fixes for this toolchain; if compilation
fails, resolve that failure before proceeding. Do not substitute MMCV 2.x
or MMDetection 3.x: this project uses the older runner and registry APIs.

Verify an actual MMCV CUDA operator:

```bash
python - <<'PYTHON'
import torch
import mmcv
from mmcv.ops import nms
boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.]], device="cuda")
scores = torch.tensor([0.9, 0.8], device="cuda")
_, keep = nms(boxes, scores, 0.5)
torch.cuda.synchronize()
assert keep.tolist() == [0]
print("MMCV CUDA NMS OK:", mmcv.__version__)
PYTHON
python -m pip install -r requirements.txt
python -m pip check
python -c "import mmcv, mmdet, numpy, numba; print(mmcv.__version__, mmdet.__version__, numpy.__version__, numba.__version__)"
```

MPI is not universally optional: inspect the selected launcher and torchpack
usage. Even without a direct mpi4py pin, dependencies may install it. If that
path needs MPI, install a compatible MPI runtime/development toolchain and
validate mpi4py loading. The legacy OpenMPI / mpi4py instructions below are
archived, not the default installation path.

## 4. Complete source migration before building DAOcc

**Stop here until the following source changes have been completed:**

- Change the bundled spconv C++14 compilation setting in `setup.py` to C++17.
- Migrate obsolete THC headers/state references in custom CUDA extensions.
- Update `from numba import errors` to `from numba.core import errors` in
  `mmdet3d/datasets/pipelines/utils.py`.
- Check the remaining PyTorch / MMDetection API usage during build and runtime.

Rebuild all DAOcc extensions for the new environment. Existing `.so` and build
outputs may belong to the original environment: inventory and back them up or
use an isolated checkout before rebuilding, since both environments can refer
to the same source directory.

After those prerequisites are satisfied, run from the DAOcc checkout:

```bash
python -m pip install -v --no-build-isolation -e .
python -m pip check
python -c "import mmdet3d.ops; print('DAOcc ops import OK')"
```

An import check alone is insufficient: validate core CUDA operators and a
single model inference with the intended checkpoint/data, then the required
training or ONNX/PTQ workflow. Record failures and validated versions before
calling this environment supported. CUDA/TensorRT deployment dependencies
in the parent repository are a separate validation step.

References:

- [PyTorch official versions](https://docs.pytorch.org/get-started/previous-versions/)
- [MMDetection 2.28.2 MMCV constraints](https://github.com/open-mmlab/mmdetection/blob/v2.28.2/mmdet/__init__.py)
- [Numba 0.58.1 dependency constraints](https://github.com/numba/numba/blob/0.58.1/setup.py)

<!-- Original installation instructions preserved below for reference.
# Step-by-step installation instructions

**a. Create a conda virtual environment and activate it.**
```shell
conda create -n daocc python=3.8 -y
conda activate daocc
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**
```shell
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```

**c. Install [openmpi-4.0.4](https://www.open-mpi.org/software/ompi/v4.0/) (optional and deprecated)**
```shell
# Create an installation path. eg: INSTALL_PATH=~/local/openmpi-4.0.4
tar -xzvf openmpi-4.0.4.tar.gz
cd openmpi-4.0.4
./configure --prefix=INSTALL_PATH
make -j32
make install
# Add the following to your ~/.bashrc or ~/.bash_aliases file.
export PATH=$PATH:INSTALL_PATH/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:INSTALL_PATH/lib
```

**d. Install other dependencies.**
```shell
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
pip install mmdet==2.20.0
pip install nuscenes-devkit==1.1.11
pip install torchpack==0.3.1
pip install numba==0.48.0
pip install numpy==1.22.0
pip install prettytable==3.11.0
pip install ninja==1.11.1.1
pip install einops==0.8.0
pip install yapf==0.40.0
pip install pillow==8.4.0
pip install mpi4py==3.0.3 (optional and deprecated)
pip install Shapely==1.7.1
```

**e. Clone DAOcc and install the codebase.**
```shell
git clone https://github.com/AlphaPlusTT/DAOcc.git
cd DAOcc
python setup.py develop
```

-->
