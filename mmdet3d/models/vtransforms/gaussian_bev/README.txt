GaussianBEVTransform for DAOcc
==============================

Scope
-----
Camera FPN [B,N,256,32,88] -> Gaussian BEV -> [B,128,180,180].
The original BEVFusion model is retained; ConvFuser input channels are [128,256].
No temporal fusion, height layers, extra losses or architecture improvements.
Existing model implementations and training scripts are unchanged.
models/vtransforms/__init__.py imports GaussianBEVTransform for normal registration.

Reference
---------
/home/wangyushen/forks/GaussianLSS/GaussianLSS/model/GaussianLSS.py
/home/wangyushen/forks/GaussianLSS/GaussianLSS/model/backbones/agp.py
/home/wangyushen/forks/GaussianLSS/GaussianLSS/model/diff-gaussian-rasterization
Official project: https://github.com/HCIS-Lab/GaussianLSS

PrepareChannel's three prediction branches are copied from the reference.
Uniform depth-bin centers (1..61 m, 64 bins), depth softmax, weighted 3D
mean/covariance (k^2/9, k=.5), opacity>.01 filtering, +.3 pixel covariance,
opacity-descending sorting and alpha compositing are retained.
Feature pixel endpoints intentionally follow GaussianLSS: 0..image_width
and 0..image_height (not a new feature-center sampling convention).

Adaptations
-----------
geometry.py: DAOcc camera intrinsics, camera2lidar, inverse image augmentation
and forward LiDAR augmentation. Every Gaussian is in augmented LiDAR space.
renderer.py: [X,Y] tensor axes; columns correspond to physical Y, rows to X.
Grid centers: ix=(x+54)/.6-.5, iy=(y+54)/.6-.5.
Geometry and rasterization run in FP32, including under native autocast.
Empty filtered samples return zeros retaining an autograd connection.
Covariance backward scale indices are corrected for unequal X/Y cell sizes;
this does not change the configured equal .6 m cell-size case.
The backward center derivative uses pixel units (factor 1), removing the
legacy NDC-to-pixel W/2,H/2 factors because BEV projection already outputs pixels.
CUDA rendering otherwise retains the reference algorithm and 128 channels.
Source license and GLM license are retained under ops/gaussian_rasterization.

Build (from DAOcc, in the daocc-cu121 environment)
------------------------------------------------
The machine default /usr/local/cuda points to CUDA 11.8 and system GCC is 13.
Use the installed CUDA 12.2 and conda GCC 12 for this PyTorch 2.1/cu121 env:

mkdir -p /tmp/daocc_gaussian_include
ln -sf /usr/include/crypt.h /tmp/daocc_gaussian_include/crypt.h

CPATH=/tmp/daocc_gaussian_include CUDA_HOME=/usr/local/cuda-12.2 \
CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc" \
CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++" \
TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=2 \
python mmdet3d/ops/gaussian_rasterization/setup.py build_ext --inplace

The temporary include directory supplies crypt.h missing from the local conda sysroot.
The architecture above is for the local RTX 3090. Choose the appropriate
architecture when building on another GPU. Build uses a separate extension
name mmdet3d.ops.gaussian_rasterization._C; no upstream installed package needed.

Configuration and registration
------------------------------
configs/nuscenes/occ3d/daocc_occ3d_nus_w_mask_inin_spconv2_voxel_gaussian_bev_iter.yaml

This is a copy of the requested iter YAML. Only camera.vtransform and the
fuser input channels change. All other effective settings, including the
iter hooks, optimizer, losses and data pipeline, remain identical.
Registration uses the standard import in models/vtransforms/__init__.py.
The configuration uses type: GaussianBEVTransform. Source and dumped configs
both work through the existing train/test entry points; no wrapper is needed.

python tools/train.py configs/nuscenes/occ3d/daocc_occ3d_nus_w_mask_inin_spconv2_voxel_gaussian_bev_iter.yaml

The original BEVTransformV2/fuser weights are not equivalent to the new module;
use the new source config's normal initialization, not strict resume from an
old-model checkpoint. No trained Gaussian checkpoint is supplied.

Verification (from DAOcc)
------------------------
python -m unittest discover -s tests/test_models -p test_gaussian_bev_transform.py -v

Checks: unchanged effective baseline settings, augmented camera geometry,
BEV center/axis convention, CUDA output and analytical gradients, opacity
sorting, empty high-threshold result, actual 6-camera 32x88 transform forward
and backward with MMCV mixed-precision wrapping and native autocast.

Delivery status
---------------
Source implementation is complete. CUDA compilation succeeded before the final
backward pixel-scale correction. Rebuild the extension before use. Testing was
stopped at user request; the final source revision has not been fully validated.

LiDAR-centred mode
------------------
Set center_source: lidar to use the augmented LiDAR XYZ coordinates as Gaussian
centres. Points are projected through lidar2image and img_aug_matrix and grouped
by camera feature cell. Each cell emits at most one Gaussian: a current-frame
point (time lag zero) is preferred and the nearest camera depth is selected;
only cells without a current-frame point fall back to the nearest historical
sweep point. The selected cell supplies its image feature and opacity. Points
unseen by every camera do not contribute to the camera BEV. gaussian_std
specifies the fixed XYZ standard deviation in metres; the current configuration
uses [0.3, 0.3, 0.3]. center_source: depth retains the original GaussianLSS
depth-distribution implementation.
