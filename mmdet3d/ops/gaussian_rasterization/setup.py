"""Build independently without changing DAOcc/setup.py.
Run from the DAOcc directory: python mmdet3d/ops/gaussian_rasterization/setup.py build_ext --inplace
"""
from pathlib import Path
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
HERE = Path(__file__).resolve().parent
os.chdir(HERE.parents[2])
setup(
    name="daocc-gaussian-rasterization",
    ext_modules=[CUDAExtension(
        name="mmdet3d.ops.gaussian_rasterization._C",
        sources=[str(HERE / "src" / p) for p in [
            "cuda_rasterizer/rasterizer_impl.cu", "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu", "rasterize_points.cu", "ext.cpp"]],
        include_dirs=[str(HERE / "third_party/glm")],
        extra_compile_args={"nvcc": ["-Xcompiler", "-fno-gnu-unique"]},
    )],
    cmdclass={"build_ext": BuildExtension},
)
