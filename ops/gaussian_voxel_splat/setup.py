import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="gaussian_voxel_splat",
    packages=["gaussian_voxel_splat"],
    package_dir={"gaussian_voxel_splat": "."},
    ext_modules=[CUDAExtension(
        name="gaussian_voxel_splat._C",
        sources=["src/aggregator_impl.cu", "src/forward.cu", "src/backward.cu",
                 "local_aggregate.cu", "ext.cpp"],
        # conda交叉编译器使用独立sysroot，显式加入当前环境头文件（crypt.h等）。
        include_dirs=[os.path.join(os.environ.get("CONDA_PREFIX", ""), "include")],
        extra_compile_args={
            "cxx": ["-O2", "-std=c++17"],
            # 当前服务器系统GCC 13高于CUDA 12.2官方上限；该开关仅跳过
            # nvcc版本检查，实际ABI仍由PyTorch扩展的编译参数保持一致。
            "nvcc": ["-O2", "-std=c++17", "-allow-unsupported-compiler",
                     "-Xcompiler", "-fno-gnu-unique"],
        },
    )],
    cmdclass={"build_ext": BuildExtension},
)
