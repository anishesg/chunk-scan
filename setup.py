import os
import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

# Collect all CUDA source files
src_dir = os.path.join(os.path.dirname(__file__), "src")
cu_files = [
    os.path.join(src_dir, f)
    for f in os.listdir(src_dir)
    if f.endswith(".cu")
]
cu_files.append(os.path.join("csrc", "bindings.cpp"))

# Require Ampere or newer for warp-shuffle and expanded shared memory
cuda_arch_flags = [
    "-gencode=arch=compute_80,code=sm_80",
    "-gencode=arch=compute_86,code=sm_86",
    "-gencode=arch=compute_89,code=sm_89",
    "-gencode=arch=compute_90,code=sm_90",
]

nvcc_flags = cuda_arch_flags + [
    "-O3",
    "--use_fast_math",
    "-std=c++17",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
]

cxx_flags = ["-O3", "-std=c++17"]

setup(
    name="chunk_scan",
    version="0.1.0",
    description="Fused SSD kernel: tensor-core intra-chunk matmul + warp-shuffle inter-chunk scan",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="chunk_scan._C",
            sources=cu_files,
            include_dirs=[src_dir],
            extra_compile_args={
                "cxx": cxx_flags,
                "nvcc": nvcc_flags,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.8",
    install_requires=["torch>=2.0"],
)
