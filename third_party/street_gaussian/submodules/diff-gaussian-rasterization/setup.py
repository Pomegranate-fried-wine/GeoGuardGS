#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

nvcc_compiler_flags = [
    "-allow-unsupported-compiler",
    "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/"),
]
cxx_compiler_flags = []
if os.name == "nt":
    nvcc_compiler_flags.extend([
        "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH",
        "-D_CRT_SECURE_NO_WARNINGS",
        "-Xcompiler",
        "/wd4819",
    ])
    cxx_compiler_flags.extend([
        "/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH",
        "/D_CRT_SECURE_NO_WARNINGS",
        "/wd4819",
    ])

setup(
    name="diff_gaussian_rasterization",
    packages=['diff_gaussian_rasterization'],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization._C",
            sources=[
            "cuda_rasterizer/rasterizer_impl.cu",
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",
            "rasterize_points.cu",
            "ext.cpp"],
            # extra_compile_args={"nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]})
            extra_compile_args={"nvcc": nvcc_compiler_flags, "cxx": cxx_compiler_flags})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
