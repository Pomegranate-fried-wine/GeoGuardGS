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

nvcc_compiler_flags = [
    "-allow-unsupported-compiler",
]
cxx_compiler_flags = []

if os.name == 'nt':
    nvcc_compiler_flags.extend([
        "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH",
        "-D_CRT_SECURE_NO_WARNINGS",
        "-Xcompiler",
        "/wd4819",
    ])
    cxx_compiler_flags.extend([
        "/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH",
        "/D_CRT_SECURE_NO_WARNINGS",
        "/wd4624",
    ])

setup(
    name="simple_knn",
    ext_modules=[
        CUDAExtension(
            name="simple_knn._C",
            sources=[
            "spatial.cu", 
            "simple_knn.cu",
            "ext.cpp"],
            extra_compile_args={"nvcc": nvcc_compiler_flags, "cxx": cxx_compiler_flags})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
