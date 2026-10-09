# Copyright (c) 2023, Albert Gu, Tri Dao.
import warnings
import os
import re
import ast
from pathlib import Path
from packaging.version import parse, Version

from setuptools import setup, find_packages
import subprocess

from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

import torch
from torch.utils.cpp_extension import (
    BuildExtension,
    CUDAExtension,
    CUDA_HOME,
    HIP_HOME
)


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))

PACKAGE_NAME = "mamba_ssm"

# KEEP_CUDA_BUILD: Set to TRUE to build CUDA selective scan kernels (needed for Mamba-1)
KEEP_CUDA_BUILD = os.getenv("MAMBA_KEEP_CUDA_BUILD", "FALSE") == "TRUE"
# For CI, we want the option to build with C++11 ABI since the nvcr images use C++11 ABI
FORCE_CXX11_ABI = os.getenv("MAMBA_FORCE_CXX11_ABI", "FALSE") == "TRUE"



def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output(
        [cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True
    )
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_ver = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_ver


def get_hip_version(rocm_dir):

    hipcc_bin = "hipcc" if rocm_dir is None else os.path.join(rocm_dir, "bin", "hipcc")
    try:
        raw_output = subprocess.check_output(
            [hipcc_bin, "--version"], universal_newlines=True
        )
    except Exception as e:
        print(
            f"hip installation not found: {e} ROCM_PATH={os.environ.get('ROCM_PATH')}"
        )
        return None, None

    for line in raw_output.split("\n"):
        if "HIP version" in line:
            rocm_version = parse(line.split()[-1].rstrip('-').replace('-', '+')) # local version is not parsed correctly
            return line, rocm_version

    return None, None


def get_torch_hip_version():

    if torch.version.hip:
        return parse(torch.version.hip.split()[-1].rstrip('-').replace('-', '+'))
    else:
        return None


def check_if_hip_home_none(global_option: str) -> None:

    if HIP_HOME is not None:
        return
    # warn instead of error because user could be downloading prebuilt wheels, so hipcc won't be necessary
    # in that case.
    warnings.warn(
        f"{global_option} was requested, but hipcc was not found.  Are you sure your environment has hipcc available?"
    )


def check_if_cuda_home_none(global_option: str) -> None:
    if CUDA_HOME is not None:
        return
    # warn instead of error because user could be downloading prebuilt wheels, so nvcc won't be necessary
    # in that case.
    warnings.warn(
        f"{global_option} was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  "
        "If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, "
        "only images whose names contain 'devel' will provide nvcc."
    )


def append_nvcc_threads(nvcc_extra_args):
    return nvcc_extra_args + ["--threads", "4"]


cmdclass = {}
ext_modules = []


HIP_BUILD = bool(torch.version.hip)

if KEEP_CUDA_BUILD:
    print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
    TORCH_MAJOR = int(torch.__version__.split(".")[0])
    TORCH_MINOR = int(torch.__version__.split(".")[1])

    cc_flag = []

    if HIP_BUILD:
        check_if_hip_home_none(PACKAGE_NAME)

        rocm_home = os.getenv("ROCM_PATH")
        _, hip_version = get_hip_version(rocm_home)

        if HIP_HOME is not None:
            if hip_version < Version("6.0"):
                raise RuntimeError(
                    f"{PACKAGE_NAME} is only supported on ROCm 6.0 and above.  "
                    "Note: make sure HIP has a supported version by running hipcc --version."
                )
            if hip_version == Version("6.0"):
                warnings.warn(
                    f"{PACKAGE_NAME} requires a patch to be applied when running on ROCm 6.0. "
                    "Refer to the README.md for detailed instructions.",
                    UserWarning
                )

        cc_flag.append("-DBUILD_PYTHON_PACKAGE")

    else:
        check_if_cuda_home_none(PACKAGE_NAME)
        # Check, if CUDA11 is installed for compute capability 8.0

        if CUDA_HOME is not None:
            _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
            if bare_metal_version < Version("11.6"):
                raise RuntimeError(
                    f"{PACKAGE_NAME} is only supported on CUDA 11.6 and above.  "
                    "Note: make sure nvcc has a supported version by running nvcc -V."
                )

        # If system CUDA and PyTorch CUDA have different major versions,
        # clear TORCH_CUDA_ARCH_LIST to prevent cpp_extension from erroring
        torch_cuda_version = parse(torch.version.cuda)
        if bare_metal_version.major != torch_cuda_version.major:
            os.environ["TORCH_CUDA_ARCH_LIST"] = ""

        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_75,code=sm_75")
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_80,code=sm_80")
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_87,code=sm_87")
        if bare_metal_version >= Version("11.8"):
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_90,code=sm_90")
        if bare_metal_version >= Version("12.8"):
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_100,code=sm_100")
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_120,code=sm_120")
        if bare_metal_version >= Version("13.0"):
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_103,code=sm_103")
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_110,code=sm_110")
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_121,code=sm_121")


    # HACK: The compiler flag -D_GLIBCXX_USE_CXX11_ABI is set to be the same as
    # torch._C._GLIBCXX_USE_CXX11_ABI
    # https://github.com/pytorch/pytorch/blob/8472c24e3b5b60150096486616d98b7bea01500b/torch/utils/cpp_extension.py#L920
    if FORCE_CXX11_ABI:
        torch._C._GLIBCXX_USE_CXX11_ABI = True

    if HIP_BUILD:

        extra_compile_args = {
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": [
                "-O3",
                "-std=c++17",
                f"--offload-arch={os.getenv('HIP_ARCHITECTURES', 'native')}",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-fgpu-flush-denormals-to-zero",
            ]
            + cc_flag,
        }
    else:
        extra_compile_args = {
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": append_nvcc_threads(
                [
                    "-O3",
                    "-std=c++17",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",
                    "--ptxas-options=-v",
                    "-lineinfo",
                ]
                + cc_flag
            ),
        }

    ext_modules.append(
        CUDAExtension(
            name="selective_scan_cuda",
            sources=[
                "csrc/selective_scan/selective_scan.cpp",
                "csrc/selective_scan/selective_scan_fwd_fp32.cu",
                "csrc/selective_scan/selective_scan_fwd_fp16.cu",
                "csrc/selective_scan/selective_scan_fwd_bf16.cu",
                "csrc/selective_scan/selective_scan_bwd_fp32_real.cu",
                "csrc/selective_scan/selective_scan_bwd_fp32_complex.cu",
                "csrc/selective_scan/selective_scan_bwd_fp16_real.cu",
                "csrc/selective_scan/selective_scan_bwd_fp16_complex.cu",
                "csrc/selective_scan/selective_scan_bwd_bf16_real.cu",
                "csrc/selective_scan/selective_scan_bwd_bf16_complex.cu",
            ],
            extra_compile_args=extra_compile_args,
            include_dirs=[Path(this_dir) / "csrc" / "selective_scan"],
        )
    )


def get_package_version():
    with open(Path(this_dir) / PACKAGE_NAME / "__init__.py", "r") as f:
        version_match = re.search(r"^__version__\s*=\s*(.*)$", f.read(), re.MULTILINE)
    public_version = ast.literal_eval(version_match.group(1))
    local_version = os.environ.get("MAMBA_LOCAL_VERSION")
    if local_version:
        return f"{public_version}+{local_version}"
    else:
        return str(public_version)


# Always build this shared research source; upstream wheels omit local changes.

setup(
    name=PACKAGE_NAME,
    version=get_package_version(),
    packages=find_packages(include=("mamba_ssm", "mamba_ssm.*")),
    package_data={
        "mamba_ssm": ["SHARED_CORE.md", "paper_kernel_policy.json"],
        "mamba_ssm.ops.interventions": ["*.diff", "*.json"],
    },
    author="Tri Dao, Albert Gu",
    author_email="tri@tridao.me, agu@cs.cmu.edu",
    description="Mamba state-space model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/state-spaces/mamba",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: Unix",
    ],
    ext_modules=ext_modules,
    cmdclass={"bdist_wheel": _bdist_wheel, "build_ext": BuildExtension}
    if ext_modules
    else {
        "bdist_wheel": _bdist_wheel,
    },
    python_requires=">=3.10",
)
