from pathlib import Path
import os
from torch.utils.cpp_extension import CUDA_HOME, load

PROJECT = Path(__file__).resolve().parents[1]

CUTLASS_ROOT = Path(
    os.environ.get(
        "CUTLASS_PATH",
        PROJECT.parent / "cutlass",
    )
).expanduser().resolve()

CUDA_ROOT = Path(
    os.environ.get("CUDA_HOME")
    or CUDA_HOME
    or "/usr/local/cuda"
).expanduser().resolve()

if not (CUTLASS_ROOT / "include/cutlass/cutlass.h").exists():
    raise RuntimeError(
        "CUTLASS was not found. Set CUTLASS_PATH to the pinned "
        "CUTLASS 3.9.2 checkout."
    )

if not (CUDA_ROOT / "include/cuda_runtime.h").exists():
    raise RuntimeError(
        "CUDA development files were not found. Set CUDA_HOME."
    )

COMMON = dict(
    extra_include_paths=[
        str(CUTLASS_ROOT / "include"),
        str(CUTLASS_ROOT / "tools/util/include"),
        str(CUTLASS_ROOT / "examples/common"),
    ],
    extra_cflags=["-O3", "-std=c++17"],
    extra_cuda_cflags=[
        "-O3",
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "-ftemplate-backtrace-limit=0",
        "-DCUTLASS_ENABLE_GDC_FOR_SM100",
        "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
        "-Xcompiler=-fno-strict-aliasing",
    ],
    extra_ldflags=[
        f"-L{CUDA_ROOT / 'lib64/stubs'}",
        "-lcuda",
    ],
    with_cuda=True,
    verbose=False,
)

def build_extensions():
    os.environ.setdefault("MAX_JOBS", "8")
    os.environ.setdefault(
        "TORCH_CUDA_ARCH_LIST", "10.0a"
    )

    direct_build = (
        PROJECT / "build"
        / "torch_extension_swiglu_direct_release_v1"
    )
    fallback_build = (
        PROJECT / "build"
        / "torch_extension_swiglu_interleaved_fallback_v1"
    )
    down_build = (
        PROJECT / "build"
        / "torch_extension_down_n64_v1"
    )

    for directory in [
        direct_build,
        fallback_build,
        down_build,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    direct = load(
        name="sm100_fp8_swiglu_direct_release_ext_v1",
        sources=[
            str(
                PROJECT / "integration"
                / "sm100_fp8_swiglu_direct_torch.cu"
            )
        ],
        build_directory=str(direct_build),
        **COMMON,
    )

    fallback = load(
        name="sm100_fp8_swiglu_interleaved_fallback_ext_v1",
        sources=[
            str(
                PROJECT / "integration"
                / "sm100_fp8_gate_up_swiglu_interleaved_torch.cu"
            )
        ],
        build_directory=str(fallback_build),
        **COMMON,
    )

    down = load(
        name="sm100_fp8_down_n64_ext_v1",
        sources=[
            str(
                PROJECT / "integration"
                / "sm100_fp8_gemm_torch_n64.cu"
            )
        ],
        build_directory=str(down_build),
        **COMMON,
    )

    return direct, fallback, down
