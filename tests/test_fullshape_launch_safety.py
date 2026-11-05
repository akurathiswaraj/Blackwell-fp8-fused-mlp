import os

# Must be set before importing torch.
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
from torch.utils.cpp_extension import load

from integration.build_extensions import (
    COMMON,
    PROJECT,
    build_extensions,
)
from integration.fp8_mlp import cublaslt_mlp


FP8 = torch.float8_e4m3fn

M = 64
K = 8192
N = 28672


def synchronize_and_check(name, output):
    torch.cuda.synchronize()

    finite = bool(
        torch.isfinite(
            output.float()
        ).all().item()
    )

    if not finite:
        raise RuntimeError(
            f"{name} produced non-finite output"
        )

    print(
        f"{name}: Passed",
        f"shape={tuple(output.shape)}",
        f"dtype={output.dtype}",
        flush=True,
    )


def main():
    if torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("SM100 GPU required")

    print(
        "GPU:",
        torch.cuda.get_device_name(),
        flush=True,
    )

    original, _, down = build_extensions()

    variant_build = (
        PROJECT
        / "build"
        / "torch_extension_swiglu_warp_amax_v1"
    )

    variant_build.mkdir(
        parents=True,
        exist_ok=True,
    )

    variant = load(
        name="sm100_fp8_swiglu_warp_amax_ext_v1",
        sources=[
            str(
                PROJECT
                / "integration"
                / "sm100_fp8_swiglu_direct_warp_amax_torch.cu"
            )
        ],
        build_directory=str(variant_build),
        **COMMON,
    )

    torch.manual_seed(20261105)

    print(
        "Creating full Llama weights...",
        flush=True,
    )

    gate_up_weight = (
        torch.randn(
            2 * N,
            K,
            device="cuda",
            dtype=torch.float16,
        )
        * 32
    ).to(FP8)

    down_weight = (
        torch.randn(
            K,
            N,
            device="cuda",
            dtype=torch.float16,
        )
        * 32
    ).to(FP8)

    activation = (
        torch.randn(
            M,
            K,
            device="cuda",
            dtype=torch.float16,
        )
        * 32
    ).to(FP8)

    torch.cuda.synchronize()

    gemm_scale = 1.0e-5
    quant_multiplier = 45.69757799928416
    down_alpha = 1.0e-2 / quant_multiplier

    original_amax = torch.zeros(
        1,
        device="cuda",
        dtype=torch.float32,
    )

    variant_amax = torch.zeros_like(
        original_amax
    )

    print(
        "Testing original gate/up...",
        flush=True,
    )

    original_product = None

    for iteration in range(5):
        original_product = (
            original.swiglu_direct(
                activation,
                gate_up_weight,
                gemm_scale,
                quant_multiplier,
                original_amax,
            )[0]
        )

        synchronize_and_check(
            f"original gate/up iteration {iteration + 1}",
            original_product,
        )

    print(
        "Testing warp-amax gate/up...",
        flush=True,
    )

    variant_product = None

    for iteration in range(5):
        variant_product = (
            variant.swiglu_direct(
                activation,
                gate_up_weight,
                gemm_scale,
                quant_multiplier,
                variant_amax,
            )[0]
        )

        synchronize_and_check(
            f"variant gate/up iteration {iteration + 1}",
            variant_product,
        )

    if not torch.equal(
        original_product,
        variant_product,
    ):
        raise RuntimeError(
            "Variant differs from original"
        )

    if not torch.equal(
        original_amax,
        variant_amax,
    ):
        raise RuntimeError(
            "Variant amax differs from original"
        )

    print(
        "Original/variant exact comparison: Passed",
        flush=True,
    )

    print(
        "Testing down projection...",
        flush=True,
    )

    for iteration in range(5):
        down_output = down.gemm(
            variant_product,
            down_weight,
            down_alpha,
        )

        synchronize_and_check(
            f"down projection iteration {iteration + 1}",
            down_output,
        )

    print(
        "Testing cuBLASLt pipeline...",
        flush=True,
    )

    scale_a = torch.tensor(
        1.0e-3,
        device="cuda",
        dtype=torch.float32,
    )

    scale_gate_up = torch.tensor(
        1.0e-2,
        device="cuda",
        dtype=torch.float32,
    )

    scale_down = torch.tensor(
        1.0e-2,
        device="cuda",
        dtype=torch.float32,
    )

    baseline_amax = torch.zeros_like(
        original_amax
    )

    for iteration in range(5):
        baseline_output = cublaslt_mlp(
            activation,
            gate_up_weight,
            down_weight,
            scale_a,
            scale_gate_up,
            scale_down,
            quant_multiplier,
            baseline_amax,
        )

        synchronize_and_check(
            f"cuBLASLt pipeline iteration {iteration + 1}",
            baseline_output,
        )

    print()
    print(
        "FULL-SHAPE LAUNCH-SAFETY TEST PASSED",
        flush=True,
    )


if __name__ == "__main__":
    main()