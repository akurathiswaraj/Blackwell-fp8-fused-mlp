from pathlib import Path

import torch
from torch.utils.cpp_extension import load

from integration.build_extensions import (
    COMMON,
    PROJECT,
    build_extensions,
)
from integration.fp8_mlp import (
    interleave_gate_up_weight,
)


FP8 = torch.float8_e4m3fn


def relative_l2(actual, reference):
    numerator = torch.linalg.vector_norm(
        actual.float() - reference.float()
    )
    denominator = torch.linalg.vector_norm(
        reference.float()
    ).clamp_min(1.0e-12)

    return float((numerator / denominator).item())


def main():
    if torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("SM100 GPU required")

    baseline, _, _ = build_extensions()

    build_directory = (
        PROJECT
        / "build"
        / "torch_extension_swiglu_warp_amax_v1"
    )

    build_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_path = (
        PROJECT
        / "integration"
        / "sm100_fp8_swiglu_direct_warp_amax_torch.cu"
    )

    print("Compiling warp-amax variant...")

    variant = load(
        name="sm100_fp8_swiglu_warp_amax_ext_v1",
        sources=[str(source_path)],
        build_directory=str(build_directory),
        **COMMON,
    )

    print("Variant compiled successfully")

    torch.manual_seed(20261105)

    K = 256
    N = 256
    gemm_scale = 1.0e-5
    quant_multiplier = 32.0

    blocked_weight = (
        torch.randn(
            2 * N,
            K,
            device="cuda",
            dtype=torch.float16,
        )
        * 32
    ).to(FP8)

    interleaved_weight = (
        interleave_gate_up_weight(
            blocked_weight
        )
    )

    for M in [1, 4, 16, 64]:
        activation = (
            torch.randn(
                M,
                K,
                device="cuda",
                dtype=torch.float16,
            )
            * 32
        ).to(FP8)

        baseline_amax = torch.zeros(
            1,
            device="cuda",
            dtype=torch.float32,
        )

        variant_amax = torch.zeros_like(
            baseline_amax
        )

        baseline_output = baseline.swiglu_direct(
            activation,
            interleaved_weight,
            gemm_scale,
            quant_multiplier,
            baseline_amax,
        )[0]

        variant_output = variant.swiglu_direct(
            activation,
            interleaved_weight,
            gemm_scale,
            quant_multiplier,
            variant_amax,
        )[0]

        torch.cuda.synchronize()

        output_error = relative_l2(
            variant_output,
            baseline_output,
        )

        exact_fraction = float(
            (
                variant_output.float()
                == baseline_output.float()
            ).float().mean().item()
        )

        amax_error = float(
            (
                (variant_amax - baseline_amax).abs()
                / baseline_amax.clamp_min(1.0e-12)
            ).item()
        )

        passed = (
            output_error == 0.0
            and exact_fraction == 1.0
            and amax_error < 1.0e-6
        )

        print(
            f"M={M:2d} | "
            f"rel_l2={output_error:.6e} | "
            f"exact={exact_fraction:.6f} | "
            f"amax_error={amax_error:.6e} | "
            f"passed={passed}"
        )

        if not passed:
            raise RuntimeError(
                f"Warp-amax correctness failed at M={M}"
            )

    print()
    print("WARP-AMAX CORRECTNESS PASSED")


if __name__ == "__main__":
    main()