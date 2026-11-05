import statistics

import torch
from torch.utils.cpp_extension import load

from integration.build_extensions import (
    COMMON,
    PROJECT,
    build_extensions,
)
from integration.fp8_mlp import (
    cublaslt_mlp,
    custom_decode_mlp,
)


FP8 = torch.float8_e4m3fn

K = 8192
N = 28672

BATCHES = 7
ROUNDS = 20
WARMUPS = 10
FLUSH_BYTES = 512 * 1024 * 1024


def relative_l2(actual, reference):
    numerator = torch.linalg.vector_norm(
        actual.float() - reference.float()
    )
    denominator = torch.linalg.vector_norm(
        reference.float()
    ).clamp_min(1.0e-12)

    return float((numerator / denominator).item())


def flush_l2(buffer):
    buffer.zero_()
    torch.cuda.synchronize()


def time_once(function, flush_buffer):
    flush_l2(flush_buffer)

    start = torch.cuda.Event(
        enable_timing=True
    )
    end = torch.cuda.Event(
        enable_timing=True
    )

    start.record()
    output = function()
    end.record()
    end.synchronize()

    return float(
        start.elapsed_time(end)
    ), output


def main():
    if torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("SM100 GPU required")

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

    print("Creating Llama 70B weights...")

    interleaved_gate_up_weight = (
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

    scale_a_value = 1.0e-3
    scale_gate_up_value = 1.0e-2
    scale_down_value = 1.0e-2

    quant_multiplier = 45.69757799928416

    gate_up_alpha = (
        scale_a_value
        * scale_gate_up_value
    )

    down_alpha = (
        scale_down_value
        / quant_multiplier
    )

    scale_a = torch.tensor(
        scale_a_value,
        device="cuda",
        dtype=torch.float32,
    )

    scale_gate_up = torch.tensor(
        scale_gate_up_value,
        device="cuda",
        dtype=torch.float32,
    )

    scale_down = torch.tensor(
        scale_down_value,
        device="cuda",
        dtype=torch.float32,
    )

    flush_buffer = torch.empty(
        FLUSH_BYTES,
        device="cuda",
        dtype=torch.uint8,
    )

    summary = []

    for M in [1, 4, 16, 64]:
        print()
        print(f"Benchmarking complete MLP M={M}...")

        activation = (
            torch.randn(
                M,
                K,
                device="cuda",
                dtype=torch.float16,
            )
            * 32
        ).to(FP8)

        original_amax = torch.zeros(
            1,
            device="cuda",
            dtype=torch.float32,
        )

        variant_amax = torch.zeros_like(
            original_amax
        )

        baseline_amax = torch.zeros_like(
            original_amax
        )

        def run_original():
            return custom_decode_mlp(
                activation,
                interleaved_gate_up_weight,
                down_weight,
                gate_up_alpha,
                quant_multiplier,
                down_alpha,
                original_amax,
                original,
                down,
            )

        def run_variant():
            return custom_decode_mlp(
                activation,
                interleaved_gate_up_weight,
                down_weight,
                gate_up_alpha,
                quant_multiplier,
                down_alpha,
                variant_amax,
                variant,
                down,
            )

        def run_cublaslt():
            return cublaslt_mlp(
                activation,
                interleaved_gate_up_weight,
                down_weight,
                scale_a,
                scale_gate_up,
                scale_down,
                quant_multiplier,
                baseline_amax,
            )

        original_check = run_original()
        variant_check = run_variant()
        baseline_check = run_cublaslt()
        torch.cuda.synchronize()

        exact = torch.equal(
            original_check,
            variant_check,
        )

        baseline_error = relative_l2(
            variant_check,
            baseline_check,
        )

        if not exact:
            raise RuntimeError(
                f"Variant differs from original at M={M}"
            )

        if baseline_error >= 4.0e-2:
            raise RuntimeError(
                f"Baseline correctness failure at M={M}"
            )

        print(
            f"  correctness: exact={exact} | "
            f"vs_cuBLASLt={baseline_error:.6e}"
        )

        for _ in range(WARMUPS):
            run_original()
            run_variant()
            run_cublaslt()

        torch.cuda.synchronize()

        pooled_original = []
        pooled_variant = []
        pooled_cublaslt = []

        batch_variant_original = []
        batch_variant_cublaslt = []

        variant_original_wins = 0
        variant_cublaslt_wins = 0

        functions = {
            "original": run_original,
            "variant": run_variant,
            "cublaslt": run_cublaslt,
        }

        orders = [
            ["variant", "original", "cublaslt"],
            ["original", "cublaslt", "variant"],
            ["cublaslt", "variant", "original"],
        ]

        for batch in range(1, BATCHES + 1):
            batch_times = {
                "original": [],
                "variant": [],
                "cublaslt": [],
            }

            for round_index in range(ROUNDS):
                order = orders[
                    (batch + round_index)
                    % len(orders)
                ]

                for name in order:
                    elapsed, _ = time_once(
                        functions[name],
                        flush_buffer,
                    )

                    batch_times[name].append(
                        elapsed
                    )

            original_median = statistics.median(
                batch_times["original"]
            )

            variant_median = statistics.median(
                batch_times["variant"]
            )

            cublaslt_median = statistics.median(
                batch_times["cublaslt"]
            )

            versus_original = (
                original_median
                / variant_median
            )

            versus_cublaslt = (
                cublaslt_median
                / variant_median
            )

            batch_variant_original.append(
                versus_original
            )

            batch_variant_cublaslt.append(
                versus_cublaslt
            )

            if versus_original > 1.0:
                variant_original_wins += 1

            if versus_cublaslt > 1.0:
                variant_cublaslt_wins += 1

            pooled_original.extend(
                batch_times["original"]
            )

            pooled_variant.extend(
                batch_times["variant"]
            )

            pooled_cublaslt.extend(
                batch_times["cublaslt"]
            )

            print(
                f"  Batch {batch}: "
                f"variant={variant_median:.6f} ms | "
                f"original={original_median:.6f} ms | "
                f"cuBLASLt={cublaslt_median:.6f} ms | "
                f"v/orig={versus_original:.3f}x | "
                f"v/cuBLAS={versus_cublaslt:.3f}x"
            )

        original_median = statistics.median(
            pooled_original
        )

        variant_median = statistics.median(
            pooled_variant
        )

        cublaslt_median = statistics.median(
            pooled_cublaslt
        )

        versus_original = (
            original_median
            / variant_median
        )

        versus_cublaslt = (
            cublaslt_median
            / variant_median
        )

        latency_reduction = (
            1.0
            - variant_median / cublaslt_median
        ) * 100.0

        print(
            f"  Pooled: variant={variant_median:.6f} ms | "
            f"original={original_median:.6f} ms | "
            f"cuBLASLt={cublaslt_median:.6f} ms"
        )

        print(
            f"  Variant/original={versus_original:.3f}x | "
            f"variant/cuBLASLt={versus_cublaslt:.3f}x | "
            f"latency reduction={latency_reduction:.1f}%"
        )

        summary.append(
            {
                "M": M,
                "variant_ms": variant_median,
                "original_ms": original_median,
                "cublaslt_ms": cublaslt_median,
                "versus_original": versus_original,
                "versus_cublaslt": versus_cublaslt,
                "latency_reduction": latency_reduction,
                "original_wins": variant_original_wins,
                "cublaslt_wins": variant_cublaslt_wins,
                "original_min": min(
                    batch_variant_original
                ),
                "original_max": max(
                    batch_variant_original
                ),
            }
        )

    print()
    print("FULL-MLP WARP-AMAX SUMMARY")

    for row in summary:
        print(
            f"M={row['M']:2d} | "
            f"vs original={row['versus_original']:.3f}x | "
            f"vs cuBLASLt={row['versus_cublaslt']:.3f}x | "
            f"reduction={row['latency_reduction']:.1f}% | "
            f"original wins={row['original_wins']}/7 | "
            f"cuBLAS wins={row['cublaslt_wins']}/7"
        )


if __name__ == "__main__":
    main()