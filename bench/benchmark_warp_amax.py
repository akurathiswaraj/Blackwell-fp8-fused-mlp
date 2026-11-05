import statistics

import torch
from torch.utils.cpp_extension import load

from integration.build_extensions import (
    COMMON,
    PROJECT,
    build_extensions,
)


FP8 = torch.float8_e4m3fn
K = 8192
N = 28672

BATCHES = 7
ROUNDS = 20
WARMUPS = 10
FLUSH_BYTES = 512 * 1024 * 1024


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

    variant = load(
        name="sm100_fp8_swiglu_warp_amax_ext_v1",
        sources=[
            str(
                PROJECT
                / "integration"
                / "sm100_fp8_swiglu_direct_warp_amax_torch.cu"
            )
        ],
        build_directory=str(build_directory),
        **COMMON,
    )

    torch.manual_seed(20261105)

    print("Creating Llama 70B gate/up weights...")

    interleaved_weight = (
        torch.randn(
            2 * N,
            K,
            device="cuda",
            dtype=torch.float16,
        )
        * 32
    ).to(FP8)

    flush_buffer = torch.empty(
        FLUSH_BYTES,
        device="cuda",
        dtype=torch.uint8,
    )

    gemm_scale = 1.0e-5
    quant_multiplier = 45.69757799928416

    summary = []

    for M in [
    16,
    24,
    32,
    40,
    48,
    56,
    64,
]:
        print()
        print(f"Benchmarking M={M}...")

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

        def run_baseline():
            return baseline.swiglu_direct(
                activation,
                interleaved_weight,
                gemm_scale,
                quant_multiplier,
                baseline_amax,
            )[0]

        def run_variant():
            return variant.swiglu_direct(
                activation,
                interleaved_weight,
                gemm_scale,
                quant_multiplier,
                variant_amax,
            )[0]

        baseline_check = run_baseline()
        variant_check = run_variant()
        torch.cuda.synchronize()

        exact = torch.equal(
            baseline_check,
            variant_check,
        )

        amax_exact = torch.equal(
            baseline_amax,
            variant_amax,
        )

        if not exact or not amax_exact:
            raise RuntimeError(
                f"Correctness failure at M={M}"
            )

        for _ in range(WARMUPS):
            run_baseline()
            run_variant()

        torch.cuda.synchronize()

        pooled_baseline = []
        pooled_variant = []
        batch_speedups = []
        batches_won = 0

        for batch in range(1, BATCHES + 1):
            batch_baseline = []
            batch_variant = []

            for round_index in range(
                1,
                ROUNDS + 1,
            ):
                variant_first = (
                    (batch + round_index) % 2 == 0
                )

                if variant_first:
                    variant_ms, _ = time_once(
                        run_variant,
                        flush_buffer,
                    )
                    baseline_ms, _ = time_once(
                        run_baseline,
                        flush_buffer,
                    )
                else:
                    baseline_ms, _ = time_once(
                        run_baseline,
                        flush_buffer,
                    )
                    variant_ms, _ = time_once(
                        run_variant,
                        flush_buffer,
                    )

                batch_baseline.append(
                    baseline_ms
                )
                batch_variant.append(
                    variant_ms
                )

                pooled_baseline.append(
                    baseline_ms
                )
                pooled_variant.append(
                    variant_ms
                )

            baseline_median = statistics.median(
                batch_baseline
            )
            variant_median = statistics.median(
                batch_variant
            )

            speedup = (
                baseline_median
                / variant_median
            )

            batch_speedups.append(speedup)

            if speedup > 1.0:
                batches_won += 1

            print(
                f"  Batch {batch}: "
                f"baseline={baseline_median:.6f} ms | "
                f"variant={variant_median:.6f} ms | "
                f"speedup={speedup:.3f}x"
            )

        pooled_baseline_median = (
            statistics.median(
                pooled_baseline
            )
        )

        pooled_variant_median = (
            statistics.median(
                pooled_variant
            )
        )

        pooled_speedup = (
            pooled_baseline_median
            / pooled_variant_median
        )

        print(
            f"  Pooled: "
            f"baseline={pooled_baseline_median:.6f} ms | "
            f"variant={pooled_variant_median:.6f} ms | "
            f"speedup={pooled_speedup:.3f}x | "
            f"wins={batches_won}/{BATCHES}"
        )

        summary.append(
            {
                "M": M,
                "baseline_ms": (
                    pooled_baseline_median
                ),
                "variant_ms": (
                    pooled_variant_median
                ),
                "speedup": pooled_speedup,
                "minimum": min(
                    batch_speedups
                ),
                "maximum": max(
                    batch_speedups
                ),
                "wins": batches_won,
            }
        )

    print()
    print("WARP-AMAX BENCHMARK SUMMARY")

    for row in summary:
        print(
            f"M={row['M']:2d} | "
            f"speedup={row['speedup']:.3f}x | "
            f"range={row['minimum']:.3f}"
            f"–{row['maximum']:.3f}x | "
            f"wins={row['wins']}/{BATCHES}"
        )


if __name__ == "__main__":
    main()