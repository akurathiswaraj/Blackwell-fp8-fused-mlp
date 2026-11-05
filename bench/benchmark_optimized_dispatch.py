from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import statistics
import subprocess

import torch

from integration.build_extensions import (
    PROJECT,
    build_optimized_extensions,
)
from integration.fp8_mlp import (
    WARP_AMAX_MIN_M,
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

M_VALUES = [
    1,
    4,
    16,
    32,
    40,
    48,
    56,
    64,
]


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


def write_csv(path, rows, fieldnames):
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def main():
    if torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("SM100 GPU required")

    (
        direct,
        warp_amax,
        _,
        down,
    ) = build_optimized_extensions()

    torch.manual_seed(20261105)

    print("Creating Llama 70B weights...")

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

    raw_rows = []
    batch_rows = []
    summary_rows = []

    for M in M_VALUES:
        selected_kernel = (
            "warp_amax"
            if WARP_AMAX_MIN_M <= M <= 64
            else "original"
        )

        print()
        print(
            f"Benchmarking M={M}, "
            f"kernel={selected_kernel}..."
        )

        activation = (
            torch.randn(
                M,
                K,
                device="cuda",
                dtype=torch.float16,
            )
            * 32
        ).to(FP8)

        custom_amax = torch.zeros(
            1,
            device="cuda",
            dtype=torch.float32,
        )

        baseline_amax = torch.zeros_like(
            custom_amax
        )

        def run_custom():
            return custom_decode_mlp(
                activation,
                gate_up_weight,
                down_weight,
                gate_up_alpha,
                quant_multiplier,
                down_alpha,
                custom_amax,
                direct,
                down,
                warp_amax_extension=warp_amax,
            )

        def run_baseline():
            return cublaslt_mlp(
                activation,
                gate_up_weight,
                down_weight,
                scale_a,
                scale_gate_up,
                scale_down,
                quant_multiplier,
                baseline_amax,
            )

        custom_check = run_custom()
        baseline_check = run_baseline()
        torch.cuda.synchronize()

        correctness_error = relative_l2(
            custom_check,
            baseline_check,
        )

        if correctness_error >= 4.0e-2:
            raise RuntimeError(
                f"Correctness failure at M={M}: "
                f"{correctness_error}"
            )

        for _ in range(WARMUPS):
            run_custom()
            run_baseline()

        torch.cuda.synchronize()

        pooled_custom = []
        pooled_baseline = []
        batch_speedups = []
        batches_won = 0

        for batch in range(1, BATCHES + 1):
            batch_custom = []
            batch_baseline = []

            for round_index in range(
                1,
                ROUNDS + 1,
            ):
                custom_first = (
                    (batch + round_index) % 2 == 0
                )

                if custom_first:
                    custom_ms, _ = time_once(
                        run_custom,
                        flush_buffer,
                    )

                    baseline_ms, _ = time_once(
                        run_baseline,
                        flush_buffer,
                    )

                    order = "custom_first"

                else:
                    baseline_ms, _ = time_once(
                        run_baseline,
                        flush_buffer,
                    )

                    custom_ms, _ = time_once(
                        run_custom,
                        flush_buffer,
                    )

                    order = "baseline_first"

                batch_custom.append(custom_ms)
                batch_baseline.append(baseline_ms)

                pooled_custom.append(custom_ms)
                pooled_baseline.append(baseline_ms)

                raw_rows.append(
                    {
                        "M": M,
                        "kernel": selected_kernel,
                        "batch": batch,
                        "round": round_index,
                        "order": order,
                        "custom_ms": custom_ms,
                        "baseline_ms": baseline_ms,
                    }
                )

            custom_median = statistics.median(
                batch_custom
            )

            baseline_median = statistics.median(
                batch_baseline
            )

            speedup = (
                baseline_median
                / custom_median
            )

            latency_reduction = (
                1.0
                - custom_median / baseline_median
            ) * 100.0

            won = speedup > 1.0

            if won:
                batches_won += 1

            batch_speedups.append(speedup)

            batch_rows.append(
                {
                    "M": M,
                    "kernel": selected_kernel,
                    "batch": batch,
                    "custom_median_ms": custom_median,
                    "baseline_median_ms": baseline_median,
                    "speedup": speedup,
                    "latency_reduction_percent": (
                        latency_reduction
                    ),
                    "won": won,
                }
            )

            print(
                f"  Batch {batch}: "
                f"custom={custom_median:.6f} ms | "
                f"baseline={baseline_median:.6f} ms | "
                f"speedup={speedup:.3f}x"
            )

        custom_median = statistics.median(
            pooled_custom
        )

        baseline_median = statistics.median(
            pooled_baseline
        )

        speedup = baseline_median / custom_median

        latency_reduction = (
            1.0
            - custom_median / baseline_median
        ) * 100.0

        summary_row = {
            "M": M,
            "kernel": selected_kernel,
            "custom_ms": custom_median,
            "baseline_ms": baseline_median,
            "speedup": speedup,
            "latency_reduction_percent": (
                latency_reduction
            ),
            "batch_speedup_min": min(
                batch_speedups
            ),
            "batch_speedup_max": max(
                batch_speedups
            ),
            "batches_won": batches_won,
            "batches_total": BATCHES,
            "relative_l2": correctness_error,
        }

        summary_rows.append(summary_row)

        print(
            f"  Pooled: custom={custom_median:.6f} ms | "
            f"baseline={baseline_median:.6f} ms | "
            f"speedup={speedup:.3f}x | "
            f"reduction={latency_reduction:.1f}% | "
            f"wins={batches_won}/{BATCHES}"
        )

    bench_directory = PROJECT / "bench"
    results_directory = PROJECT / "results"

    bench_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        bench_directory
        / "optimized_dispatch_decode.csv"
    )

    batches_path = (
        bench_directory
        / "optimized_dispatch_decode_batches.csv"
    )

    raw_path = (
        bench_directory
        / "optimized_dispatch_decode_raw.csv"
    )

    result_path = (
        results_directory
        / "optimized_dispatch_decode.json"
    )

    write_csv(
        summary_path,
        summary_rows,
        list(summary_rows[0].keys()),
    )

    write_csv(
        batches_path,
        batch_rows,
        list(batch_rows[0].keys()),
    )

    write_csv(
        raw_path,
        raw_rows,
        list(raw_rows[0].keys()),
    )

    driver_version = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()

    structured_result = {
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "hardware": "NVIDIA B200",
        "driver_version": driver_version,
        "cuda_toolkit": "12.8.93",
        "pytorch": torch.__version__,
        "cutlass_commit": (
            "ad7b2f5e84fcfa124cb02b91d5bd26d238c0459e"
        ),
        "hidden_size": K,
        "intermediate_size": N,
        "warp_amax_min_m": WARP_AMAX_MIN_M,
        "batches": BATCHES,
        "rounds_per_batch": ROUNDS,
        "l2_flush_bytes": FLUSH_BYTES,
        "results": summary_rows,
    }

    result_path.write_text(
        json.dumps(
            structured_result,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("OPTIMIZED DISPATCH SUMMARY")

    for row in summary_rows:
        print(
            f"M={row['M']:2d} | "
            f"{row['kernel']:10s} | "
            f"{row['speedup']:.3f}x | "
            f"{row['latency_reduction_percent']:.1f}% | "
            f"wins={row['batches_won']}/"
            f"{row['batches_total']} | "
            f"rel_l2={row['relative_l2']:.3e}"
        )

    print()
    print("Saved:", summary_path)
    print("Saved:", batches_path)
    print("Saved:", raw_path)
    print("Saved:", result_path)


if __name__ == "__main__":
    main()