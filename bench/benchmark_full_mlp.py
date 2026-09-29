"""Reproduce paired cold-L2 full-MLP measurements on an SM100 GPU.

Run from the repository root after setting CUTLASS_PATH and CUDA_HOME.
New files use a reproduced_ prefix and do not overwrite release data.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics
import sys

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from integration.build_extensions import build_extensions
from integration.fp8_mlp import (
    cublaslt_mlp,
    custom_decode_mlp,
    interleave_gate_up_weight,
)


HIDDEN = 8192
INTERMEDIATE = 28672
FP8 = torch.float8_e4m3fn


def relative_l2(actual, reference):
    numerator = torch.linalg.vector_norm(
        actual.float() - reference.float()
    )
    denominator = torch.linalg.vector_norm(
        reference.float()
    ).clamp_min(1e-12)
    return float((numerator / denominator).item())


def flush_l2(buffer):
    buffer.zero_()
    torch.cuda.synchronize()


def time_once(function, flush_buffer):
    flush_l2(flush_buffer)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = function()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)), output


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--m",
        type=int,
        nargs="+",
        default=[1, 4, 16, 64],
    )
    parser.add_argument("--batches", type=int, default=7)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--flush-mib", type=int, default=512)
    parser.add_argument(
        "--output-prefix",
        default="reproduced_full_mlp_direct",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    if torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("NVIDIA SM100 GPU required")

    direct, _, down = build_extensions()
    torch.manual_seed(20260928)

    blocked_gate_up = (
        torch.randn(
            2 * INTERMEDIATE,
            HIDDEN,
            device="cuda",
            dtype=torch.float16,
        )
        * 32
    ).to(FP8)
    interleaved_gate_up = interleave_gate_up_weight(
        blocked_gate_up
    )
    del blocked_gate_up

    down_weight = (
        torch.randn(
            HIDDEN,
            INTERMEDIATE,
            device="cuda",
            dtype=torch.float16,
        )
        * 32
    ).to(FP8)

    scale_a_value = 1.0e-3
    scale_gate_up_value = 1.0e-2
    scale_down_value = 1.0e-2
    quant_multiplier = 45.69757799928416

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

    gate_up_alpha = scale_a_value * scale_gate_up_value
    down_alpha = scale_down_value / quant_multiplier

    flush_buffer = torch.empty(
        args.flush_mib * 1024 * 1024,
        device="cuda",
        dtype=torch.uint8,
    )

    raw_rows = []
    batch_rows = []
    summary_rows = []

    for m_value in args.m:
        activation = (
            torch.randn(
                m_value,
                HIDDEN,
                device="cuda",
                dtype=torch.float16,
            )
            * 32
        ).to(FP8)

        direct_amax = torch.zeros(
            1,
            device="cuda",
            dtype=torch.float32,
        )
        baseline_amax = torch.zeros_like(direct_amax)

        def run_direct():
            return custom_decode_mlp(
                activation,
                interleaved_gate_up,
                down_weight,
                gate_up_alpha,
                quant_multiplier,
                down_alpha,
                direct_amax,
                direct,
                down,
            )

        def run_baseline():
            return cublaslt_mlp(
                activation,
                interleaved_gate_up,
                down_weight,
                scale_a,
                scale_gate_up,
                scale_down,
                quant_multiplier,
                baseline_amax,
            )

        for _ in range(args.warmup):
            run_direct()
            run_baseline()
        torch.cuda.synchronize()

        direct_check = run_direct()
        baseline_check = run_baseline()
        torch.cuda.synchronize()
        error = relative_l2(direct_check, baseline_check)
        if error > 4.0e-2:
            raise RuntimeError(
                f"M={m_value} correctness failure: {error}"
            )

        pooled_direct = []
        pooled_baseline = []
        batch_speedups = []

        for batch in range(1, args.batches + 1):
            batch_direct = []
            batch_baseline = []

            for round_index in range(1, args.rounds + 1):
                direct_first = (
                    (batch + round_index) % 2 == 0
                )

                if direct_first:
                    direct_ms, _ = time_once(
                        run_direct,
                        flush_buffer,
                    )
                    baseline_ms, _ = time_once(
                        run_baseline,
                        flush_buffer,
                    )
                    order = "direct_first"
                else:
                    baseline_ms, _ = time_once(
                        run_baseline,
                        flush_buffer,
                    )
                    direct_ms, _ = time_once(
                        run_direct,
                        flush_buffer,
                    )
                    order = "baseline_first"

                batch_direct.append(direct_ms)
                batch_baseline.append(baseline_ms)
                pooled_direct.append(direct_ms)
                pooled_baseline.append(baseline_ms)
                raw_rows.append(
                    {
                        "M": m_value,
                        "batch": batch,
                        "round": round_index,
                        "order": order,
                        "direct_ms": direct_ms,
                        "baseline_ms": baseline_ms,
                    }
                )

            direct_median = statistics.median(batch_direct)
            baseline_median = statistics.median(
                batch_baseline
            )
            speedup = baseline_median / direct_median
            batch_speedups.append(speedup)
            batch_rows.append(
                {
                    "M": m_value,
                    "batch": batch,
                    "direct_ms": direct_median,
                    "baseline_ms": baseline_median,
                    "speedup": speedup,
                }
            )

        direct_median = statistics.median(pooled_direct)
        baseline_median = statistics.median(pooled_baseline)
        speedup = baseline_median / direct_median
        summary_rows.append(
            {
                "M": m_value,
                "direct_ms": direct_median,
                "baseline_ms": baseline_median,
                "speedup": speedup,
                "latency_reduction_percent": (
                    100.0
                    * (1.0 - direct_median / baseline_median)
                ),
                "batch_speedup_min": min(batch_speedups),
                "batch_speedup_max": max(batch_speedups),
                "batches_won": sum(
                    value > 1.0
                    for value in batch_speedups
                ),
                "batches": args.batches,
                "rounds_per_batch": args.rounds,
                "relative_l2_vs_baseline": error,
            }
        )

        print(
            f"M={m_value:4d} direct={direct_median:.6f} ms "
            f"baseline={baseline_median:.6f} ms "
            f"speedup={speedup:.3f}x "
            f"rel_l2={error:.3e}"
        )

    prefix = PROJECT / "bench" / args.output_prefix
    write_csv(
        prefix.with_name(prefix.name + "_raw.csv"),
        raw_rows,
        [
            "M",
            "batch",
            "round",
            "order",
            "direct_ms",
            "baseline_ms",
        ],
    )
    write_csv(
        prefix.with_name(prefix.name + "_batches.csv"),
        batch_rows,
        [
            "M",
            "batch",
            "direct_ms",
            "baseline_ms",
            "speedup",
        ],
    )
    write_csv(
        prefix.with_suffix(".csv"),
        summary_rows,
        list(summary_rows[0]),
    )


if __name__ == "__main__":
    main()
