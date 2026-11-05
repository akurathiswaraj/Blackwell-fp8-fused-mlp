import csv
import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]

BENCH_PATH = (
    PROJECT
    / "bench"
    / "warp_amax_ncu_m64.csv"
)

RESULT_PATH = (
    PROJECT
    / "results"
    / "warp_amax_ncu_m64.json"
)

ORIGINAL = {
    "mode": "original",
    "duration_ns": 100032,
    "global_reduction_bytes": 1835008,
    "global_reduction_sectors": 57344,
    "xbar_reduction_sectors": 57344,
    "registers_per_thread": 97,
}

WARP_AMAX = {
    "mode": "warp_amax",
    "duration_ns": 89888,
    "global_reduction_bytes": 57344,
    "global_reduction_sectors": 1792,
    "xbar_reduction_sectors": 1792,
    "registers_per_thread": 99,
}

sector_reduction_factor = (
    ORIGINAL["global_reduction_sectors"]
    / WARP_AMAX["global_reduction_sectors"]
)

byte_reduction_factor = (
    ORIGINAL["global_reduction_bytes"]
    / WARP_AMAX["global_reduction_bytes"]
)

traffic_reduction_percent = (
    1.0
    - WARP_AMAX["global_reduction_bytes"]
    / ORIGINAL["global_reduction_bytes"]
) * 100.0

profiled_speedup = (
    ORIGINAL["duration_ns"]
    / WARP_AMAX["duration_ns"]
)

profiled_latency_reduction_percent = (
    1.0
    - WARP_AMAX["duration_ns"]
    / ORIGINAL["duration_ns"]
) * 100.0

if sector_reduction_factor != 32.0:
    raise RuntimeError(
        "Expected exactly 32x sector reduction"
    )

if byte_reduction_factor != 32.0:
    raise RuntimeError(
        "Expected exactly 32x byte reduction"
    )

BENCH_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

RESULT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

fieldnames = list(ORIGINAL.keys())

with BENCH_PATH.open(
    "w",
    newline="",
    encoding="utf-8",
) as output_file:
    writer = csv.DictWriter(
        output_file,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerow(ORIGINAL)
    writer.writerow(WARP_AMAX)

result = {
    "created_at_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "hardware": "NVIDIA B200",
    "driver_version": "610.57.04",
    "cuda_toolkit": "12.8.93",
    "nsight_compute": "2025.1.1",
    "cutlass_commit": (
        "ad7b2f5e84fcfa124cb02b91d5bd26d238c0459e"
    ),
    "shape": {
        "M": 64,
        "N": 28672,
        "K": 8192,
    },
    "clock_control": "none",
    "original": ORIGINAL,
    "warp_amax": WARP_AMAX,
    "derived": {
        "global_reduction_sector_factor": (
            sector_reduction_factor
        ),
        "global_reduction_byte_factor": (
            byte_reduction_factor
        ),
        "global_reduction_traffic_decrease_percent": (
            traffic_reduction_percent
        ),
        "profiled_kernel_speedup": (
            profiled_speedup
        ),
        "profiled_kernel_latency_reduction_percent": (
            profiled_latency_reduction_percent
        ),
    },
    "interpretation": (
        "Warp-level amax reduction lowered L1TEX global "
        "reduction traffic by exactly 32x. Profiled "
        "duration is supporting evidence only because "
        "Modal did not permit GPU clock locking."
    ),
}

RESULT_PATH.write_text(
    json.dumps(result, indent=2) + "\n",
    encoding="utf-8",
)

print("Saved:", BENCH_PATH)
print("Saved:", RESULT_PATH)
print(
    "Global reduction traffic:",
    f"{sector_reduction_factor:.1f}x fewer",
)
print(
    "Traffic decrease:",
    f"{traffic_reduction_percent:.3f}%",
)
print(
    "Profiled kernel duration:",
    f"{profiled_speedup:.3f}x",
    f"({profiled_latency_reduction_percent:.1f}% lower)",
)