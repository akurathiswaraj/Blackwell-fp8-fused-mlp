import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]

README_PATH = PROJECT / "README.md"
DESIGN_PATH = PROJECT / "DESIGN.md"
RESUME_PATH = PROJECT / "RESUME.md"

OPTIMIZED_PATH = (
    PROJECT
    / "results"
    / "optimized_dispatch_decode.json"
)

PROFILE_PATH = (
    PROJECT
    / "results"
    / "warp_amax_ncu_m64.json"
)

FINAL_PATH = (
    PROJECT
    / "results"
    / "final_summary_optimized_dispatch.json"
)


def replace_section(
    text,
    start_heading,
    end_heading,
    replacement,
):
    start = text.find(start_heading)
    end = text.find(end_heading, start)

    if start == -1 or end == -1:
        raise RuntimeError(
            f"Could not locate section: {start_heading}"
        )

    return (
        text[:start]
        + replacement.rstrip()
        + "\n\n"
        + text[end:]
    )


optimized = json.loads(
    OPTIMIZED_PATH.read_text(encoding="utf-8")
)

profile = json.loads(
    PROFILE_PATH.read_text(encoding="utf-8")
)

rows = optimized["results"]
best = max(rows, key=lambda row: row["speedup"])

total_batches_won = sum(
    row["batches_won"] for row in rows
)

total_batches = sum(
    row["batches_total"] for row in rows
)

if total_batches_won != total_batches:
    raise RuntimeError(
        "Not every optimized-dispatch batch won"
    )

profile_factor = profile["derived"][
    "global_reduction_sector_factor"
]

traffic_decrease = profile["derived"][
    "global_reduction_traffic_decrease_percent"
]

if profile_factor != 32.0:
    raise RuntimeError(
        "Expected a 32x global-reduction reduction"
    )

final_summary = {
    "release": "optimized_warp_amax_dispatch",
    "hardware": optimized["hardware"],
    "driver_version": optimized["driver_version"],
    "cuda_toolkit": optimized["cuda_toolkit"],
    "pytorch": optimized["pytorch"],
    "cutlass_commit": optimized["cutlass_commit"],
    "model_shape": {
        "hidden_size": optimized["hidden_size"],
        "intermediate_size": (
            optimized["intermediate_size"]
        ),
    },
    "dispatch": {
        "M_below_40": "original direct epilogue",
        "M_40_through_64": (
            "warp-reduced amax direct epilogue"
        ),
        "M_above_64": "cuBLASLt/PyTorch fallback",
    },
    "decode_results": rows,
    "best_decode_speedup": best["speedup"],
    "best_latency_reduction_percent": (
        best["latency_reduction_percent"]
    ),
    "best_shape_M": best["M"],
    "all_batches_won": True,
    "batches_won": total_batches_won,
    "batches_total": total_batches,
    "ncu_evidence": {
        "shape_M": 64,
        "global_reduction_factor": profile_factor,
        "global_reduction_traffic_decrease_percent": (
            traffic_decrease
        ),
        "clock_control": "none",
        "note": (
            "NCU timing is supporting evidence only. "
            "Repeated paired cold-L2 measurements are "
            "the authoritative latency results."
        ),
    },
    "earlier_release": {
        "driver_version": "580.95.05",
        "best_speedup": 1.373,
        "best_latency_reduction_percent": 27.1,
        "note": (
            "Do not compare absolute timings across "
            "different driver environments."
        ),
    },
}

FINAL_PATH.write_text(
    json.dumps(final_summary, indent=2) + "\n",
    encoding="utf-8",
)


# ---------------------------------------------------------
# README
# ---------------------------------------------------------

readme = README_PATH.read_text(encoding="utf-8")

optimized_heading = (
    "## Optimized dispatch results "
    "(driver 610.57.04)"
)

if optimized_heading not in readme:
    opening_start = readme.find(
        "On an NVIDIA B200,"
    )

    opening_end = readme.find(
        "\n\n## Final results",
        opening_start,
    )

    if opening_start == -1 or opening_end == -1:
        raise RuntimeError(
            "Could not locate README opening results"
        )

    new_opening = """On an NVIDIA B200 with driver 610.57.04, the optimized
shape-aware implementation reduced complete Llama 70B MLP decode latency by
up to **38.4% (1.623x)** against a cuBLASLt/PyTorch baseline. It won
**56/56 repeated batches** across
`M={1,4,16,32,40,48,56,64}`. The runtime selects the original direct
epilogue below M=40, the warp-reduced amax epilogue for M=40 through M=64,
and the cuBLASLt/PyTorch fallback above M=64."""

    readme = (
        readme[:opening_start]
        + new_opening
        + readme[opening_end:]
    )

    current_results = """## Optimized dispatch results (driver 610.57.04)

These are paired cold-L2 measurements using seven batches of 20 rounds per
shape. Absolute timings should only be compared within this environment.

| M | Selected kernel | Custom | cuBLASLt/PyTorch | Speedup | Reduction | Wins |
|---:|---|---:|---:|---:|---:|---:|
| 1 | original | 0.167264 ms | 0.220176 ms | 1.316x | 24.0% | 7/7 |
| 4 | original | 0.167328 ms | 0.221824 ms | 1.326x | 24.6% | 7/7 |
| 16 | original | 0.150368 ms | 0.229536 ms | 1.526x | 34.5% | 7/7 |
| 32 | original | 0.151184 ms | 0.233984 ms | 1.548x | 35.4% | 7/7 |
| 40 | warp-amax | 0.150128 ms | 0.235760 ms | 1.570x | 36.3% | 7/7 |
| 48 | warp-amax | 0.150688 ms | 0.239776 ms | 1.591x | 37.2% | 7/7 |
| 56 | warp-amax | 0.150896 ms | 0.242672 ms | 1.608x | 37.8% | 7/7 |
| 64 | warp-amax | 0.151040 ms | 0.245088 ms | 1.623x | 38.4% | 7/7 |

Relative L2 error remained approximately `1.6e-2`. The raw, batch-level, and
summary measurements are in `bench/optimized_dispatch_decode*.csv` and
`results/optimized_dispatch_decode.json`.

## Earlier direct-only release (driver 580.95.05)"""

    readme = readme.replace(
        "## Final results",
        current_results,
        1,
    )

    dispatch_old = (
        "The supported dispatcher uses the custom path "
        "for `M<=64` and cuBLASLt for\n"
        "`M>=512`. No prefill speedup is claimed."
    )

    dispatch_new = (
        "The supported dispatcher uses the original direct kernel for "
        "`M<40`, the warp-amax kernel for `40<=M<=64`, and the "
        "cuBLASLt/PyTorch fallback for `M>64`. No prefill speedup "
        "is claimed."
    )

    if dispatch_old in readme:
        readme = readme.replace(
            dispatch_old,
            dispatch_new,
            1,
        )

    ncu_section = """## Warp-amax profiler evidence

The original epilogue issued a global `atomicMax` while processing every
valid output. The optimized variant accumulates one maximum per thread,
performs a warp shuffle reduction, and lets lane zero issue one global
update per warp.

At `M=64`, Nsight Compute 2025.1.1 reported:

| Metric | Original | Warp-amax | Change |
|---|---:|---:|---:|
| Global reduction sectors | 57,344 | 1,792 | 32x fewer |
| Global reduction bytes | 1,835,008 | 57,344 | 32x fewer |
| Registers per thread | 97 | 99 | +2 |
| Profiled duration | 100.032 us | 89.888 us | 10.1% lower |

SM100 classified the unsigned `atomicMax` under the `op_red` counters rather
than `op_atom`. Global-reduction traffic decreased by **96.875%**. Because
Modal did not permit clock locking, the NCU duration is supporting evidence;
the paired cold-L2 benchmark remains the authoritative latency measurement.

"""

    environment_heading = "## Environment"

    if environment_heading not in readme:
        raise RuntimeError(
            "README Environment heading missing"
        )

    readme = readme.replace(
        environment_heading,
        ncu_section + environment_heading,
        1,
    )

    readme = readme.replace(
        "The supported source is\n"
        "`integration/sm100_fp8_swiglu_direct_torch.cu`.",
        "The supported sources are\n"
        "`integration/sm100_fp8_swiglu_direct_torch.cu` and\n"
        "`integration/sm100_fp8_swiglu_direct_warp_amax_torch.cu`.",
        1,
    )

README_PATH.write_text(
    readme,
    encoding="utf-8",
)


# ---------------------------------------------------------
# DESIGN
# ---------------------------------------------------------

design = DESIGN_PATH.read_text(encoding="utf-8")

if "## Warp-reduced amax optimization" not in design:
    dispatch_section = """## Dispatch policy

- `M<40`: original direct-store epilogue.
- `40<=M<=64`: warp-reduced amax epilogue.
- `M>64`: cuBLASLt/PyTorch fallback.

The threshold comes from a paired shape sweep. The warp-reduced variant
became consistently beneficial at M=40, while the original kernel remained
preferable for smaller decode shapes."""

    design = replace_section(
        design,
        "## Dispatch policy",
        "## Correctness",
        dispatch_section,
    )

    optimization_section = """## Warp-reduced amax optimization

The original direct epilogue updated the caller-owned running amax using a
global unsigned `atomicMax` while processing each valid output. The optimized
variant instead:

1. accumulates a local maximum per epilogue thread;
2. reduces those maxima with `__shfl_down_sync`;
3. lets lane zero perform one global update per warp.

The output tensor and running amax are bit-exact with the original kernel for
the tested decode shapes. At M=64, Nsight Compute measured global reduction
traffic falling from 57,344 to 1,792 sectors and from 1,835,008 to 57,344
bytes: exactly 32x fewer transactions and a 96.875% decrease.

The additional reduction logic increased register use from 97 to 99 registers
per thread. The runtime therefore uses this variant only for measured shapes
where it improves latency.

## Optimized performance on driver 610.57.04

The shape-aware dispatcher achieved 1.316x to 1.623x versus the matched
cuBLASLt/PyTorch pipeline across M={1,4,16,32,40,48,56,64}, winning all
56/56 repeated batches. The best result was 1.623x, or 38.4% lower latency,
at M=64.

These measurements are a separate environment from the earlier driver
580.95.05 release. Absolute timings are not compared across driver versions.

"""

    design = design.replace(
        "## Honest limitations",
        optimization_section + "## Honest limitations",
        1,
    )

DESIGN_PATH.write_text(
    design,
    encoding="utf-8",
)


# ---------------------------------------------------------
# Resume bullets
# ---------------------------------------------------------

resume = """# Résumé bullets

**Blackwell FP8 Fused MLP Kernels for Llama 3.1 70B** |
CUDA, CUTLASS/CuTe, PyTorch, Nsight Compute

- Built SM100-specific FP8 Llama 70B MLP kernels using TMA, `tcgen05` tensor-core MMA, TMEM, and a direct SwiGLU epilogue that fuses `SiLU(gate) * up`, running-amax tracking, and FP8 E4M3 requantization without materializing the BF16 gate/up intermediate.

- Added shape-aware warp-reduced amax dispatch, reducing measured full-MLP decode latency by up to **38.4% (1.623x)** versus a cuBLASLt/PyTorch baseline on NVIDIA B200 and winning **56/56** repeated batches; Nsight Compute measured **32x less global-reduction traffic** with approximately `1.6e-2` end-to-end relative L2 error.
"""

RESUME_PATH.write_text(
    resume,
    encoding="utf-8",
)

print("Updated:", README_PATH)
print("Updated:", DESIGN_PATH)
print("Created:", RESUME_PATH)
print("Created:", FINAL_PATH)
print(
    "Best optimized speedup:",
    f"{best['speedup']:.3f}x",
)
print(
    "Best latency reduction:",
    f"{best['latency_reduction_percent']:.1f}%",
)
print(
    "Repeated batches won:",
    f"{total_batches_won}/{total_batches}",
)
print(
    "NCU global-reduction improvement:",
    f"{profile_factor:.0f}x fewer",
)