import gc

import torch
import torch.nn.functional as F

from integration.build_extensions import build_extensions
from integration.fp8_mlp import (
    custom_decode_mlp,
    interleave_gate_up_weight,
)


FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0


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

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(20261105)

    direct, _, down = build_extensions()

    M = 16
    K = 256
    N = 256

    scale_a = 1.0e-3
    scale_weight = 1.0e-2
    scale_down = 1.0e-2

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

    down_weight = (
        torch.randn(
            K,
            N,
            device="cuda",
            dtype=torch.float16,
        )
        * 32
    ).to(FP8)

    gate_weight, up_weight = blocked_weight.chunk(
        2,
        dim=0,
    )

    small_activation = (
        torch.randn(
            M,
            K,
            device="cuda",
            dtype=torch.float16,
        )
        * 8
    ).to(FP8)

    large_activation = (
        torch.randn(
            M,
            K,
            device="cuda",
            dtype=torch.float16,
        )
        * 32
    ).to(FP8)

    def reference(activation, quant_multiplier):
        gate = (
            activation.float() * scale_a
        ) @ (
            gate_weight.float()
            * scale_weight
        ).t()

        up = (
            activation.float() * scale_a
        ) @ (
            up_weight.float()
            * scale_weight
        ).t()

        product = F.silu(gate) * up
        amax = product.abs().amax()

        product_fp8 = torch.clamp(
            product * quant_multiplier,
            -FP8_MAX,
            FP8_MAX,
        ).to(FP8)

        output = (
            product_fp8.float()
            / quant_multiplier
        ) @ (
            down_weight.float()
            * scale_down
        ).t()

        return output, amax, product_fp8

    _, large_reference_amax, _ = reference(
        large_activation,
        1.0,
    )

    quant_multiplier = float(
        (
            FP8_MAX
            / large_reference_amax.clamp_min(1.0e-12)
        ).item()
    )

    def launch(
        activation,
        running_amax,
        multiplier=quant_multiplier,
    ):
        return custom_decode_mlp(
            activation,
            interleaved_weight,
            down_weight,
            scale_a * scale_weight,
            multiplier,
            scale_down / multiplier,
            running_amax,
            direct,
            down,
        )

    # -------------------------------------------------
    # 1. Running-amax accumulation and reset behavior
    # -------------------------------------------------

    _, small_reference_amax, _ = reference(
        small_activation,
        quant_multiplier,
    )

    _, large_reference_amax, _ = reference(
        large_activation,
        quant_multiplier,
    )

    running_amax = torch.zeros(
        1,
        device="cuda",
        dtype=torch.float32,
    )

    launch(small_activation, running_amax)
    launch(large_activation, running_amax)
    torch.cuda.synchronize()

    expected_amax = torch.maximum(
        small_reference_amax,
        large_reference_amax,
    )

    accumulation_error = float(
        (
            (running_amax[0] - expected_amax).abs()
            / expected_amax.clamp_min(1.0e-12)
        ).item()
    )

    if accumulation_error >= 2.0e-2:
        raise RuntimeError(
            "Running-amax accumulation failed"
        )

    sentinel = float(expected_amax.item() * 2.0)
    running_amax.fill_(sentinel)

    launch(small_activation, running_amax)
    torch.cuda.synchronize()

    sentinel_preserved = (
        float(running_amax.item()) == sentinel
    )

    if not sentinel_preserved:
        raise RuntimeError(
            "Running amax incorrectly decreased"
        )

    running_amax.zero_()

    if float(running_amax.item()) != 0.0:
        raise RuntimeError("Amax reset failed")

    print(
        "Amax accumulation/reset: Passed",
        f"error={accumulation_error:.6e}",
    )

    # -------------------------------------------------
    # 2. Deliberate FP8 saturation
    # -------------------------------------------------

    saturated_multiplier = quant_multiplier * 4.0

    saturated_reference, _, reference_fp8 = reference(
        large_activation,
        saturated_multiplier,
    )

    saturation_amax = torch.zeros(
        1,
        device="cuda",
        dtype=torch.float32,
    )

    saturated_output = launch(
        large_activation,
        saturation_amax,
        saturated_multiplier,
    )

    torch.cuda.synchronize()

    saturation_error = relative_l2(
        saturated_output,
        saturated_reference,
    )

    saturation_fraction = float(
        (
            reference_fp8.float().abs()
            == FP8_MAX
        ).float().mean().item()
    )

    saturation_finite = bool(
        torch.isfinite(
            saturated_output.float()
        ).all().item()
    )

    if saturation_fraction <= 0.0:
        raise RuntimeError(
            "Saturation test did not saturate values"
        )

    if saturation_error >= 4.0e-2:
        raise RuntimeError(
            "Saturated output correctness failed"
        )

    if not saturation_finite:
        raise RuntimeError(
            "Saturated output contains non-finite values"
        )

    print(
        "FP8 saturation: Passed",
        f"fraction={saturation_fraction:.6f}",
        f"rel_l2={saturation_error:.6e}",
    )

    # -------------------------------------------------
    # 3. Memory stability over repeated execution
    # -------------------------------------------------

    memory_amax = torch.zeros(
        1,
        device="cuda",
        dtype=torch.float32,
    )

    for _ in range(10):
        temporary_output = launch(
            large_activation,
            memory_amax,
        )

    del temporary_output
    torch.cuda.synchronize()
    gc.collect()

    allocated_before = torch.cuda.memory_allocated()

    for _ in range(200):
        temporary_output = launch(
            large_activation,
            memory_amax,
        )

    del temporary_output
    torch.cuda.synchronize()
    gc.collect()

    allocated_after = torch.cuda.memory_allocated()
    allocation_growth = (
        allocated_after - allocated_before
    )

    if allocation_growth > 2 * 1024 * 1024:
        raise RuntimeError(
            "Possible memory leak: "
            f"{allocation_growth} bytes"
        )

    print(
        "Memory stability: Passed",
        f"growth={allocation_growth} bytes",
    )

    # -------------------------------------------------
    # 4. CUDA Graph capture and replay
    # -------------------------------------------------

    graph_amax = torch.zeros(
        1,
        device="cuda",
        dtype=torch.float32,
    )

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(
        torch.cuda.current_stream()
    )

    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            warmup_output = launch(
                large_activation,
                graph_amax,
            )

    torch.cuda.current_stream().wait_stream(
        warmup_stream
    )
    torch.cuda.synchronize()

    graph_amax.zero_()

    eager_output = launch(
        large_activation,
        graph_amax,
    )

    torch.cuda.synchronize()
    graph_amax.zero_()

    graph = torch.cuda.CUDAGraph()

    with torch.cuda.graph(graph):
        graph_output = launch(
            large_activation,
            graph_amax,
        )

    graph.replay()
    torch.cuda.synchronize()

    graph_error = relative_l2(
        graph_output,
        eager_output,
    )

    graph_exact = torch.equal(
        graph_output,
        eager_output,
    )

    if graph_error != 0.0 or not graph_exact:
        raise RuntimeError(
            "CUDA Graph replay differs from eager output"
        )

    print(
        "CUDA Graph capture/replay: Passed",
        f"rel_l2={graph_error:.6e}",
        f"exact={graph_exact}",
    )

    print()
    print("RUNTIME ROBUSTNESS TEST PASSED")


if __name__ == "__main__":
    main()