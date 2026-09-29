import torch

from integration.build_extensions import (
    build_extensions,
)
from integration.llama_mlp import (
    BlackwellFP8LlamaMLP,
)

def relative_l2(actual, reference):
    return (
        torch.linalg.vector_norm(
            actual.float() - reference.float()
        )
        / torch.linalg.vector_norm(
            reference.float()
        ).clamp_min(1e-12)
    ).item()

def main():
    if torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("SM100 GPU required")

    direct, fallback, down = build_extensions()

    torch.manual_seed(4321)

    batch = 2
    sequence = 2
    hidden = 256
    intermediate = 256
    FP8 = torch.float8_e4m3fn

    blocked_gate_up = (
        torch.randn(
            2 * intermediate,
            hidden,
            device="cuda",
            dtype=torch.float16,
        )
        * 32
    ).to(FP8)

    down_weight = (
        torch.randn(
            hidden,
            intermediate,
            device="cuda",
            dtype=torch.float16,
        )
        * 32
    ).to(FP8)

    scale_gate_up = 1e-2
    scale_down = 1e-2
    quant_multiplier = 32.0

    module = BlackwellFP8LlamaMLP(
        blocked_gate_up,
        down_weight,
        scale_gate_up,
        scale_down,
        quant_multiplier,
        direct,
        down,
    )

    activation = torch.randn(
        batch,
        sequence,
        hidden,
        device="cuda",
        dtype=torch.bfloat16,
    )

    output = module(activation)
    torch.cuda.synchronize()

    assert output.shape == activation.shape
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output.float()).all()
    assert module.running_amax.item() > 0

    # Validate the module's quantized computation independently.
    scale_a = (
        activation.float().abs().amax()
        / 448.0
    ).clamp_min(
        torch.finfo(torch.float32).tiny
    )

    activation_fp8 = torch.clamp(
        activation.float() / scale_a,
        -448,
        448,
    ).to(FP8)

    gate_weight, up_weight = (
        blocked_gate_up.chunk(2, dim=0)
    )

    flat = activation_fp8.reshape(
        -1, hidden
    )

    gate = (
        flat.float() * scale_a
    ) @ (
        gate_weight.float()
        * scale_gate_up
    ).t()

    up = (
        flat.float() * scale_a
    ) @ (
        up_weight.float()
        * scale_gate_up
    ).t()

    product = torch.nn.functional.silu(
        gate
    ) * up

    product_fp8 = torch.clamp(
        product * quant_multiplier,
        -448,
        448,
    ).to(FP8)

    reference = (
        product_fp8.float()
        / quant_multiplier
    ) @ (
        down_weight.float()
        * scale_down
    ).t()

    reference = reference.reshape_as(output)

    error = relative_l2(output, reference)

    assert error < 4e-2, error

    print(
        "Llama-compatible module test passed:",
        f"shape={tuple(output.shape)}",
        f"rel_l2={error:.6e}",
    )

if __name__ == "__main__":
    main()
