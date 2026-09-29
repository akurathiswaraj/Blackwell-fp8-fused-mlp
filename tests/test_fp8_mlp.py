import torch
import torch.nn.functional as F

from integration.build_extensions import (
    build_extensions,
)
from integration.fp8_mlp import (
    custom_decode_mlp,
    dispatch_mlp,
    interleave_gate_up_weight,
    two_kernel_decode_mlp,
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
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    if torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("SM100 GPU required")

    direct, fallback, down = build_extensions()

    torch.manual_seed(1234)

    M = 4
    K = 256
    N = 256
    FP8 = torch.float8_e4m3fn

    a = (
        torch.randn(
            (M, K),
            device="cuda",
            dtype=torch.float16,
        )
        * 32
    ).to(FP8)

    blocked_weight = (
        torch.randn(
            (2 * N, K),
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
            (K, N),
            device="cuda",
            dtype=torch.float16,
        )
        * 32
    ).to(FP8)

    scale_a = 1e-3
    scale_gate_up = 1e-2
    scale_down = 1e-2

    gate_weight, up_weight = (
        blocked_weight.chunk(2, dim=0)
    )

    gate_reference = (
        a.float() * scale_a
    ) @ (
        gate_weight.float()
        * scale_gate_up
    ).t()

    up_reference = (
        a.float() * scale_a
    ) @ (
        up_weight.float()
        * scale_gate_up
    ).t()

    product = (
        F.silu(gate_reference)
        * up_reference
    )

    quant_multiplier = float(
        (
            448.0
            / product.abs().max().clamp_min(1e-12)
        ).item()
    )

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

    direct_amax = torch.zeros(
        1,
        device="cuda",
        dtype=torch.float32,
    )
    fallback_amax = torch.zeros_like(
        direct_amax
    )

    direct_output = custom_decode_mlp(
        a,
        interleaved_weight,
        down_weight,
        scale_a * scale_gate_up,
        quant_multiplier,
        (1.0 / quant_multiplier)
        * scale_down,
        direct_amax,
        direct,
        down,
    )

    fallback_output = two_kernel_decode_mlp(
        a,
        interleaved_weight,
        down_weight,
        scale_a * scale_gate_up,
        quant_multiplier,
        (1.0 / quant_multiplier)
        * scale_down,
        fallback_amax,
        fallback,
        down,
    )

    dispatch_amax = torch.zeros_like(
        direct_amax
    )

    dispatch_output = dispatch_mlp(
        a,
        interleaved_weight,
        down_weight,
        torch.tensor(
            scale_a,
            device="cuda",
            dtype=torch.float32,
        ),
        torch.tensor(
            scale_gate_up,
            device="cuda",
            dtype=torch.float32,
        ),
        torch.tensor(
            scale_down,
            device="cuda",
            dtype=torch.float32,
        ),
        quant_multiplier,
        dispatch_amax,
        direct,
        down,
    )

    torch.cuda.synchronize()

    direct_error = relative_l2(
        direct_output, reference
    )
    fallback_error = relative_l2(
        fallback_output, reference
    )
    dispatch_error = relative_l2(
        dispatch_output, reference
    )

    assert direct_output.shape == (M, K)
    assert direct_output.dtype == torch.bfloat16
    assert torch.isfinite(
        direct_output.float()
    ).all()

    assert direct_error < 4e-2, direct_error
    assert fallback_error < 4e-2, fallback_error
    assert dispatch_error < 4e-2, dispatch_error
    assert direct_amax.item() > 0
    assert dispatch_amax.item() > 0

    print(
        "SM100 direct FP8 MLP test passed:",
        f"direct_rel_l2={direct_error:.6e}",
        f"fallback_rel_l2={fallback_error:.6e}",
        f"dispatch_rel_l2={dispatch_error:.6e}",
    )

if __name__ == "__main__":
    main()
