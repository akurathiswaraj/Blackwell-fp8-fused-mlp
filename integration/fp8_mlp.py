import torch
import torch.nn.functional as F

DECODE_MAX_M = 64
FP8_MAX = 448.0

def interleave_gate_up_weight(
    packed_gate_up_weight,
):
    """Offline/preload conversion from [gate; up] to adjacent pairs."""
    if packed_gate_up_weight.dim() != 2:
        raise ValueError(
            "gate/up weight must have shape [2N,K]"
        )

    if packed_gate_up_weight.shape[0] % 2 != 0:
        raise ValueError(
            "gate/up output dimension must be even"
        )

    gate, up = packed_gate_up_weight.chunk(
        2, dim=0
    )

    return torch.stack(
        [gate, up],
        dim=1,
    ).reshape(
        packed_gate_up_weight.shape
    ).contiguous()

def custom_decode_mlp(
    activation,
    interleaved_gate_up_weight,
    down_weight,
    gate_up_alpha,
    quant_multiplier,
    down_alpha,
    running_amax,
    direct_extension,
    down_extension,
):
    product_fp8 = (
        direct_extension.swiglu_direct(
            activation,
            interleaved_gate_up_weight,
            float(gate_up_alpha),
            float(quant_multiplier),
            running_amax,
        )[0]
    )

    return down_extension.gemm(
        product_fp8,
        down_weight,
        float(down_alpha),
    )

def two_kernel_decode_mlp(
    activation,
    interleaved_gate_up_weight,
    down_weight,
    gate_up_alpha,
    quant_multiplier,
    down_alpha,
    running_amax,
    fallback_extension,
    down_extension,
):
    product_fp8 = (
        fallback_extension.gate_up_swiglu_fp8(
            activation,
            interleaved_gate_up_weight,
            float(gate_up_alpha),
            float(quant_multiplier),
            running_amax,
        )[0]
    )

    return down_extension.gemm(
        product_fp8,
        down_weight,
        float(down_alpha),
    )

def cublaslt_mlp(
    activation,
    interleaved_gate_up_weight,
    down_weight,
    scale_a,
    scale_gate_up,
    scale_down,
    quant_multiplier,
    running_amax,
):
    combined = torch._scaled_mm(
        activation,
        interleaved_gate_up_weight.t(),
        scale_a=scale_a,
        scale_b=scale_gate_up,
        out_dtype=torch.bfloat16,
    )

    gate = combined[:, 0::2]
    up = combined[:, 1::2]
    product = F.silu(gate) * up

    candidate = (
        product.abs()
        .amax()
        .reshape(1)
        .float()
    )

    running_amax.copy_(
        torch.maximum(
            running_amax,
            candidate,
        )
    )

    product_fp8 = torch.clamp(
        product.float()
        * float(quant_multiplier),
        -FP8_MAX,
        FP8_MAX,
    ).to(torch.float8_e4m3fn)

    product_scale = torch.tensor(
        1.0 / float(quant_multiplier),
        device=activation.device,
        dtype=torch.float32,
    )

    return torch._scaled_mm(
        product_fp8,
        down_weight.t(),
        scale_a=product_scale,
        scale_b=scale_down,
        out_dtype=torch.bfloat16,
    )

def dispatch_mlp(
    activation,
    interleaved_gate_up_weight,
    down_weight,
    scale_a,
    scale_gate_up,
    scale_down,
    quant_multiplier,
    running_amax,
    direct_extension,
    down_extension,
):
    if activation.shape[0] <= DECODE_MAX_M:
        gate_up_alpha = (
            float(scale_a.item())
            * float(scale_gate_up.item())
        )

        down_alpha = (
            (1.0 / float(quant_multiplier))
            * float(scale_down.item())
        )

        return custom_decode_mlp(
            activation,
            interleaved_gate_up_weight,
            down_weight,
            gate_up_alpha,
            quant_multiplier,
            down_alpha,
            running_amax,
            direct_extension,
            down_extension,
        )

    return cublaslt_mlp(
        activation,
        interleaved_gate_up_weight,
        down_weight,
        scale_a,
        scale_gate_up,
        scale_down,
        quant_multiplier,
        running_amax,
    )
