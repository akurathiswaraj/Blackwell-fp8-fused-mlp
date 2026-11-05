import torch

from integration.build_extensions import (
    build_optimized_extensions,
)
from integration.llama_mlp import (
    BlackwellFP8LlamaMLP,
)


FP8 = torch.float8_e4m3fn


class RecordingExtension:
    def __init__(self, extension):
        self.extension = extension
        self.calls = 0

    def swiglu_direct(self, *args):
        self.calls += 1

        return self.extension.swiglu_direct(
            *args
        )


def run_case(
    flat_m,
    shape,
    blocked_gate_up,
    down_weight,
    direct,
    warp_amax,
    down,
):
    if (
        shape[0] * shape[1]
        != flat_m
    ):
        raise ValueError(
            "Shape does not match flattened M"
        )

    direct_recorder = RecordingExtension(
        direct
    )

    warp_recorder = RecordingExtension(
        warp_amax
    )

    optimized_module = BlackwellFP8LlamaMLP(
        blocked_gate_up,
        down_weight,
        1.0e-2,
        1.0e-2,
        32.0,
        direct_recorder,
        down,
        warp_amax_extension=(
            warp_recorder
        ),
    )

    reference_module = BlackwellFP8LlamaMLP(
        blocked_gate_up,
        down_weight,
        1.0e-2,
        1.0e-2,
        32.0,
        direct,
        down,
    )

    activation_fp8 = (
        torch.randn(
            *shape,
            256,
            device="cuda",
            dtype=torch.float16,
        )
        * 32
    ).to(FP8)

    scale_activation = torch.tensor(
        1.0e-3,
        device="cuda",
        dtype=torch.float32,
    )

    optimized_output = (
        optimized_module.forward_fp8(
            activation_fp8,
            scale_activation,
        )
    )

    reference_output = (
        reference_module.forward_fp8(
            activation_fp8,
            scale_activation,
        )
    )

    torch.cuda.synchronize()

    should_use_warp = (
        40 <= flat_m <= 64
    )

    expected_direct_calls = (
        0 if should_use_warp else 1
    )

    expected_warp_calls = (
        1 if should_use_warp else 0
    )

    if (
        direct_recorder.calls
        != expected_direct_calls
    ):
        raise RuntimeError(
            f"Incorrect direct calls at M={flat_m}"
        )

    if (
        warp_recorder.calls
        != expected_warp_calls
    ):
        raise RuntimeError(
            f"Incorrect warp calls at M={flat_m}"
        )

    if not torch.equal(
        optimized_output,
        reference_output,
    ):
        raise RuntimeError(
            f"Output mismatch at M={flat_m}"
        )

    if (
        optimized_output.shape
        != activation_fp8.shape
    ):
        raise RuntimeError(
            f"Shape mismatch at M={flat_m}"
        )

    if (
        optimized_output.dtype
        != torch.bfloat16
    ):
        raise RuntimeError(
            f"Dtype mismatch at M={flat_m}"
        )

    if not torch.isfinite(
        optimized_output.float()
    ).all():
        raise RuntimeError(
            f"Non-finite output at M={flat_m}"
        )

    selected = (
        "warp_amax"
        if should_use_warp
        else "original"
    )

    print(
        f"flat_M={flat_m:2d} | "
        f"input_shape={tuple(activation_fp8.shape)} | "
        f"selected={selected} | "
        f"exact=True | "
        f"passed=True"
    )


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

    hidden = 256
    intermediate = 256

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

    run_case(
        flat_m=39,
        shape=(1, 39),
        blocked_gate_up=blocked_gate_up,
        down_weight=down_weight,
        direct=direct,
        warp_amax=warp_amax,
        down=down,
    )

    run_case(
        flat_m=40,
        shape=(2, 20),
        blocked_gate_up=blocked_gate_up,
        down_weight=down_weight,
        direct=direct,
        warp_amax=warp_amax,
        down=down,
    )

    run_case(
        flat_m=64,
        shape=(2, 32),
        blocked_gate_up=blocked_gate_up,
        down_weight=down_weight,
        direct=direct,
        warp_amax=warp_amax,
        down=down,
    )

    print()
    print(
        "LLAMA OPTIMIZED DISPATCH TEST PASSED"
    )


if __name__ == "__main__":
    main()