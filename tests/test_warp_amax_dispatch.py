import torch

from integration.build_extensions import (
    build_optimized_extensions,
)
from integration.fp8_mlp import (
    WARP_AMAX_MIN_M,
    custom_decode_mlp,
    interleave_gate_up_weight,
)


FP8 = torch.float8_e4m3fn


class RecordingExtension:
    def __init__(self, extension, name):
        self.extension = extension
        self.name = name
        self.calls = 0

    def swiglu_direct(self, *args):
        self.calls += 1
        return self.extension.swiglu_direct(
            *args
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

    K = 256
    N = 256

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

    for M in [
        WARP_AMAX_MIN_M - 1,
        WARP_AMAX_MIN_M,
        48,
        64,
    ]:
        activation = (
            torch.randn(
                M,
                K,
                device="cuda",
                dtype=torch.float16,
            )
            * 32
        ).to(FP8)

        direct_recorder = RecordingExtension(
            direct,
            "original",
        )

        warp_recorder = RecordingExtension(
            warp_amax,
            "warp_amax",
        )

        dispatch_amax = torch.zeros(
            1,
            device="cuda",
            dtype=torch.float32,
        )

        output = custom_decode_mlp(
            activation,
            interleaved_weight,
            down_weight,
            1.0e-5,
            32.0,
            1.0e-2 / 32.0,
            dispatch_amax,
            direct_recorder,
            down,
            warp_amax_extension=(
                warp_recorder
            ),
        )

        torch.cuda.synchronize()

        should_use_warp = (
            WARP_AMAX_MIN_M
            <= M
            <= 64
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
                f"Incorrect original dispatch at M={M}"
            )

        if (
            warp_recorder.calls
            != expected_warp_calls
        ):
            raise RuntimeError(
                f"Incorrect warp dispatch at M={M}"
            )

        if not torch.isfinite(
            output.float()
        ).all():
            raise RuntimeError(
                f"Non-finite output at M={M}"
            )

        selected = (
            "warp_amax"
            if should_use_warp
            else "original"
        )

        print(
            f"M={M:2d} | "
            f"selected={selected} | "
            f"direct_calls={direct_recorder.calls} | "
            f"warp_calls={warp_recorder.calls} | "
            "passed=True"
        )

    print()
    print("WARP-AMAX DISPATCH TEST PASSED")


if __name__ == "__main__":
    main()