import argparse

import torch

from integration.build_extensions import (
    build_extensions,
    build_warp_amax_extension,
)


FP8 = torch.float8_e4m3fn

M = 64
K = 8192
N = 28672

GEMM_SCALE = 1.0e-5
QUANT_MULTIPLIER = 45.69757799928416


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        required=True,
        choices=["original", "warp_amax"],
    )

    arguments = parser.parse_args()

    if torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("NVIDIA SM100 GPU required")

    if arguments.mode == "original":
        extension, _, _ = build_extensions()
    else:
        extension = build_warp_amax_extension()

    torch.manual_seed(20261105)

    activation = (
        torch.randn(
            M,
            K,
            device="cuda",
            dtype=torch.float16,
        )
        * 32
    ).to(FP8)

    interleaved_weight = (
        torch.randn(
            2 * N,
            K,
            device="cuda",
            dtype=torch.float16,
        )
        * 32
    ).to(FP8)

    running_amax = torch.zeros(
        1,
        device="cuda",
        dtype=torch.float32,
    )

    torch.cuda.synchronize()

    output, returned_amax = extension.swiglu_direct(
        activation,
        interleaved_weight,
        GEMM_SCALE,
        QUANT_MULTIPLIER,
        running_amax,
    )

    torch.cuda.synchronize()

    print(
        "mode=",
        arguments.mode,
        "output_shape=",
        tuple(output.shape),
        "output_dtype=",
        output.dtype,
        "amax=",
        float(returned_amax.item()),
    )


if __name__ == "__main__":
    main()