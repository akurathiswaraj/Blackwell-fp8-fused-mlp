
import argparse

import torch

from integration.build_extensions import build_extensions

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    choices=["direct", "two_kernel"],
    required=True,
)
parser.add_argument("--m", type=int, required=True)
args = parser.parse_args()

direct, fallback, _ = build_extensions()
module = direct if args.mode == "direct" else fallback

torch.manual_seed(132)

M = args.m
K = 8192
N = 28672

a = torch.randn(
    M, K,
    device="cuda",
    dtype=torch.float32,
).to(torch.float8_e4m3fn)

weight = torch.randn(
    2 * N, K,
    device="cuda",
    dtype=torch.float32,
).to(torch.float8_e4m3fn)

running_amax = torch.zeros(
    1,
    device="cuda",
    dtype=torch.float32,
)

if args.mode == "direct":
    output = module.swiglu_direct(
        a,
        weight,
        1.0e-5,
        45.69757799928416,
        running_amax,
    )[0]
else:
    output = module.gate_up_swiglu_fp8(
        a,
        weight,
        1.0e-5,
        45.69757799928416,
        running_amax,
    )[0]

torch.cuda.synchronize()

print(
    "mode=", args.mode,
    "M=", M,
    "output=", tuple(output.shape),
)
