
import argparse

import torch

from integration.build_extensions import build_extensions

parser = argparse.ArgumentParser()
parser.add_argument("--m", type=int, required=True)
args = parser.parse_args()

direct, _, down = build_extensions()

torch.manual_seed(131)

device = "cuda"
M = args.m
K = 8192
N = 28672
H = 8192

a = torch.randn(
    M, K, device=device, dtype=torch.float32
).to(torch.float8_e4m3fn)

gate_up_weight = torch.randn(
    2 * N, K,
    device=device,
    dtype=torch.float32,
).to(torch.float8_e4m3fn)

down_weight = torch.randn(
    H, N,
    device=device,
    dtype=torch.float32,
).to(torch.float8_e4m3fn)

running_amax = torch.zeros(
    1, device=device, dtype=torch.float32
)

swiglu = direct.swiglu_direct(
    a,
    gate_up_weight,
    1.0e-5,
    45.69757799928416,
    running_amax,
)[0]

output = down.gemm(
    swiglu,
    down_weight,
    0.00021882996952626854,
)

torch.cuda.synchronize()

print(
    "M=", M,
    "output=", tuple(output.shape),
    "dtype=", output.dtype,
    "amax=", float(running_amax.item()),
)
