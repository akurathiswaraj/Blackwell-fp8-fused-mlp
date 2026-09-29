
import argparse
import torch

from integration.build_extensions import build_extensions

parser = argparse.ArgumentParser()
parser.add_argument("--m", type=int, required=True)
args = parser.parse_args()

_, swiglu, down = build_extensions()

M = args.m
K = 8192
N = 28672
FP8 = torch.float8_e4m3fn
device = torch.device("cuda")

torch.manual_seed(9000 + M)

a = (
    torch.randn(
        (M, K),
        device=device,
        dtype=torch.float16,
    )
    * 32.0
).to(FP8)

gate_up_weight = (
    torch.randn(
        (2 * N, K),
        device=device,
        dtype=torch.float16,
    )
    * 32.0
).to(FP8)

down_weight = (
    torch.randn(
        (K, N),
        device=device,
        dtype=torch.float16,
    )
    * 32.0
).to(FP8)

gate_up_alpha = 1.0e-5
quant_multiplier = 64.0
down_alpha = (1.0 / quant_multiplier) * 1.0e-2

running_amax = torch.zeros(
    1,
    device=device,
    dtype=torch.float32,
)

# Initialize CUDA and allocators before profiling.
temporary = swiglu.gate_up_swiglu_fp8(
    a,
    gate_up_weight,
    gate_up_alpha,
    quant_multiplier,
    running_amax,
)[0]

down.gemm(
    temporary,
    down_weight,
    down_alpha,
)

torch.cuda.synchronize()
running_amax.zero_()

torch.cuda.cudart().cudaProfilerStart()

intermediate = swiglu.gate_up_swiglu_fp8(
    a,
    gate_up_weight,
    gate_up_alpha,
    quant_multiplier,
    running_amax,
)[0]

output = down.gemm(
    intermediate,
    down_weight,
    down_alpha,
)

torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()

print(
    f"M={M} output={tuple(output.shape)} "
    f"dtype={output.dtype}"
)
