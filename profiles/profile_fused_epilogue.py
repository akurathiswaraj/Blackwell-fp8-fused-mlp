import argparse
import importlib.util
from pathlib import Path

import torch


def load_extension(name, path):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not load extension spec: {path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def quantize_e4m3(tensor):
    tensor32 = tensor.float()
    scale = (
        tensor32.abs().amax() / 448.0
    ).clamp_min(1e-12)

    quantized = (
        (tensor32 / scale)
        .clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
    )
    return quantized, scale.float()


parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    choices=["with_amax", "no_amax"],
    required=True,
)
parser.add_argument(
    "--library",
    type=Path,
    required=True,
    help="Path to the already-built diagnostic extension",
)
args = parser.parse_args()

module_names = {
    "with_amax": (
        "sm100_fp8_bias_silu_fp8_n64_ext"
    ),
    "no_amax": (
        "sm100_fp8_bias_silu_fp8_n64_no_amax_ext"
    ),
}

module_name = module_names[args.mode]
module_path = args.library.expanduser().resolve()
module = load_extension(
    module_name,
    module_path,
)

torch.manual_seed(8600)

M = 16
N = 28672
K = 8192

a_original = (
    0.1
    * torch.randn(
        (M, K),
        device="cuda",
        dtype=torch.bfloat16,
    )
)
b_original = (
    0.1
    * torch.randn(
        (N, K),
        device="cuda",
        dtype=torch.bfloat16,
    )
)
bias = torch.linspace(
    -1.0,
    1.0,
    N,
    device="cuda",
    dtype=torch.bfloat16,
)

a_fp8, scale_a = quantize_e4m3(a_original)
b_fp8, scale_b = quantize_e4m3(b_original)

scale_a_value = float(scale_a.item())
scale_b_value = float(scale_b.item())
scale_d_value = 100.0

del a_original, b_original
torch.cuda.empty_cache()

if args.mode == "with_amax":
    def run():
        return module.gemm_bias_silu_fp8(
            a_fp8,
            b_fp8,
            bias,
            scale_a_value,
            scale_b_value,
            scale_d_value,
        )
else:
    def run():
        return module.gemm_bias_silu_fp8_no_amax(
            a_fp8,
            b_fp8,
            bias,
            scale_a_value,
            scale_b_value,
            scale_d_value,
        )

# Warm up module loading, workspace allocation, and CUDA state.
warmup_output = run()
torch.cuda.synchronize()
del warmup_output

# Only this range should be profiled.
torch.cuda.nvtx.range_push("profile")
profile_output = run()
torch.cuda.nvtx.range_pop()
torch.cuda.synchronize()

if args.mode == "with_amax":
    output, amax = profile_output
    print(
        "mode=with_amax",
        "output_shape=",
        tuple(output.shape),
        "amax=",
        float(amax.item()),
    )
else:
    output = profile_output[0]
    print(
        "mode=no_amax",
        "output_shape=",
        tuple(output.shape),
    )
