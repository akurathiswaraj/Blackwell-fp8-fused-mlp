# Blackwell FP8 Fused MLP Kernels

SM100-specific FP8 SwiGLU kernels for Llama 3.1 70B, implemented with
CUDA, CUTLASS/CuTe, TMA, `tcgen05` tensor-core MMA, and TMEM.

On an NVIDIA B200, the final direct-epilogue implementation reduced complete
MLP decode latency by up to **27.1% (1.373x)** against a cuBLASLt/PyTorch
baseline. It won all **28/28 repeated decode batches** across
`M={1,4,16,64}`. The decode-specialized kernel loses on prefill, so the
runtime deliberately falls back to cuBLASLt for `M>=512`.

## Final results

The benchmark uses the Llama 3.1 70B MLP dimensions: hidden size 8192 and
intermediate size 28672.

| M | Direct custom | cuBLASLt/PyTorch | Speedup | Latency reduction | Batches won |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.156704 ms | 0.173136 ms | 1.105x | 9.5% | 7/7 |
| 4 | 0.156704 ms | 0.183296 ms | 1.170x | 14.5% | 7/7 |
| 16 | 0.140320 ms | 0.185344 ms | 1.321x | 24.3% | 7/7 |
| 64 | 0.148512 ms | 0.203840 ms | 1.373x | 27.1% | 7/7 |

End-to-end relative L2 error remained approximately `1.4e-2`. Full raw and
batch-level measurements are under `bench/`.

### Prefill

| M | Direct decode kernel | cuBLASLt pipeline | Ratio |
|---:|---:|---:|---:|
| 512 | 0.615360 ms | 0.486416 ms | 0.790x |
| 2048 | 2.319376 ms | 1.873952 ms | 0.808x |
| 8192 | 11.914752 ms | 7.585264 ms | 0.637x |

The supported dispatcher uses the custom path for `M<=64` and cuBLASLt for
`M>=512`. No prefill speedup is claimed.

## Architecture

Gate and up weights are packed once as interleaved rows:

```text
[gate0, up0, gate1, up1, ...]
```

One combined FP8 GEMM consumes the activation matrix and the packed weights.
A project-owned SM100 direct-store collective moves adjacent gate/up
accumulators from TMEM into register fragments and computes:

```text
FP8 output = quantize_fp8(SiLU(gate) * up)
```

The same epilogue atomically updates a caller-owned running amax and stores the
compact `[M,N]` FP8 result. It does not materialize the former global BF16
`[M,2N]` gate/up tensor. A decode-tuned FP8 down projection produces BF16
output.

The supported source is
`integration/sm100_fp8_swiglu_direct_torch.cu`. CUTLASS itself remains pinned
and unmodified.

## Traffic measurements

Eliminating the BF16 `[M,2N]` write and reread removes exactly `8*M*N`
logical bytes. At M=64 this is **14 MiB**. These are tensor-size calculations,
not DRAM-counter claims.

Targeted Nsight Compute measurements found:

| Shape | Direct | Previous two-kernel path | Reduction |
|---|---:|---:|---:|
| M=64 | 475.52 MB | 487.12 MB | 2.38% |
| M=8192 | 2.69 GB | 4.84 GB | 44.4% |

The M=8192 traffic reduction does not translate to a latency win because the
scalar epilogue is inefficient in the compute-bound prefill regime.

## Environment

- NVIDIA B200, compute capability 10.0
- CUDA Toolkit 12.8.93
- PyTorch 2.8.0+cu129
- CUTLASS 3.9.2 at commit
  `ad7b2f5e84fcfa124cb02b91d5bd26d238c0459e`
- Ninja
- Nsight Compute 2025.1.1 for the included profiles

## Setup

Clone the pinned CUTLASS revision outside this repository:

```bash
git clone https://github.com/NVIDIA/cutlass.git
cd cutlass
git checkout ad7b2f5e84fcfa124cb02b91d5bd26d238c0459e
export CUTLASS_PATH="$PWD"
```

Point the extension builder at CUDA and select SM100a:

```bash
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST=10.0a
export MAX_JOBS=8
pip install -r requirements-build.txt
```

The CUDA 12.8 development installation must include cuBLAS, cuSPARSE, and
cuSOLVER headers.

## Build and test

Run commands from the repository root:

```bash
export PYTHONPATH="$PWD"
python tests/test_fp8_mlp.py
python tests/test_llama_mlp_module.py
```

The tests compile the direct epilogue, two-kernel fallback, and down-projection
extensions. An NVIDIA SM100 GPU is required.

Run the reproducible paired benchmark with:

```bash
python bench/benchmark_full_mlp.py --m 1 4 16 64 --batches 7 --rounds 20
```

The script writes new files with a `reproduced_` prefix so the verified B200
release measurements are not overwritten.

## Measurement methodology

- Identical FP8 tensors for custom and baseline paths
- CUDA-event timing
- 512 MiB L2 flush before every timed invocation
- Alternating custom/baseline invocation order
- Seven independent batches of 20 paired rounds for final decode claims
- Correctness checked against the cuBLASLt/PyTorch pipeline
- Scales converted to host scalars outside timed custom-kernel closures

The earlier `.item()`-inside-timing mistake is documented in
`results/timing_methodology_correction.json`; invalid raw measurements are not
part of this public package.

## Repository layout

```text
integration/  PyTorch extension sources and Llama-compatible module
kernels/      Standalone CUTLASS kernel variants and tuning artifacts
tests/        SM100 correctness and integration tests
bench/        Reproduction script and measured CSV data
profiles/     Nsight Compute drivers, text exports, and reports
results/      Structured release summaries
```

See [DESIGN.md](DESIGN.md) for the architecture and profiler interpretation,
and [EXPERIMENTS.md](EXPERIMENTS.md) for neutral or abandoned investigations.

## Limitations

- The direct implementation uses one combined interleaved gate/up GEMM, not
  two independent accumulator matrices.
- The custom kernel is decode-specialized and loses to cuBLASLt on prefill.
- `torch._scaled_mm` is an internal PyTorch API and may change between releases.
- Results currently cover one B200 system and the listed software versions.

## License

Project-authored code is released under BSD-3-Clause. CUTLASS-derived files
retain NVIDIA's original BSD-3-Clause notices. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
