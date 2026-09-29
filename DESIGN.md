# Design: Blackwell FP8 Llama 70B MLP

## Problem

Llama 3.1 70B decode repeatedly executes gate projection, up projection,
SwiGLU, requantization, and down projection at small token counts. A framework
pipeline launches several short kernels and materializes a large gate/up
intermediate, making decode sensitive to launch and memory-traffic overhead.

## Supported implementation

The decode path consists of two GPU kernels:

1. A combined gate/up FP8 GEMM with a custom direct-store SwiGLU epilogue.
2. An FP8 down-projection GEMM with BF16 output.

This is different from the earlier implementation, which used a gate/up GEMM,
a separate SwiGLU/amax/quantization CUDA kernel, and the down projection.

### Combined gate/up GEMM

Gate and up rows are interleaved offline as
`[gate0, up0, gate1, up1, ...]`. The SM100 mainloop uses CUTLASS/CuTe, TMA,
`tcgen05` MMA, and TMEM. The kernel uses a `64x128x128` MMA tile and a
`1x2x1` CTA cluster.

### Direct-store epilogue

The project-owned `Sm100NoSmem` collective is wrapped by CUTLASS's
warp-specialized adapter. It loads adjacent accumulator values from TMEM into
the same thread's register fragment, applies GEMM scaling, computes
`SiLU(gate) * up`, updates a caller-owned FP32 running amax, converts to FP8
E4M3, and stores compact `[M,N]` output.

The epilogue therefore avoids the global BF16 `[M,2N]` write, its reread, and
the separate post-processing launch. CUTLASS headers are not modified.

### Down projection

The compact FP8 SwiGLU result feeds a decode-tuned `64x64x128` SM100 FP8 GEMM
that produces BF16 output.

## Dispatch policy

- `M<=64`: direct custom decode path.
- `M>=512`: cuBLASLt/PyTorch path.

The explicit split is based on measurements rather than an assumption that one
kernel should serve both regimes.

## Correctness

The direct epilogue was checked against an independently computed FP32/FP8
reference. Full-MLP relative L2 error is approximately `1.4e-2`, dominated by
the intended FP8 quantization. Standalone and Llama-compatible integration
tests cover the direct path, fallback path, dispatch, output shape, dtype,
finite values, and amax updates.

## Performance

| M | Direct ms | Baseline ms | Speedup | Reduction | Batches won |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.156704 | 0.173136 | 1.105x | 9.5% | 7/7 |
| 4 | 0.156704 | 0.183296 | 1.170x | 14.5% | 7/7 |
| 16 | 0.140320 | 0.185344 | 1.321x | 24.3% | 7/7 |
| 64 | 0.148512 | 0.203840 | 1.373x | 27.1% | 7/7 |

## Profiler interpretation

At decode M=64, the direct gate/up kernel used 97 registers per thread,
221.70 KiB dynamic shared memory per block, and achieved 10.59% occupancy.
Despite the resource cost, removing an entire post-op launch and intermediate
write/read improved complete-MLP latency.

At prefill M=8192, the direct gate/up kernel took 10.91 ms in the NCU replay,
with 54.92% SM throughput. The direct scalar activation work is less efficient
than the cuBLASLt-based pipeline in this compute-bound regime, so prefill falls
back to cuBLASLt.

## Traffic accounting

The eliminated logical tensor traffic is exactly twice the size of a BF16
`[M,2N]` tensor:

```text
2 accesses * 2 bytes * M * (2N) = 8*M*N bytes
```

For `M=64`, `N=28672`, this is 14 MiB. Physical NCU DRAM counters are reported
separately because caching and memory transactions mean physical traffic need
not equal logical tensor bytes.

## Honest limitations

- This is one combined GEMM over interleaved weights, not two independent
  gate/up accumulator matrices.
- The decode-specialized direct epilogue is slower for measured prefill shapes.
- Results are from one NVIDIA B200 system.
- The Llama-compatible wrapper is a focused MLP module, not a complete model
  server integration.
