# Résumé bullets

**Blackwell FP8 Fused MLP Kernels for Llama 3.1 70B** |
CUDA, CUTLASS/CuTe, PyTorch, Nsight Compute

- Built SM100-specific FP8 Llama 70B MLP kernels using TMA, `tcgen05` tensor-core MMA, TMEM, and a direct SwiGLU epilogue that fuses `SiLU(gate) * up`, running-amax tracking, and FP8 E4M3 requantization without materializing the BF16 gate/up intermediate.

- Optimized the epilogue with warp-reduced amax tracking, cutting global-reduction traffic **32x** and lowering same-environment full-MLP latency by **4.4%** over the original kernel at M=64; the shape-aware pipeline ran **1.32-1.62x faster** than cuBLASLt/PyTorch across eight B200 decode shapes, winning **56/56** repeated batches at approximately `0.016` relative L2 error.
