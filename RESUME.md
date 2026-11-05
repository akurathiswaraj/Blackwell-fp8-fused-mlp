# Résumé bullets

**Blackwell FP8 Fused MLP Kernels for Llama 3.1 70B** |
CUDA, CUTLASS/CuTe, PyTorch, Nsight Compute

- Built SM100-specific FP8 Llama 70B MLP kernels using TMA, `tcgen05` tensor-core MMA, TMEM, and a direct SwiGLU epilogue that fuses `SiLU(gate) * up`, running-amax tracking, and FP8 E4M3 requantization without materializing the BF16 gate/up intermediate.

- Added shape-aware warp-reduced amax dispatch, reducing measured full-MLP decode latency by up to **38.4% (1.623x)** versus a cuBLASLt/PyTorch baseline on NVIDIA B200 and winning **56/56** repeated batches; Nsight Compute measured **32x less global-reduction traffic** with approximately `1.6e-2` end-to-end relative L2 error.
