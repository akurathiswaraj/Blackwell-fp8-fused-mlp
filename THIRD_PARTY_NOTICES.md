# Third-Party Notices

This project builds against NVIDIA CUTLASS 3.9.2 at commit:

```text
ad7b2f5e84fcfa124cb02b91d5bd26d238c0459e
```

CUTLASS is distributed under the BSD 3-Clause license. See the upstream
[CUTLASS license](https://github.com/NVIDIA/cutlass/blob/main/LICENSE.txt).

Files under `kernels/` that were adapted from NVIDIA CUTLASS examples retain
their original NVIDIA copyright and BSD-3-Clause license headers. CUTLASS is
not vendored in this repository and must be obtained separately.

PyTorch is distributed under its own BSD-style license. NVIDIA CUDA, cuBLAS,
Nsight Compute, and related components are governed by their respective
licenses.
