# Experimental Kernel Paths

These investigations are retained as engineering evidence but are not selected
by the supported build unless explicitly identified as a fallback.

## TMA multicast cluster-N2

Source: `integration/sm100_fp8_gemm_torch_cluster_n2.cu`

The `1x2x1` cluster was correct but ranged from neutral to slightly slower than
the `1x1x1` kernel in the isolated gate/up experiment. Multicast is therefore
not presented as the source of the final speedup.

## Interleaved gate/up layout

Sources:

- `integration/sm100_fp8_gate_up_swiglu_interleaved_torch.cu`
- `integration/sm100_fp8_gate_up_swiglu_interleaved_nosmem_probe.cu`

The interleaved layout matched the blocked reference exactly. Layout conversion
alone was performance-neutral, but it enabled adjacent gate/up accumulator
pairing in the later direct epilogue. The two-kernel interleaved implementation
remains a debug/reference fallback.

## TMEM register-pairing diagnostics

Sources:

- `integration/sm100_fp8_gate_up_swiglu_pair_swap_probe.cu`
- `integration/sm100_fp8_gate_up_swiglu_pair_swap_active_probe.cu`

The diagnostic established that adjacent logical gate/up columns are
register-local after the TMEM-to-register copy. A pinned CUTLASS header was
temporarily instrumented during this isolated investigation and immediately
restored. The final release was built and tested against a pristine checkout at
commit `ad7b2f5e84fcfa124cb02b91d5bd26d238c0459e`.

## Split-K and Stream-K

Deterministic Split-K=2 and Stream-K were explored for skinny-M parallelism.
Reduction and workspace-reset overhead outweighed the additional scheduling
parallelism, so neither is used by the final dispatcher.

## Amax variants

The project evaluated internal allocation, `cudaMemsetAsync`, no-amax, and
caller-owned running-amax variants. Passing an external running-amax buffer
removed allocation/reset overhead and became the supported contract.

## Promoted direct epilogue

`integration/sm100_fp8_swiglu_direct_torch.cu` began as an experimental
collective and was promoted after it compiled without CUTLASS modifications,
passed independent correctness tests, beat the two-kernel predecessor, and won
all 28 repeated full-MLP decode batches. It is production code in this
repository, not an unverified draft.

## Authoritative release data

- `results/release_manifest_checkpoint13.json`
- `results/final_summary_direct_epilogue.json`
- `bench/full_mlp_direct_decode_repeatability.csv`
- `bench/full_mlp_direct_prefill.csv`
- `results/direct_epilogue_physical_traffic.json`

Older CSV and JSON files document intermediate milestones and should not be
used as final headline results.
