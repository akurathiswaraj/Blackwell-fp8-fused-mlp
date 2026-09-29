# Benchmark data

The authoritative checkpoint-13 measurements are:

- `full_mlp_direct_decode_repeatability.csv`
- `full_mlp_direct_decode_repeatability_batches.csv`
- `full_mlp_direct_decode_repeatability_raw.csv`
- `full_mlp_direct_prefill.csv`
- `full_mlp_direct_prefill_raw.csv`
- `direct_epilogue_exact_traffic.csv`

Run `benchmark_full_mlp.py` on an NVIDIA SM100 GPU to generate a new paired
cold-L2 measurement set. Older files document intermediate tuning milestones
and are not the source of final headline claims.
