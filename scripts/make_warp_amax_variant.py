from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]

source_path = (
    PROJECT
    / "integration"
    / "sm100_fp8_swiglu_direct_torch.cu"
)

destination_path = (
    PROJECT
    / "integration"
    / "sm100_fp8_swiglu_direct_warp_amax_torch.cu"
)

source = source_path.read_text()


old_loop_start = """\
        CUTLASS_PRAGMA_UNROLL
        for (
            int i = 0;
"""

new_loop_start = """\
        // Accumulate one local maximum per epilogue thread.
        // A warp reduction below replaces per-output atomics.
        float thread_amax = 0.0f;

        CUTLASS_PRAGMA_UNROLL
        for (
            int i = 0;
"""


old_atomic = """\
                    atomicMax(
                        reinterpret_cast<unsigned int*>(
                            params.running_amax
                        ),
                        __float_as_uint(fabsf(value))
                    );
"""

new_atomic = """\
                    thread_amax = fmaxf(
                        thread_amax,
                        fabsf(value)
                    );
"""


old_loop_end = """\
            }
        }

        return cute::make_tuple(acc_state);
"""

new_loop_end = """\
            }
        }

        // All lanes participate in the epilogue loop, including
        // lanes whose coordinates were out of bounds. Reduce the
        // thread-local maxima and issue one global atomic per warp.
        unsigned int active_mask = __activemask();

        CUTLASS_PRAGMA_UNROLL
        for (
            int offset = 16;
            offset > 0;
            offset >>= 1
        ) {
            float other = __shfl_down_sync(
                active_mask,
                thread_amax,
                offset
            );

            thread_amax = fmaxf(
                thread_amax,
                other
            );
        }

        if (
            (int(threadIdx.x) & 31) == 0
            && thread_amax > 0.0f
        ) {
            atomicMax(
                reinterpret_cast<unsigned int*>(
                    params.running_amax
                ),
                __float_as_uint(thread_amax)
            );
        }

        return cute::make_tuple(acc_state);
"""


replacements = [
    (
        old_loop_start,
        new_loop_start,
        "loop start",
    ),
    (
        old_atomic,
        new_atomic,
        "per-output atomic",
    ),
    (
        old_loop_end,
        new_loop_end,
        "loop end",
    ),
]

for old, new, description in replacements:
    count = source.count(old)

    if count != 1:
        raise RuntimeError(
            f"Expected one {description}; found {count}"
        )

    source = source.replace(
        old,
        new,
        1,
    )

destination_path.write_text(source)

print("Created:", destination_path)
print(
    "Per-output atomic removed:",
    old_atomic not in source,
)
print(
    "Warp reduction present:",
    "__shfl_down_sync" in source,
)
print(
    "Original source unchanged:",
    source_path.exists(),
)