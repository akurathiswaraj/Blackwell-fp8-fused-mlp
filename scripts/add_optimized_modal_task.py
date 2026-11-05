from pathlib import Path
import ast


path = Path("modal_b200.py")
source = path.read_text()


function_marker = """\
# ---------------------------------------------------------
# Local command dispatcher
# ---------------------------------------------------------
"""


optimized_function = """\
# ---------------------------------------------------------
# Final optimized-dispatch benchmark
# ---------------------------------------------------------

@app.function(
    gpu="B200",
    cpu=16,
    memory=65536,
    timeout=3600,
)
def run_optimized_dispatch_benchmark():
    print_environment()

    print(
        "===== OPTIMIZED DISPATCH BENCHMARK ====="
    )

    run_command(
        [
            "python",
            "bench/benchmark_optimized_dispatch.py",
        ]
    )

    artifact_paths = [
        "bench/optimized_dispatch_decode.csv",
        "bench/optimized_dispatch_decode_batches.csv",
        "bench/optimized_dispatch_decode_raw.csv",
        "results/optimized_dispatch_decode.json",
    ]

    artifacts = {}

    for relative_path in artifact_paths:
        remote_path = (
            Path(REMOTE_PROJECT)
            / relative_path
        )

        if not remote_path.exists():
            raise RuntimeError(
                f"Missing benchmark artifact: "
                f"{relative_path}"
            )

        artifacts[relative_path] = (
            remote_path.read_text(
                encoding="utf-8"
            )
        )

    return artifacts


"""


if (
    "def run_optimized_dispatch_benchmark():"
    not in source
):
    if function_marker not in source:
        raise RuntimeError(
            "Could not locate local dispatcher marker"
        )

    source = source.replace(
        function_marker,
        optimized_function + function_marker,
        1,
    )


old_branch = """\
    elif task == "full-benchmark":
        run_full_mlp_warp_benchmark.remote()

    else:
"""

new_branch = """\
    elif task == "full-benchmark":
        run_full_mlp_warp_benchmark.remote()

    elif task == "optimized-benchmark":
        artifacts = (
            run_optimized_dispatch_benchmark.remote()
        )

        for (
            relative_path,
            contents,
        ) in artifacts.items():
            destination = (
                LOCAL_PROJECT
                / relative_path
            )

            destination.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            destination.write_text(
                contents,
                encoding="utf-8",
            )

            print(
                "Downloaded:",
                destination,
            )

    else:
"""


if (
    'elif task == "optimized-benchmark":'
    not in source
):
    if old_branch not in source:
        raise RuntimeError(
            "Could not locate main task branch"
        )

    source = source.replace(
        old_branch,
        new_branch,
        1,
    )


source = source.replace(
    """\
            "release, safety, gateup-benchmark, "
            "full-benchmark"
""",
    """\
            "release, safety, gateup-benchmark, "
            "full-benchmark, optimized-benchmark"
""",
)


ast.parse(
    source,
    filename=str(path),
)

path.write_text(source)

print("Updated:", path)
print(
    "Optimized function present:",
    (
        "def run_optimized_dispatch_benchmark():"
        in source
    ),
)
print(
    "Optimized task present:",
    (
        'elif task == "optimized-benchmark":'
        in source
    ),
)
print("Python syntax: Passed")