from pathlib import Path
import subprocess

import modal


APP_NAME = "blackwell-fp8-mlp"
REMOTE_PROJECT = "/workspace/blackwell_fp8_mlp"
LOCAL_PROJECT = Path(__file__).resolve().parent

CUTLASS_COMMIT = (
    "ad7b2f5e84fcfa124cb02b91d5bd26d238c0459e"
)


# ---------------------------------------------------------
# Reproducible CUDA environment
# ---------------------------------------------------------

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .apt_install(
        "git",
        "build-essential",
        "cmake",
    )
    .pip_install(
        "ninja==1.13.0",
        "packaging",
        "numpy==2.1.3",
    )
    .pip_install(
        "torch==2.8.0",
        index_url=(
            "https://download.pytorch.org/whl/cu129"
        ),
    )
    .run_commands(
        (
            "git clone "
            "https://github.com/NVIDIA/cutlass.git "
            "/opt/cutlass"
        ),
        (
            "git -C /opt/cutlass checkout "
            f"{CUTLASS_COMMIT}"
        ),
    )
    .env(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "CUTLASS_PATH": "/opt/cutlass",
            "TORCH_CUDA_ARCH_LIST": "10.0a",
            "MAX_JOBS": "8",
            "PYTHONPATH": REMOTE_PROJECT,
        }
    )
    .add_local_dir(
        LOCAL_PROJECT,
        remote_path=REMOTE_PROJECT,
        ignore=[
            ".git/**",
            "build/**",
            "**/__pycache__/**",
            "*.zip",
        ],
    )
)


app = modal.App(
    APP_NAME,
    image=image,
)


# ---------------------------------------------------------
# Command helpers
# ---------------------------------------------------------

def run_command(command):
    print()
    print("$", " ".join(command), flush=True)

    result = subprocess.run(
        command,
        cwd=REMOTE_PROJECT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    print(result.stdout, flush=True)

    if result.returncode != 0:
        raise RuntimeError(
            "Command failed with return code "
            f"{result.returncode}"
        )

    return result.stdout


def print_environment():
    import torch

    print("===== ENVIRONMENT =====")
    print("PyTorch:", torch.__version__)
    print("PyTorch CUDA:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "Compute capability:",
        torch.cuda.get_device_capability(0),
    )
    print(
        "GPU memory GiB:",
        round(
            torch.cuda.get_device_properties(
                0
            ).total_memory
            / (1024**3),
            1,
        ),
    )

    run_command(
        [
            "nvidia-smi",
            "--query-gpu="
            "name,uuid,driver_version,memory.total",
            "--format=csv,noheader",
        ]
    )

    run_command(
        [
            "nvcc",
            "--version",
        ]
    )

    run_command(
        [
            "git",
            "-C",
            "/opt/cutlass",
            "rev-parse",
            "HEAD",
        ]
    )


# ---------------------------------------------------------
# Release and production tests
# ---------------------------------------------------------

@app.function(
    gpu="B200",
    cpu=16,
    memory=65536,
    timeout=3600,
)
def run_release_tests():
    print_environment()

    tests = [
        (
            "RELEASE FP8 MLP TEST",
            "tests/test_fp8_mlp.py",
        ),
        (
            "LLAMA MODULE TEST",
            "tests/test_llama_mlp_module.py",
        ),
        (
            "EXTENDED DECODE TEST",
            "tests/test_decode_extended.py",
        ),
        (
            "RUNTIME ROBUSTNESS TEST",
            "tests/test_runtime_robustness.py",
        ),
        (
            "WARP-AMAX CORRECTNESS TEST",
            "tests/test_warp_amax_variant.py",
        ),
        (
            "WARP-AMAX DISPATCH TEST",
            "tests/test_warp_amax_dispatch.py",
        ),
        
                (
            "LLAMA OPTIMIZED DISPATCH TEST",
            "tests/test_llama_optimized_dispatch.py",
        ),
    ]

    for title, path in tests:
        print()
        print(f"===== {title} =====")

        run_command(
            [
                "python",
                path,
            ]
        )

    print()
    print("ALL MODAL B200 TESTS PASSED")


# ---------------------------------------------------------
# Full-shape launch-safety test
# ---------------------------------------------------------

@app.function(
    gpu="B200",
    cpu=16,
    memory=65536,
    timeout=3600,
)
def run_launch_safety_test():
    print_environment()

    print("===== FULL-SHAPE LAUNCH SAFETY =====")

    run_command(
        [
            "python",
            "tests/test_fullshape_launch_safety.py",
        ]
    )

    print()
    print("FULL-SHAPE SAFETY TEST PASSED")


# ---------------------------------------------------------
# Gate/up-only warp-amax benchmark
# ---------------------------------------------------------

@app.function(
    gpu="B200",
    cpu=16,
    memory=65536,
    timeout=3600,
)
def run_gateup_warp_benchmark():
    print_environment()

    print("===== GATE/UP WARP-AMAX BENCHMARK =====")

    run_command(
        [
            "python",
            "bench/benchmark_warp_amax.py",
        ]
    )

    print()
    print("GATE/UP BENCHMARK COMPLETED")


# ---------------------------------------------------------
# Complete MLP warp-amax benchmark
# ---------------------------------------------------------

@app.function(
    gpu="B200",
    cpu=16,
    memory=65536,
    timeout=3600,
)
def run_full_mlp_warp_benchmark():
    print_environment()

    print("===== FULL-MLP WARP-AMAX BENCHMARK =====")

    run_command(
        [
            "python",
            "bench/benchmark_full_mlp_warp_amax.py",
        ]
    )

    print()
    print("FULL-MLP WARP-AMAX BENCHMARK COMPLETED")


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


# ---------------------------------------------------------
# Local command dispatcher
# ---------------------------------------------------------
@app.local_entrypoint()
def main(task: str = "release"):
    print("Selected task:", task)

    if task == "release":
        run_release_tests.remote()
    elif task == "safety":
        run_launch_safety_test.remote()
    elif task == "gateup-benchmark":
        run_gateup_warp_benchmark.remote()
    elif task == "full-benchmark":
        run_full_mlp_warp_benchmark.remote()
    elif task == "optimized-benchmark":
        run_optimized_dispatch_benchmark.remote()
    elif task == "inspect-profiler":
        inspect_profiler.remote()
    elif task == "inspect-ncu-metrics":
        inspect_ncu_metrics.remote()  
    elif task == "profile-warp-amax":
        run_warp_amax_profile.remote()      
    else:
        raise ValueError(
            "Unknown task. Choose one of: release, safety, "
            "gateup-benchmark, full-benchmark, "
            "optimized-benchmark, inspect-profiler"
            "inspect-ncu-metrics",  "profile-warp-amax"
        )
        
        
@app.function(
    cpu=2,
    memory=4096,
    timeout=300,
)
def inspect_profiler():
    run_command(
        [
            "bash",
            "-lc",
            """
set -e
echo "===== NCU PATH ====="
command -v ncu || true

echo "===== NCU FILE SEARCH ====="
find /opt /usr/local -type f -name ncu 2>/dev/null | head -20

echo "===== AVAILABLE PACKAGES ====="
apt-cache search nsight-compute 2>/dev/null | head -20
""",
        ]
    )    
    
    
@app.function(
    gpu="B200",
    cpu=4,
    memory=8192,
    timeout=600,
)
def inspect_ncu_metrics():
    run_command(["ncu", "--version"])

    run_command(
        [
            "bash",
            "-lc",
            """
set -o pipefail

echo "===== ATOMIC/REDUCTION METRICS ====="

ncu --query-metrics 2>/dev/null |
grep -Ei \
'l1tex.*(atom|red)|lts.*(atom|red)|atomic' |
head -120
""",
        ]
    )        
    
    
@app.function(
    gpu="B200",
    cpu=16,
    memory=65536,
    timeout=3600,
)
def run_warp_amax_profile():
    print_environment()

    metrics = ",".join(
        [
            "gpu__time_duration.sum",
            (
                "l1tex__t_bytes_pipe_lsu_"
                "mem_global_op_atom.sum"
            ),
            (
                "l1tex__t_bytes_pipe_lsu_"
                "mem_global_op_red.sum"
            ),
            (
                "l1tex__t_sectors_pipe_lsu_"
                "mem_global_op_atom.sum"
            ),
            (
                "l1tex__t_sectors_pipe_lsu_"
                "mem_global_op_red.sum"
            ),
            (
                "l1tex__m_l1tex2xbar_write_sectors_"
                "mem_global_op_atom.sum"
            ),
            (
                "l1tex__m_l1tex2xbar_write_sectors_"
                "mem_global_op_red.sum"
            ),
        ]
    )

    for mode in ["original", "warp_amax"]:
        print()
        print(
            f"===== PROFILING {mode.upper()} ====="
        )

        report_path = (
            f"/tmp/warp_amax_{mode}_m64"
        )

        run_command(
            [
                
                "ncu",
                "--clock-control",
                "none",
                "--target-processes",
                "all",
                "--kernel-name",
                "regex:.*device_kernel.*",
                "--launch-count",
                "1",
                "--metrics",
                metrics,
                "--page",
                "raw",
                "--csv",
                "--force-overwrite",
                "-o",
                report_path,
                "python",
                "profiles/profile_warp_amax_atomic.py",
                "--mode",
                mode,
            ]
        )

        run_command(
            [
                "ncu",
                "--import",
                f"{report_path}.ncu-rep",
                "--page",
                "raw",
                "--csv",
            ]
        )

    print()
    print("WARP-AMAX NCU PROFILING COMPLETED")    