import torch
import torch.nn.functional as F

from integration.build_extensions import build_extensions
from integration.fp8_mlp import (
    custom_decode_mlp,
    interleave_gate_up_weight,
)


FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0


def relative_l2(actual, reference):
    numerator = torch.linalg.vector_norm(
        actual.float() - reference.float()
    )
    denominator = torch.linalg.vector_norm(
        reference.float()
    ).clamp_min(1.0e-12)

    return float((numerator / denominator).item())


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    if torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("SM100 GPU required")

    torch.backends.cuda.matmul.allow_tf32 = False

    print("Loading extensions...")
    direct, _, down = build_extensions()

    torch.manual_seed(20261105)

    K = 256
    N = 256

    blocked_weight = (
        torch.randn(
            2 * N,
            K,
            device="cuda",
            dtype=torch.float16,
        )
        * 32
    ).to(FP8)

    interleaved_weight = (
        interleave_gate_up_weight(
            blocked_weight
        )
    )

    down_weight = (
        torch.randn(
            K,
            N,
            device="cuda",
            dtype=torch.float16,
        )
        * 32
    ).to(FP8)

    gate_weight, up_weight = blocked_weight.chunk(
        2,
        dim=0,
    )

    scale_a = 1.0e-3
    scale_weight = 1.0e-2
    scale_down = 1.0e-2

    test_sizes = [
        1, 2, 3, 4, 5, 7, 8,
        15, 16, 31, 32, 63, 64,
    ]

    results = []

    for M in test_sizes:
        activation = (
            torch.randn(
                M,
                K,
                device="cuda",
                dtype=torch.float16,
            )
            * 32
        ).to(FP8)

        gate_reference = (
            activation.float() * scale_a
        ) @ (
            gate_weight.float() * scale_weight
        ).t()

        up_reference = (
            activation.float() * scale_a
        ) @ (
            up_weight.float() * scale_weight
        ).t()

        product_reference = (
            F.silu(gate_reference)
            * up_reference
        )

        reference_amax = (
            product_reference.abs().amax()
        )

        quant_multiplier = float(
            (
                FP8_MAX
                / reference_amax.clamp_min(1.0e-12)
            ).item()
        )

        product_fp8 = torch.clamp(
            product_reference * quant_multiplier,
            -FP8_MAX,
            FP8_MAX,
        ).to(FP8)

        output_reference = (
            product_fp8.float()
            / quant_multiplier
        ) @ (
            down_weight.float()
            * scale_down
        ).t()

        def run_custom(stream=None):
            running_amax = torch.zeros(
                1,
                device="cuda",
                dtype=torch.float32,
            )

            def launch():
                return custom_decode_mlp(
                    activation,
                    interleaved_weight,
                    down_weight,
                    scale_a * scale_weight,
                    quant_multiplier,
                    scale_down / quant_multiplier,
                    running_amax,
                    direct,
                    down,
                )

            if stream is None:
                output = launch()
            else:
                with torch.cuda.stream(stream):
                    output = launch()

                torch.cuda.current_stream().wait_stream(
                    stream
                )

            return output, running_amax

        output_1, amax_1 = run_custom()
        output_2, amax_2 = run_custom()

        test_stream = torch.cuda.Stream()
        stream_output, stream_amax = run_custom(
            test_stream
        )

        torch.cuda.synchronize()

        output_error = relative_l2(
            output_1,
            output_reference,
        )

        amax_error = float(
            (
                (amax_1[0] - reference_amax).abs()
                / reference_amax.clamp_min(1.0e-12)
            ).item()
        )

        deterministic = torch.equal(
            output_1,
            output_2,
        )

        stream_match = torch.equal(
            output_1,
            stream_output,
        )

        amax_match = (
            torch.equal(amax_1, amax_2)
            and torch.equal(
                amax_1,
                stream_amax,
            )
        )

        finite = bool(
            torch.isfinite(
                output_1.float()
            ).all().item()
        )

        passed = (
            output_1.shape == (M, K)
            and output_1.dtype == torch.bfloat16
            and finite
            and output_error < 4.0e-2
            and amax_error < 2.0e-2
            and deterministic
            and stream_match
            and amax_match
        )

        print(
            f"M={M:2d} | "
            f"rel_l2={output_error:.6e} | "
            f"amax_error={amax_error:.6e} | "
            f"deterministic={deterministic} | "
            f"stream={stream_match} | "
            f"passed={passed}"
        )

        if not passed:
            raise RuntimeError(
                f"Extended test failed at M={M}"
            )

        results.append(
            (output_error, amax_error)
        )

    print()
    print("EXTENDED DECODE TEST PASSED")
    print("Cases:", len(results))
    print(
        "Maximum relative L2:",
        f"{max(x[0] for x in results):.6e}",
    )
    print(
        "Maximum amax error:",
        f"{max(x[1] for x in results):.6e}",
    )


if __name__ == "__main__":
    main()