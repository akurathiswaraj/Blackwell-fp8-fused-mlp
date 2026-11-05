"""Repository checks that do not require CUDA or a GPU."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]


def validate_python():
    errors = []

    for path in sorted(
        ROOT.rglob("*.py")
    ):
        if any(
            part in {
                "build",
                "__pycache__",
            }
            for part in path.parts
        ):
            continue

        try:
            ast.parse(
                path.read_text(
                    encoding="utf-8"
                ),
                filename=str(path),
            )

        except (
            SyntaxError,
            UnicodeDecodeError,
        ) as error:
            errors.append(
                "Python parse failure: "
                f"{path.relative_to(ROOT)}: "
                f"{error}"
            )

    return errors


def validate_json():
    errors = []

    for path in sorted(
        ROOT.rglob("*.json")
    ):
        try:
            json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )

        except (
            json.JSONDecodeError,
            UnicodeDecodeError,
        ) as error:
            errors.append(
                "JSON parse failure: "
                f"{path.relative_to(ROOT)}: "
                f"{error}"
            )

    return errors


def validate_markdown_links():
    errors = []

    pattern = re.compile(
        r"\[[^]]*\]\(([^)]+)\)"
    )

    for path in sorted(
        ROOT.rglob("*.md")
    ):
        text = path.read_text(
            encoding="utf-8"
        )

        for target in pattern.findall(text):
            if target.startswith(
                (
                    "http://",
                    "https://",
                    "#",
                    "mailto:",
                )
            ):
                continue

            clean_target = target.split(
                "#",
                1,
            )[0]

            if not clean_target:
                continue

            resolved = (
                path.parent
                / clean_target
            ).resolve()

            if not resolved.exists():
                errors.append(
                    "Broken local link: "
                    f"{path.relative_to(ROOT)} "
                    f"-> {target}"
                )

    return errors


def validate_portability():
    errors = []

    source_suffixes = {
        ".py",
        ".md",
        ".cu",
    }

    validator_path = Path(
        __file__
    ).resolve()

    for path in sorted(
        ROOT.rglob("*")
    ):
        if not path.is_file():
            continue

        if path.suffix not in source_suffixes:
            continue

        if path.resolve() == validator_path:
            continue

        if any(
            part in {
                "build",
                "__pycache__",
            }
            for part in path.parts
        ):
            continue

        try:
            text = path.read_text(
                encoding="utf-8"
            )
        except UnicodeDecodeError:
            continue

        if "/root/" in text:
            errors.append(
                "Hard-coded /root path: "
                f"{path.relative_to(ROOT)}"
            )

    return errors


def validate_required_files():
    required = [
        "integration/"
        "sm100_fp8_swiglu_direct_torch.cu",

        "integration/"
        "sm100_fp8_swiglu_direct_"
        "warp_amax_torch.cu",

        "integration/build_extensions.py",
        "integration/fp8_mlp.py",
        "integration/llama_mlp.py",

        "tests/test_fp8_mlp.py",
        "tests/test_decode_extended.py",
        "tests/test_runtime_robustness.py",
        "tests/test_warp_amax_variant.py",
        "tests/test_warp_amax_dispatch.py",
        "tests/test_llama_optimized_dispatch.py",

        "bench/benchmark_optimized_dispatch.py",
        "bench/optimized_dispatch_decode.csv",
        "bench/optimized_dispatch_decode_batches.csv",
        "bench/optimized_dispatch_decode_raw.csv",

        "results/optimized_dispatch_decode.json",
        "results/full_mlp_warp_amax_ab_m64.json",
        
        "profiles/profile_warp_amax_atomic.py",

        "bench/warp_amax_ncu_m64.csv",

        "results/warp_amax_ncu_m64.json",
        "results/final_summary_optimized_dispatch.json",

        "scripts/record_warp_amax_profile.py",
        "scripts/finalize_optimized_release.py",

        "README.md",
        "DESIGN.md",
        "RESUME.md",
        "modal_b200.py",
    ]

    return [
        f"Missing required file: {relative}"
        for relative in required
        if not (ROOT / relative).exists()
    ]


def main():
    errors = (
        validate_python()
        + validate_json()
        + validate_markdown_links()
        + validate_portability()
        + validate_required_files()
    )

    if errors:
        print(
            "Repository validation failed:"
        )

        for error in errors:
            print("-", error)

        return 1

    print("Repository validation passed")
    print(
        "- Python sources parse successfully"
    )
    print("- JSON files are valid")
    print(
        "- Local Markdown links resolve"
    )
    print(
        "- No hard-coded /root paths "
        "were found in source files"
    )
    print(
        "- Optimized warp-amax release "
        "files are present"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())