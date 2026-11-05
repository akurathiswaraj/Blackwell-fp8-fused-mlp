from pathlib import Path
import ast
import re


PROJECT = Path(__file__).resolve().parents[1]


def replace_exactly_once(
    source,
    old,
    new,
    description,
):
    count = source.count(old)

    if count != 1:
        raise RuntimeError(
            f"{description}: expected one match, "
            f"found {count}"
        )

    return source.replace(old, new, 1)


# ---------------------------------------------------------
# 1. Extension builder
# ---------------------------------------------------------

build_path = (
    PROJECT
    / "integration"
    / "build_extensions.py"
)

build_source = build_path.read_text()

if "def build_warp_amax_extension():" not in build_source:
    build_source = build_source.rstrip() + r'''


def build_warp_amax_extension():
    """Build the warp-reduced amax direct epilogue."""

    build_directory = (
        PROJECT
        / "build"
        / "torch_extension_swiglu_warp_amax_release_v1"
    )

    build_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return load(
        name="sm100_fp8_swiglu_warp_amax_release_ext_v1",
        sources=[
            str(
                PROJECT
                / "integration"
                / "sm100_fp8_swiglu_direct_warp_amax_torch.cu"
            )
        ],
        build_directory=str(build_directory),
        **COMMON,
    )


def build_optimized_extensions():
    """Build original, optimized, fallback, and down kernels."""

    direct, fallback, down = build_extensions()
    warp_amax = build_warp_amax_extension()

    return (
        direct,
        warp_amax,
        fallback,
        down,
    )
''' + "\n"

build_path.write_text(build_source)


# ---------------------------------------------------------
# 2. Runtime dispatch
# ---------------------------------------------------------

runtime_path = (
    PROJECT
    / "integration"
    / "fp8_mlp.py"
)

runtime_source = runtime_path.read_text()


# Add threshold constant.
if "WARP_AMAX_MIN_M" not in runtime_source:
    runtime_source, replacement_count = re.subn(
        r"^DECODE_MAX_M\s*=\s*64\s*$",
        (
            "DECODE_MAX_M = 64\n"
            "WARP_AMAX_MIN_M = 40"
        ),
        runtime_source,
        count=1,
        flags=re.MULTILINE,
    )

    if replacement_count != 1:
        raise RuntimeError(
            "Could not locate DECODE_MAX_M"
        )


# Modify custom_decode_mlp section.
custom_start = runtime_source.index(
    "def custom_decode_mlp("
)

custom_end = runtime_source.index(
    "\ndef two_kernel_decode_mlp(",
    custom_start,
)

custom_block = runtime_source[
    custom_start:custom_end
]


if "warp_amax_extension=None" not in custom_block:
    custom_block = replace_exactly_once(
        custom_block,
        """\
    direct_extension,
    down_extension,
):
""",
        """\
    direct_extension,
    down_extension,
    warp_amax_extension=None,
):
""",
        "custom_decode_mlp signature",
    )


if "selected_direct_extension = (" not in custom_block:
    custom_block = replace_exactly_once(
        custom_block,
        """\
    product_fp8 = (
""",
        """\
    m_value = int(activation.shape[0])

    selected_direct_extension = (
        direct_extension
    )

    if (
        warp_amax_extension is not None
        and WARP_AMAX_MIN_M
        <= m_value
        <= DECODE_MAX_M
    ):
        selected_direct_extension = (
            warp_amax_extension
        )

    product_fp8 = (
""",
        "custom extension selection",
    )


if "direct_extension.swiglu_direct(" in custom_block:
    custom_block = replace_exactly_once(
        custom_block,
        "direct_extension.swiglu_direct(",
        "selected_direct_extension.swiglu_direct(",
        "selected direct call",
    )


if (
    "        direct_extension.swiglu_direct("
    in custom_block
):
    raise RuntimeError(
        "Selected extension call is missing"
    )


runtime_source = (
    runtime_source[:custom_start]
    + custom_block
    + runtime_source[custom_end:]
)


# Modify dispatch_mlp section.
dispatch_start = runtime_source.index(
    "def dispatch_mlp("
)

dispatch_block = runtime_source[dispatch_start:]


dispatch_header_end = dispatch_block.index(
    "):"
) + 2

dispatch_header = dispatch_block[
    :dispatch_header_end
]

dispatch_body = dispatch_block[
    dispatch_header_end:
]


if "warp_amax_extension=None" not in dispatch_header:
    dispatch_header = replace_exactly_once(
        dispatch_header,
        """\
    direct_extension,
    down_extension,
):
""",
        """\
    direct_extension,
    down_extension,
    warp_amax_extension=None,
):
""",
        "dispatch_mlp signature",
    )


dispatch_block = (
    dispatch_header
    + dispatch_body
)


custom_call_start = dispatch_block.index(
    "        return custom_decode_mlp("
)

custom_call_end = dispatch_block.index(
    "\n\n    return cublaslt_mlp(",
    custom_call_start,
)

dispatch_custom_call = dispatch_block[
    custom_call_start:custom_call_end
]


if (
    "warp_amax_extension="
    not in dispatch_custom_call
):
    dispatch_custom_call = (
        replace_exactly_once(
            dispatch_custom_call,
            """\
            direct_extension,
            down_extension,
        )
""",
            """\
            direct_extension,
            down_extension,
            warp_amax_extension=(
                warp_amax_extension
            ),
        )
""",
            "dispatch custom call",
        )
    )


dispatch_block = (
    dispatch_block[:custom_call_start]
    + dispatch_custom_call
    + dispatch_block[custom_call_end:]
)


runtime_source = (
    runtime_source[:dispatch_start]
    + dispatch_block
)

runtime_path.write_text(runtime_source)


# ---------------------------------------------------------
# 3. Llama-compatible module
# ---------------------------------------------------------

llama_path = (
    PROJECT
    / "integration"
    / "llama_mlp.py"
)

llama_source = llama_path.read_text()


constructor_start = llama_source.index(
    "    def __init__("
)

constructor_end = llama_source.index(
    "    ):",
    constructor_start,
) + len("    ):")

constructor = llama_source[
    constructor_start:constructor_end
]


if "warp_amax_extension=None" not in constructor:
    constructor = replace_exactly_once(
        constructor,
        """\
        direct_extension,
        down_extension,
""",
        """\
        direct_extension,
        down_extension,
        warp_amax_extension=None,
""",
        "Llama constructor",
    )


llama_source = (
    llama_source[:constructor_start]
    + constructor
    + llama_source[constructor_end:]
)


if "_warp_amax_extension" not in llama_source:
    llama_source = replace_exactly_once(
        llama_source,
        """\
        object.__setattr__(
            self,
            "_down_extension",
            down_extension,
        )
""",
        """\
        object.__setattr__(
            self,
            "_down_extension",
            down_extension,
        )
        object.__setattr__(
            self,
            "_warp_amax_extension",
            warp_amax_extension,
        )
""",
        "Llama extension storage",
    )


dispatch_call_start = llama_source.index(
    "        output = dispatch_mlp("
)

dispatch_call_end = llama_source.index(
    "\n\n        return output.reshape(",
    dispatch_call_start,
)

llama_dispatch_call = llama_source[
    dispatch_call_start:dispatch_call_end
]


if (
    "warp_amax_extension="
    not in llama_dispatch_call
):
    llama_dispatch_call = (
        replace_exactly_once(
            llama_dispatch_call,
            """\
            self._direct_extension,
            self._down_extension,
        )
""",
            """\
            self._direct_extension,
            self._down_extension,
            warp_amax_extension=(
                self._warp_amax_extension
            ),
        )
""",
            "Llama dispatch call",
        )
    )


llama_source = (
    llama_source[:dispatch_call_start]
    + llama_dispatch_call
    + llama_source[dispatch_call_end:]
)

llama_path.write_text(llama_source)


# ---------------------------------------------------------
# 4. Validate Python syntax
# ---------------------------------------------------------

for path in [
    build_path,
    runtime_path,
    llama_path,
]:
    ast.parse(
        path.read_text(),
        filename=str(path),
    )

    print("Validated:", path)


print()
print("Warp-amax production dispatch installed")
print("Original kernel: M < 40")
print("Warp-amax kernel: 40 <= M <= 64")
print("cuBLASLt fallback: M > 64")