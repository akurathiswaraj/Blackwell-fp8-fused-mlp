
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"

#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

namespace {

using ElementA = cutlass::float_e4m3_t;
using LayoutA = cutlass::layout::RowMajor;
constexpr int AlignmentA =
    128 / cutlass::sizeof_bits<ElementA>::value;

using ElementB = cutlass::float_e4m3_t;
using LayoutB = cutlass::layout::ColumnMajor;
constexpr int AlignmentB =
    128 / cutlass::sizeof_bits<ElementB>::value;

using ElementC = cutlass::bfloat16_t;
using LayoutC = cutlass::layout::RowMajor;
constexpr int AlignmentC =
    128 / cutlass::sizeof_bits<ElementC>::value;

using ElementAccumulator = float;
using ElementCompute = float;

using MmaTileShape = Shape<_64, _128, _128>;
// Two CTAs cooperate along N. CUTLASS selects
// SM90_TMA_LOAD_MULTICAST for operand A, so both CTAs
// receive the same activation tile from one TMA transaction.
using ClusterShape = Shape<_1, _2, _1>;

using CollectiveEpilogue =
    typename cutlass::epilogue::collective::
    CollectiveBuilder<
        cutlass::arch::Sm100,
        cutlass::arch::OpClassTensorOp,
        MmaTileShape,
        ClusterShape,
        cutlass::epilogue::collective::
            EpilogueTileAuto,
        ElementAccumulator,
        ElementCompute,
        ElementC,
        LayoutC,
        AlignmentC,
        ElementC,
        LayoutC,
        AlignmentC,
        cutlass::epilogue::
            NoSmemWarpSpecialized1Sm
    >::CollectiveOp;

using CollectiveMainloop =
    typename cutlass::gemm::collective::
    CollectiveBuilder<
        cutlass::arch::Sm100,
        cutlass::arch::OpClassTensorOp,
        ElementA,
        LayoutA,
        AlignmentA,
        ElementB,
        LayoutB,
        AlignmentB,
        ElementAccumulator,
        MmaTileShape,
        ClusterShape,
        cutlass::gemm::collective::
            StageCountAutoCarveout<
                static_cast<int>(
                    sizeof(
                        typename CollectiveEpilogue::
                            SharedStorage
                    )
                )
            >,
        cutlass::gemm::
            KernelTmaWarpSpecialized1SmSm100
    >::CollectiveOp;

using GemmKernel =
    cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        void
    >;

using Gemm =
    cutlass::gemm::device::
    GemmUniversalAdapter<GemmKernel>;

using StrideA = typename GemmKernel::StrideA;
using StrideB = typename GemmKernel::StrideB;
using StrideC = typename GemmKernel::StrideC;
using StrideD = typename GemmKernel::StrideD;

void check_cutlass(
    cutlass::Status status,
    const char* operation
) {
    TORCH_CHECK(
        status == cutlass::Status::kSuccess,
        operation,
        " failed with CUTLASS status ",
        static_cast<int>(status)
    );
}

}  // namespace


torch::Tensor sm100_fp8_gemm(
    torch::Tensor a,
    torch::Tensor b,
    double output_scale
) {
    TORCH_CHECK(a.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "B must be a CUDA tensor");

    TORCH_CHECK(
        a.is_contiguous(),
        "A must be contiguous"
    );

    TORCH_CHECK(
        b.is_contiguous(),
        "B must be contiguous"
    );

    TORCH_CHECK(
        a.dim() == 2,
        "A must have shape [M, K]"
    );

    TORCH_CHECK(
        b.dim() == 2,
        "B must have shape [N, K]"
    );

    TORCH_CHECK(
        a.element_size() == 1,
        "A must contain 8-bit FP8 elements"
    );

    TORCH_CHECK(
        b.element_size() == 1,
        "B must contain 8-bit FP8 elements"
    );

    TORCH_CHECK(
        a.size(1) == b.size(1),
        "A and B must have the same K dimension"
    );

    TORCH_CHECK(
        a.get_device() == b.get_device(),
        "A and B must be on the same GPU"
    );

    int M = static_cast<int>(a.size(0));
    int N = static_cast<int>(b.size(0));
    int K = static_cast<int>(a.size(1));

    TORCH_CHECK(
        K % AlignmentA == 0,
        "K must satisfy FP8 alignment"
    );

    TORCH_CHECK(
        N % AlignmentC == 0,
        "N must satisfy BF16 output alignment"
    );

    auto output = torch::empty(
        {M, N},
        a.options().dtype(torch::kBFloat16)
    );

    int32_t m = static_cast<int32_t>(M);
    int32_t n = static_cast<int32_t>(N);
    int32_t k = static_cast<int32_t>(K);
    int32_t l = 1;

    StrideA stride_a =
        cutlass::make_cute_packed_stride(
            StrideA{},
            cute::make_shape(m, k, l)
        );

    StrideB stride_b =
        cutlass::make_cute_packed_stride(
            StrideB{},
            cute::make_shape(n, k, l)
        );

    StrideC stride_c =
        cutlass::make_cute_packed_stride(
            StrideC{},
            cute::make_shape(m, n, l)
        );

    StrideD stride_d =
        cutlass::make_cute_packed_stride(
            StrideD{},
            cute::make_shape(m, n, l)
        );

    auto* a_ptr = reinterpret_cast<ElementA*>(
        a.data_ptr()
    );

    auto* b_ptr = reinterpret_cast<ElementB*>(
        b.data_ptr()
    );

    auto* output_ptr =
        reinterpret_cast<ElementC*>(
            output.data_ptr()
        );

    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {
            a_ptr,
            stride_a,
            b_ptr,
            stride_b
        },
        {
            {
                static_cast<float>(output_scale),
                0.0f
            },
            output_ptr,
            stride_c,
            output_ptr,
            stride_d
        }
    };

    Gemm gemm;

    size_t workspace_size =
        Gemm::get_workspace_size(arguments);

    auto workspace = torch::empty(
        {
            static_cast<int64_t>(
                workspace_size
            )
        },
        a.options().dtype(torch::kUInt8)
    );

    void* workspace_ptr =
        workspace_size == 0
        ? nullptr
        : workspace.data_ptr();

    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(
            a.get_device()
        ).stream();

    check_cutlass(
        gemm.can_implement(arguments),
        "can_implement"
    );

    check_cutlass(
        gemm.initialize(
            arguments,
            workspace_ptr,
            stream
        ),
        "initialize"
    );

    check_cutlass(
        gemm.run(stream),
        "run"
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}




// Fuses the post-GEMM Llama SwiGLU operations:
//   output = quantize_fp8(SiLU(gate) * up)
// while also updating a caller-owned running amax.
//
// The combined GEMM output uses an interleaved layout:
//   [gate0, up0, gate1, up1, ...]
// This makes each SwiGLU pair adjacent in the accumulator
// N dimension, preparing for a custom on-chip epilogue.
__global__ void swiglu_amax_quantize_kernel(
    cutlass::bfloat16_t const* combined,
    cutlass::float_e4m3_t* output,
    int64_t rows,
    int64_t intermediate_size,
    float quant_multiplier,
    float* running_amax
) {
    int64_t linear_index =
        static_cast<int64_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;

    int64_t total_elements =
        rows * intermediate_size;

    if (linear_index >= total_elements) {
        return;
    }

    int64_t row =
        linear_index / intermediate_size;

    int64_t column =
        linear_index - row * intermediate_size;

    int64_t combined_stride =
        intermediate_size * 2;

    int64_t gate_index =
        row * combined_stride
        + 2 * column;

    int64_t up_index =
        gate_index + 1;

    float gate =
        static_cast<float>(combined[gate_index]);

    float up =
        static_cast<float>(combined[up_index]);

    float silu_gate =
        gate / (1.0f + expf(-gate));

    float value = silu_gate * up;
    float absolute_value = fabsf(value);

    // Values are non-negative, so integer atomicMax preserves
    // floating-point ordering for finite FP32 values.
    atomicMax(
        reinterpret_cast<unsigned int*>(running_amax),
        __float_as_uint(absolute_value)
    );

    float scaled = value * quant_multiplier;

    // E4M3FN maximum finite magnitude.
    scaled = fminf(448.0f, fmaxf(-448.0f, scaled));

    output[linear_index] =
        cutlass::float_e4m3_t(scaled);
}


std::vector<torch::Tensor>
sm100_fp8_gate_up_swiglu_pair_swap_probe(
    torch::Tensor a,
    torch::Tensor packed_gate_up_weight,
    double gemm_scale,
    double quant_multiplier,
    torch::Tensor running_amax
) {
    TORCH_CHECK(
        packed_gate_up_weight.dim() == 2,
        "Packed gate/up weight must have shape [2N, K]"
    );

    TORCH_CHECK(
        packed_gate_up_weight.size(0) % 2 == 0,
        "Packed gate/up output dimension must be even"
    );

    TORCH_CHECK(
        running_amax.is_cuda(),
        "running_amax must be a CUDA tensor"
    );

    TORCH_CHECK(
        running_amax.is_contiguous(),
        "running_amax must be contiguous"
    );

    TORCH_CHECK(
        running_amax.scalar_type() == torch::kFloat32,
        "running_amax must be FP32"
    );

    TORCH_CHECK(
        running_amax.numel() == 1,
        "running_amax must contain one FP32 value"
    );

    TORCH_CHECK(
        running_amax.get_device() == a.get_device(),
        "running_amax and A must be on the same GPU"
    );

    // Existing SM100 CUTLASS path. With ClusterShape 1x2x1,
    // CUTLASS selects TMA multicast for operand A.
    torch::Tensor combined =
        sm100_fp8_gemm(
            a,
            packed_gate_up_weight,
            gemm_scale
        );

    int64_t rows = combined.size(0);
    int64_t intermediate_size =
        combined.size(1) / 2;

    auto output = torch::empty(
        {
            rows,
            intermediate_size
        },
        a.options()
    );

    auto* combined_ptr =
        reinterpret_cast<cutlass::bfloat16_t const*>(
            combined.data_ptr()
        );

    auto* output_ptr =
        reinterpret_cast<cutlass::float_e4m3_t*>(
            output.data_ptr()
        );

    auto* amax_ptr =
        reinterpret_cast<float*>(
            running_amax.data_ptr()
        );

    int64_t total_elements =
        rows * intermediate_size;

    constexpr int threads = 256;

    int blocks = static_cast<int>(
        (total_elements + threads - 1) / threads
    );

    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(
            a.get_device()
        ).stream();

    swiglu_amax_quantize_kernel<<<
        blocks,
        threads,
        0,
        stream
    >>>(
        combined_ptr,
        output_ptr,
        rows,
        intermediate_size,
        static_cast<float>(quant_multiplier),
        amax_ptr
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {
        output,
        running_amax
    };
}

PYBIND11_MODULE(
    TORCH_EXTENSION_NAME,
    module
) {
    module.def(
        "gemm",
        &sm100_fp8_gemm,
        "SM100 FP8 GEMM with BF16 output"
    );
    module.def(
        "gate_up_swiglu_fp8_pair_swap_probe",
        &sm100_fp8_gate_up_swiglu_pair_swap_probe,
        "SM100 interleaved FP8 gate/up GEMM + fused "
        "SiLU multiply + running amax + FP8 output"
    );

}
