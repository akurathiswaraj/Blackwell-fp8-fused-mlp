
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

using MmaTileShape = Shape<_64, _64, _128>;
using ClusterShape = Shape<_1, _1, _1>;

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
        cutlass::epilogue::collective::
            EpilogueScheduleAuto
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


PYBIND11_MODULE(
    TORCH_EXTENSION_NAME,
    module
) {
    module.def(
        "gemm",
        &sm100_fp8_gemm,
        "SM100 FP8 GEMM with BF16 output"
    );
}
