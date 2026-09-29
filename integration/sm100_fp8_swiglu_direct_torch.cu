
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/sm100_epilogue_nosmem.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"


namespace cutlass::epilogue::collective {

using ProjectSwiGLUPrototype =
    typename CollectiveBuilder<
        cutlass::arch::Sm100,
        cutlass::arch::OpClassTensorOp,
        cute::Shape<cute::_64, cute::_128, cute::_128>,
        cute::Shape<cute::_1, cute::_2, cute::_1>,
        EpilogueTileAuto,
        float,
        float,
        cutlass::bfloat16_t,
        cutlass::layout::RowMajor,
        8,
        cutlass::bfloat16_t,
        cutlass::layout::RowMajor,
        8,
        cutlass::epilogue::NoSmemWarpSpecialized1Sm
    >::CollectiveOp;

using ProjectSwiGLUEpilogueTile =
    typename ProjectSwiGLUPrototype::EpilogueTile;

using ProjectSwiGLUCopyOpT2R =
    typename ProjectSwiGLUPrototype::CopyOpT2R;

using ProjectSwiGLUThreadOp =
    typename ProjectSwiGLUPrototype::ThreadEpilogueOp;

using ProjectSwiGLUStrideC =
    typename ProjectSwiGLUPrototype::StrideC;

using ProjectSwiGLUStrideD =
    typename ProjectSwiGLUPrototype::StrideD;

} // namespace cutlass::epilogue::collective

namespace {

using ElementA = cutlass::float_e4m3_t;
using LayoutA = cutlass::layout::RowMajor;
constexpr int AlignmentA =
    128 / cutlass::sizeof_bits<ElementA>::value;

using ElementB = cutlass::float_e4m3_t;
using LayoutB = cutlass::layout::ColumnMajor;
constexpr int AlignmentB =
    128 / cutlass::sizeof_bits<ElementB>::value;

using ElementAccumulator = float;
using ElementCompute = float;
using OutputElement = cutlass::float_e4m3_t;
using LayoutD = cutlass::layout::RowMajor;

using MmaTileShape = cute::Shape<cute::_64, cute::_128, cute::_128>;
using ClusterShape = cute::Shape<cute::_1, cute::_2, cute::_1>;

// Build a stock no-SMEM epilogue only to obtain CUTLASS's
// validated SM100 epilogue tile and TMEM-to-register copy atom.
using PrototypeEpilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm100,
        cutlass::arch::OpClassTensorOp,
        MmaTileShape,
        ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator,
        ElementCompute,
        cutlass::bfloat16_t,
        cutlass::layout::RowMajor,
        8,
        cutlass::bfloat16_t,
        cutlass::layout::RowMajor,
        8,
        cutlass::epilogue::NoSmemWarpSpecialized1Sm
    >::CollectiveOp;

class DirectSwiGLUEpilogue {
public:
    using DispatchPolicy =
        cutlass::epilogue::Sm100NoSmem;

    using EpilogueTile =
        cutlass::epilogue::collective::ProjectSwiGLUEpilogueTile;

    using CopyOpT2R =
        cutlass::epilogue::collective::ProjectSwiGLUCopyOpT2R;

    // Compatibility alias required by GemmUniversalAdapter.
    // The custom collective performs its own output operation.
    using ThreadEpilogueOp =
        cutlass::epilogue::collective::ProjectSwiGLUThreadOp;

    using ElementC = void;
    using StrideC =
        cutlass::epilogue::collective::ProjectSwiGLUStrideC;

    using ElementD = OutputElement;
    using ElementOutput = OutputElement;
    using StrideD =
        cutlass::epilogue::collective::ProjectSwiGLUStrideD;

    using GmemTiledCopyC = void;
    using GmemTiledCopyD = void;

    static constexpr int ThreadCount = 128;
    static constexpr int kOutputAlignment = 1;
    static constexpr bool isEpilogueBiasSupported = false;
    static constexpr uint32_t TmaTransactionBytes = 0;

    struct SharedStorage {};

    struct Arguments {
        float gemm_scale = 1.0f;
        float quant_multiplier = 1.0f;
        ElementD* ptr_D = nullptr;
        float* running_amax = nullptr;
    };

    using Params = Arguments;

    template <class ProblemShape>
    static constexpr Params to_underlying_arguments(
        ProblemShape const&,
        Arguments const& args,
        void*
    ) {
        return args;
    }

    template <class ProblemShape>
    static size_t get_workspace_size(
        ProblemShape const&,
        Arguments const&
    ) {
        return 0;
    }

    template <class ProblemShape>
    static cutlass::Status initialize_workspace(
        ProblemShape const&,
        Arguments const&,
        void*,
        cudaStream_t,
        cutlass::CudaHostAdapter* = nullptr
    ) {
        return cutlass::Status::kSuccess;
    }

    template <class ProblemShape>
    static bool can_implement(
        ProblemShape const&,
        Arguments const& args
    ) {
        return (
            args.ptr_D != nullptr
            && args.running_amax != nullptr
        );
    }

    CUTLASS_DEVICE
    DirectSwiGLUEpilogue(
        Params const& params_,
        SharedStorage&
    ) : params(params_) {}

private:
    Params const& params;

public:
    template <
        bool ReuseTmem = false,
        class AccumulatorPipeline,
        class AccumulatorPipelineState,
        class ProblemShapeMNKL,
        class TileShapeMNK,
        class TileCoordMNKL,
        class AccEngine,
        class AccLayout
    >
    CUTLASS_DEVICE auto operator()(
        AccumulatorPipeline acc_pipeline,
        AccumulatorPipelineState acc_state,
        ProblemShapeMNKL problem_shape,
        TileShapeMNK cta_tile_shape,
        TileCoordMNKL cta_coord,
        cute::Tensor<AccEngine, AccLayout> const& accumulators,
        SharedStorage&
    ) {
        using namespace cute;

        static_assert(
            is_tmem<AccEngine>::value,
            "Accumulator must reside in TMEM"
        );

        auto [M, combined_N, K, L] = problem_shape;
        int output_N = int(combined_N) / 2;

        auto problem_mnl =
            select<0, 1, 3>(problem_shape);
        auto cta_coord_mnl =
            select<0, 1, 3>(cta_coord);
        auto cta_tiler =
            take<0, 2>(cta_tile_shape);

        auto tiled_t2r = make_tmem_copy(
            CopyOpT2R{},
            tensor<0>(accumulators)
        );

        int thread_idx =
            int(threadIdx.x) % int(size(tiled_t2r));

        auto thread_t2r =
            tiled_t2r.get_slice(thread_idx);

        Tensor coord = make_identity_tensor(
            problem_mnl
        );

        Tensor coord_tile = local_tile(
            coord,
            cta_tiler,
            cta_coord_mnl
        );

        Tensor thread_coord =
            thread_t2r.partition_D(coord_tile);

        using Accumulator =
            typename AccEngine::value_type;

        Tensor register_accumulator =
            make_tensor<Accumulator>(
                shape(thread_coord)
            );

        Tensor tmem_accumulator =
            accumulators(
                make_coord(_, _),
                _0{},
                _0{}
            );

        Tensor thread_tmem =
            thread_t2r.partition_S(
                tmem_accumulator
            );

        copy(
            tiled_t2r,
            thread_tmem,
            register_accumulator
        );

        cutlass::arch::fence_view_async_tmem_load();
        acc_pipeline.consumer_release(acc_state);
        ++acc_state;

        CUTLASS_PRAGMA_UNROLL
        for (
            int i = 0;
            i < int(size(register_accumulator));
            ++i
        ) {
            auto logical_coord = thread_coord(i);

            if (
                elem_less(
                    logical_coord,
                    problem_mnl
                )
            ) {
                int row =
                    int(get<0>(logical_coord));
                int combined_column =
                    int(get<1>(logical_coord));
                int batch =
                    int(get<2>(logical_coord));

                // Only the gate member of each adjacent pair
                // writes an output. i^1 is the matching up value.
                if ((combined_column & 1) == 0) {
                    float gate =
                        float(register_accumulator(i))
                        * params.gemm_scale;

                    float up =
                        float(register_accumulator(i ^ 1))
                        * params.gemm_scale;

                    float silu_gate =
                        gate / (1.0f + expf(-gate));

                    float value = silu_gate * up;

                    atomicMax(
                        reinterpret_cast<unsigned int*>(
                            params.running_amax
                        ),
                        __float_as_uint(fabsf(value))
                    );

                    float scaled =
                        value * params.quant_multiplier;

                    scaled = fminf(
                        448.0f,
                        fmaxf(-448.0f, scaled)
                    );

                    int output_column =
                        combined_column / 2;

                    int64_t output_index =
                        (
                            (
                                int64_t(batch)
                                * int64_t(M)
                                + int64_t(row)
                            )
                            * int64_t(output_N)
                        )
                        + int64_t(output_column);

                    params.ptr_D[output_index] =
                        ElementD(scaled);
                }
            }
        }

        return cute::make_tuple(acc_state);
    }
};

using CollectiveEpilogue =
    cutlass::epilogue::collective::detail::
        Sm100TmaWarpSpecializedAdapter<
            DirectSwiGLUEpilogue
        >;

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
        cute::Shape<int, int, int, int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        void
    >;

using Gemm =
    cutlass::gemm::device::
        GemmUniversalAdapter<GemmKernel>;

using StrideA = typename GemmKernel::StrideA;
using StrideB = typename GemmKernel::StrideB;

void check_cutlass(
    cutlass::Status status,
    char const* operation
) {
    TORCH_CHECK(
        status == cutlass::Status::kSuccess,
        operation,
        " failed with CUTLASS status ",
        static_cast<int>(status)
    );
}

} // namespace

std::vector<torch::Tensor>
sm100_fp8_swiglu_direct(
    torch::Tensor a,
    torch::Tensor interleaved_weight,
    double gemm_scale,
    double quant_multiplier,
    torch::Tensor running_amax
) {
    TORCH_CHECK(
        a.is_cuda() && interleaved_weight.is_cuda(),
        "Inputs must be CUDA tensors"
    );
    TORCH_CHECK(
        a.is_contiguous()
        && interleaved_weight.is_contiguous(),
        "Inputs must be contiguous"
    );
    TORCH_CHECK(
        a.dim() == 2
        && interleaved_weight.dim() == 2,
        "Expected rank-2 inputs"
    );
    TORCH_CHECK(
        a.element_size() == 1
        && interleaved_weight.element_size() == 1,
        "Inputs must contain 8-bit FP8 values"
    );
    TORCH_CHECK(
        a.size(1) == interleaved_weight.size(1),
        "K dimensions must match"
    );
    TORCH_CHECK(
        interleaved_weight.size(0) % 2 == 0,
        "Packed gate/up N must be even"
    );
    TORCH_CHECK(
        running_amax.is_cuda()
        && running_amax.scalar_type() == torch::kFloat32
        && running_amax.numel() == 1,
        "running_amax must be one CUDA FP32 value"
    );

    int M = int(a.size(0));
    int combined_N = int(interleaved_weight.size(0));
    int output_N = combined_N / 2;
    int K = int(a.size(1));

    TORCH_CHECK(
        K % AlignmentA == 0,
        "K does not satisfy FP8 alignment"
    );

    auto output = torch::empty(
        {M, output_N},
        a.options().dtype(torch::kFloat8_e4m3fn)
    );

    auto stride_a =
        cutlass::make_cute_packed_stride(
            StrideA{},
            cute::make_shape(M, K, 1)
        );

    auto stride_b =
        cutlass::make_cute_packed_stride(
            StrideB{},
            cute::make_shape(combined_N, K, 1)
        );

    auto* a_ptr =
        reinterpret_cast<ElementA*>(a.data_ptr());

    auto* b_ptr =
        reinterpret_cast<ElementB*>(
            interleaved_weight.data_ptr()
        );

    auto* output_ptr =
        reinterpret_cast<OutputElement*>(
            output.data_ptr()
        );

    auto* amax_ptr =
        running_amax.data_ptr<float>();

    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, combined_N, K, 1},
        {
            a_ptr,
            stride_a,
            b_ptr,
            stride_b
        },
        {
            float(gemm_scale),
            float(quant_multiplier),
            output_ptr,
            amax_ptr
        }
    };

    Gemm gemm;

    size_t workspace_size =
        Gemm::get_workspace_size(arguments);

    auto workspace = torch::empty(
        {int64_t(workspace_size)},
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

    return {output, running_amax};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "swiglu_direct",
        &sm100_fp8_swiglu_direct,
        "Single-kernel SM100 FP8 gate/up GEMM with "
        "direct SwiGLU, amax, and compact FP8 output"
    );
}
