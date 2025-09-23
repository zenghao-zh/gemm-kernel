#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_universal.h>
#include <cutlass/util/device_memory.h>
#include <gemm.h>

#define CUTLASS_CHECK(status)                                                                    \
  {                                                                                              \
    cutlass::Status error = status;                                                              \
    if (error != cutlass::Status::kSuccess) {                                                    \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ \
                << std::endl;                                                                    \
      exit(EXIT_FAILURE);                                                                        \
    }                                                                                            \
  }


void matmul_w4a4_host(
    const Int4Storage *A,
    const Int4Storage *B,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    int32_t *C)
{
    using Gemm = cutlass::gemm::device::Gemm<
        cutlass::int4b_t,               // ElementA
        cutlass::layout::RowMajor,      // LayoutA
        cutlass::int4b_t,               // ElementB
        cutlass::layout::ColumnMajor,   // LayoutB
        int32_t,                        // ElementOutput
        cutlass::layout::RowMajor,      // LayoutOutput
        int32_t,                        // ElementAccumulator
        cutlass::arch::OpClassTensorOp, // tag indicating Tensor Cores
        cutlass::arch::Sm80             // Hopper architecture with native INT4 Tensor Core support
        >;

    Gemm gemmOp;

    using GemmCoord = cutlass::gemm::GemmCoord;

    typename Gemm::Arguments arguments{
        {static_cast<GemmCoord::Index>(M), static_cast<GemmCoord::Index>(N), static_cast<GemmCoord::Index>(K)},
        {(cutlass::int4b_t *)A, K},
        {(cutlass::int4b_t *)B, K},
        {C, N},
        {C, N},
        {1, 0}};

    auto status = gemmOp(arguments);

    ensure(status == cutlass::Status::kSuccess,
           cutlassGetStatusString(status));
}

void matmul_w4a8_host(
    const int8_t *A,
    const Int4Storage *B,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    int32_t *C)
{
    using ElementA = int8_t;
    using ElementB = cutlass::int4b_t;
    using ElementOutput = int32_t;
    using ElementAccumulator = int32_t;
    constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
    constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

    using DeviceGemmT = cutlass::gemm::device::GemmUniversal<
        ElementA,
        cutlass::layout::RowMajor,
        ElementB,
        cutlass::layout::ColumnMajor,
        ElementOutput,
        cutlass::layout::RowMajor,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        // cutlass::gemm::GemmShape<128, 128, 64>,
        cutlass::gemm::GemmShape<128, 128, 64>,
        // cutlass::gemm::GemmShape<64, 64, 64>,
        cutlass::gemm::GemmShape<64, 64, 64>,
        cutlass::gemm::GemmShape<16, 8, 32>,
        cutlass::epilogue::thread::LinearCombination<
            ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
            ElementAccumulator, ElementAccumulator>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        4,  // Stages
        AlignmentA, // AlignmentA
        AlignmentB, // AlignmentB
        cutlass::arch::OpMultiplyAddMixedInputUpcast,
        cutlass::ComplexTransform::kNone,
        cutlass::ComplexTransform::kNone>;

    cutlass::gemm::GemmCoord problem_size(M, N, K);
    constexpr auto SplitKFactor = 1;

    typename DeviceGemmT::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem_size, // problem size
        SplitKFactor,
        {1, 0},
        A,
        B,
        C,
        C,
        M * K,
        N * K,
        M * N,
        M * N, // batch strides
        K, // lda
        K, // ldb
        N, // ldc
        N, // ldd
    };
    // Using the arguments, query for extra workspace required for matrix multiplication computation
    size_t workspace_size = DeviceGemmT::get_workspace_size(arguments);
    // Allocate workspace memory
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    DeviceGemmT device_gemm;
    // Check the problem size is supported or not
    CUTLASS_CHECK(device_gemm.can_implement(arguments));

    // Initialize CUTLASS kernel with arguments and workspace pointer
    CUTLASS_CHECK(device_gemm.initialize(arguments, workspace.get()));

    // Correctness / Warmup iteration
    CUTLASS_CHECK(device_gemm());
}


void matmul_w8a8_host(
    const int8_t *A,
    const int8_t *B,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    int32_t *C)
{
    using Gemm = cutlass::gemm::device::Gemm<
        int8_t,               // ElementA
        cutlass::layout::RowMajor,      // LayoutA
        int8_t,               // ElementB
        cutlass::layout::ColumnMajor,   // LayoutB
        int32_t,                        // ElementOutput
        cutlass::layout::RowMajor,      // LayoutOutput
        int32_t,                        // ElementAccumulator
        cutlass::arch::OpClassTensorOp, // tag indicating Tensor Cores
        cutlass::arch::Sm80             // Hopper architecture with enhanced INT8 Tensor Core support
        >;

    Gemm gemmOp;

    using GemmCoord = cutlass::gemm::GemmCoord;

    typename Gemm::Arguments arguments{
        {static_cast<GemmCoord::Index>(M), static_cast<GemmCoord::Index>(N), static_cast<GemmCoord::Index>(K)},
        {A, K},
        {B, K},
        {C, N},
        {C, N},
        {1, 0}};

    auto status = gemmOp(arguments);

    ensure(status == cutlass::Status::kSuccess,
           cutlassGetStatusString(status));
}

void matmul_w8a8_host_accum(
    const int8_t *A, const int8_t *B,
    uint32_t M, uint32_t N, uint32_t K,
    int32_t *C,
    int alpha /*=1*/, int beta /*=0*/)
{
    using Gemm = cutlass::gemm::device::Gemm<
        int8_t, cutlass::layout::RowMajor,
        int8_t, cutlass::layout::ColumnMajor,
        int32_t, cutlass::layout::RowMajor,
        int32_t,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80>;

    Gemm gemmOp;
    using GemmCoord = cutlass::gemm::GemmCoord;

    typename Gemm::Arguments args{
        { (GemmCoord::Index)M, (GemmCoord::Index)N, (GemmCoord::Index)K },
        { A, K },
        { B, K },
        { C, N },
        { C, N },
        { alpha, beta }      // ← 关键：允许累加到已有的 C
    };

    auto status = gemmOp(args);
    ensure(status == cutlass::Status::kSuccess, cutlassGetStatusString(status));
}

void matmul_w4a4_host_accum(
    const Int4Storage *A, const Int4Storage *B,
    uint32_t M, uint32_t N, uint32_t K,
    int32_t *C,
    int alpha /*=1*/, int beta /*=0*/)
{
    using Gemm = cutlass::gemm::device::Gemm<
        cutlass::int4b_t, cutlass::layout::RowMajor,
        cutlass::int4b_t, cutlass::layout::ColumnMajor,
        int32_t,           cutlass::layout::RowMajor,
        int32_t,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80>;

    Gemm gemmOp;
    using GemmCoord = cutlass::gemm::GemmCoord;

    typename Gemm::Arguments args{
        { (GemmCoord::Index)M, (GemmCoord::Index)N, (GemmCoord::Index)K },
        { (const cutlass::int4b_t*)A, K },
        { (const cutlass::int4b_t*)B, K },
        { C, N },
        { C, N },
        { alpha, beta }      // ← 同样支持累加
    };

    auto status = gemmOp(args);
    ensure(status == cutlass::Status::kSuccess, cutlassGetStatusString(status));
}

void matmul_w4a8x4_host(
    const int8_t*       A1, const int8_t*       B1,   // w8a8
    const Int4Storage*  A2, const Int4Storage*  B2,   // w4a4
    uint32_t M1, uint32_t N1, uint32_t K1,            // 尺寸1
    uint32_t M2, uint32_t N2, uint32_t K2,            // 尺寸2
    int32_t* C)                                        // 输出 C（大小至少 M1×N1 或 M2×N2）
{
    // 输出维度必须一致，否则无法相加
    ensure(M1 == M2 && N1 == N2, "Output dimensions must match: (M1,N1)==(M2,N2)");

    // 第一次：C = A1*B1（beta=0 覆写/初始化 C）
    matmul_w8a8_host_accum(A1, B1, M1, N1, K1, C, /*alpha=*/1, /*beta=*/0);

    // 第二次：C = A2*B2 + C（beta=1 直接在 epilogue 中累加）
    matmul_w4a4_host_accum(A2, B2, M2, N2, K2, C, /*alpha=*/1, /*beta=*/1);

}
