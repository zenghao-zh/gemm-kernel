/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Template for a double-buffered threadblock-scoped GEMM kernel.
*/

#pragma once

#include "cutlass/aligned_buffer.h"
#include "cutlass/arch/memory.h"
#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"

#include "cutlass/gemm/threadblock/mma_base.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product targeting CUDA cores and SIMT math
/// instructions.
template <
    /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename Shape_,
    /// Policy describing tuning details (concept: MmaPolicy)
    typename Policy0_,
    /// B1-specific version of the policy (concept: MmaPolicy)
    typename Policy1_,
    /// Number of stages,
    int Stages,
    /// Used for partial specialization
    typename Enable = bool>
class DualMmaBase {
 public:
  ///< Size of the Gemm problem - concept: gemm::GemmShape<>
  using Shape = Shape_;

  ///< Policy describing tuning details
  using Policy0 = Policy0_;
  using Policy1 = Policy1_;

  //
  // Dependent types
  //

  /// Warp-level Mma
  using Operator0 = typename Policy0::Operator;
  using Operator1 = typename Policy1::Operator;

  /// Shape describing the overall GEMM computed from shared memory
  /// by each warp.
  using WarpGemm = typename Policy0::Operator::Shape;

  /// Shape describing the number of warps filling the CTA
  using WarpCount = GemmShape<Shape::kM / WarpGemm::kM,
                              Shape::kN / WarpGemm::kN,
                              Shape::kK / WarpGemm::kK>;

  /// Number of warp-level GEMM oeprations
  static int const kWarpGemmIterations =
      (WarpGemm::kK / Operator0::Policy::MmaShape::kK);

  /// Number of stages
  static int const kStages = Stages;

  /// Tensor reference to the A operands
  using TensorRefA0 = TensorRef<typename Operator0::ElementA, typename Operator0::LayoutA>;
  using TensorRefA1 = TensorRef<typename Operator1::ElementA, typename Operator1::LayoutA>;

  /// Tensor reference to the B operands
  using TensorRefB0 = TensorRef<typename Operator0::ElementB, typename Operator0::LayoutB>;
  using TensorRefB1 = TensorRef<typename Operator1::ElementB, typename Operator1::LayoutB>;

  static_assert(kWarpGemmIterations > 1,
                "The pipelined structure requires at least two warp-level "
                "GEMM operations.");

  static_assert((kWarpGemmIterations % 2) == 0,
                "Inner loop iteration must be an even number.");

  //
  // Nested structs
  //

  /// Shared storage object needed by threadblock-scoped GEMM
  class SharedStorage {
   public:
    //
    // Type definitions
    //

    /// Shape of the A matrix operands in shared memory
    using ShapeA0 = MatrixShape<Shape::kM + Policy0::SmemPaddingA::kRow,
                                Shape::kK * kStages +
                                    Policy0::SmemPaddingA::kColumn>;
    using ShapeA1 = MatrixShape<Shape::kM + Policy1::SmemPaddingA::kRow,
                                Shape::kK * kStages +
                                    Policy1::SmemPaddingA::kColumn>;

    /// Shape of the B matrix operand in shared memory
    using ShapeB0 =
        MatrixShape<Shape::kK * kStages + Policy0::SmemPaddingB::kRow,
                    Shape::kN + Policy0::SmemPaddingB::kColumn>;
    using ShapeB1 =
        MatrixShape<Shape::kK * kStages + Policy1::SmemPaddingB::kRow,
                    Shape::kN + Policy1::SmemPaddingB::kColumn>;

   public:
    //
    // Data members
    //

    /// Buffer for A operands
    AlignedBuffer<typename Operator0::ElementA, ShapeA0::kCount> operand_A0;
    AlignedBuffer<typename Operator1::ElementA, ShapeA1::kCount> operand_A1;

    /// Buffer for B operand
    AlignedBuffer<typename Operator0::ElementB, ShapeB0::kCount> operand_B0;
    AlignedBuffer<typename Operator1::ElementB, ShapeB1::kCount> operand_B1;

   public:

    //
    // Methods
    //

    /// Returns a layout object for the A matrices
    CUTLASS_DEVICE
    static typename Operator0::LayoutA LayoutA0() {
      return Operator0::LayoutA::packed({ShapeA0::kRow, ShapeA0::kColumn});
    }

    CUTLASS_DEVICE
    static typename Operator1::LayoutA LayoutA1() {
      return Operator1::LayoutA::packed({ShapeA1::kRow, ShapeA1::kColumn});
    }

    /// Returns a layout object for the B matrix
    CUTLASS_HOST_DEVICE
    static typename Operator0::LayoutB LayoutB0() {
      return Operator0::LayoutB::packed({ShapeB0::kRow, ShapeB0::kColumn});
    }

    /// Returns a layout object for the B matrix
    CUTLASS_HOST_DEVICE
    static typename Operator1::LayoutB LayoutB1() {
      return Operator1::LayoutB::packed({ShapeB1::kRow, ShapeB1::kColumn});
    }

    /// Returns a TensorRef to the A operands
    CUTLASS_HOST_DEVICE
    TensorRefA0 operand_A0_ref() {
      return TensorRefA0{operand_A0.data(), LayoutA0()};
    }

    CUTLASS_HOST_DEVICE
    TensorRefA1 operand_A1_ref() {
      return TensorRefA1{operand_A1.data(), LayoutA1()};
    }

    /// Returns a TensorRef to the B operand
    CUTLASS_HOST_DEVICE
    TensorRefB0 operand_B0_ref() {
      return TensorRefB0{operand_B0.data(), LayoutB0()};
    }
    CUTLASS_HOST_DEVICE
    TensorRefB1 operand_B1_ref() {
      return TensorRefB1{operand_B1.data(), LayoutB1()};
    }
  };

 protected:

  //
  // Data members
  //

  /// Iterator to load a warp-scoped tile of A operands from shared memory
  typename Operator0::IteratorA warp_tile_iterator_A0_;
  typename Operator1::IteratorA warp_tile_iterator_A1_;

  /// Iterator to load a warp-scoped tile of B operand from shared memory
  typename Operator0::IteratorB warp_tile_iterator_B0_;
  typename Operator1::IteratorB warp_tile_iterator_B1_;

public:

  /// Construct from tensor references
  CUTLASS_DEVICE
  DualMmaBase(
      ///< Shared storage needed for internal use by threadblock-scoped GEMM
      SharedStorage &shared_storage,
      ///< ID within the threadblock
      int thread_idx,
      ///< ID of warp
      int warp_idx,
      ///< ID of each thread within a warp
      int lane_idx
    ):
      warp_tile_iterator_A0_(shared_storage.operand_A0_ref(), lane_idx),
      warp_tile_iterator_A1_(shared_storage.operand_A1_ref(), lane_idx),
      warp_tile_iterator_B0_(shared_storage.operand_B0_ref(), lane_idx),
      warp_tile_iterator_B1_(shared_storage.operand_B1_ref(), lane_idx) {

  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace gemm
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
