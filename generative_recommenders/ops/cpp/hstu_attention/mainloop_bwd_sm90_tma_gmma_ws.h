/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar,
 *Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cutlass/array.h>
#include <cutlass/barrier.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include "cutlass/pipeline/pipeline.hpp"

#include "cute/tensor.hpp"

#include "cutlass/gemm/collective/builders/sm90_common.inl"

#include "copy_sm90_bulk_reduce.h"
#include "mask.h"
#include "named_barrier.h"
#include "seqlen.h"
#include "utils.h"

namespace hstu {

using namespace cute;

template <
    int Stages,
    int Stages_dO,
    int Stages_dS,
    class ClusterShape_,
    class TileShape_MNK_,
    class Element_,
    class ElementAccum_,
    class ArchTag_,
    bool Causal,
    bool Local,
    bool Contexual_mask,
    bool Jagged,
    bool Has_targets,
    bool Deterministic,
    bool SdP_swapAB_,
    bool dKV_swapAB_,
    bool dQ_swapAB_,
    int NumMmaWarpGroups = 2,
    int AtomLayoutMSdP = 1,
    int AtomLayoutNdKV = 2,
    int AtomLayoutMdQ = 1,
    bool Mma_dP_is_RS = false,
    bool Cross = false,
    bool Softmax = false>
struct CollectiveMainloopBwdSm90 {
  static constexpr int kStages = Stages;
  static constexpr int kStages_dO = Stages_dO;
  static constexpr int kStages_dS = Stages_dS;
  static_assert(kStages >= kStages_dO);
  static_assert(Stages_dS == 1 || Stages_dS == kStages);
  static_assert(
      !Mma_dP_is_RS || SdP_swapAB_); // If Mma_dP_is_RS, we need SdP_SwapAB
  using ClusterShape = ClusterShape_;
  using TileShape_MNK = TileShape_MNK_;
  using Element = Element_;
  using ElementAccum = ElementAccum_;
  using ArchTag = ArchTag_;
  using SeqlenInfo_t = hstu::SeqlenInfoQKBwd<
      Jagged,
      Cross,
      Has_targets,
      CUTE_STATIC_V(get<0>(TileShape_MNK{}))>;

  static constexpr bool SdP_swapAB = SdP_swapAB_;
  static constexpr bool dKV_swapAB = dKV_swapAB_;
  static constexpr bool dQ_swapAB = dQ_swapAB_;

  static constexpr bool Q_dO_same_stages = kStages == kStages_dO;

  static constexpr int kBlockM = get<0>(TileShape_MNK{});
  static constexpr int kBlockN = get<1>(TileShape_MNK{});
  static constexpr int kHeadDim = get<2>(TileShape_MNK{});

  static_assert(ArchTag::kMinComputeCapability >= 90);
  static_assert(get<0>(ClusterShape{}) == 1 && get<2>(ClusterShape{}) == 1);

  static constexpr int NumMmaThreads =
      NumMmaWarpGroups * cutlass::NumThreadsPerWarpGroup;
  static constexpr int NumProducerThreads = cutlass::NumThreadsPerWarp * 2;

  static_assert(NumMmaWarpGroups % AtomLayoutMSdP == 0);
  static_assert(NumMmaWarpGroups % AtomLayoutNdKV == 0);
  static_assert(NumMmaWarpGroups % AtomLayoutMdQ == 0);
  static constexpr bool Mma_dKV_is_RS = AtomLayoutMSdP == 1 &&
      AtomLayoutNdKV == NumMmaWarpGroups && SdP_swapAB && !dKV_swapAB;
  static constexpr bool Mma_dQ_is_RS = AtomLayoutMSdP == NumMmaWarpGroups &&
      AtomLayoutMdQ == NumMmaWarpGroups && !SdP_swapAB &&
      !dQ_swapAB; // If dQ_swapAB we can't use RS

  static constexpr GMMA::Major PdS_Major = GMMA::Major::K;
  // static constexpr GMMA::Major PdS_Major = GMMA::Major::MN;
  static constexpr GMMA::Major PdSt_Major =
      PdS_Major == GMMA::Major::K ? GMMA::Major::MN : GMMA::Major::K;

  using TileShapeAtomSdP = std::conditional_t<
      !SdP_swapAB,
      Shape<
          Int<kBlockM>,
          Int<kBlockN / (NumMmaWarpGroups / AtomLayoutMSdP)>,
          Int<kHeadDim>>,
      Shape<Int<kBlockN>, Int<kBlockM / AtomLayoutMSdP>, Int<kHeadDim>>>;
  using AtomLayoutSdP = std::conditional_t<
      !SdP_swapAB,
      Layout<Shape<
          Int<AtomLayoutMSdP>,
          Int<NumMmaWarpGroups / AtomLayoutMSdP>,
          _1>>,
      Layout<Shape<
          Int<NumMmaWarpGroups / AtomLayoutMSdP>,
          Int<AtomLayoutMSdP>,
          _1>>>;
  using TiledMmaSdP = decltype(cute::make_tiled_mma(
      cute::GMMA::
          ss_op_selector<Element, Element, ElementAccum, TileShapeAtomSdP>(),
      AtomLayoutSdP{}));

  using TiledMmadPRS = decltype(cute::make_tiled_mma(
      cute::GMMA::
          rs_op_selector<Element, Element, ElementAccum, TileShapeAtomSdP>(),
      AtomLayoutSdP{}));

  using TileShapeAtomdKV = std::conditional_t<
      !dKV_swapAB,
      Shape<
          Int<kBlockN>,
          Int<kHeadDim / (NumMmaWarpGroups / AtomLayoutNdKV)>,
          Int<kBlockM>>,
      Shape<Int<kHeadDim>, Int<kBlockN / AtomLayoutNdKV>, Int<kBlockM>>>;
  using AtomLayoutdKV = std::conditional_t<
      !dKV_swapAB,
      Layout<Shape<
          Int<AtomLayoutNdKV>,
          Int<NumMmaWarpGroups / AtomLayoutNdKV>,
          _1>>,
      Layout<Shape<
          Int<NumMmaWarpGroups / AtomLayoutNdKV>,
          Int<AtomLayoutNdKV>,
          _1>>>;
  using TiledMmadKV = decltype(cute::make_tiled_mma(
      std::conditional_t<
          Mma_dKV_is_RS,
          decltype(cute::GMMA::rs_op_selector<
                   Element,
                   Element,
                   ElementAccum,
                   TileShapeAtomdKV,
                   GMMA::Major::K,
                   GMMA::Major::MN>()),
          decltype(cute::GMMA::ss_op_selector < Element, Element, ElementAccum, TileShapeAtomdKV, !dKV_swapAB ? PdSt_Major : GMMA::Major::MN, !dKV_swapAB ? GMMA::Major::MN : PdSt_Major > ())>{},
      AtomLayoutdKV{}));

  static constexpr bool dQacc_use_TMA = kHeadDim < 256;
  // For hdim256, we want to slice the dQ MMA (64 x 256 on 2 WGs) into two (64 x
  // 128 on 2 WGs) so that we can do atomic add on one half before doing the
  // other half of the MMA, to reduce register pressure.
  static constexpr bool Slice_dQKV_Mma = kHeadDim == 256 && !dQacc_use_TMA &&
      dQ_swapAB && AtomLayoutMdQ == 1 && NumMmaWarpGroups == 2;
  static_assert(
      !(Deterministic && Slice_dQKV_Mma),
      "Deterministic mode not supported with Slice_dQKV_Mma");

  static constexpr int TileShapeAtomdQ_BlockM = kBlockM / AtomLayoutMdQ;
  static constexpr int TileShapeAtomdQ_HeadDim =
      (Slice_dQKV_Mma ? kHeadDim / 2 : kHeadDim) /
      (NumMmaWarpGroups / AtomLayoutMdQ);
  static_assert(
      !dQ_swapAB ? TileShapeAtomdQ_BlockM == 64 : TileShapeAtomdQ_HeadDim == 64,
      "Tile_M must be 64.");
  using TileShapeAtomdQ = std::conditional_t<
      !dQ_swapAB,
      Shape<
          Int<TileShapeAtomdQ_BlockM>,
          Int<TileShapeAtomdQ_HeadDim>,
          Int<kBlockN>>,
      Shape<
          Int<TileShapeAtomdQ_HeadDim>,
          Int<TileShapeAtomdQ_BlockM>,
          Int<kBlockN>>>;
  using AtomLayoutdQ = std::conditional_t<
      !dQ_swapAB,
      Layout<
          Shape<Int<AtomLayoutMdQ>, Int<NumMmaWarpGroups / AtomLayoutMdQ>, _1>>,
      Layout<Shape<
          Int<NumMmaWarpGroups / AtomLayoutMdQ>,
          Int<AtomLayoutMdQ>,
          _1>>>;
  using TiledMmadQ = decltype(cute::make_tiled_mma(
      std::conditional_t<
          Mma_dQ_is_RS,
          decltype(cute::GMMA::rs_op_selector<
                   Element,
                   Element,
                   ElementAccum,
                   TileShapeAtomdQ,
                   GMMA::Major::K,
                   GMMA::Major::MN>()),
          decltype(cute::GMMA::ss_op_selector < Element, Element, ElementAccum, TileShapeAtomdQ, !dQ_swapAB ? PdS_Major : GMMA::Major::MN, !dQ_swapAB ? GMMA::Major::MN : PdS_Major > ())>{},
      AtomLayoutdQ{}));

  // We need to accommodate both Q and Q^T (and dO and dO^T) in shared memory.
  // Q & dO are used in the SdP Mma and Q^T and dO^T are used in the dKV Mma.
  // Since this is GMMA::Major::K, the M dimension (kBlockM) doesn't matter for
  // the layout, only the K dimension changes the layout.
  using SmemLayoutAtomQdO =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GMMA::Major::K,
               Element,
               Int<kBlockM>,
               Int<kHeadDim /
                   (NumMmaWarpGroups / AtomLayoutNdKV)>>()); // for dKV_Mma
  using SmemLayoutQ = decltype(tile_to_shape(
      SmemLayoutAtomQdO{},
      make_shape(
          shape<0>(TileShape_MNK{}),
          shape<2>(TileShape_MNK{}),
          Int<kStages>{})));
  using SmemLayoutdO = decltype(tile_to_shape(
      SmemLayoutAtomQdO{},
      make_shape(
          shape<0>(TileShape_MNK{}),
          shape<2>(TileShape_MNK{}),
          Int<kStages_dO>{})));

  using SmemLayoutAtomK =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GMMA::Major::K,
               Element,
               Int<kBlockN>,
               Int<kHeadDim / (NumMmaWarpGroups / AtomLayoutMdQ)>>());
  using SmemLayoutK =
      decltype(tile_to_shape(SmemLayoutAtomK{}, select<1, 2>(TileShape_MNK{})));

  using SmemLayoutAtomV =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GMMA::Major::K,
               Element,
               decltype(cute::get<1>(TileShape_MNK{})),
               decltype(cute::get<2>(TileShape_MNK{}))>());
  using SmemLayoutV =
      decltype(tile_to_shape(SmemLayoutAtomV{}, select<1, 2>(TileShape_MNK{})));

  using SmemLayoutAtomPdS =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               PdS_Major,
               Element,
               Int<kBlockM / AtomLayoutMSdP>,
               Int<kBlockN / (NumMmaWarpGroups / AtomLayoutMSdP)>>());
  using SmemLayoutPdS = decltype(tile_to_shape(
      SmemLayoutAtomPdS{},
      make_shape(Int<kBlockM>{}, Int<kBlockN>{}, Int<kStages_dS>{}),
      std::conditional_t<
          PdS_Major == GMMA::Major::K,
          cute::Step<_1, _2, _3>,
          cute::Step<_2, _1, _3>>{}));
  // Need stride to be multiple of 32, otherwise we get error (misaligned
  // address) when doing TMA if e.g. kBlockM=80 We set stride to be multiple of
  // 64 so that if ShuffleLSE, even if threads read from sLSE but out of bounds,
  // it's still a valid smem address.
  using SmemLayoutLSE = cute::Layout<
      cute::Shape<Int<kBlockM>, Int<kStages>>,
      cute::Stride<_1, Int<cute::round_up(kBlockM, 64)>>>;
  using SmemLayoutLSEMma = std::conditional_t<
      SdP_swapAB,
      cute::Layout<
          cute::Shape<Int<kBlockN>, Int<kBlockM>, Int<kStages>>,
          cute::Stride<_0, _1, Int<cute::round_up(kBlockM, 64)>>>,
      cute::Layout<
          cute::Shape<Int<kBlockM>, Int<kBlockN>, Int<kStages>>,
          cute::Stride<_1, _0, Int<cute::round_up(kBlockM, 64)>>>>;

  // Note this is the transpose in terms of the view, not in terms of memory.
  using SmemLayoutQt = decltype(cute::composition(
      SmemLayoutQ{},
      make_layout(
          make_shape(
              get<2>(TileShape_MNK{}),
              get<0>(TileShape_MNK{}),
              Int<kStages>{}),
          make_stride(Int<kBlockM>{}, _1{}, Int<kBlockM * kHeadDim>{}))));
  using SmemLayoutdOt = decltype(cute::composition(
      SmemLayoutdO{},
      make_layout(
          make_shape(
              get<2>(TileShape_MNK{}),
              get<0>(TileShape_MNK{}),
              Int<kStages_dO>{}),
          make_stride(Int<kBlockM>{}, _1{}, Int<kBlockM * kHeadDim>{}))));
  using SmemLayoutKt = decltype(cute::composition(
      SmemLayoutK{},
      make_layout(
          make_shape(get<2>(TileShape_MNK{}), get<1>(TileShape_MNK{})),
          make_stride(Int<kBlockN>{}, _1{}))));
  using SmemLayoutPdSt = decltype(cute::composition(
      SmemLayoutPdS{},
      make_layout(
          make_shape(Int<kBlockN>{}, Int<kBlockM>{}, Int<kStages_dS>{}),
          make_stride(Int<kBlockM>{}, _1{}, Int<kBlockM * kBlockN>{}))));

  // Thread layout, 256 or 384 threads per row
  // We split into NumMmaWarpGroups so that we can do Bulk reduce add for each
  // WG separately.
  using R2SLayoutAtomdQaccum = Layout<
      Shape<Int<cutlass::NumThreadsPerWarpGroup>, Int<NumMmaWarpGroups>>>;
  using R2STiledCopydQaccum = decltype(make_tiled_copy(
      Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccum>{},
      R2SLayoutAtomdQaccum{},
      Layout<Shape<_4>>{})); // Val layout, 4 vals per store
  using SmemLayoutdQaccum = Layout<
      Shape<Int<kBlockM * kHeadDim / NumMmaWarpGroups>, Int<NumMmaWarpGroups>>>;

  static constexpr int kNumPdSStore = kBlockM * kBlockN / NumMmaThreads;
  // If !SdP_swapAB, the accum registers hold P / dS, otherwise they hold Pt /
  // dSt. If PdS_major is MN, then we need to "transpose" the write.
  using SmemCopyAtomPdS = Copy_Atom<
      std::conditional_t<
          (!SdP_swapAB) ^ (PdS_Major == GMMA::Major::MN),
          std::conditional_t<
              kNumPdSStore % 8 == 0,
              cute::SM90_U32x4_STSM_N,
              cute::SM90_U32x2_STSM_N>,
          std::conditional_t<
              kNumPdSStore % 8 == 0,
              cute::SM90_U16x8_STSM_T,
              cute::SM90_U16x4_STSM_T>>,
      Element>;

  using GmemTiledCopyQdO =
      decltype(cutlass::gemm::collective::detail::
                   sm90_cluster_shape_to_tma_atom(shape<1>(ClusterShape{})));
  using GmemTiledCopyKV = cute::SM90_TMA_LOAD;

  using ShapeQKV =
      cute::Shape<int32_t, int32_t, int32_t, int32_t>; // (seqlen, d, head,
                                                       // batch)
  using StrideQKV = cute::Stride<int64_t, _1, int64_t, int64_t>;
  using ShapeLSE =
      cute::Shape<int32_t, int32_t, int32_t>; // (seqlen, head, batch)
  using StrideLSE = cute::Stride<_1, int64_t, int64_t>; // (seqlen, head, batch)
  using ShapedQaccum =
      cute::Shape<int32_t, int32_t, int32_t>; // (seqlen * d, head, batch)
  using StridedQaccum = cute::Stride<_1, int64_t, int64_t>;

  using TMA_QdO = decltype(make_tma_copy_A_sm90(
      GmemTiledCopyQdO{},
      make_tensor(
          make_gmem_ptr(static_cast<Element const*>(nullptr)),
          ShapeQKV{},
          StrideQKV{}),
      take<0, 2>(SmemLayoutQ{}),
      TileShape_MNK{},
      ClusterShape{})); // mcast along N mode for this M load, if any

  using TMA_K = decltype(make_tma_copy_B_sm90(
      GmemTiledCopyKV{},
      make_tensor(
          make_gmem_ptr(static_cast<Element const*>(nullptr)),
          ShapeQKV{},
          StrideQKV{}),
      SmemLayoutK{},
      TileShape_MNK{},
      ClusterShape{})); // no mcast for KV

  using TMA_V = decltype(make_tma_copy_B_sm90(
      GmemTiledCopyKV{},
      make_tensor(
          make_gmem_ptr(static_cast<Element const*>(nullptr)),
          ShapeQKV{},
          StrideQKV{}),
      SmemLayoutV{},
      TileShape_MNK{},
      ClusterShape{})); // no mcast for KV

  using MainloopPipeline = typename cutlass::PipelineTmaAsync<kStages>;
  using PipelineState = typename MainloopPipeline::PipelineState;
  using MainloopPipeline_dO = typename cutlass::PipelineTmaAsync<kStages_dO>;
  using PipelineState_dO = typename MainloopPipeline_dO::PipelineState;

  // Set the bytes transferred in this TMA transaction (may involve multiple
  // issues)
  static constexpr uint32_t TmaTransactionBytesQ = static_cast<uint32_t>(
      size(take<0, 2>(SmemLayoutQ{})) * cutlass::sizeof_bits_v<Element> / 8);
  static constexpr uint32_t TmaTransactionBytesK = static_cast<uint32_t>(
      size(SmemLayoutK{}) * cutlass::sizeof_bits_v<Element> / 8);
  static constexpr uint32_t TmaTransactionBytesV = static_cast<uint32_t>(
      size(SmemLayoutV{}) * cutlass::sizeof_bits_v<Element> / 8);
  static constexpr uint32_t TmaTransactionBytesLSE = static_cast<uint32_t>(
      size(select<0>(SmemLayoutLSE{})) * cutlass::sizeof_bits_v<ElementAccum> /
      8);

  // These are tuned for speed. They don't affect correctness.
  // We have separate iterations with causal masking. Not necessary for hdim 128
  // but for hdim 64 this helps quite a bit to not have to do causal masking for
  // most of the iterations. For hdim 192, separating masking iterations results
  // in register spills.
  static constexpr bool SeparateMaskingIterations = false;
  // Do we keep the LSE and dPsum in each thread, or split them across 8 threads
  // that share them and then shuffle to get the value whenever we need? This
  // can reduce register pressure when SdP_swapAB, where each thread needs to
  // keep statistics for (kBlockM / 4) rows. If !SdP_swapAB, each thread only
  // needs to keep statistic for 2 rows.
  static constexpr bool ShuffleLSE = SdP_swapAB && kHeadDim <= 64;
  static constexpr bool ShuffledPsum = SdP_swapAB && kHeadDim <= 64;
  static constexpr size_t SmemAlignmentP =
      cutlass::detail::alignment_for_swizzle(SmemLayoutPdS{});
  static constexpr size_t SmemAlignmentdS =
      cutlass::detail::alignment_for_swizzle(SmemLayoutPdS{});
  // Without this SmemAlignment, with hdim 256 we get "misaligned address" error
  // in TMA
  static constexpr size_t SmemAlignmentQKVdO = kHeadDim % 256 == 0 ? 256 : 128;
  static constexpr size_t SmemAlignmentV = !Mma_dP_is_RS
      ? SmemAlignmentQKVdO
      : cutlass::detail::alignment_for_swizzle(SmemLayoutV{});
  static_assert(
      SmemAlignmentP >= 128 && SmemAlignmentdS >= 128,
      "Require at least 128B alignment");

  // TODO: do we have to worry that smem_dk and smem_dv in the epilogue don't
  // line up w smem_k and smem_v due to alignment?
  using SmemdQacc_t = std::conditional_t<
      !dQacc_use_TMA,
      cute::array<ElementAccum, 0>,
      cute::array_aligned<ElementAccum, cute::cosize_v<SmemLayoutdQaccum>>>;
  using SmemP_t = std::conditional_t<
      Mma_dKV_is_RS,
      cute::array<Element, 0>,
      cute::array_aligned<
          Element,
          cute::cosize_v<SmemLayoutPdS>,
          SmemAlignmentP>>;
  struct TensorStorage
      : cute::aligned_struct<
            cute::max(SmemAlignmentP, SmemAlignmentdS, SmemAlignmentQKVdO)> {
    cute::
        array_aligned<Element, cute::cosize_v<SmemLayoutK>, SmemAlignmentQKVdO>
            smem_k;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>, SmemAlignmentV>
        smem_v;
    SmemdQacc_t smem_dqacc;
    cute::
        array_aligned<Element, cute::cosize_v<SmemLayoutQ>, SmemAlignmentQKVdO>
            smem_q;
    cute::
        array_aligned<Element, cute::cosize_v<SmemLayoutdO>, SmemAlignmentQKVdO>
            smem_do;
    cute::array_aligned<ElementAccum, cute::cosize_v<SmemLayoutLSE>, 128>
        smem_lse;
    cute::array_aligned<ElementAccum, cute::cosize_v<SmemLayoutLSE>, 128>
        smem_dpsum;
    SmemP_t smem_p;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutPdS>, SmemAlignmentdS>
        smem_ds;
  };

  // Host side kernel arguments
  struct Arguments {
    Element const* const ptr_Q;
    ShapeQKV const shape_Q;
    StrideQKV const stride_Q;
    Element const* const ptr_K;
    ShapeQKV const shape_K;
    StrideQKV const stride_K;
    Element const* const ptr_V;
    ShapeQKV const shape_V;
    StrideQKV const stride_V;
    Element const* const ptr_dO;
    ShapeQKV const shape_dO;
    StrideQKV const stride_dO;
    ElementAccum* const ptr_dQaccum;
    ShapedQaccum const shape_dQaccum;
    StridedQaccum const stride_dQaccum;
    float const* const ptr_LSE_log2;
    ShapeLSE const shape_LSE;
    StrideLSE const stride_LSE_log2;
    float const* const ptr_dPsum;
    StrideLSE const stride_dPsum;
    int const max_attn_len;
    int const min_full_attn_seq_len;
    int const contextual_seq_len;
    float const max_seq_len_inv;
    float const alpha;
    int const num_batch;
    int const num_softmax_heads;
    int* const dq_semaphore;
    int const* const seq_offsets = nullptr;
    int const* const seq_offsets_q = nullptr;
    int const* const num_targets = nullptr;
    float const* const attn_scale = nullptr;
    bool const scalar_scale = true;
  };

  // Device side kernel params
  struct Params {
    ShapeQKV const shape_Q;
    ShapeQKV const shape_K;
    ShapeQKV const shape_V;
    ShapeQKV const shape_dO;
    ElementAccum* const ptr_dQaccum;
    ShapedQaccum const shape_dQaccum;
    StridedQaccum stride_dQaccum;
    TMA_QdO tma_load_Q, tma_load_dO;
    TMA_K tma_load_K;
    TMA_V tma_load_V;
    float const* const ptr_LSE_log2;
    ShapeLSE const shape_LSE;
    StrideLSE const stride_LSE_log2;
    float const* const ptr_dPsum;
    StrideLSE const stride_dPsum;
    int const max_attn_len;
    int const min_full_attn_seq_len;
    int const contextual_seq_len;
    float const max_seq_len_inv;
    float const alpha;
    float const alpha_log2;
    int const num_batch;
    int const num_softmax_heads;
    int* const dq_semaphore;
    int const* const seq_offsets = nullptr;
    int const* const seq_offsets_q = nullptr;
    int const* const num_targets;
    float const* const attn_scale;
    bool const scalar_scale = true;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    Tensor mQ =
        make_tensor(make_gmem_ptr(args.ptr_Q), args.shape_Q, args.stride_Q);
    TMA_QdO tma_load_Q = make_tma_copy_A_sm90(
        GmemTiledCopyQdO{},
        mQ,
        SmemLayoutQ{}(_, _, _0{}),
        TileShape_MNK{},
        ClusterShape{}); // mcast along N mode for this M load, if any
    Tensor mdO =
        make_tensor(make_gmem_ptr(args.ptr_dO), args.shape_Q, args.stride_dO);
    TMA_QdO tma_load_dO = make_tma_copy_A_sm90(
        GmemTiledCopyQdO{},
        mdO,
        SmemLayoutdO{}(_, _, _0{}),
        TileShape_MNK{},
        ClusterShape{}); // mcast along N mode for this M load, if any
    Tensor mK =
        make_tensor(make_gmem_ptr(args.ptr_K), args.shape_K, args.stride_K);
    TMA_K tma_load_K = make_tma_copy_B_sm90(
        GmemTiledCopyKV{},
        mK,
        SmemLayoutK{},
        TileShape_MNK{},
        ClusterShape{}); // no mcast for KV
    Tensor mV =
        make_tensor(make_gmem_ptr(args.ptr_V), args.shape_K, args.stride_V);
    TMA_V tma_load_V = make_tma_copy_B_sm90(
        GmemTiledCopyKV{},
        mV,
        SmemLayoutV{},
        TileShape_MNK{},
        ClusterShape{}); // no mcast for KV
    if constexpr (Deterministic) {
      assert(args.dq_semaphore != nullptr);
    }
    return {
        args.shape_Q,
        args.shape_K,
        args.shape_V,
        args.shape_dO,
        args.ptr_dQaccum,
        args.shape_dQaccum,
        args.stride_dQaccum,
        tma_load_Q,
        tma_load_dO,
        tma_load_K,
        tma_load_V,
        args.ptr_LSE_log2,
        args.shape_LSE,
        args.stride_LSE_log2,
        args.ptr_dPsum,
        args.stride_dPsum,
        args.max_attn_len,
        args.min_full_attn_seq_len,
        args.contextual_seq_len,
        args.max_seq_len_inv,
        args.alpha,
        float(args.alpha * M_LOG2E),
        args.num_batch,
        args.num_softmax_heads,
        args.dq_semaphore,
        args.seq_offsets,
        args.seq_offsets_q,
        args.num_targets,
        args.attn_scale,
        args.scalar_scale};
  }

  /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best
  /// performance
  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& params) {
    cute::prefetch_tma_descriptor(params.tma_load_Q.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_dO.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_K.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_V.get_tma_descriptor());
  }

  CUTLASS_DEVICE
  cute::tuple<int, int> get_m_block_min_max(
      int const max_attn_len,
      int const contextual_seq_len,
      int const uihlen,
      int const seqlen,
      int const n_block) {
    static constexpr int kBlockM = get<0>(TileShape_MNK{});
    static constexpr int kBlockN = get<1>(TileShape_MNK{});
    if constexpr (Has_targets) {
      int n_idx_min = n_block * kBlockN;
      if (n_idx_min >= uihlen) {
        int n_idx_max = (n_block + 1) * kBlockN;
        return {
            std::max(0, n_idx_min / kBlockM),
            cute::ceil_div(std::min(n_idx_max, seqlen), kBlockM)};
      }
    }
    // uih part
    int m_block_max = cute::ceil_div(seqlen, kBlockM);
    if constexpr (Local) {
      int local_m_block_max =
          cute::ceil_div((n_block + 1) * kBlockN + max_attn_len, kBlockM);
      if constexpr (Contexual_mask) {
        // row contexual without sink
        if (n_block * kBlockN < contextual_seq_len) {
          local_m_block_max = std::max(
              local_m_block_max,
              cute::ceil_div(contextual_seq_len + max_attn_len, kBlockM));
        }
      }
      m_block_max = std::min(m_block_max, local_m_block_max);
    }
    int m_block_min = 0;
    if constexpr (Causal || Local) {
      m_block_min = std::max(m_block_min, (n_block * kBlockN) / kBlockM);
    }
    return {m_block_min, m_block_max};
  }

  CUTLASS_DEVICE
  cute::tuple<int, int> get_full_m_block_min_max(
      int const uihlen,
      int const seqlen,
      int const min_full_attn_seq_len,
      int const m_block_max,
      int const n_block) {
    static constexpr int kBlockM = get<0>(TileShape_MNK{});
    static constexpr int kBlockN = get<1>(TileShape_MNK{});
    if constexpr (Cross) {
      return {0, 0};
    }
    if constexpr (!Local) {
      return {0, 0};
    }
    if constexpr (Has_targets) {
      int n_idx_min = n_block * kBlockN;
      if (n_idx_min >= uihlen) {
        return {0, 0};
      }
    }
    if constexpr (Local) {
      int full_m_block_max = cute::ceil_div(seqlen, kBlockM);
      int full_m_block_min =
          std::max(m_block_max, (uihlen - min_full_attn_seq_len) / kBlockM);
      return {full_m_block_min, full_m_block_max};
    }
    return {0, 0};
  }

  CUTLASS_DEVICE
  int get_contexual_m_block_max(
      int const uihlen,
      int const contextual_seq_len,
      int const m_block_min,
      int const n_block) {
    static constexpr int kBlockM = get<0>(TileShape_MNK{});
    static constexpr int kBlockN = get<1>(TileShape_MNK{});
    if constexpr (Cross) {
      return 0;
    }
    if constexpr (!Contexual_mask) {
      return 0;
    }
    if constexpr (Has_targets) {
      int n_idx_min = n_block * kBlockN;
      if (n_idx_min >= uihlen) {
        return 0;
      }
    }
    if constexpr (Causal || Local) {
      int contexual_m_block_max =
          std::min(m_block_min, cute::ceil_div(contextual_seq_len, kBlockM));
      return contexual_m_block_max;
    }
    return 0;
  }

  CUTLASS_DEVICE
  int get_next_m_block(
      int const m_block,
      int const m_block_min,
      int const m_block_max,
      int const contexual_m_block_max,
      int const full_m_block_min,
      int const full_m_block_max) {
    int const out_m_block = m_block + 1;
    if constexpr (Contexual_mask || Local) {
      if (out_m_block == m_block_max) {
        if (contexual_m_block_max > 0) {
          return 0;
        }
        if (full_m_block_max > full_m_block_min) {
          return full_m_block_min;
        }
        return -1;
      }
      if (out_m_block == contexual_m_block_max) {
        if (full_m_block_max > full_m_block_min) {
          return full_m_block_min;
        }
        return -1;
      }
      if (out_m_block == full_m_block_max) {
        return -1;
      }
      return out_m_block;
    }
    if (out_m_block == m_block_max) {
      return -1;
    }
    return out_m_block;
  }

  CUTLASS_DEVICE
  cute::tuple<int, int> get_cross_m_block_min_max(
      int const uihlen_q,
      int const seqlen_q,
      int const seqlen_kv,
      int const n_block) {
    static constexpr int kBlockM = get<0>(TileShape_MNK{});
    static constexpr int kBlockN = get<1>(TileShape_MNK{});
    int m_block_max = cute::ceil_div(seqlen_q, kBlockM);
    if constexpr (!Causal) {
      return {0, m_block_max};
    }
    int m_block_min =
        std::max(0, (n_block * kBlockN + uihlen_q - seqlen_kv) / kBlockM);
    return {m_block_min, m_block_max};
  }

  template <typename SchedulerPrefetch, typename SharedStorage>
  CUTLASS_DEVICE void load(
      Params const& params,
      MainloopPipeline pipeline_q,
      MainloopPipeline_dO pipeline_do,
      PipelineState& smem_pipe_write,
      PipelineState_dO& smem_pipe_write_do,
      SharedStorage& shared_storage,
      SchedulerPrefetch const& scheduler_prefetch,
      cute::tuple<int32_t, int32_t, int32_t> block_coord) {
    auto [n_block, bidh, bidb] = block_coord;
    SeqlenInfo_t seqlen_info{
        bidb,
        get<0>(params.shape_Q),
        get<0>(params.shape_K),
        params.seq_offsets,
        params.seq_offsets_q,
        params.num_targets};
    if constexpr (Jagged) {
      static constexpr int kBlockN = get<1>(TileShape_MNK{});
      if (n_block * kBlockN >= seqlen_info.seqlen_kv) {
        scheduler_prefetch();
        return;
      }
    }
    int m_block_min, m_block_max;
    if constexpr (Cross) {
      auto m_block_min_max = get_cross_m_block_min_max(
          seqlen_info.uihlen_q,
          seqlen_info.seqlen_q,
          seqlen_info.seqlen_kv,
          n_block);
      m_block_min = get<0>(m_block_min_max);
      m_block_max = get<1>(m_block_min_max);
    } else {
      auto m_block_min_max = get_m_block_min_max(
          params.max_attn_len,
          params.contextual_seq_len,
          seqlen_info.uihlen_q,
          seqlen_info.seqlen_q,
          n_block);
      m_block_min = get<0>(m_block_min_max);
      m_block_max = get<1>(m_block_min_max);
    }
    auto full_m_block_min_max = get_full_m_block_min_max(
        seqlen_info.uihlen_q,
        seqlen_info.seqlen_q,
        params.min_full_attn_seq_len,
        m_block_max,
        n_block);
    int const full_m_block_min = get<0>(full_m_block_min_max);
    int const full_m_block_max = get<1>(full_m_block_min_max);
    int contexual_m_block_max = get_contexual_m_block_max(
        seqlen_info.uihlen_q, params.contextual_seq_len, m_block_min, n_block);

    Tensor sQ = make_tensor(
        make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()),
        SmemLayoutQ{});
    Tensor sdO = make_tensor(
        make_smem_ptr(shared_storage.tensors.mainloop.smem_do.data()),
        SmemLayoutdO{});
    Tensor sK = make_tensor(
        make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()),
        SmemLayoutK{});
    Tensor sV = make_tensor(
        make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()),
        SmemLayoutV{});
    Tensor sLSE = make_tensor(
        make_smem_ptr(shared_storage.tensors.mainloop.smem_lse.data()),
        SmemLayoutLSE{});
    Tensor sdPsum = make_tensor(
        make_smem_ptr(shared_storage.tensors.mainloop.smem_dpsum.data()),
        SmemLayoutLSE{});

    // Prepare the TMA loads
    uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();
    constexpr uint32_t cluster_shape_x = get<0>(ClusterShape());
    uint2 cluster_local_block_id = {
        block_rank_in_cluster % cluster_shape_x,
        block_rank_in_cluster / cluster_shape_x};
    Tensor mQ = params.tma_load_Q.get_tma_tensor(params.shape_Q)(
        _, _, bidh, !Jagged ? bidb : 0);
    Tensor mdO = params.tma_load_dO.get_tma_tensor(params.shape_Q)(
        _, _, bidh, !Jagged ? bidb : 0);
    Tensor mK = params.tma_load_K.get_tma_tensor(params.shape_K)(
        _, _, bidh, !Jagged ? bidb : 0);
    Tensor mV = params.tma_load_V.get_tma_tensor(params.shape_K)(
        _, _, bidh, !Jagged ? bidb : 0);
    Tensor mLSE = make_tensor(
        make_gmem_ptr(params.ptr_LSE_log2),
        params.shape_LSE,
        params.stride_LSE_log2)(_, bidh, !Jagged ? bidb : 0);
    Tensor mdPsum = make_tensor(
        make_gmem_ptr(params.ptr_dPsum), params.shape_LSE, params.stride_dPsum)(
        _, bidh, !Jagged ? bidb : 0);

    Tensor gQ = local_tile(
        domain_offset(make_coord(seqlen_info.offset_q, _0{}), mQ),
        select<0, 2>(TileShape_MNK{}),
        make_coord(_, _0{})); // (M, K, _)
    Tensor gdO = local_tile(
        domain_offset(make_coord(seqlen_info.offset_q, _0{}), mdO),
        select<0, 2>(TileShape_MNK{}),
        make_coord(_, _0{})); // (M, K, _)
    Tensor gK = local_tile(
        domain_offset(make_coord(seqlen_info.offset_k, _0{}), mK),
        select<1, 2>(TileShape_MNK{}),
        make_coord(n_block, _0{})); // (N, K)
    Tensor gV = local_tile(
        domain_offset(make_coord(seqlen_info.offset_k, _0{}), mV),
        select<1, 2>(TileShape_MNK{}),
        make_coord(n_block, _0{})); // (N, K)
    Tensor gLSE = local_tile(
        domain_offset(make_coord(seqlen_info.offset_q_padded), mLSE),
        select<0>(TileShape_MNK{}),
        make_coord(_)); // (M, _)
    Tensor gdPsum = local_tile(
        domain_offset(make_coord(seqlen_info.offset_q_padded), mdPsum),
        select<0>(TileShape_MNK{}),
        make_coord(_)); // (M, _)

    Tensor sK_x =
        make_tensor(sK.data(), make_layout(sK.layout(), Layout<_1>{}));
    Tensor gK_x =
        make_tensor(gK.data(), make_layout(gK.layout(), Layout<_1>{}));
    Tensor sV_x =
        make_tensor(sV.data(), make_layout(sV.layout(), Layout<_1>{}));
    Tensor gV_x =
        make_tensor(gV.data(), make_layout(gV.layout(), Layout<_1>{}));
    // auto [tQgQ, tQsQ] = tma_partition(params.tma_load_Q,
    // block_rank_in_cluster, Layout<ClusterShape>{},
    //                                   group_modes<0, 2>(sQ), group_modes<0,
    //                                   2>(gQ));  // (TMA, k), (TMA, PIPE)
    // auto [tdOgdO, tdOsdO] = tma_partition(params.tma_load_dO,
    // block_rank_in_cluster, Layout<ClusterShape>{},
    //                                   group_modes<0, 2>(sdO), group_modes<0,
    //                                   2>(gdO));  // (TMA, k), (TMA, PIPE)
    auto block_tma_Q = params.tma_load_Q.get_slice(cluster_local_block_id.y);
    auto block_tma_dO = params.tma_load_dO.get_slice(cluster_local_block_id.y);
    Tensor tQgQ = group_modes<0, 3>(block_tma_Q.partition_S(gQ));
    Tensor tQsQ = group_modes<0, 3>(block_tma_Q.partition_D(sQ));
    Tensor tdOgdO = group_modes<0, 3>(block_tma_dO.partition_S(gdO));
    Tensor tdOsdO = group_modes<0, 3>(block_tma_dO.partition_D(sdO));
    auto [tKgK, tKsK] = tma_partition(
        params.tma_load_K,
        _0{},
        Layout<_1>{},
        group_modes<0, 2>(sK_x),
        group_modes<0, 2>(gK_x)); // (TMA), (TMA)
    auto [tVgV, tVsV] = tma_partition(
        params.tma_load_V,
        _0{},
        Layout<_1>{},
        group_modes<0, 2>(sV_x),
        group_modes<0, 2>(gV_x)); // (TMA), (TMA)
    auto bulk_copy = Copy_Traits<SM90_BULK_COPY_AUTO>{};

    uint16_t mcast_mask_qdo = 0;
    if constexpr (cute::is_same_v<GmemTiledCopyQdO, SM90_TMA_LOAD_MULTICAST>) {
      auto block_layout = Layout<ClusterShape>{}; // (m,n) -> block_id
      for (int n = 0; n < size<1>(block_layout); ++n) {
        mcast_mask_qdo |=
            (uint16_t(1) << block_layout(cluster_local_block_id.x, n, _0{}));
      }
    }

    int m_block = m_block_min;
    int next_m_block = -1;
    int lane_predicate = cute::elect_one_sync();

    if (lane_predicate) {
      pipeline_q.producer_acquire(smem_pipe_write);
      copy(
          params.tma_load_Q.with(
              *pipeline_q.producer_get_barrier(smem_pipe_write),
              mcast_mask_qdo,
              TMA::CacheHintSm90::EVICT_LAST),
          tQgQ(_, m_block),
          tQsQ(_, smem_pipe_write.index()));
      if constexpr (Softmax) {
        copy(
            bulk_copy.with(*pipeline_q.producer_get_barrier(smem_pipe_write)),
            gLSE(_, m_block),
            sLSE(_, smem_pipe_write.index()));
      }
    }

    // // Wait for the MMA warpgroups to say that smem_k and smem_v are ready
    // cutlass::arch::NamedBarrier::sync(NumMmaThreads +
    // cutlass::NumThreadsPerWarp,
    // static_cast<uint32_t>(BwdNamedBarriers::KVEmpty) /*id*/);

    auto load_step = [&](int m_block) {
      // If Q and dO have the same number of stages, we can use the same
      // pipeline state variable to reduce registers
      PipelineState_dO smem_pipe_write_do_cur =
          cute::conditional_return<Q_dO_same_stages>(
              smem_pipe_write, smem_pipe_write_do);
      pipeline_do.producer_acquire(smem_pipe_write_do_cur);
      copy(
          params.tma_load_dO.with(
              *pipeline_do.producer_get_barrier(smem_pipe_write_do_cur),
              mcast_mask_qdo,
              TMA::CacheHintSm90::EVICT_LAST),
          tdOgdO(_, m_block),
          tdOsdO(_, smem_pipe_write_do_cur.index()));
      if constexpr (Softmax) {
        copy(
            bulk_copy.with(
                *pipeline_do.producer_get_barrier(smem_pipe_write_do_cur)),
            gdPsum(_, m_block),
            sdPsum(_, smem_pipe_write_do_cur.index()));
      }
      if constexpr (!Q_dO_same_stages) {
        ++smem_pipe_write_do;
      }
      ++smem_pipe_write;
      next_m_block = get_next_m_block(
          m_block,
          m_block_min,
          m_block_max,
          contexual_m_block_max,
          full_m_block_min,
          full_m_block_max);
      if (next_m_block != -1) {
        pipeline_q.producer_acquire(smem_pipe_write);
        copy(
            params.tma_load_Q.with(
                *pipeline_q.producer_get_barrier(smem_pipe_write),
                mcast_mask_qdo,
                TMA::CacheHintSm90::EVICT_LAST),
            tQgQ(_, next_m_block),
            tQsQ(_, smem_pipe_write.index()));
        if constexpr (Softmax) {
          copy(
              bulk_copy.with(*pipeline_q.producer_get_barrier(smem_pipe_write)),
              gLSE(_, next_m_block),
              sLSE(_, smem_pipe_write.index()));
        }
      }
    };

    if (lane_predicate) {
      // Copy K tile and V tile from GMEM to SMEM.
      shared_storage.pipelines.barrier_KV.arrive_and_expect_tx(
          TmaTransactionBytesK + TmaTransactionBytesV);
      copy(
          params.tma_load_K.with(
              reinterpret_cast<
                  cutlass::arch::ClusterTransactionBarrier::ValueType&>(
                  shared_storage.pipelines.barrier_KV),
              0 /*mcast_mask*/),
          tKgK,
          tKsK);
      copy(
          params.tma_load_V.with(
              reinterpret_cast<
                  cutlass::arch::ClusterTransactionBarrier::ValueType&>(
                  shared_storage.pipelines.barrier_KV),
              0 /*mcast_mask*/),
          tVgV,
          tVsV);

#pragma unroll(kHeadDim < 256 ? 2 : 1)
      for (; m_block < m_block_max; ++m_block) {
        load_step(m_block);
      }
    }
    scheduler_prefetch();
    m_block = next_m_block;
    if constexpr (Contexual_mask) {
      if (lane_predicate) {
        if (m_block >= 0) {
#pragma unroll(kHeadDim < 256 ? 2 : 1)
          for (; m_block < contexual_m_block_max; ++m_block) {
            load_step(m_block);
          }
        }
      }
    }
    m_block = next_m_block;
    if constexpr (Local) {
      if (lane_predicate) {
        if (m_block >= 0) {
#pragma unroll(kHeadDim < 256 ? 2 : 1)
          for (; m_block < full_m_block_max; ++m_block) {
            load_step(m_block);
          }
        }
      }
    }
    if constexpr (Q_dO_same_stages) {
      smem_pipe_write_do = smem_pipe_write;
    }
  }

  /// Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
  CUTLASS_DEVICE void load_tail(
      MainloopPipeline pipeline_q,
      MainloopPipeline_dO pipeline_do,
      PipelineState& smem_pipe_write) {
    static_assert(
        Q_dO_same_stages, "Q and dO must have the same number of stages");
    // Need to copy since pipeline_q.producer_tail(smem_pipe_write) will
    // increment smem_pipe_write
    PipelineState smem_pipe_write_do = smem_pipe_write;
    // Issue the epilogue waits
    if (cute::elect_one_sync()) {
      /* This helps avoid early exit of blocks in Cluster
       * Waits for all stages to either be released (all Consumer UNLOCKs), or
       * if the stage was never used then would just be acquired since the phase
       * was still inverted from make_producer_start_state
       */
      pipeline_q.producer_tail(smem_pipe_write);
      pipeline_do.producer_tail(smem_pipe_write_do);
    }
  }

  /// Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
  CUTLASS_DEVICE void load_tail(
      MainloopPipeline pipeline_q,
      MainloopPipeline_dO pipeline_do,
      PipelineState& smem_pipe_write,
      PipelineState_dO& smem_pipe_write_do) {
    // Issue the epilogue waits
    if (cute::elect_one_sync()) {
      /* This helps avoid early exit of blocks in Cluster
       * Waits for all stages to either be released (all Consumer UNLOCKs), or
       * if the stage was never used then would just be acquired since the phase
       * was still inverted from make_producer_start_state
       */
      pipeline_q.producer_tail(smem_pipe_write);
      pipeline_do.producer_tail(smem_pipe_write_do);
    }
  }

  template <typename SharedStorage>
  CUTLASS_DEVICE void store_dq(
      Params const& params,
      SharedStorage& shared_storage,
      cute::tuple<int32_t, int32_t, int32_t> block_coord) {
    if constexpr (!dQacc_use_TMA) {
      return;
    }

    auto [n_block, bidh, bidb] = block_coord;
    SeqlenInfo_t seqlen_info{
        bidb,
        get<0>(params.shape_Q),
        get<0>(params.shape_K),
        params.seq_offsets,
        params.seq_offsets_q,
        params.num_targets};
    if constexpr (Jagged) {
      static constexpr int kBlockN = get<1>(TileShape_MNK{});
      if (n_block * kBlockN >= seqlen_info.seqlen_kv) {
        return;
      }
    }
    int m_block_min, m_block_max;
    if constexpr (Cross) {
      auto m_block_min_max = get_cross_m_block_min_max(
          seqlen_info.uihlen_q,
          seqlen_info.seqlen_q,
          seqlen_info.seqlen_kv,
          n_block);
      m_block_min = get<0>(m_block_min_max);
      m_block_max = get<1>(m_block_min_max);
    } else {
      auto m_block_min_max = get_m_block_min_max(
          params.max_attn_len,
          params.contextual_seq_len,
          seqlen_info.uihlen_q,
          seqlen_info.seqlen_q,
          n_block);
      m_block_min = get<0>(m_block_min_max);
      m_block_max = get<1>(m_block_min_max);
    }
    auto full_m_block_min_max = get_full_m_block_min_max(
        seqlen_info.uihlen_q,
        seqlen_info.seqlen_q,
        params.min_full_attn_seq_len,
        m_block_max,
        n_block);
    int const full_m_block_min = get<0>(full_m_block_min_max);
    int const full_m_block_max = get<1>(full_m_block_min_max);
    int contexual_m_block_max = get_contexual_m_block_max(
        seqlen_info.uihlen_q, params.contextual_seq_len, m_block_min, n_block);

    Tensor sdQ = make_tensor(
        make_smem_ptr(shared_storage.tensors.mainloop.smem_dqacc.data()),
        SmemLayoutdQaccum{});
    static constexpr int dQ_TMA_num_bytes =
        CUTE_STATIC_V(size<0>(sdQ)) * sizeof(ElementAccum);

    Tensor mdQaccum = make_tensor(
        make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.ptr_dQaccum)),
        params.shape_dQaccum,
        params.stride_dQaccum)(_, bidh, !Jagged ? bidb : 0);
    Tensor gdQaccum_ = local_tile(
        domain_offset(
            make_coord(seqlen_info.offset_q_padded * kHeadDim), mdQaccum),
        Shape<Int<kBlockM * kHeadDim>>{},
        make_coord(_)); // (M * K, _)
    Tensor gdQaccum = cute::flat_divide(
        gdQaccum_,
        Int<kBlockM * kHeadDim / NumMmaWarpGroups>{}); // (M * K / WG, WG, _)

    int const num_batch = params.num_batch;
    int const num_head = get<2>(params.shape_Q);
    int* lock_ptr =
        !Deterministic ? nullptr : params.dq_semaphore + bidb * num_head + bidh;
    using Barrier = cutlass::GenericBarrier<cutlass::detail::SyncwarpSync>;
    bool const lane_predicate = cute::elect_one_sync();

    auto store_dq_step = [&](int m_block) {
      if constexpr (Deterministic) {
        Barrier::wait_eq(
            lock_ptr,
            threadIdx.x % cutlass::NumThreadsPerWarp,
            m_block * num_batch * num_head,
            n_block);
      }
#pragma unroll
      for (int warpgroup_idx = 0; warpgroup_idx < NumMmaWarpGroups;
           ++warpgroup_idx) {
        cutlass::arch::NamedBarrier::sync(
            cutlass::NumThreadsPerWarpGroup + cutlass::NumThreadsPerWarp,
            static_cast<uint32_t>(BwdNamedBarriers::dQFullWG1) +
                warpgroup_idx /*id*/); // sdQ full, to be written to gmem
        if (lane_predicate) {
          SM90_BULK_REDUCE_ADD::copy(
              raw_pointer_cast(sdQ(_, warpgroup_idx).data()),
              raw_pointer_cast(gdQaccum(_, warpgroup_idx, m_block).data()),
              dQ_TMA_num_bytes,
              static_cast<uint64_t>(TMA::CacheHintSm90::EVICT_LAST));
          tma_store_arrive();
        }
      }
      // Note, the for_each() function is required here to ensure
      // `warpgroup_idx` is of type Int<x>.
      for_each(make_int_sequence<NumMmaWarpGroups>{}, [&](auto warpgroup_idx) {
        if (lane_predicate) {
          tma_store_wait<NumMmaWarpGroups - 1 - CUTE_STATIC_V(warpgroup_idx)>();
        }
        cutlass::arch::NamedBarrier::arrive(
            cutlass::NumThreadsPerWarpGroup + cutlass::NumThreadsPerWarp,
            static_cast<uint32_t>(BwdNamedBarriers::dQEmptyWG1) +
                warpgroup_idx /*id*/); // sdQ empty, ready to be written to
      });
      if constexpr (Deterministic) {
        Barrier::arrive_inc(
            lock_ptr,
            threadIdx.x % cutlass::NumThreadsPerWarp,
            m_block * num_batch * num_head);
      }
    };

#pragma unroll 2
    for (int m_block = m_block_min; m_block < m_block_max; ++m_block) {
      store_dq_step(m_block);
    }
    if constexpr (Contexual_mask) {
#pragma unroll 2
      for (int m_block = 0; m_block < contexual_m_block_max; ++m_block) {
        store_dq_step(m_block);
      }
    }
    if constexpr (Local) {
#pragma unroll 2
      for (int m_block = full_m_block_min; m_block < full_m_block_max;
           ++m_block) {
        store_dq_step(m_block);
      }
    }
    if constexpr (Local && Deterministic) {
      constexpr int kBlockM = get<0>(TileShape_MNK{});
      int const m_block_global_max =
          cute::ceil_div(seqlen_info.seqlen_q, kBlockM);
#pragma unroll 2
      for (int m_block = m_block_max; m_block < m_block_global_max; ++m_block) {
        Barrier::arrive_inc(
            lock_ptr,
            threadIdx.x % cutlass::NumThreadsPerWarp,
            m_block * num_batch * num_head);
      }
    }
  }

  CUTLASS_DEVICE void mma_init() {
    // We're not currently using this bc we're not using persistent scheduler
    // // Tell producer (warp 0) that smem_k and smem_v are ready
    // cutlass::arch::NamedBarrier::arrive(NumMmaThreads +
    // cutlass::NumThreadsPerWarp,
    // static_cast<uint32_t>(BwdNamedBarriers::KVEmpty) /*id*/);
    int warp_idx_in_warpgroup =
        __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
    if constexpr (dQacc_use_TMA) {
      if (warp_idx_in_warpgroup == 0) {
        cutlass::arch::NamedBarrier::arrive(
            cutlass::NumThreadsPerWarpGroup + cutlass::NumThreadsPerWarp,
            static_cast<uint32_t>(BwdNamedBarriers::dQEmptyWG1) - 1 +
                hstu::canonical_warp_group_idx_nosync() /*id*/); // sdQ empty,
                                                                 // ready to be
                                                                 // written to
      }
    }
  }

  template <typename SharedStorage, typename FrgTensordKV>
  CUTLASS_DEVICE bool mma(
      Params const& params,
      MainloopPipeline pipeline_q,
      MainloopPipeline_dO pipeline_do,
      PipelineState& smem_pipe_read,
      PipelineState_dO& smem_pipe_read_do,
      FrgTensordKV& tdKrdK,
      FrgTensordKV& tdVrdV,
      int thread_idx,
      int& work_idx,
      cute::tuple<int32_t, int32_t, int32_t> block_coord,
      SharedStorage& shared_storage) {
    static_assert(
        is_rmem<FrgTensordKV>::value,
        "dK and dV tensor must be rmem resident.");

    int n_block = get<0>(block_coord);
    int bidb = get<2>(block_coord);
    SeqlenInfo_t seqlen_info{
        bidb,
        get<0>(params.shape_Q),
        get<0>(params.shape_K),
        params.seq_offsets,
        params.seq_offsets_q,
        params.num_targets};
    if constexpr (Jagged) {
      static constexpr int kBlockN = get<1>(TileShape_MNK{});
      if (n_block * kBlockN >= seqlen_info.seqlen_kv) {
        return false;
      }
    }
    int m_block_min, m_block_max;
    if constexpr (Cross) {
      auto m_block_min_max = get_cross_m_block_min_max(
          seqlen_info.uihlen_q,
          seqlen_info.seqlen_q,
          seqlen_info.seqlen_kv,
          n_block);
      m_block_min = get<0>(m_block_min_max);
      m_block_max = get<1>(m_block_min_max);
    } else {
      auto m_block_min_max = get_m_block_min_max(
          params.max_attn_len,
          params.contextual_seq_len,
          seqlen_info.uihlen_q,
          seqlen_info.seqlen_q,
          n_block);
      m_block_min = get<0>(m_block_min_max);
      m_block_max = get<1>(m_block_min_max);
    }
    auto full_m_block_min_max = get_full_m_block_min_max(
        seqlen_info.uihlen_q,
        seqlen_info.seqlen_q,
        params.min_full_attn_seq_len,
        m_block_max,
        n_block);
    int const full_m_block_min = get<0>(full_m_block_min_max);
    int const full_m_block_max = get<1>(full_m_block_min_max);
    int contexual_m_block_max = get_contexual_m_block_max(
        seqlen_info.uihlen_q, params.contextual_seq_len, m_block_min, n_block);

    Tensor sQ = make_tensor(
        make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()),
        SmemLayoutQ{});
    Tensor sdO = make_tensor(
        make_smem_ptr(shared_storage.tensors.mainloop.smem_do.data()),
        SmemLayoutdO{});
    Tensor sK = make_tensor(
        make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()),
        SmemLayoutK{});
    Tensor sV = make_tensor(
        make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()),
        SmemLayoutV{});
    Tensor sQt = make_tensor(
        make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()),
        SmemLayoutQt{});
    Tensor sdOt = make_tensor(
        make_smem_ptr(shared_storage.tensors.mainloop.smem_do.data()),
        SmemLayoutdOt{});
    Tensor sKt = make_tensor(
        make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()),
        SmemLayoutKt{});
    Tensor sP = make_tensor(
        make_smem_ptr(shared_storage.tensors.mainloop.smem_p.data()),
        SmemLayoutPdS{});
    Tensor sP_pi = cute::as_position_independent_swizzle_tensor(sP);
    Tensor sPt = make_tensor(
        make_smem_ptr(shared_storage.tensors.mainloop.smem_p.data()),
        SmemLayoutPdSt{});
    Tensor sPt_pi = cute::as_position_independent_swizzle_tensor(sPt);
    Tensor sdS = make_tensor(
        make_smem_ptr(shared_storage.tensors.mainloop.smem_ds.data()),
        SmemLayoutPdS{});
    Tensor sdS_pi = cute::as_position_independent_swizzle_tensor(sdS);
    Tensor sdSt = make_tensor(
        make_smem_ptr(shared_storage.tensors.mainloop.smem_ds.data()),
        SmemLayoutPdSt{});
    Tensor sdSt_pi = cute::as_position_independent_swizzle_tensor(sdSt);
    Tensor sdQ = make_tensor(
        make_smem_ptr(shared_storage.tensors.mainloop.smem_dqacc.data()),
        SmemLayoutdQaccum{});

    static_assert(
        stride<0>(typename TiledMmaSdP::ALayout{}) == 0 and
            stride<0>(typename TiledMmaSdP::BLayout{}) == 0 and
            size<0>(typename TiledMmaSdP::ALayout{}) ==
                cutlass::NumThreadsPerWarpGroup and
            size<0>(typename TiledMmaSdP::BLayout{}) ==
                cutlass::NumThreadsPerWarpGroup,
        "Stride of the first mode must be 0 and the size of the mode must be NumThreadsPerWarpGroup");
    constexpr int MmaWarpGroups =
        NumMmaThreads / cutlass::NumThreadsPerWarpGroup;
    Layout warp_group_thread_layout = make_layout(
        make_shape(Int<MmaWarpGroups>{}),
        make_stride(Int<cutlass::NumThreadsPerWarpGroup>{}));
    Layout warp_group_thread_layout_dq = make_layout(
        make_shape(Int<NumMmaWarpGroups>{}),
        make_stride(Int<cutlass::NumThreadsPerWarpGroup>{}));

    int warp_group_idx = __shfl_sync(
        0xFFFFFFFF, thread_idx / cutlass::NumThreadsPerWarpGroup, 0);
    TiledMmaSdP tiled_mma_SdP;
    using TiledMmadP =
        std::conditional_t<!Mma_dP_is_RS, TiledMmaSdP, TiledMmadPRS>;
    TiledMmadP tiled_mma_dP;
    TiledMmadKV tiled_mma_dKV;
    TiledMmadQ tiled_mma_dQ;

    auto wg_mma_SdP =
        tiled_mma_SdP.get_slice(warp_group_thread_layout(warp_group_idx));
    auto wg_mma_dP =
        tiled_mma_dP.get_slice(warp_group_thread_layout(warp_group_idx));
    auto thread_mma_SdP = tiled_mma_SdP.get_thread_slice(thread_idx);
    auto wg_mma_dKV =
        tiled_mma_dKV.get_slice(warp_group_thread_layout(warp_group_idx));
    auto wg_mma_dQ =
        tiled_mma_dQ.get_slice(warp_group_thread_layout_dq(warp_group_idx));

    auto smem_tiled_copy_PdS =
        make_tiled_copy_C(SmemCopyAtomPdS{}, tiled_mma_SdP);
    auto smem_thr_copy_PdS = smem_tiled_copy_PdS.get_thread_slice(thread_idx);

    R2STiledCopydQaccum r2s_tiled_copy_dQaccum;
    auto r2s_thr_copy_dQaccum =
        r2s_tiled_copy_dQaccum.get_thread_slice(thread_idx);
    Tensor tdQsdQaccum = r2s_thr_copy_dQaccum.partition_D(sdQ);
    // if (thread_idx == 0) { print(sdQ); printf("\n"); print(tdQsdQaccum);
    // printf("\n"); }

    // Allocate "fragments/descriptors"
    // We have to use the templated mma_partition_fragment_AB instead of
    // cute::conditional_return or lambda, because some partition_fragment_A/B
    // don't compile.
    // https://stackoverflow.com/questions/50051473/if-constexpr-in-c17-does-not-work-in-a-non-templated-function
    Tensor tSrQ = mma_partition_fragment_AB</*A=*/!SdP_swapAB>(wg_mma_SdP, sQ);
    Tensor tSrK = mma_partition_fragment_AB</*A=*/SdP_swapAB>(wg_mma_SdP, sK);
    Tensor tdPrdO =
        mma_partition_fragment_AB</*A=*/!SdP_swapAB>(wg_mma_SdP, sdO);
    Tensor tdPrV = mma_partition_fragment_AB</*A=*/SdP_swapAB>(wg_mma_dP, sV);
    Tensor tdVrdO =
        mma_partition_fragment_AB</*A=*/dKV_swapAB>(wg_mma_dKV, sdOt);
    Tensor tdKrQ = mma_partition_fragment_AB</*A=*/dKV_swapAB>(wg_mma_dKV, sQt);
    Tensor tdQrdS = mma_partition_fragment_AB</*A=*/!dQ_swapAB>(wg_mma_dQ, sdS);
    Tensor tdQrK = mma_partition_fragment_AB</*A=*/dQ_swapAB>(wg_mma_dQ, sKt);

    Tensor tPsP =
        smem_thr_copy_PdS.partition_D(cute::conditional_return<!SdP_swapAB>(
            sP_pi, sPt_pi)); // ((Atom,AtomNum),PIPE_M,PIPE_N)
    Tensor tdSsdS =
        smem_thr_copy_PdS.partition_D(cute::conditional_return<!SdP_swapAB>(
            sdS_pi, sdSt_pi)); // ((Atom,AtomNum),PIPE_M,PIPE_N)
    // if (blockIdx.x == 0 && threadIdx.x == 128) { print(smem_thr_copy_PdS);
    // print(sP_pi); printf("\n"); print(sPt_pi); printf("\n"); print(tPsP);
    // printf("\n"); print(tdSsdS); printf("\n"); }

    auto consumer_wait = [](auto& pipeline, auto& smem_pipe_read) {
      auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      pipeline.consumer_wait(smem_pipe_read, barrier_token);
    };

    int bidh = get<1>(block_coord);
    // For the case where we do atomicAdd directly to gdQaccum instead of using
    // TMA
    Tensor mdQaccum = make_tensor(
        make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.ptr_dQaccum)),
        params.shape_dQaccum,
        params.stride_dQaccum)(_, bidh, !Jagged ? bidb : 0);
    Tensor gdQaccum_ = local_tile(
        domain_offset(
            make_coord(seqlen_info.offset_q_padded * kHeadDim), mdQaccum),
        Shape<Int<kBlockM * kHeadDim>>{},
        make_coord(_)); // (M * K, _)
    Tensor gdQaccum = cute::flat_divide(
        gdQaccum_,
        Int<kBlockM * kHeadDim / NumMmaWarpGroups>{}); // (M * K / WG, WG, _)
    // We can reuse r2s_thr_copy_dQaccum for this partitioning
    Tensor tdQgdQaccum = r2s_thr_copy_dQaccum.partition_D(gdQaccum);
    // if (blockIdx.x == 0 && threadIdx.x == 128) { print(mdQaccum);
    // printf("\n"); print(gdQaccum_); printf("\n"); print(gdQaccum);
    // printf("\n"); print(tdQgdQaccum); printf("\n"); }

    hstu::Mask<kBlockM, kBlockN, TiledMmaSdP, SdP_swapAB> mask(
        thread_idx,
        seqlen_info.seqlen_q,
        seqlen_info.seqlen_kv,
        params.max_attn_len,
        params.min_full_attn_seq_len,
        params.contextual_seq_len,
        seqlen_info.uihlen_q);

    int m_block = m_block_min;

    clear(tdKrdK);
    clear(tdVrdV);
    // tiled_mma_dKV.accumulate_ = GMMA::ScaleOut::Zero;

    cutlass::ConsumerToken barrier_token = static_cast<cutlass::BarrierStatus>(
        shared_storage.pipelines.barrier_KV.try_wait(work_idx % 2));
    if (barrier_token == cutlass::BarrierStatus::WaitAgain) {
      shared_storage.pipelines.barrier_KV.wait(work_idx % 2);
    }

    if constexpr (Mma_dP_is_RS) {
      using SmemCopyAtomV = Copy_Atom<cute::SM75_U32x4_LDSM_N, Element>;
      auto smem_tiled_copy_V = make_tiled_copy_A(SmemCopyAtomV{}, tiled_mma_dP);
      auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(thread_idx);
      Tensor tdPrV_copy_view = smem_thr_copy_V.retile_D(tdPrV);
      Tensor tdPsV_copy_view = smem_thr_copy_V.partition_S(
          cute::as_position_independent_swizzle_tensor(sV));
      cute::copy(smem_tiled_copy_V, tdPsV_copy_view, tdPrV_copy_view);
    }
    // attention scale
    float scalar_scale_val = params.scalar_scale
        ? (params.attn_scale == nullptr ? params.max_seq_len_inv
                                        : params.attn_scale[0])
        : 0;
    static constexpr int Qdim = !SdP_swapAB ? 0 : 1;
    auto thread0_mma_SdP = tiled_mma_SdP.get_thread_slice(_0{});
    Tensor cS =
        cute::make_identity_tensor(Shape<
                                   Int<!SdP_swapAB ? kBlockM : kBlockN>,
                                   Int<!SdP_swapAB ? kBlockN : kBlockM>>{});
    Tensor tScS = thread_mma_SdP.partition_C(cS);
    Tensor tScS_rowcol = make_tensor(
        tScS.data(),
        hstu::convert_layout_acc_rowcol</*Transposed=*/SdP_swapAB>(
            tScS.layout()));
    Tensor t0ScS = thread0_mma_SdP.partition_C(cS);
    Tensor t0ScS_rowcol = make_tensor(
        t0ScS.data(),
        hstu::convert_layout_acc_rowcol</*Transposed=*/SdP_swapAB>(
            t0ScS.layout()));
    int const thread_qdim_offset = get<Qdim>(tScS_rowcol(_0{}, _0{}));

    auto bwd_step = [&](int m_block, auto mask_fn) {
      Tensor tSrS = partition_fragment_C(
          tiled_mma_SdP,
          select < !SdP_swapAB ? 0 : 1,
          !SdP_swapAB ? 1 : 0 > (TileShape_MNK{}));
      consumer_wait(pipeline_q, smem_pipe_read);
      hstu::gemm</*zero_init=*/true, /*wg_wait=*/-1, /*SwapAB=*/SdP_swapAB>(
          tiled_mma_SdP, tSrQ(_, _, _, smem_pipe_read.index()), tSrK, tSrS);
      Tensor tdPrdP = partition_fragment_C(
          tiled_mma_SdP,
          select < !SdP_swapAB ? 0 : 1,
          !SdP_swapAB ? 1 : 0 > (TileShape_MNK{}));
      PipelineState_dO smem_pipe_read_do_cur =
          cute::conditional_return<Q_dO_same_stages>(
              smem_pipe_read, smem_pipe_read_do);
      consumer_wait(pipeline_do, smem_pipe_read_do_cur);
      hstu::gemm</*zero_init=*/true, /*wg_wait=*/-1, /*SwapAB=*/SdP_swapAB>(
          tiled_mma_dP,
          tdPrdO(_, _, _, smem_pipe_read_do_cur.index()),
          tdPrV,
          tdPrdP);
      warpgroup_wait<1>();
      // Reshape tSrS from ((2, 2, V), MMA_N, MMA_M) to (nrow=(2, V, MMA_M),
      // ncol=(2, MMA_N))
      Tensor scores = make_tensor(
          tSrS.data(),
          hstu::convert_layout_acc_rowcol</*Transposed=*/SdP_swapAB>(
              tSrS.layout()));
      Tensor tSrS_sigmoid = make_tensor_like(tSrS);
      Tensor sigmoid = make_tensor(
          tSrS_sigmoid.data(),
          hstu::convert_layout_acc_rowcol</*Transposed=*/SdP_swapAB>(
              tSrS_sigmoid.layout()));
      int qdim_offset = params.scalar_scale
          ? 0
          : m_block * kBlockM + thread_qdim_offset + seqlen_info.offset_q;
      mask_fn(tSrS, m_block);
#pragma unroll
      for (int mi = 0; mi < size<0>(scores); ++mi) {
        float scale = scalar_scale_val;
        if (!params.scalar_scale) {
          int q_index = qdim_offset + int(get<Qdim>(t0ScS_rowcol(mi, _0{})));
          if (q_index < seqlen_info.seqlen_q) {
            scale = params.attn_scale[q_index];
          }
        }
#pragma unroll
        for (int ni = 0; ni < size<1>(scores); ++ni) {
          scores(mi, ni) = scores(mi, ni) * params.alpha;
          sigmoid(mi, ni) =
              __fdividef(1., 1.0f + cutlass::fast_exp(-scores(mi, ni)));
          scores(mi, ni) = sigmoid(mi, ni) * scores(mi, ni) * scale;
        }
      }
      mask_fn(tSrS_sigmoid, m_block);

      warpgroup_wait<0>();
      // Reshape tdPrdP from ((2, 2, V), MMA_N, MMA_M) to (nrow=(2, V, MMA_M),
      // ncol=(2, MMA_N))
      Tensor dS = make_tensor(tdPrdP.data(), scores.layout());
#pragma unroll
      for (int mi = 0; mi < size<0>(dS); ++mi) {
        float scale = scalar_scale_val;
        if (!params.scalar_scale) {
          int q_index = qdim_offset + int(get<Qdim>(t0ScS_rowcol(mi, _0{})));
          if (q_index < seqlen_info.seqlen_q) {
            scale = params.attn_scale[q_index];
          }
        }
#pragma unroll
        for (int ni = 0; ni < size<1>(dS); ++ni) {
          dS(mi, ni) = dS(mi, ni) * sigmoid(mi, ni) * scale +
              dS(mi, ni) * scores(mi, ni) * (1.f - sigmoid(mi, ni));
          dS(mi, ni) = dS(mi, ni) * params.alpha;
          //   if (dS(mi, ni) > 0.0001) {
          //     std::printf(
          //         "dS(mi, ni) is (%f), (m, n) is (%d, %d), thread_idx is
          //         (%d), blockIdx.z is (%d)\n", dS(mi, ni), mi, ni,
          //         threadIdx.x,
          //         blockIdx.z);
          //   }
        }
      }
      // Convert scores from fp32 to fp16/bf16
      Tensor rP = make_tensor_like<Element>(tSrS);
      hstu::convert_type_out(tSrS, rP);
      if constexpr (!Mma_dKV_is_RS) {
        // Need to sync to make sure P has already been used in the previous
        // iteration before writing new values
        if constexpr (kStages_dS == 1) {
          cutlass::arch::NamedBarrier::sync(
              NumMmaThreads,
              static_cast<uint32_t>(BwdNamedBarriers::PdS) /*id*/);
        }
        Tensor tPaP =
            smem_thr_copy_PdS.retile_S(rP); // ((Atom,AtomNum), MMA_N, MMA_N)
        cute::copy(
            smem_tiled_copy_PdS,
            tPaP,
            tPsP(
                _,
                _,
                _,
                cute::conditional_return<kStages_dS == 1>(
                    _0{}, smem_pipe_read.index())));
      }
      Tensor rdS = make_tensor_like<Element>(tdPrdP);
      hstu::convert_type_out(tdPrdP, rdS);
      // If there's double buffering on dS, we don't need to sync here.
      // Otherwise we might have WG1 writing to dS before WG2 is done reading
      // from it during MmadQ. But because both WGs have to sync at the end of
      // the loop and double buffering, this race condition is not possible.
      // This sync is to ensure (1) P is written in case of !Mma_dKV_is_RS and
      // (2) dS is already read by the Mma in the previous iteration in case of
      // Mma_dKV_is_RS.
      if constexpr (!Mma_dKV_is_RS || (kStages_dS == 1 && Mma_dKV_is_RS)) {
        cutlass::arch::fence_view_async_shared();
        cutlass::arch::NamedBarrier::sync(
            NumMmaThreads, static_cast<uint32_t>(BwdNamedBarriers::PdS) /*id*/);
      }
      // For hdim 64, It's faster to write to smem_dS first before the dV gemm
      Tensor tdSadS =
          smem_thr_copy_PdS.retile_S(rdS); // ((Atom,AtomNum), MMA_N, MMA_N)
      cute::copy(
          smem_tiled_copy_PdS,
          tdSadS,
          tdSsdS(
              _,
              _,
              _,
              cute::conditional_return<kStages_dS == 1>(
                  _0{}, smem_pipe_read.index())));

      if constexpr (!Slice_dQKV_Mma) {
        // Most cases take this path, except for hdim256 where we want to slice
        // to reduce register pressure
        if constexpr (Mma_dKV_is_RS) {
          Tensor tdVrP = make_tensor(
              rP.data(), convert_layout_acc_Aregs<TiledMmadKV>(tSrS.layout()));
          hstu::gemm</*zero_init=*/false, /*wg_wait=*/-1>(
              tiled_mma_dKV,
              tdVrP,
              tdVrdO(_, _, _, smem_pipe_read_do_cur.index()),
              tdVrdV);
        } else {
          Tensor tdVrP =
              mma_partition_fragment_AB</*A=*/!dKV_swapAB>(wg_mma_dKV, sPt);
          Tensor tdVrP_cur = tdVrP(
              _,
              _,
              _,
              cute::conditional_return<kStages_dS == 1>(
                  _0{}, smem_pipe_read.index()));
          hstu::
              gemm</*zero_init=*/false, /*wg_wait=*/-1, /*SwapAB=*/dKV_swapAB>(
                  tiled_mma_dKV,
                  tdVrP_cur,
                  tdVrdO(_, _, _, smem_pipe_read_do_cur.index()),
                  tdVrdV);
        }
        // SMEM fence to make sure sdS is written before it's read by WGMMA
        cutlass::arch::fence_view_async_shared();
        cutlass::arch::NamedBarrier::sync(
            NumMmaThreads, static_cast<uint32_t>(BwdNamedBarriers::PdS) /*id*/);
        Tensor tdQrdQ = partition_fragment_C(
            tiled_mma_dQ,
            select < !dQ_swapAB ? 0 : 2,
            !dQ_swapAB ? 2 : 0 > (TileShape_MNK{}));
        Tensor tdQrdS_cur = tdQrdS(
            _,
            _,
            _,
            cute::conditional_return<kStages_dS == 1>(
                _0{}, smem_pipe_read.index()));
        hstu::gemm</*zero_init=*/true, /*wg_wait=*/1, /*SwapAB=*/dQ_swapAB>(
            tiled_mma_dQ, tdQrdS_cur, tdQrK, tdQrdQ);
        pipeline_do.consumer_release(smem_pipe_read_do_cur); // release dQ

        if constexpr (Mma_dKV_is_RS) {
          Tensor tdKrdS = make_tensor(
              rdS.data(),
              convert_layout_acc_Aregs<TiledMmadKV>(tdPrdP.layout()));
          hstu::gemm</*zero_init=*/false, /*wg_wait=*/1>(
              tiled_mma_dKV,
              tdKrdS,
              tdKrQ(_, _, _, smem_pipe_read.index()),
              tdKrdK);
        } else {
          Tensor tdKrdS =
              mma_partition_fragment_AB</*A=*/!dKV_swapAB>(wg_mma_dKV, sdSt);
          Tensor tdKrdS_cur = tdKrdS(
              _,
              _,
              _,
              cute::conditional_return<kStages_dS == 1>(
                  _0{}, smem_pipe_read.index()));
          hstu::gemm</*zero_init=*/false, /*wg_wait=*/1, /*SwapAB=*/dKV_swapAB>(
              tiled_mma_dKV,
              tdKrdS_cur,
              tdKrQ(_, _, _, smem_pipe_read.index()),
              tdKrdK);
        }
        if constexpr (dQacc_use_TMA) {
          int const warp_group_idx =
              hstu::canonical_warp_group_idx_nosync() - 1;
          cutlass::arch::NamedBarrier::sync(
              cutlass::NumThreadsPerWarpGroup + cutlass::NumThreadsPerWarp,
              static_cast<uint32_t>(BwdNamedBarriers::dQEmptyWG1) +
                  warp_group_idx /*id*/); // sdQ full, to be written to gmem
          Tensor taccdQrdQ = r2s_thr_copy_dQaccum.retile_S(tdQrdQ);
          cute::copy(r2s_tiled_copy_dQaccum, taccdQrdQ, tdQsdQaccum);
          cutlass::arch::fence_view_async_shared();
          cutlass::arch::NamedBarrier::arrive(
              cutlass::NumThreadsPerWarpGroup + cutlass::NumThreadsPerWarp,
              static_cast<uint32_t>(BwdNamedBarriers::dQFullWG1) +
                  warp_group_idx /*id*/); // sdQ full, to be written to gmem
        } else {
          // We can reuse r2s_thr_copy_dQaccum for this partitioning
          Tensor tdQrdQ_atomic =
              recast<float4>(r2s_thr_copy_dQaccum.retile_S(tdQrdQ));
          Tensor tdQgdQaccum_atomic =
              recast<float4>(tdQgdQaccum(_, _, _, m_block));
          static_assert(
              CUTE_STATIC_V(size(tdQrdQ_atomic)) ==
              CUTE_STATIC_V(size(tdQgdQaccum_atomic)));
#pragma unroll
          for (int i = 0; i < size(tdQrdQ_atomic); ++i) {
            atomicAdd(&tdQgdQaccum_atomic(i), tdQrdQ_atomic(i));
          }
        }

      } else { // Slice_dQKV_Mma

        static_assert(!(Slice_dQKV_Mma && Mma_dKV_is_RS));
        Tensor tdVrP =
            mma_partition_fragment_AB</*A=*/!dKV_swapAB>(wg_mma_dKV, sPt);
        Tensor tdVrP_cur = tdVrP(
            _,
            _,
            _,
            cute::conditional_return<kStages_dS == 1>(
                _0{}, smem_pipe_read.index()));
        hstu::gemm<
            /*zero_init=*/false,
            /*wg_wait=*/-1,
            /*SwapAB=*/dKV_swapAB,
            /*M_slice=*/0>(
            tiled_mma_dKV,
            tdVrP_cur,
            tdVrdO(_, _, _, smem_pipe_read_do_cur.index()),
            tdVrdV);

        cutlass::arch::fence_view_async_shared();
        cutlass::arch::NamedBarrier::sync(
            NumMmaThreads, static_cast<uint32_t>(BwdNamedBarriers::PdS) /*id*/);
        Tensor tdQrdQ = partition_fragment_C(
            tiled_mma_dQ,
            select < !dQ_swapAB ? 0 : 2,
            !dQ_swapAB ? 2 : 0 > (TileShape_MNK{}));
        Tensor tdQrdS_cur = tdQrdS(
            _,
            _,
            _,
            cute::conditional_return<kStages_dS == 1>(
                _0{}, smem_pipe_read.index()));
        hstu::gemm<
            /*zero_init=*/true,
            /*wg_wait=*/-1,
            /*SwapAB=*/dQ_swapAB,
            /*M_slice=*/0>(tiled_mma_dQ, tdQrdS_cur, tdQrK, tdQrdQ);
        hstu::gemm<
            /*zero_init=*/false,
            /*wg_wait=*/1,
            /*SwapAB=*/dKV_swapAB,
            /*M_slice=*/1>(
            tiled_mma_dKV,
            tdVrP_cur,
            tdVrdO(_, _, _, smem_pipe_read_do_cur.index()),
            tdVrdV);
        Tensor tdQrdQ_atomic =
            recast<float4>(r2s_thr_copy_dQaccum.retile_S(tdQrdQ));
        Tensor tdQgdQaccum_atomic =
            recast<float4>(tdQgdQaccum(_, _, _, m_block));
#pragma unroll
        for (int i = 0; i < size(tdQrdQ_atomic) / 2; ++i) {
          atomicAdd(&tdQgdQaccum_atomic(i), tdQrdQ_atomic(i));
        }

        Tensor tdKrdS =
            mma_partition_fragment_AB</*A=*/!dKV_swapAB>(wg_mma_dKV, sdSt);
        Tensor tdKrdS_cur = tdKrdS(
            _,
            _,
            _,
            cute::conditional_return<kStages_dS == 1>(
                _0{}, smem_pipe_read.index()));
        hstu::gemm<
            /*zero_init=*/false,
            /*wg_wait=*/1,
            /*SwapAB=*/dKV_swapAB,
            /*M_slice=*/0>(
            tiled_mma_dKV,
            tdKrdS_cur,
            tdKrQ(_, _, _, smem_pipe_read.index()),
            tdKrdK);
        pipeline_do.consumer_release(smem_pipe_read_do_cur); // release dO

        hstu::gemm<
            /*zero_init=*/true,
            /*wg_wait=*/0,
            /*SwapAB=*/dQ_swapAB,
            /*M_slice=*/1>(tiled_mma_dQ, tdQrdS_cur, tdQrK, tdQrdQ);
#pragma unroll
        for (int i = size(tdQrdQ_atomic) / 2; i < size(tdQrdQ_atomic); ++i) {
          atomicAdd(&tdQgdQaccum_atomic(i), tdQrdQ_atomic(i));
        }

        hstu::gemm<
            /*zero_init=*/false,
            /*wg_wait=*/-1,
            /*SwapAB=*/dKV_swapAB,
            /*M_slice=*/1>(
            tiled_mma_dKV,
            tdKrdS_cur,
            tdKrQ(_, _, _, smem_pipe_read.index()),
            tdKrdK);
      }

      warpgroup_wait<0>();
      pipeline_q.consumer_release(smem_pipe_read); // release Q
      ++smem_pipe_read;
      if constexpr (!Q_dO_same_stages) {
        ++smem_pipe_read_do;
      }
    };
    static constexpr int kBlockM = get<0>(TileShape_MNK{});
    static constexpr int kBlockN = get<1>(TileShape_MNK{});

    if constexpr (Cross) {
      auto mask_fn = [&](auto& tSrS, int m_block) {
        mask.template apply<
            true /*Seqlenq_mask*/,
            true /*Seqlenk_mask*/,
            Causal,
            false /*Local*/,
            false /*Contexual_mask*/,
            false /*Target_mask*/,
            Cross,
            false /*Softmax*/>(tSrS, m_block, n_block);
      };
      CUTLASS_PRAGMA_NO_UNROLL
      for (; m_block < m_block_max; ++m_block) {
        bwd_step(m_block, mask_fn);
      }
      if constexpr (Q_dO_same_stages) {
        smem_pipe_read_do = smem_pipe_read;
      }
      ++work_idx;
      return true;
    }
    if constexpr (Has_targets) {
      if (n_block * kBlockN >= seqlen_info.uihlen_q) {
        auto mask_fn = [&](auto& tSrS, int m_block) {
          mask.template apply<
              true /*Seqlenq_mask*/,
              true /*Seqlenk_mask*/,
              false /*Causal*/,
              false /*Local*/,
              false /*Contexual_mask*/,
              Has_targets /*Target_mask*/,
              false /*Cross*/,
              false /*Softmax*/>(tSrS, m_block, n_block);
        };
        CUTLASS_PRAGMA_NO_UNROLL
        for (; m_block < m_block_max; ++m_block) {
          bwd_step(m_block, mask_fn);
        }
        if constexpr (Q_dO_same_stages) {
          smem_pipe_read_do = smem_pipe_read;
        }
        ++work_idx;
        return true;
      } else if ((n_block + 1) * kBlockN >= seqlen_info.uihlen_q) {
        if constexpr ((Causal || Local) && SeparateMaskingIterations) {
          auto mask_fn = [&](auto& tSrS, int m_block) {
            mask.template apply<
                true /*Seqlenq_mask*/,
                true /*Seqlenk_mask*/,
                Causal,
                Local,
                Contexual_mask,
                Has_targets /*Target_mask*/,
                false /*Cross*/,
                false /*Softmax*/>(tSrS, m_block, n_block);
          };
          int const m_block_masking_max =
              ((n_block + 1) * kBlockN - 1) / kBlockM + 1;
          CUTLASS_PRAGMA_NO_UNROLL
          for (; m_block < std::min(m_block_max, m_block_masking_max);
               ++m_block) {
            bwd_step(m_block, mask_fn);
          }
        }

        auto mask_fn = [&](auto& tSrS, int m_block) {
          mask.template apply<
              true /*Seqlenq_mask*/,
              true /*Seqlenk_mask*/,
              Causal && !SeparateMaskingIterations,
              Local && !SeparateMaskingIterations,
              Contexual_mask,
              Has_targets /*Target_mask*/,
              false /*Cross*/,
              false /*Softmax*/>(tSrS, m_block, n_block);
        };
        if constexpr (SeparateMaskingIterations) {
          int const m_block_max_before_local_mask =
              !Local || !SeparateMaskingIterations
              ? m_block_max
              : std::min(
                    m_block_max,
                    (n_block * kBlockN + params.max_attn_len) / kBlockM);
          CUTLASS_PRAGMA_NO_UNROLL
          for (; m_block < m_block_max_before_local_mask; ++m_block) {
            bwd_step(m_block, mask_fn);
          }
        } else {
          int num_m_block = m_block_max - m_block_min;
          CUTLASS_PRAGMA_NO_UNROLL
          for (int i = 0; i < num_m_block + full_m_block_max -
                   full_m_block_min + contexual_m_block_max;
               ++i) {
            if (i < num_m_block) {
              m_block = m_block_min + i;
            } else if (i < num_m_block + contexual_m_block_max) {
              m_block = i - num_m_block;
            } else {
              m_block =
                  i - num_m_block - contexual_m_block_max + full_m_block_min;
            }
            bwd_step(m_block, mask_fn);
          }
        }

        if constexpr (Local && SeparateMaskingIterations) {
          auto mask_fn = [&](auto& tSrS, int m_block) {
            mask.template apply<
                true /*Seqlenq_mask*/,
                true /*Seqlenk_mask*/,
                false /*Causal_mask*/,
                Local,
                Contexual_mask,
                Has_targets /*Target_mask*/,
                false /*Cross*/,
                false /*Softmax*/>(tSrS, m_block, n_block);
          };
          CUTLASS_PRAGMA_NO_UNROLL
          for (; m_block < m_block_max; ++m_block) {
            bwd_step(m_block, mask_fn);
          }
        }
        if constexpr (Contexual_mask && SeparateMaskingIterations) {
          auto mask_fn = [&](auto& tSrS, int m_block) {
            mask.template apply<
                true /*Seqlenq_mask*/,
                true /*Seqlenk_mask*/,
                Causal /*Causal_mask*/,
                Local /*Local_mask*/,
                Contexual_mask,
                Has_targets,
                false /*Cross*/,
                false /*Softmax*/>(tSrS, m_block, n_block);
          };
          CUTLASS_PRAGMA_NO_UNROLL
          for (m_block = 0; m_block < contexual_m_block_max; ++m_block) {
            bwd_step(m_block, mask_fn);
          }
        }

        if constexpr (Local && SeparateMaskingIterations) {
          auto mask_fn = [&](auto& tSrS, int m_block) {
            mask.template apply<
                true /*Seqlenq_mask*/,
                true /*Seqlenk_mask*/,
                false /*Causal_mask*/,
                Local,
                Contexual_mask,
                Has_targets,
                false /*Cross*/,
                false /*Softmax*/>(tSrS, m_block, n_block);
          };
          CUTLASS_PRAGMA_NO_UNROLL
          for (m_block = full_m_block_min; m_block < full_m_block_max;
               ++m_block) {
            bwd_step(m_block, mask_fn);
          }
        }
        if constexpr (Q_dO_same_stages) {
          smem_pipe_read_do = smem_pipe_read;
        }
        ++work_idx;
        return true;
      }
    }
    // We have separate iterations with causal masking. Not necessary for hdim
    // 128 but for hdim 64 this helps quite a bit to not have to do causal
    // masking for most of the iterations.
    if constexpr ((Causal || Local) && SeparateMaskingIterations) {
      auto mask_fn = [&](auto& tSrS, int m_block) {
        mask.template apply<
            true /*Seqlenq_mask*/,
            true /*Seqlenk_mask*/,
            Causal,
            Local,
            Contexual_mask,
            false /*Target_mask*/,
            false /*Cross*/,
            false /*Softmax*/>(tSrS, m_block, n_block);
      };
      static constexpr int kBlockM = get<0>(TileShape_MNK{});
      int const m_block_masking_max =
          ((n_block + 1) * kBlockN - 1) / kBlockM + 1;
      CUTLASS_PRAGMA_NO_UNROLL
      for (; m_block < std::min(m_block_max, m_block_masking_max); ++m_block) {
        bwd_step(m_block, mask_fn);
      }
    }

    auto mask_fn = [&](auto& tSrS, int m_block) {
      mask.template apply<
          true /*Seqlenq_mask*/,
          true /*Seqlenk_mask*/,
          Causal && !SeparateMaskingIterations,
          Local && !SeparateMaskingIterations,
          Contexual_mask,
          false /*Target_mask*/,
          false /*Cross*/,
          false /*Softmax*/>(tSrS, m_block, n_block);
    };
    if constexpr (SeparateMaskingIterations) {
      int const m_block_max_before_local_mask =
          !Local || !SeparateMaskingIterations
          ? m_block_max
          : std::min(
                m_block_max,
                (n_block * kBlockN + params.max_attn_len) / kBlockM);
      CUTLASS_PRAGMA_NO_UNROLL
      for (; m_block < m_block_max_before_local_mask; ++m_block) {
        bwd_step(m_block, mask_fn);
      }
    } else {
      int num_m_block = m_block_max - m_block_min;
      CUTLASS_PRAGMA_NO_UNROLL
      for (int i = 0; i < num_m_block + full_m_block_max - full_m_block_min +
               contexual_m_block_max;
           ++i) {
        if (i < num_m_block) {
          m_block = m_block_min + i;
        } else if (i < num_m_block + contexual_m_block_max) {
          m_block = i - num_m_block;
        } else {
          m_block = i - num_m_block - contexual_m_block_max + full_m_block_min;
        }
        bwd_step(m_block, mask_fn);
      }
    }

    if constexpr (Local && SeparateMaskingIterations) {
      auto mask_fn = [&](auto& tSrS, int m_block) {
        mask.template apply<
            true /*Seqlenq_mask*/,
            true /*Seqlenk_mask*/,
            false /*Causal_mask*/,
            Local,
            Contexual_mask,
            false /*Target_mask*/,
            false /*Cross*/,
            false /*Softmax*/>(tSrS, m_block, n_block);
      };
      CUTLASS_PRAGMA_NO_UNROLL
      for (; m_block < m_block_max; ++m_block) {
        bwd_step(m_block, mask_fn);
      }
    }
    if constexpr (Contexual_mask && SeparateMaskingIterations) {
      auto mask_fn = [&](auto& tSrS, int m_block) {
        mask.template apply<
            true /*Seqlenq_mask*/,
            true /*Seqlenk_mask*/,
            Causal /*Causal_mask*/,
            Local /*Local_mask*/,
            Contexual_mask,
            false /*Target_mask*/,
            false /*Cross*/,
            false /*Softmax*/>(tSrS, m_block, n_block);
      };
      CUTLASS_PRAGMA_NO_UNROLL
      for (m_block = 0; m_block < contexual_m_block_max; ++m_block) {
        bwd_step(m_block, mask_fn);
      }
    }

    if constexpr (Local && SeparateMaskingIterations) {
      auto mask_fn = [&](auto& tSrS, int m_block) {
        mask.template apply<
            true /*Seqlenq_mask*/,
            true /*Seqlenk_mask*/,
            false /*Causal_mask*/,
            Local,
            Contexual_mask,
            false /*Target_mask*/,
            false /*Cross*/,
            false /*Softmax*/>(tSrS, m_block, n_block);
      };
      CUTLASS_PRAGMA_NO_UNROLL
      for (m_block = full_m_block_min; m_block < full_m_block_max; ++m_block) {
        bwd_step(m_block, mask_fn);
      }
    }

    // if (blockIdx.x == 0 && threadIdx.x == 128) { print_tensor(tdVrdV); }
    if constexpr (Q_dO_same_stages) {
      smem_pipe_read_do = smem_pipe_read;
    }
    ++work_idx;
    return true;
  }

  template <typename SharedStorage, typename FrgTensordKV>
  CUTLASS_DEVICE bool mma_softmax(
      Params const& params,
      MainloopPipeline pipeline_q,
      MainloopPipeline_dO pipeline_do,
      PipelineState& smem_pipe_read,
      PipelineState_dO& smem_pipe_read_do,
      FrgTensordKV& tdKrdK,
      FrgTensordKV& tdVrdV,
      int thread_idx,
      int& work_idx,
      cute::tuple<int32_t, int32_t, int32_t> block_coord,
      SharedStorage& shared_storage) {
    static_assert(
        is_rmem<FrgTensordKV>::value,
        "dK and dV tensor must be rmem resident.");

    int n_block = get<0>(block_coord);
    int bidb = get<2>(block_coord);
    SeqlenInfo_t seqlen_info{
        bidb,
        get<0>(params.shape_Q),
        get<0>(params.shape_K),
        params.seq_offsets,
        params.seq_offsets_q,
        params.num_targets};
    if constexpr (Jagged) {
      static constexpr int kBlockN = get<1>(TileShape_MNK{});
      if (n_block * kBlockN >= seqlen_info.seqlen_kv) {
        return false;
      }
    }
    int m_block_min, m_block_max;
    if constexpr (Cross) {
      auto m_block_min_max = get_cross_m_block_min_max(
          seqlen_info.uihlen_q,
          seqlen_info.seqlen_q,
          seqlen_info.seqlen_kv,
          n_block);
      m_block_min = get<0>(m_block_min_max);
      m_block_max = get<1>(m_block_min_max);
    } else {
      auto m_block_min_max = get_m_block_min_max(
          params.max_attn_len,
          params.contextual_seq_len,
          seqlen_info.uihlen_q,
          seqlen_info.seqlen_q,
          n_block);
      m_block_min = get<0>(m_block_min_max);
      m_block_max = get<1>(m_block_min_max);
    }
    auto full_m_block_min_max = get_full_m_block_min_max(
        seqlen_info.uihlen_q,
        seqlen_info.seqlen_q,
        params.min_full_attn_seq_len,
        m_block_max,
        n_block);
    int const full_m_block_min = get<0>(full_m_block_min_max);
    int const full_m_block_max = get<1>(full_m_block_min_max);
    int contexual_m_block_max = get_contexual_m_block_max(
        seqlen_info.uihlen_q, params.contextual_seq_len, m_block_min, n_block);

    Tensor sQ = make_tensor(
        make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()),
        SmemLayoutQ{});
    Tensor sdO = make_tensor(
        make_smem_ptr(shared_storage.tensors.mainloop.smem_do.data()),
        SmemLayoutdO{});
    Tensor sK = make_tensor(
        make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()),
        SmemLayoutK{});
    Tensor sV = make_tensor(
        make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()),
        SmemLayoutV{});
    Tensor sQt = make_tensor(
        make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()),
        SmemLayoutQt{});
    Tensor sdOt = make_tensor(
        make_smem_ptr(shared_storage.tensors.mainloop.smem_do.data()),
        SmemLayoutdOt{});
    Tensor sKt = make_tensor(
        make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()),
        SmemLayoutKt{});
    Tensor sP = make_tensor(
        make_smem_ptr(shared_storage.tensors.mainloop.smem_p.data()),
        SmemLayoutPdS{});
    Tensor sP_pi = cute::as_position_independent_swizzle_tensor(sP);
    Tensor sPt = make_tensor(
        make_smem_ptr(shared_storage.tensors.mainloop.smem_p.data()),
        SmemLayoutPdSt{});
    Tensor sPt_pi = cute::as_position_independent_swizzle_tensor(sPt);
    Tensor sdS = make_tensor(
        make_smem_ptr(shared_storage.tensors.mainloop.smem_ds.data()),
        SmemLayoutPdS{});
    Tensor sdS_pi = cute::as_position_independent_swizzle_tensor(sdS);
    Tensor sdSt = make_tensor(
        make_smem_ptr(shared_storage.tensors.mainloop.smem_ds.data()),
        SmemLayoutPdSt{});
    Tensor sdSt_pi = cute::as_position_independent_swizzle_tensor(sdSt);
    Tensor sdQ = make_tensor(
        make_smem_ptr(shared_storage.tensors.mainloop.smem_dqacc.data()),
        SmemLayoutdQaccum{});
    Tensor sLSEMma = make_tensor(
        make_smem_ptr(shared_storage.tensors.mainloop.smem_lse.data()),
        SmemLayoutLSEMma{});
    Tensor sdPsumMma = make_tensor(
        make_smem_ptr(shared_storage.tensors.mainloop.smem_dpsum.data()),
        SmemLayoutLSEMma{});

    static_assert(
        stride<0>(typename TiledMmaSdP::ALayout{}) == 0 and
            stride<0>(typename TiledMmaSdP::BLayout{}) == 0 and
            size<0>(typename TiledMmaSdP::ALayout{}) ==
                cutlass::NumThreadsPerWarpGroup and
            size<0>(typename TiledMmaSdP::BLayout{}) ==
                cutlass::NumThreadsPerWarpGroup,
        "Stride of the first mode must be 0 and the size of the mode must be NumThreadsPerWarpGroup");
    constexpr int MmaWarpGroups =
        NumMmaThreads / cutlass::NumThreadsPerWarpGroup;
    Layout warp_group_thread_layout = make_layout(
        make_shape(Int<MmaWarpGroups>{}),
        make_stride(Int<cutlass::NumThreadsPerWarpGroup>{}));
    Layout warp_group_thread_layout_dq = make_layout(
        make_shape(Int<NumMmaWarpGroups>{}),
        make_stride(Int<cutlass::NumThreadsPerWarpGroup>{}));

    int warp_group_idx = __shfl_sync(
        0xFFFFFFFF, thread_idx / cutlass::NumThreadsPerWarpGroup, 0);
    TiledMmaSdP tiled_mma_SdP;
    using TiledMmadP =
        std::conditional_t<!Mma_dP_is_RS, TiledMmaSdP, TiledMmadPRS>;
    TiledMmadP tiled_mma_dP;
    TiledMmadKV tiled_mma_dKV;
    TiledMmadQ tiled_mma_dQ;

    auto wg_mma_SdP =
        tiled_mma_SdP.get_slice(warp_group_thread_layout(warp_group_idx));
    auto wg_mma_dP =
        tiled_mma_dP.get_slice(warp_group_thread_layout(warp_group_idx));
    auto thread_mma_SdP = tiled_mma_SdP.get_thread_slice(thread_idx);
    auto wg_mma_dKV =
        tiled_mma_dKV.get_slice(warp_group_thread_layout(warp_group_idx));
    auto wg_mma_dQ =
        tiled_mma_dQ.get_slice(warp_group_thread_layout_dq(warp_group_idx));

    auto smem_tiled_copy_PdS =
        make_tiled_copy_C(SmemCopyAtomPdS{}, tiled_mma_SdP);
    auto smem_thr_copy_PdS = smem_tiled_copy_PdS.get_thread_slice(thread_idx);

    R2STiledCopydQaccum r2s_tiled_copy_dQaccum;
    auto r2s_thr_copy_dQaccum =
        r2s_tiled_copy_dQaccum.get_thread_slice(thread_idx);
    Tensor tdQsdQaccum = r2s_thr_copy_dQaccum.partition_D(sdQ);
    // if (thread_idx == 0) { print(sdQ); printf("\n"); print(tdQsdQaccum);
    // printf("\n"); }

    // Allocate "fragments/descriptors"
    // We have to use the templated mma_partition_fragment_AB instead of
    // cute::conditional_return or lambda, because some partition_fragment_A/B
    // don't compile.
    // https://stackoverflow.com/questions/50051473/if-constexpr-in-c17-does-not-work-in-a-non-templated-function
    Tensor tSrQ = mma_partition_fragment_AB</*A=*/!SdP_swapAB>(wg_mma_SdP, sQ);
    Tensor tSrK = mma_partition_fragment_AB</*A=*/SdP_swapAB>(wg_mma_SdP, sK);
    Tensor tdPrdO =
        mma_partition_fragment_AB</*A=*/!SdP_swapAB>(wg_mma_SdP, sdO);
    Tensor tdPrV = mma_partition_fragment_AB</*A=*/SdP_swapAB>(wg_mma_dP, sV);
    Tensor tdVrdO =
        mma_partition_fragment_AB</*A=*/dKV_swapAB>(wg_mma_dKV, sdOt);
    Tensor tdKrQ = mma_partition_fragment_AB</*A=*/dKV_swapAB>(wg_mma_dKV, sQt);
    Tensor tdQrdS = mma_partition_fragment_AB</*A=*/!dQ_swapAB>(wg_mma_dQ, sdS);
    Tensor tdQrK = mma_partition_fragment_AB</*A=*/dQ_swapAB>(wg_mma_dQ, sKt);

    Tensor tPsP =
        smem_thr_copy_PdS.partition_D(cute::conditional_return<!SdP_swapAB>(
            sP_pi, sPt_pi)); // ((Atom,AtomNum),PIPE_M,PIPE_N)
    Tensor tdSsdS =
        smem_thr_copy_PdS.partition_D(cute::conditional_return<!SdP_swapAB>(
            sdS_pi, sdSt_pi)); // ((Atom,AtomNum),PIPE_M,PIPE_N)
    // if (blockIdx.x == 0 && threadIdx.x == 128) { print(smem_thr_copy_PdS);
    // print(sP_pi); printf("\n"); print(sPt_pi); printf("\n"); print(tPsP);
    // printf("\n"); print(tdSsdS); printf("\n"); }

    // thread_mma_SdP.partition_C(sLSEMma) has shape ((2, 2, V), MMA_M, MMA_N,
    // PIPE), we only take the col indices or row indices, depending on whether
    // SdP_swapAB.
    Tensor tLSEsLSE = cute::conditional_return<!SdP_swapAB>(
        group_modes<0, 2>(thread_mma_SdP.partition_C(sLSEMma)(
            make_coord(_0{}, _, _0{}), _, _0{}, _)), // (2, MMA_M, PIPE)
        group_modes<0, 3>(thread_mma_SdP.partition_C(sLSEMma)(
            make_coord(_, _0{}, _), _0{}, _, _))); // (2, V, MMA_N, PIPE)
    Tensor tLSEsdPsum = cute::conditional_return<!SdP_swapAB>(
        group_modes<0, 2>(thread_mma_SdP.partition_C(sdPsumMma)(
            make_coord(_0{}, _, _0{}), _, _0{}, _)),
        group_modes<0, 3>(thread_mma_SdP.partition_C(sdPsumMma)(
            make_coord(_, _0{}, _), _0{}, _, _)));
    // if (blockIdx.x == 0 && threadIdx.x == 128) { print(sLSEMma);
    // printf("\n"); print(tLSEsLSE); printf("\n"); } If we want to split the
    // stats among the 8 threads that share the same rows.
    static constexpr int kStatsPerThread =
        cute::ceil_div(decltype(size(tLSEsLSE))::value, 8);

    auto consumer_wait = [](auto& pipeline, auto& smem_pipe_read) {
      auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      pipeline.consumer_wait(smem_pipe_read, barrier_token);
    };

    int bidh = get<1>(block_coord);
    // For the case where we do atomicAdd directly to gdQaccum instead of using
    // TMA
    Tensor mdQaccum = make_tensor(
        make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.ptr_dQaccum)),
        params.shape_dQaccum,
        params.stride_dQaccum)(_, bidh, !Jagged ? bidb : 0);
    Tensor gdQaccum_ = local_tile(
        domain_offset(
            make_coord(seqlen_info.offset_q_padded * kHeadDim), mdQaccum),
        Shape<Int<kBlockM * kHeadDim>>{},
        make_coord(_)); // (M * K, _)
    Tensor gdQaccum = cute::flat_divide(
        gdQaccum_,
        Int<kBlockM * kHeadDim / NumMmaWarpGroups>{}); // (M * K / WG, WG, _)
    // We can reuse r2s_thr_copy_dQaccum for this partitioning
    Tensor tdQgdQaccum = r2s_thr_copy_dQaccum.partition_D(gdQaccum);
    // if (blockIdx.x == 0 && threadIdx.x == 128) { print(mdQaccum);
    // printf("\n"); print(gdQaccum_); printf("\n"); print(gdQaccum);
    // printf("\n"); print(tdQgdQaccum); printf("\n"); }

    hstu::Mask<kBlockM, kBlockN, TiledMmaSdP, SdP_swapAB> mask(
        thread_idx,
        seqlen_info.seqlen_q,
        seqlen_info.seqlen_kv,
        params.max_attn_len,
        params.min_full_attn_seq_len,
        params.contextual_seq_len,
        seqlen_info.uihlen_q);

    int m_block = m_block_min;

    clear(tdKrdK);
    clear(tdVrdV);
    // tiled_mma_dKV.accumulate_ = GMMA::ScaleOut::Zero;

    cutlass::ConsumerToken barrier_token = static_cast<cutlass::BarrierStatus>(
        shared_storage.pipelines.barrier_KV.try_wait(work_idx % 2));
    if (barrier_token == cutlass::BarrierStatus::WaitAgain) {
      shared_storage.pipelines.barrier_KV.wait(work_idx % 2);
    }

    if constexpr (Mma_dP_is_RS) {
      using SmemCopyAtomV = Copy_Atom<cute::SM75_U32x4_LDSM_N, Element>;
      auto smem_tiled_copy_V = make_tiled_copy_A(SmemCopyAtomV{}, tiled_mma_dP);
      auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(thread_idx);
      Tensor tdPrV_copy_view = smem_thr_copy_V.retile_D(tdPrV);
      Tensor tdPsV_copy_view = smem_thr_copy_V.partition_S(
          cute::as_position_independent_swizzle_tensor(sV));
      cute::copy(smem_tiled_copy_V, tdPsV_copy_view, tdPrV_copy_view);
    }
    static constexpr int Qdim = !SdP_swapAB ? 0 : 1;
    auto thread0_mma_SdP = tiled_mma_SdP.get_thread_slice(_0{});
    Tensor cS =
        cute::make_identity_tensor(Shape<
                                   Int<!SdP_swapAB ? kBlockM : kBlockN>,
                                   Int<!SdP_swapAB ? kBlockN : kBlockM>>{});
    Tensor tScS = thread_mma_SdP.partition_C(cS);
    Tensor tScS_rowcol = make_tensor(
        tScS.data(),
        hstu::convert_layout_acc_rowcol</*Transposed=*/SdP_swapAB>(
            tScS.layout()));
    Tensor t0ScS = thread0_mma_SdP.partition_C(cS);
    Tensor t0ScS_rowcol = make_tensor(
        t0ScS.data(),
        hstu::convert_layout_acc_rowcol</*Transposed=*/SdP_swapAB>(
            t0ScS.layout()));
    int const thread_qdim_offset = get<Qdim>(tScS_rowcol(_0{}, _0{}));

    auto bwd_step = [&](int m_block, auto mask_fn) {
      Tensor tSrS = partition_fragment_C(
          tiled_mma_SdP,
          select < !SdP_swapAB ? 0 : 1,
          !SdP_swapAB ? 1 : 0 > (TileShape_MNK{}));
      consumer_wait(pipeline_q, smem_pipe_read);
      hstu::gemm</*zero_init=*/true, /*wg_wait=*/-1, /*SwapAB=*/SdP_swapAB>(
          tiled_mma_SdP, tSrQ(_, _, _, smem_pipe_read.index()), tSrK, tSrS);
      Tensor tLSErLSE = cute::conditional_return<!ShuffleLSE>(
          make_fragment_like(tLSEsLSE(_, _0{})),
          make_tensor<ElementAccum>(Int<kStatsPerThread>{}));
      if constexpr (!ShuffleLSE) {
        cute::copy(tLSEsLSE(_, smem_pipe_read.index()), tLSErLSE);
      } else {
#pragma unroll
        for (int i = 0; i < kStatsPerThread; ++i) {
          // It's ok to read OOB, since we made sure sLSE is large enough and we
          // won't use the OOB values
          tLSErLSE(i) =
              tLSEsLSE((thread_idx % 32) / 4 + i * 8, smem_pipe_read.index());
        }
      }
      Tensor tdPrdP = partition_fragment_C(
          tiled_mma_SdP,
          select < !SdP_swapAB ? 0 : 1,
          !SdP_swapAB ? 1 : 0 > (TileShape_MNK{}));
      PipelineState_dO smem_pipe_read_do_cur =
          cute::conditional_return<Q_dO_same_stages>(
              smem_pipe_read, smem_pipe_read_do);
      consumer_wait(pipeline_do, smem_pipe_read_do_cur);
      hstu::gemm</*zero_init=*/true, /*wg_wait=*/-1, /*SwapAB=*/SdP_swapAB>(
          tiled_mma_dP,
          tdPrdO(_, _, _, smem_pipe_read_do_cur.index()),
          tdPrV,
          tdPrdP);
      warpgroup_wait<1>();
      // Reshape tSrS from ((2, 2, V), MMA_N, MMA_M) to (nrow=(2, V, MMA_M),
      // ncol=(2, MMA_N))
      Tensor scores = make_tensor(
          tSrS.data(),
          hstu::convert_layout_acc_rowcol</*Transposed=*/SdP_swapAB>(
              tSrS.layout()));
      mask_fn(tSrS, m_block);
#pragma unroll
      for (int mi = 0; mi < size<0>(scores); ++mi) {
        float const lse_scaled = [&] {
          if constexpr (!ShuffleLSE)
            return tLSErLSE(mi);
          else
            return __shfl_sync(
                0xffffffff, tLSErLSE(mi / 8), (mi % 8) * 4 + (thread_idx % 4));
        }();
#pragma unroll
        for (int ni = 0; ni < size<1>(scores); ++ni) {
          scores(mi, ni) =
              exp2f(scores(mi, ni) * params.alpha_log2 - lse_scaled);
        }
      }
      Tensor tLSErdPsum = cute::conditional_return<!ShuffledPsum>(
          make_fragment_like(tLSEsdPsum(_, _0{})),
          make_tensor<ElementAccum>(Int<kStatsPerThread>{}));
      if constexpr (!ShuffledPsum) {
        cute::copy(tLSEsdPsum(_, smem_pipe_read_do_cur.index()), tLSErdPsum);
      } else {
#pragma unroll
        for (int i = 0; i < kStatsPerThread; ++i) {
          tLSErdPsum(i) = tLSEsdPsum(
              (thread_idx % 32) / 4 + i * 8, smem_pipe_read_do_cur.index());
        }
      }

      warpgroup_wait<0>();
      // Reshape tdPrdP from ((2, 2, V), MMA_N, MMA_M) to (nrow=(2, V, MMA_M),
      // ncol=(2, MMA_N))
      Tensor dS = make_tensor(tdPrdP.data(), scores.layout());
#pragma unroll
      for (int mi = 0; mi < size<0>(dS); ++mi) {
        float const dP_sum_cur = [&] {
          if constexpr (!ShuffledPsum)
            return tLSErdPsum(mi);
          else
            return __shfl_sync(
                0xffffffff,
                tLSErdPsum(mi / 8),
                (mi % 8) * 4 + (thread_idx % 4));
        }();
#pragma unroll
        for (int ni = 0; ni < size<1>(dS); ++ni) {
          dS(mi, ni) =
              scores(mi, ni) * (dS(mi, ni) - dP_sum_cur) * params.alpha;
        }
      }
      // Convert scores from fp32 to fp16/bf16
      Tensor rP = make_tensor_like<Element>(tSrS);
      hstu::convert_type_out(tSrS, rP);
      if constexpr (!Mma_dKV_is_RS) {
        // Need to sync to make sure P has already been used in the previous
        // iteration before writing new values
        if constexpr (kStages_dS == 1) {
          cutlass::arch::NamedBarrier::sync(
              NumMmaThreads,
              static_cast<uint32_t>(BwdNamedBarriers::PdS) /*id*/);
        }
        Tensor tPaP =
            smem_thr_copy_PdS.retile_S(rP); // ((Atom,AtomNum), MMA_N, MMA_N)
        cute::copy(
            smem_tiled_copy_PdS,
            tPaP,
            tPsP(
                _,
                _,
                _,
                cute::conditional_return<kStages_dS == 1>(
                    _0{}, smem_pipe_read.index())));
      }
      Tensor rdS = make_tensor_like<Element>(tdPrdP);
      hstu::convert_type_out(tdPrdP, rdS);
      // If there's double buffering on dS, we don't need to sync here.
      // Otherwise we might have WG1 writing to dS before WG2 is done reading
      // from it during MmadQ. But because both WGs have to sync at the end of
      // the loop and double buffering, this race condition is not possible.
      // This sync is to ensure (1) P is written in case of !Mma_dKV_is_RS and
      // (2) dS is already read by the Mma in the previous iteration in case of
      // Mma_dKV_is_RS.
      if constexpr (!Mma_dKV_is_RS || (kStages_dS == 1 && Mma_dKV_is_RS)) {
        cutlass::arch::fence_view_async_shared();
        cutlass::arch::NamedBarrier::sync(
            NumMmaThreads, static_cast<uint32_t>(BwdNamedBarriers::PdS) /*id*/);
      }
      // For hdim 64, It's faster to write to smem_dS first before the dV gemm
      Tensor tdSadS =
          smem_thr_copy_PdS.retile_S(rdS); // ((Atom,AtomNum), MMA_N, MMA_N)
      cute::copy(
          smem_tiled_copy_PdS,
          tdSadS,
          tdSsdS(
              _,
              _,
              _,
              cute::conditional_return<kStages_dS == 1>(
                  _0{}, smem_pipe_read.index())));

      if constexpr (!Slice_dQKV_Mma) {
        // Most cases take this path, except for hdim256 where we want to slice
        // to reduce register pressure
        if constexpr (Mma_dKV_is_RS) {
          Tensor tdVrP = make_tensor(
              rP.data(), convert_layout_acc_Aregs<TiledMmadKV>(tSrS.layout()));
          hstu::gemm</*zero_init=*/false, /*wg_wait=*/-1>(
              tiled_mma_dKV,
              tdVrP,
              tdVrdO(_, _, _, smem_pipe_read_do_cur.index()),
              tdVrdV);
        } else {
          Tensor tdVrP =
              mma_partition_fragment_AB</*A=*/!dKV_swapAB>(wg_mma_dKV, sPt);
          Tensor tdVrP_cur = tdVrP(
              _,
              _,
              _,
              cute::conditional_return<kStages_dS == 1>(
                  _0{}, smem_pipe_read.index()));
          hstu::
              gemm</*zero_init=*/false, /*wg_wait=*/-1, /*SwapAB=*/dKV_swapAB>(
                  tiled_mma_dKV,
                  tdVrP_cur,
                  tdVrdO(_, _, _, smem_pipe_read_do_cur.index()),
                  tdVrdV);
        }
        // SMEM fence to make sure sdS is written before it's read by WGMMA
        cutlass::arch::fence_view_async_shared();
        cutlass::arch::NamedBarrier::sync(
            NumMmaThreads, static_cast<uint32_t>(BwdNamedBarriers::PdS) /*id*/);
        Tensor tdQrdQ = partition_fragment_C(
            tiled_mma_dQ,
            select < !dQ_swapAB ? 0 : 2,
            !dQ_swapAB ? 2 : 0 > (TileShape_MNK{}));
        Tensor tdQrdS_cur = tdQrdS(
            _,
            _,
            _,
            cute::conditional_return<kStages_dS == 1>(
                _0{}, smem_pipe_read.index()));
        hstu::gemm</*zero_init=*/true, /*wg_wait=*/1, /*SwapAB=*/dQ_swapAB>(
            tiled_mma_dQ, tdQrdS_cur, tdQrK, tdQrdQ);
        pipeline_do.consumer_release(smem_pipe_read_do_cur); // release dQ

        if constexpr (Mma_dKV_is_RS) {
          Tensor tdKrdS = make_tensor(
              rdS.data(),
              convert_layout_acc_Aregs<TiledMmadKV>(tdPrdP.layout()));
          hstu::gemm</*zero_init=*/false, /*wg_wait=*/1>(
              tiled_mma_dKV,
              tdKrdS,
              tdKrQ(_, _, _, smem_pipe_read.index()),
              tdKrdK);
        } else {
          Tensor tdKrdS =
              mma_partition_fragment_AB</*A=*/!dKV_swapAB>(wg_mma_dKV, sdSt);
          Tensor tdKrdS_cur = tdKrdS(
              _,
              _,
              _,
              cute::conditional_return<kStages_dS == 1>(
                  _0{}, smem_pipe_read.index()));
          hstu::gemm</*zero_init=*/false, /*wg_wait=*/1, /*SwapAB=*/dKV_swapAB>(
              tiled_mma_dKV,
              tdKrdS_cur,
              tdKrQ(_, _, _, smem_pipe_read.index()),
              tdKrdK);
        }
        if constexpr (dQacc_use_TMA) {
          int const warp_group_idx =
              hstu::canonical_warp_group_idx_nosync() - 1;
          cutlass::arch::NamedBarrier::sync(
              cutlass::NumThreadsPerWarpGroup + cutlass::NumThreadsPerWarp,
              static_cast<uint32_t>(BwdNamedBarriers::dQEmptyWG1) +
                  warp_group_idx /*id*/); // sdQ full, to be written to gmem
          Tensor taccdQrdQ = r2s_thr_copy_dQaccum.retile_S(tdQrdQ);
          cute::copy(r2s_tiled_copy_dQaccum, taccdQrdQ, tdQsdQaccum);
          cutlass::arch::fence_view_async_shared();
          cutlass::arch::NamedBarrier::arrive(
              cutlass::NumThreadsPerWarpGroup + cutlass::NumThreadsPerWarp,
              static_cast<uint32_t>(BwdNamedBarriers::dQFullWG1) +
                  warp_group_idx /*id*/); // sdQ full, to be written to gmem
        } else {
          // We can reuse r2s_thr_copy_dQaccum for this partitioning
          Tensor tdQrdQ_atomic =
              recast<float4>(r2s_thr_copy_dQaccum.retile_S(tdQrdQ));
          Tensor tdQgdQaccum_atomic =
              recast<float4>(tdQgdQaccum(_, _, _, m_block));
          static_assert(
              CUTE_STATIC_V(size(tdQrdQ_atomic)) ==
              CUTE_STATIC_V(size(tdQgdQaccum_atomic)));
#pragma unroll
          for (int i = 0; i < size(tdQrdQ_atomic); ++i) {
            atomicAdd(&tdQgdQaccum_atomic(i), tdQrdQ_atomic(i));
          }
        }

      } else { // Slice_dQKV_Mma

        static_assert(!(Slice_dQKV_Mma && Mma_dKV_is_RS));
        Tensor tdVrP =
            mma_partition_fragment_AB</*A=*/!dKV_swapAB>(wg_mma_dKV, sPt);
        Tensor tdVrP_cur = tdVrP(
            _,
            _,
            _,
            cute::conditional_return<kStages_dS == 1>(
                _0{}, smem_pipe_read.index()));
        hstu::gemm<
            /*zero_init=*/false,
            /*wg_wait=*/-1,
            /*SwapAB=*/dKV_swapAB,
            /*M_slice=*/0>(
            tiled_mma_dKV,
            tdVrP_cur,
            tdVrdO(_, _, _, smem_pipe_read_do_cur.index()),
            tdVrdV);

        cutlass::arch::fence_view_async_shared();
        cutlass::arch::NamedBarrier::sync(
            NumMmaThreads, static_cast<uint32_t>(BwdNamedBarriers::PdS) /*id*/);
        Tensor tdQrdQ = partition_fragment_C(
            tiled_mma_dQ,
            select < !dQ_swapAB ? 0 : 2,
            !dQ_swapAB ? 2 : 0 > (TileShape_MNK{}));
        Tensor tdQrdS_cur = tdQrdS(
            _,
            _,
            _,
            cute::conditional_return<kStages_dS == 1>(
                _0{}, smem_pipe_read.index()));
        hstu::gemm<
            /*zero_init=*/true,
            /*wg_wait=*/-1,
            /*SwapAB=*/dQ_swapAB,
            /*M_slice=*/0>(tiled_mma_dQ, tdQrdS_cur, tdQrK, tdQrdQ);
        hstu::gemm<
            /*zero_init=*/false,
            /*wg_wait=*/1,
            /*SwapAB=*/dKV_swapAB,
            /*M_slice=*/1>(
            tiled_mma_dKV,
            tdVrP_cur,
            tdVrdO(_, _, _, smem_pipe_read_do_cur.index()),
            tdVrdV);
        Tensor tdQrdQ_atomic =
            recast<float4>(r2s_thr_copy_dQaccum.retile_S(tdQrdQ));
        Tensor tdQgdQaccum_atomic =
            recast<float4>(tdQgdQaccum(_, _, _, m_block));
#pragma unroll
        for (int i = 0; i < size(tdQrdQ_atomic) / 2; ++i) {
          atomicAdd(&tdQgdQaccum_atomic(i), tdQrdQ_atomic(i));
        }

        Tensor tdKrdS =
            mma_partition_fragment_AB</*A=*/!dKV_swapAB>(wg_mma_dKV, sdSt);
        Tensor tdKrdS_cur = tdKrdS(
            _,
            _,
            _,
            cute::conditional_return<kStages_dS == 1>(
                _0{}, smem_pipe_read.index()));
        hstu::gemm<
            /*zero_init=*/false,
            /*wg_wait=*/1,
            /*SwapAB=*/dKV_swapAB,
            /*M_slice=*/0>(
            tiled_mma_dKV,
            tdKrdS_cur,
            tdKrQ(_, _, _, smem_pipe_read.index()),
            tdKrdK);
        pipeline_do.consumer_release(smem_pipe_read_do_cur); // release dO

        hstu::gemm<
            /*zero_init=*/true,
            /*wg_wait=*/0,
            /*SwapAB=*/dQ_swapAB,
            /*M_slice=*/1>(tiled_mma_dQ, tdQrdS_cur, tdQrK, tdQrdQ);
#pragma unroll
        for (int i = size(tdQrdQ_atomic) / 2; i < size(tdQrdQ_atomic); ++i) {
          atomicAdd(&tdQgdQaccum_atomic(i), tdQrdQ_atomic(i));
        }

        hstu::gemm<
            /*zero_init=*/false,
            /*wg_wait=*/-1,
            /*SwapAB=*/dKV_swapAB,
            /*M_slice=*/1>(
            tiled_mma_dKV,
            tdKrdS_cur,
            tdKrQ(_, _, _, smem_pipe_read.index()),
            tdKrdK);
      }

      warpgroup_wait<0>();
      pipeline_q.consumer_release(smem_pipe_read); // release Q
      ++smem_pipe_read;
      if constexpr (!Q_dO_same_stages) {
        ++smem_pipe_read_do;
      }
    };
    static constexpr int kBlockM = get<0>(TileShape_MNK{});
    static constexpr int kBlockN = get<1>(TileShape_MNK{});

    if constexpr (Cross) {
      auto mask_fn = [&](auto& tSrS, int m_block) {
        mask.template apply<
            true /*Seqlenq_mask*/,
            true /*Seqlenk_mask*/,
            Causal,
            false /*Local*/,
            false /*Contexual_mask*/,
            false /*Target_mask*/,
            Cross,
            true /*Softmax*/>(tSrS, m_block, n_block);
      };
      CUTLASS_PRAGMA_NO_UNROLL
      for (; m_block < m_block_max; ++m_block) {
        bwd_step(m_block, mask_fn);
      }
      if constexpr (Q_dO_same_stages) {
        smem_pipe_read_do = smem_pipe_read;
      }
      ++work_idx;
      return true;
    }
    if constexpr (Has_targets) {
      if (n_block * kBlockN >= seqlen_info.uihlen_q) {
        auto mask_fn = [&](auto& tSrS, int m_block) {
          mask.template apply<
              true /*Seqlenq_mask*/,
              true /*Seqlenk_mask*/,
              false /*Causal*/,
              false /*Local*/,
              false /*Contexual_mask*/,
              Has_targets /*Target_mask*/,
              false /*Cross*/,
              true /*Softmax*/>(tSrS, m_block, n_block);
        };
        CUTLASS_PRAGMA_NO_UNROLL
        for (; m_block < m_block_max; ++m_block) {
          bwd_step(m_block, mask_fn);
        }
        if constexpr (Q_dO_same_stages) {
          smem_pipe_read_do = smem_pipe_read;
        }
        ++work_idx;
        return true;
      } else if ((n_block + 1) * kBlockN >= seqlen_info.uihlen_q) {
        if constexpr ((Causal || Local) && SeparateMaskingIterations) {
          auto mask_fn = [&](auto& tSrS, int m_block) {
            mask.template apply<
                true /*Seqlenq_mask*/,
                true /*Seqlenk_mask*/,
                Causal,
                Local,
                Contexual_mask,
                Has_targets /*Target_mask*/,
                false /*Cross*/,
                true /*Softmax*/>(tSrS, m_block, n_block);
          };
          int const m_block_masking_max =
              ((n_block + 1) * kBlockN - 1) / kBlockM + 1;
          CUTLASS_PRAGMA_NO_UNROLL
          for (; m_block < std::min(m_block_max, m_block_masking_max);
               ++m_block) {
            bwd_step(m_block, mask_fn);
          }
        }

        auto mask_fn = [&](auto& tSrS, int m_block) {
          mask.template apply<
              true /*Seqlenq_mask*/,
              true /*Seqlenk_mask*/,
              Causal && !SeparateMaskingIterations,
              Local && !SeparateMaskingIterations,
              Contexual_mask,
              Has_targets /*Target_mask*/,
              false /*Cross*/,
              true /*Softmax*/>(tSrS, m_block, n_block);
        };
        if constexpr (SeparateMaskingIterations) {
          int const m_block_max_before_local_mask =
              !Local || !SeparateMaskingIterations
              ? m_block_max
              : std::min(
                    m_block_max,
                    (n_block * kBlockN + params.max_attn_len) / kBlockM);
          CUTLASS_PRAGMA_NO_UNROLL
          for (; m_block < m_block_max_before_local_mask; ++m_block) {
            bwd_step(m_block, mask_fn);
          }
        } else {
          int num_m_block = m_block_max - m_block_min;
          CUTLASS_PRAGMA_NO_UNROLL
          for (int i = 0; i < num_m_block + full_m_block_max -
                   full_m_block_min + contexual_m_block_max;
               ++i) {
            if (i < num_m_block) {
              m_block = m_block_min + i;
            } else if (i < num_m_block + contexual_m_block_max) {
              m_block = i - num_m_block;
            } else {
              m_block =
                  i - num_m_block - contexual_m_block_max + full_m_block_min;
            }
            bwd_step(m_block, mask_fn);
          }
        }

        if constexpr (Local && SeparateMaskingIterations) {
          auto mask_fn = [&](auto& tSrS, int m_block) {
            mask.template apply<
                true /*Seqlenq_mask*/,
                true /*Seqlenk_mask*/,
                false /*Causal_mask*/,
                Local,
                Contexual_mask,
                Has_targets /*Target_mask*/,
                false /*Cross*/,
                true /*Softmax*/>(tSrS, m_block, n_block);
          };
          CUTLASS_PRAGMA_NO_UNROLL
          for (; m_block < m_block_max; ++m_block) {
            bwd_step(m_block, mask_fn);
          }
        }
        if constexpr (Contexual_mask && SeparateMaskingIterations) {
          auto mask_fn = [&](auto& tSrS, int m_block) {
            mask.template apply<
                true /*Seqlenq_mask*/,
                true /*Seqlenk_mask*/,
                Causal /*Causal_mask*/,
                Local /*Local_mask*/,
                Contexual_mask,
                Has_targets,
                false /*Cross*/,
                true /*Softmax*/>(tSrS, m_block, n_block);
          };
          CUTLASS_PRAGMA_NO_UNROLL
          for (m_block = 0; m_block < contexual_m_block_max; ++m_block) {
            bwd_step(m_block, mask_fn);
          }
        }

        if constexpr (Local && SeparateMaskingIterations) {
          auto mask_fn = [&](auto& tSrS, int m_block) {
            mask.template apply<
                true /*Seqlenq_mask*/,
                true /*Seqlenk_mask*/,
                false /*Causal_mask*/,
                Local,
                Contexual_mask,
                Has_targets,
                false /*Cross*/,
                true /*Softmax*/>(tSrS, m_block, n_block);
          };
          CUTLASS_PRAGMA_NO_UNROLL
          for (m_block = full_m_block_min; m_block < full_m_block_max;
               ++m_block) {
            bwd_step(m_block, mask_fn);
          }
        }
        if constexpr (Q_dO_same_stages) {
          smem_pipe_read_do = smem_pipe_read;
        }
        ++work_idx;
        return true;
      }
    }
    // We have separate iterations with causal masking. Not necessary for hdim
    // 128 but for hdim 64 this helps quite a bit to not have to do causal
    // masking for most of the iterations.
    if constexpr ((Causal || Local) && SeparateMaskingIterations) {
      auto mask_fn = [&](auto& tSrS, int m_block) {
        mask.template apply<
            true /*Seqlenq_mask*/,
            true /*Seqlenk_mask*/,
            Causal,
            Local,
            Contexual_mask,
            false /*Target_mask*/,
            false /*Cross*/,
            true /*Softmax*/>(tSrS, m_block, n_block);
      };
      static constexpr int kBlockM = get<0>(TileShape_MNK{});
      int const m_block_masking_max =
          ((n_block + 1) * kBlockN - 1) / kBlockM + 1;
      CUTLASS_PRAGMA_NO_UNROLL
      for (; m_block < std::min(m_block_max, m_block_masking_max); ++m_block) {
        bwd_step(m_block, mask_fn);
      }
    }

    auto mask_fn = [&](auto& tSrS, int m_block) {
      mask.template apply<
          true /*Seqlenq_mask*/,
          true /*Seqlenk_mask*/,
          Causal && !SeparateMaskingIterations,
          Local && !SeparateMaskingIterations,
          Contexual_mask,
          false /*Target_mask*/,
          false /*Cross*/,
          true /*Softmax*/>(tSrS, m_block, n_block);
    };
    if constexpr (SeparateMaskingIterations) {
      int const m_block_max_before_local_mask =
          !Local || !SeparateMaskingIterations
          ? m_block_max
          : std::min(
                m_block_max,
                (n_block * kBlockN + params.max_attn_len) / kBlockM);
      CUTLASS_PRAGMA_NO_UNROLL
      for (; m_block < m_block_max_before_local_mask; ++m_block) {
        bwd_step(m_block, mask_fn);
      }
    } else {
      int num_m_block = m_block_max - m_block_min;
      CUTLASS_PRAGMA_NO_UNROLL
      for (int i = 0; i < num_m_block + full_m_block_max - full_m_block_min +
               contexual_m_block_max;
           ++i) {
        if (i < num_m_block) {
          m_block = m_block_min + i;
        } else if (i < num_m_block + contexual_m_block_max) {
          m_block = i - num_m_block;
        } else {
          m_block = i - num_m_block - contexual_m_block_max + full_m_block_min;
        }
        bwd_step(m_block, mask_fn);
      }
    }

    if constexpr (Local && SeparateMaskingIterations) {
      auto mask_fn = [&](auto& tSrS, int m_block) {
        mask.template apply<
            true /*Seqlenq_mask*/,
            true /*Seqlenk_mask*/,
            false /*Causal_mask*/,
            Local,
            Contexual_mask,
            false /*Target_mask*/,
            false /*Cross*/,
            true /*Softmax*/>(tSrS, m_block, n_block);
      };
      CUTLASS_PRAGMA_NO_UNROLL
      for (; m_block < m_block_max; ++m_block) {
        bwd_step(m_block, mask_fn);
      }
    }
    if constexpr (Contexual_mask && SeparateMaskingIterations) {
      auto mask_fn = [&](auto& tSrS, int m_block) {
        mask.template apply<
            true /*Seqlenq_mask*/,
            true /*Seqlenk_mask*/,
            Causal /*Causal_mask*/,
            Local /*Local_mask*/,
            Contexual_mask,
            false /*Target_mask*/,
            false /*Cross*/,
            true /*Softmax*/>(tSrS, m_block, n_block);
      };
      CUTLASS_PRAGMA_NO_UNROLL
      for (m_block = 0; m_block < contexual_m_block_max; ++m_block) {
        bwd_step(m_block, mask_fn);
      }
    }

    if constexpr (Local && SeparateMaskingIterations) {
      auto mask_fn = [&](auto& tSrS, int m_block) {
        mask.template apply<
            true /*Seqlenq_mask*/,
            true /*Seqlenk_mask*/,
            false /*Causal_mask*/,
            Local,
            Contexual_mask,
            false /*Target_mask*/,
            false /*Cross*/,
            true /*Softmax*/>(tSrS, m_block, n_block);
      };
      CUTLASS_PRAGMA_NO_UNROLL
      for (m_block = full_m_block_min; m_block < full_m_block_max; ++m_block) {
        bwd_step(m_block, mask_fn);
      }
    }

    // if (blockIdx.x == 0 && threadIdx.x == 128) { print_tensor(tdVrdV); }
    if constexpr (Q_dO_same_stages) {
      smem_pipe_read_do = smem_pipe_read;
    }
    ++work_idx;
    return true;
  }
};

} // namespace hstu
