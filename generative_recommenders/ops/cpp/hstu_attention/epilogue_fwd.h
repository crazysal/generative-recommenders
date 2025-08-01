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

#include <cutlass/cutlass.h>
#include <cutlass/fast_math.h> // For FastDivMod
#include "cute/tensor.hpp"

#include "cutlass/epilogue/collective/builders/sm90_common.inl"
#include "cutlass/gemm/collective/builders/sm90_common.inl"

#include "named_barrier.h"
#include "seqlen.h"
#include "utils.h"

namespace hstu {

using namespace cute;

template <
    class TileShape_MNK_,
    class ClusterShape_,
    class Element_,
    class ArchTag_,
    int NumEpilogueThreads_,
    bool Jagged,
    bool FP8PermuteCol = false>
struct CollectiveEpilogueFwd {
  using TileShape_MNK = TileShape_MNK_;
  using ClusterShape = ClusterShape_;
  using Element = Element_;
  using ArchTag = ArchTag_;
  static constexpr int NumEpilogueThreads = NumEpilogueThreads_;
  static constexpr bool Use_smem = sizeof(Element) <= 2;
  static constexpr bool Use_TMA_O =
      ArchTag::kMinComputeCapability >= 90 && !Jagged && Use_smem;

  static_assert(ArchTag::kMinComputeCapability >= 80);
  static_assert(
      ArchTag::kMinComputeCapability >= 90 ||
      CUTE_STATIC_V(size(ClusterShape{})) == 1);

  static constexpr int kBlockM = get<0>(TileShape_MNK{});
  static constexpr int kHeadDim = get<2>(TileShape_MNK{});

  using GmemTiledCopyOTMA = cute::SM90_TMA_STORE;

  // These are for storing the output tensor without TMA (e.g., for setting
  // output to zero)
  static constexpr int kGmemElemsPerStore =
      sizeof(cute::uint128_t) / sizeof(Element);
  static_assert(
      kHeadDim % kGmemElemsPerStore == 0,
      "Headdim must be a multiple of kGmemElemsPerStore");
  // We want each "row" to have 64 elements (128 bytes, i.e. 1 cache line). We
  // want each thread to have 4 elements in the M direction and 2 elements in
  // the K direction. In the case of PackGQA, this reduces the number of times
  // we need to call divmod.
  static constexpr int kBytePerRow = kHeadDim * sizeof(Element);
  static constexpr int kBlockKGmem =
      (kBytePerRow % 128 == 0 ? 128 : (kBytePerRow % 64 == 0 ? 64 : 32)) /
      sizeof(Element);
  // static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? 128 : (kHeadDim %
  // 64 == 0 ? 64 : 32); static constexpr int kGmemThreadsPerRow =
  // cutlass::gcd(kHeadDim / kGmemElemsPerStore, NumEpilogueThreads);
  static constexpr int kGmemThreadsPerRow = kBlockKGmem / kGmemElemsPerStore;
  // If PackGQA, we split the work of compute O_ptr among threads in the same
  // row, so we need this to within a warp
  static_assert(cutlass::NumThreadsPerWarp % kGmemThreadsPerRow == 0);
  static_assert(
      NumEpilogueThreads % kGmemThreadsPerRow == 0,
      "NumEpilogueThreads must be a multiple of kGmemThreadsPerRow");
  using GmemLayoutAtom = Layout<
      Shape<
          Int<NumEpilogueThreads / kGmemThreadsPerRow>,
          Int<kGmemThreadsPerRow>>,
      Stride<Int<kGmemThreadsPerRow>, _1>>;
  static_assert(
      kBlockM % CUTE_STATIC_V(shape<0>(GmemLayoutAtom{})) == 0,
      "kBlockM must be a multiple of NumEpilogueThreads / kGmemThreadsPerRow");
  using GmemTiledCopyO = decltype(make_tiled_copy(
      Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
      GmemLayoutAtom{},
      Layout<Shape<_1, Int<kGmemElemsPerStore>>>{})); // Val layout, 8 or 16
                                                      // vals per store

  using SmemLayoutAtomOTMA =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GMMA::Major::K,
               Element,
               decltype(cute::get<0>(TileShape_MNK{})),
               decltype(cute::get<2>(TileShape_MNK{}))>());
  using SmemLayoutOTMA = decltype(tile_to_shape(
      SmemLayoutAtomOTMA{},
      select<0, 2>(TileShape_MNK{})));
  static constexpr int kSwizzle = kBlockKGmem == 128
      ? 4
      : (kBlockKGmem == 64 ? 3 : (kBlockKGmem == 32 ? 2 : 1));
  static constexpr int kSwizzleBase =
      sizeof(Element) == 4 ? 2 : (sizeof(Element) == 2 ? 3 : 4);
  using SmemLayoutAtomO = decltype(composition(
      Swizzle<kSwizzle, kSwizzleBase, kSwizzleBase>{},
      Layout<Shape<_8, Int<kBlockKGmem>>, Stride<Int<kBlockKGmem>, _1>>{}));
  using SmemLayoutOSTS =
      decltype(tile_to_shape(SmemLayoutAtomO{}, select<0, 2>(TileShape_MNK{})));
  using SmemLayoutO = std::conditional_t<
      ArchTag::kMinComputeCapability >= 90,
      SmemLayoutOTMA,
      SmemLayoutOSTS>;

  using ShapeO =
      cute::Shape<int32_t, int32_t, int32_t, int32_t, int32_t>; // (seqlen_q, d,
                                                                // head, batch,
                                                                // num_splits)
  using StrideO = cute::Stride<int64_t, _1, int64_t, int64_t, int64_t>;
  // ((qhead_per_khead, seqlen_q), d, nheads, batch, num_splits)
  using ShapeOPacked = ShapeO;
  using StrideOPacked = StrideO;
  // ((qhead_per_khead, seqlen_q), nheads, batch, num_splits)
  using StrideLSE =
      cute::Stride<_1, int64_t, int64_t, int64_t>; // (seqlen_q, head, batch,
  // num_splits)
  using ShapeLSEPacked = cute::Shape<int32_t, int32_t, int32_t, int32_t>;
  using StrideLSEPacked = StrideLSE;
  using EpilogueTile_MN = decltype(select<0, 1>(TileShape_MNK{}));
  using CopyOpR2S = std::conditional_t<
      ArchTag::kMinComputeCapability >= 90,
      // cute::SM90_U32x4_STSM_N if Element size is 2 bytes (fp16, bf16)
      decltype(cutlass::epilogue::collective::detail::
                   sm90_get_smem_store_op_for_accumulator<
                       StrideO,
                       Element,
                       EpilogueTile_MN>()),
      AutoVectorizingCopyWithAssumedAlignment<128>>;
  using SmemCopyAtomO = Copy_Atom<CopyOpR2S, Element>;

  // static constexpr size_t SmemAlignmentO =
  // cutlass::detail::alignment_for_swizzle(SmemLayoutO{});
  // static_assert(SmemAlignmentO >= 128, "Require at least 128B alignment");
  // struct TensorStorage : cute::aligned_struct<SmemAlignmentO> {
  //     cute::array_aligned<Element, Use_smem ? cute::cosize_v<SmemLayoutO> :
  //     0, SmemAlignmentO> smem_o;
  // };
  struct TensorStorage : cute::aligned_struct<128> {
    cute::array_aligned<Element, Use_smem ? cute::cosize_v<SmemLayoutO> : 0>
        smem_o;
  };

  using TMA_O = std::conditional_t<
      Use_TMA_O,
      decltype(make_tma_copy(
          GmemTiledCopyOTMA{},
          make_tensor(
              make_gmem_ptr(static_cast<Element*>(nullptr)),
              ShapeO{},
              StrideO{}),
          SmemLayoutOTMA{},
          select<0, 2>(TileShape_MNK{}),
          _1{})), // no mcast for O
      std::nullptr_t>;

  // Host side kernel arguments
  struct Arguments {
    Element* ptr_O;
    ShapeO const shape_O;
    StrideO const stride_O;
    int32_t const nheads;
    int32_t const num_softmax_heads;
    StrideLSE const stride_lse;
    float* ptr_lse = nullptr;
    int const* seq_offsets = nullptr;
  };

  // Device side kernel params
  struct Params {
    Element* ptr_O;
    ShapeO const shape_O;
    StrideO const stride_O;
    ShapeOPacked const shape_O_packed;
    StrideOPacked const stride_O_packed;
    float* ptr_lse;
    StrideLSE const stride_lse;
    ShapeLSEPacked const shape_lse_packed;
    StrideLSEPacked const stride_lse_packed;
    TMA_O tma_store_O;
    int const* seq_offsets = nullptr;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    Tensor mO =
        make_tensor(make_gmem_ptr(args.ptr_O), args.shape_O, args.stride_O);
    TMA_O tma_store_O = [&] {
      if constexpr (Use_TMA_O) {
        return make_tma_copy(
            GmemTiledCopyOTMA{},
            mO,
            SmemLayoutO{},
            select<0, 2>(TileShape_MNK{}),
            _1{}); // no mcast
      } else {
        return nullptr;
      }
    }();
    // If PackGQA, reshape O to be ((qhead_per_khead, seqlen_q), head_size,
    // nhead_k, batch_size, num_splits)
    int const qhead_per_khead = 1;
    auto const shape_O_packed = cute::conditional_return<true>(
        args.shape_O,
        make_shape(
            make_shape(qhead_per_khead, get<0>(args.shape_O)),
            get<1>(args.shape_O),
            args.nheads,
            get<3>(args.shape_O),
            get<4>(args.shape_O)));
    auto const stride_O_packed = cute::conditional_return<true>(
        args.stride_O,
        make_stride(
            make_stride(get<2>(args.stride_O), get<0>(args.stride_O)),
            get<1>(args.stride_O),
            get<2>(args.stride_O) * qhead_per_khead,
            get<3>(args.stride_O),
            get<4>(args.stride_O)));
    auto const shape_lse_packed = select<0, 2, 3, 4>(args.shape_O);
    auto const stride_lse_packed = args.stride_lse;
    return {
        args.ptr_O,
        args.shape_O,
        args.stride_O,
        shape_O_packed,
        stride_O_packed,
        args.ptr_lse,
        args.stride_lse,
        shape_lse_packed,
        stride_lse_packed,
        tma_store_O,
        args.seq_offsets};
  }

  /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best
  /// performance
  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& params) {
    if constexpr (Use_TMA_O) {
      cute::prefetch_tma_descriptor(params.tma_store_O.get_tma_descriptor());
    }
  }

  template <typename SharedStorage, typename FrgTensorO, typename TiledMma>
  CUTLASS_DEVICE void store(
      Params const& params,
      FrgTensorO const& tOrO,
      SharedStorage& shared_storage,
      TiledMma tiled_mma,
      int thread_idx,
      cute::tuple<int32_t, int32_t, int32_t, int32_t> const& block_coord) {
    auto [m_block, bidh, bidb, split_idx] = block_coord;
    Tensor sO = make_tensor(
        make_smem_ptr(shared_storage.tensors.epilogue.smem_o.data()),
        SmemLayoutO{});
    // Tensor sO_pi = cute::as_position_independent_swizzle_tensor(sO);

    Tensor tOrO_out = make_tensor_like<Element>(tOrO);
    hstu::convert_type_out(tOrO, tOrO_out);
    if constexpr (
        FP8PermuteCol && (sizeof(Element) == 2 || sizeof(Element) == 4)) {
      hstu::permute_output_fp8_Vcolmajor(tOrO_out);
    }

    // Make sure all WGs have finished reading V
    // Technically we don't need this if we're not using smem, but the mainloop
    // makes the assumption that all epilogue threads sync at least once during
    // the epilogue (so that we can start loading Q with cp.async if we need).
    hstu::named_barrier_sync(
        NumEpilogueThreads,
        cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);

    // Step 1: Write O from rmem -> smem
    if constexpr (Use_smem) {
      auto smem_tiled_copy_O = make_tiled_copy_C(SmemCopyAtomO{}, tiled_mma);
      auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(thread_idx);
      Tensor taccOrO =
          smem_thr_copy_O.retile_S(tOrO_out); // ((Atom,AtomNum), MMA_M, MMA_N)
      Tensor taccOsO =
          smem_thr_copy_O.partition_D(sO); // ((Atom,AtomNum),PIPE_M,PIPE_N)
      // Tensor taccOsO = smem_thr_copy_O.partition_D(sO_pi);     //
      // ((Atom,AtomNum),PIPE_M,PIPE_N)
      cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);
      if constexpr (Use_TMA_O) {
        cutlass::arch::fence_view_async_shared(); // ensure smem writes are
                                                  // visible to TMA
        cutlass::arch::NamedBarrier::arrive(
            NumEpilogueThreads + cutlass::NumThreadsPerWarp,
            cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
      } else {
        hstu::named_barrier_sync(
            NumEpilogueThreads,
            cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
      }
    } else {
      if constexpr (ArchTag::kMinComputeCapability >= 90) {
#pragma unroll
        for (uint32_t cta_id = 0; cta_id < size(ClusterShape{}); ++cta_id) {
          shared_storage.pipelines.barrier_O.arrive(cta_id);
        }
      }
    }

    hstu::SeqlenInfo<Jagged, kBlockM> seqlen_info{
        bidb, size<0>(params.shape_O), params.seq_offsets};
    int offset_o = seqlen_info.offset;
    int seqlen_o = seqlen_info.seqlen;

    // Step 2: Write LSE from rmem -> gmem
    auto thread_mma = tiled_mma.get_thread_slice(thread_idx);
    // (MMA,MMA_M,MMA_K)
    Tensor taccOcO = thread_mma.partition_C(
        cute::make_identity_tensor(select<0, 2>(TileShape_MNK{})));
    static_assert(decltype(size<0, 0>(taccOcO))::value == 2);
    static_assert(decltype(size<0, 1>(taccOcO))::value == 2);
    Tensor taccOcO_rowcol = make_tensor(
        taccOcO.data(), hstu::convert_layout_acc_rowcol(taccOcO.layout()));
    Tensor taccOcO_row = taccOcO_rowcol(_, _0{});
    // Step 3: Write O from smem -> gmem
    if constexpr (Use_TMA_O) {
      Tensor mO = params.tma_store_O.get_tma_tensor(params.shape_O)(
          _, _, bidh, bidb, split_idx);
      Tensor gO = local_tile(
          mO,
          select<0, 2>(TileShape_MNK{}),
          make_coord(m_block, _0{})); // (M, K)
      auto block_tma_O = params.tma_store_O.get_slice(_0{});
      Tensor tOgO = block_tma_O.partition_D(gO); // (TMA, TMA_M, TMA_K)
      Tensor tOsO = block_tma_O.partition_S(sO); // (TMA, TMA_M, TMA_K)
      int warp_idx_sync =
          __shfl_sync(0xffffffff, thread_idx / cutlass::NumThreadsPerWarp, 0);
      if (warp_idx_sync ==
          NumEpilogueThreads / cutlass::NumThreadsPerWarp - 1) {
        cutlass::arch::NamedBarrier::sync(
            NumEpilogueThreads + cutlass::NumThreadsPerWarp,
            cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
        if (cute::elect_one_sync()) {
          cute::copy(params.tma_store_O, tOsO, tOgO);
          tma_store_arrive();
          tma_store_wait<0>();
#pragma unroll
          for (uint32_t cta_id = 0; cta_id < size(ClusterShape{}); ++cta_id) {
            shared_storage.pipelines.barrier_O.arrive(cta_id);
          }
        }
      }
    } else { // Don't use TMA in Jagged case since we don't want to overwrite
             // the output of another sequence
      Tensor mO = make_tensor(
          make_gmem_ptr(params.ptr_O + offset_o * get<0>(params.stride_O)),
          params.shape_O_packed,
          params.stride_O_packed)(_, _, bidh, !Jagged ? bidb : 0, split_idx);
      Tensor gO = local_tile(
          mO,
          select<0, 2>(TileShape_MNK{}),
          make_coord(m_block, _0{})); // (M, K)
      // if (thread_idx == 0) { printf("Before O write, m_block: %d, bidh: %d,
      // bidb: %d, split_idx: %d, offset_o: %d, seqlen_o: %d, mO_addr = %p, addr
      // diff = %d\n", m_block, bidh, bidb, split_idx, offset_o, seqlen_o,
      // mO.data(), reinterpret_cast<int>(&mO(0)) -
      // reinterpret_cast<int>(params.ptr_O)); }
      if constexpr (Use_smem) {
        GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(thread_idx);
        Tensor tOsO =
            gmem_thr_copy_O.partition_S(sO); // ((Atom,AtomNum),ATOM_M,ATOM_N)
        // Tensor tOsO = gmem_thr_copy_O.partition_S(sO_pi);        //
        // ((Atom,AtomNum),ATOM_M,ATOM_N)
        Tensor tOrO = make_fragment_like(tOsO);
        cute::copy(gmem_tiled_copy_O, tOsO, tOrO);
        if constexpr (ArchTag::kMinComputeCapability >= 90) {
          cutlass::arch::fence_view_async_shared(); // ensure smem reads are
                                                    // done before next TMA to
                                                    // smem_v
#pragma unroll
          for (uint32_t cta_id = 0; cta_id < size(ClusterShape{}); ++cta_id) {
            shared_storage.pipelines.barrier_O.arrive(cta_id);
          }
        }
        // (BLK_M,BLK_K) -> (blk_m,blk_k)
        Tensor tOcO = gmem_thr_copy_O.partition_D(
            cute::make_identity_tensor(select<0, 2>(TileShape_MNK{})));
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOsO)));
#pragma unroll
        for (int k = 0; k < size(tOpO); ++k) {
          tOpO(k) = get<1>(tOcO(_0{}, _0{}, k)) < get<1>(params.shape_O);
        }
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
        // Clear_OOB_K must be false since we don't want to write zeros to
        // gmem
        hstu::copy<
            /*Is_even_MN=*/false,
            /*Is_even_K=*/false,
            /*Clear_OOB_MN=*/false,
            /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_O,
            tOrO,
            tOgO,
            tOcO,
            tOpO,
            seqlen_o - m_block * kBlockM);
      } else {
        // We already arrived on barrier_O earlier
        static constexpr int kGmemElemsPerStoreDirect = 2;
        cute::Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>
            gmem_copy_direct;
        // Reshape acc from ((2, 2, V), MMA_M, MMA_N) to (nrow=(2, MMA_M),
        // ncol=(2, V, MMA_N))
        Tensor tOrO_rowcol = make_tensor(
            tOrO_out.data(), hstu::convert_layout_acc_rowcol(tOrO.layout()));
        Tensor tOrO_copy = cute::tiled_divide(
            tOrO_rowcol, Shape<_1, Int<kGmemElemsPerStoreDirect>>{});
        Tensor tOgO = thread_mma.partition_C(gO);
        Tensor tOgO_rowcol = make_tensor(
            tOgO.data(), hstu::convert_layout_acc_rowcol(tOgO.layout()));
        Tensor tOgO_copy = cute::tiled_divide(
            tOgO_rowcol, Shape<_1, Int<kGmemElemsPerStoreDirect>>{});
        Tensor taccOcO_col = taccOcO_rowcol(_0{}, _);
#pragma unroll
        for (int m = 0; m < size(taccOcO_row); ++m) {
          if (get<0>(taccOcO_row(m)) < seqlen_o - m_block * kBlockM) {
#pragma unroll
            for (int k = 0; k < size(taccOcO_col) / kGmemElemsPerStoreDirect;
                 ++k) {
              if (get<1>(taccOcO_col(k * kGmemElemsPerStoreDirect)) <
                  get<1>(params.shape_O)) {
                cute::copy(
                    gmem_copy_direct, tOrO_copy(_, m, k), tOgO_copy(_, m, k));
              }
            }
          }
        }
      }
    }
  }

  template <typename FrgTensorLSE, typename TiledMma>
  CUTLASS_DEVICE void store_softmax(
      Params const& params,
      FrgTensorLSE const& lse,
      TiledMma tiled_mma,
      int thread_idx,
      cute::tuple<int32_t, int32_t, int32_t, int32_t> const& block_coord) {
    auto [m_block, bidh, bidb, split_idx] = block_coord;
    hstu::SeqlenInfo<Jagged, kBlockM> seqlen_info{
        bidb, size<0>(params.shape_O), params.seq_offsets};
    int offset_o = seqlen_info.offset;
    int seqlen_o = seqlen_info.seqlen;
    // Step 2: Write LSE from rmem -> gmem
    auto thread_mma = tiled_mma.get_thread_slice(thread_idx);
    // (MMA,MMA_M,MMA_K)
    Tensor taccOcO = thread_mma.partition_C(
        cute::make_identity_tensor(select<0, 2>(TileShape_MNK{})));
    static_assert(decltype(size<0, 0>(taccOcO))::value == 2);
    static_assert(decltype(size<0, 1>(taccOcO))::value == 2);
    Tensor taccOcO_rowcol = make_tensor(
        taccOcO.data(), hstu::convert_layout_acc_rowcol(taccOcO.layout()));
    Tensor taccOcO_row = taccOcO_rowcol(_, _0{});
    CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row)); // MMA_M
    Tensor mLSE = make_tensor(
        make_gmem_ptr(params.ptr_lse + offset_o * get<0>(params.stride_lse)),
        params.shape_lse_packed,
        params.stride_lse_packed)(_, bidh, !Jagged ? bidb : 0, 0);
#pragma unroll
    for (int mi = 0; mi < size(lse); ++mi) {
      int const row = m_block * kBlockM + get<0>(taccOcO_row(mi));
      if (get<1>(taccOcO_row(_0{})) == 0 && row < seqlen_o) {
        mLSE(row) = lse(mi);
      }
    }
  }

  CUTLASS_DEVICE void store_tail() {
    // Don't need to do tma_store_wait<0>() here since we already did in @store
  }

  // Write 0 to output and -inf to LSE
  template <bool Clear_O = true>
  CUTLASS_DEVICE void store_zero(
      Params const& params,
      int thread_idx,
      cute::tuple<int32_t, int32_t, int32_t, int32_t> const& block_coord) {
    static constexpr int kBlockM = get<0>(TileShape_MNK{});
    auto [m_block, bidh, bidb, split_idx] = block_coord;
    hstu::SeqlenInfo<Jagged, kBlockM> seqlen_info{
        bidb, size<0>(params.shape_O), params.seq_offsets};
    int offset_o = seqlen_info.offset;
    int seqlen_o = seqlen_info.seqlen;
    Tensor mO = make_tensor(
        make_gmem_ptr(params.ptr_O + offset_o * get<0>(params.stride_O)),
        params.shape_O_packed,
        params.stride_O_packed)(_, _, bidh, !Jagged ? bidb : 0, split_idx);

    static_assert(kBlockM <= NumEpilogueThreads);
    if constexpr (!Clear_O) {
      return;
    }

    GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(thread_idx);
    Tensor tOcO = gmem_thr_copy_O.partition_D(
        cute::make_identity_tensor(select<0, 2>(TileShape_MNK{})));
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOcO)));
#pragma unroll
    for (int k = 0; k < size(tOpO); ++k) {
      tOpO(k) = get<1>(tOcO(_0{}, _0{}, k)) < get<1>(params.shape_O);
    }
    Tensor gO = local_tile(
        mO, select<0, 2>(TileShape_MNK{}), make_coord(m_block, _0{})); // (M, K)
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
    Tensor tOrO = make_fragment_like(tOgO);
    cute::clear(tOrO);
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    hstu::copy<
        /*Is_even_MN=*/false,
        /*Is_even_K=*/false,
        /*Clear_OOB_MN=*/false,
        /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_O,
        tOrO,
        tOgO,
        tOcO,
        tOpO,
        seqlen_o - m_block * kBlockM);
  }
};

} // namespace hstu
