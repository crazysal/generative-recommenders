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

#include <cutlass/barrier.h>
#include <cutlass/cutlass.h>

#include "cutlass/gemm/collective/builders/sm90_common.inl"

#include "copy_sm90_bulk_reduce.h"
#include "named_barrier.h"
#include "seqlen.h"
#include "utils.h"

namespace hstu {

using namespace cute;

template <
    class TileShape_MNK_,
    class Element_,
    class ArchTag_,
    int NumEpilogueThreads_,
    bool Jagged,
    bool dKV_swapAB_,
    int AtomLayoutKdKV = 1>
struct CollectiveEpilogueBwd {
  using TileShape_MNK = TileShape_MNK_;
  using Element = Element_;
  using ArchTag = ArchTag_;
  static constexpr int NumEpilogueThreads = NumEpilogueThreads_;
  static constexpr bool dKV_swapAB = dKV_swapAB_;
  static constexpr bool Use_TMA =
      !Jagged && ArchTag::kMinComputeCapability >= 90;

  static_assert(ArchTag::kMinComputeCapability >= 80);

  using GmemTiledCopydKVTMA = cute::SM90_TMA_STORE;

  // These are for storing the output tensor without TMA (e.g., for setting
  // output to zero)
  static constexpr int kGmemElemsPerLoad =
      sizeof(cute::uint128_t) / sizeof(Element);
  static_assert(
      get<2>(TileShape_MNK{}) % kGmemElemsPerLoad == 0,
      "Headdim must be a multiple of kGmemElemsPerLoad");
  static constexpr int kHeadDim = get<2>(TileShape_MNK{});
  static constexpr int kGmemThreadsPerRow =
      cutlass::gcd(kHeadDim / kGmemElemsPerLoad, NumEpilogueThreads);
  static_assert(
      NumEpilogueThreads % kGmemThreadsPerRow == 0,
      "NumEpilogueThreads must be a multiple of kGmemThreadsPerRow");
  using GmemLayoutAtom = Layout<
      Shape<
          Int<NumEpilogueThreads / kGmemThreadsPerRow>,
          Int<kGmemThreadsPerRow>>,
      Stride<Int<kGmemThreadsPerRow>, _1>>;
  using GmemTiledCopydKV = decltype(make_tiled_copy(
      Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
      GmemLayoutAtom{},
      Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{})); // Val layout, 8 or 16 vals
                                                     // per store

  using SmemLayoutAtomdKVTMA =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GMMA::Major::K,
               Element,
               // TODO: do we have to change this if dKV_swapAB is true?
               decltype(cute::get<1>(TileShape_MNK{})),
               Int<CUTE_STATIC_V(cute::get<2>(TileShape_MNK{})) /
                   AtomLayoutKdKV>>());
  using SmemLayoutdKVTMA = decltype(tile_to_shape(
      SmemLayoutAtomdKVTMA{},
      select<1, 2>(TileShape_MNK{})));
  using SmemLayoutdKVtTMA = decltype(cute::composition(
      SmemLayoutdKVTMA{},
      make_layout(
          make_shape(get<2>(TileShape_MNK{}), get<1>(TileShape_MNK{})),
          make_stride(decltype(get<1>(TileShape_MNK{})){}, _1{}))));

  // If we don't use TMA
  static constexpr int kBlockKSmem =
      kHeadDim % 64 == 0 ? 64 : (kHeadDim % 32 == 0 ? 32 : 16);
  static constexpr int kSwizzle =
      kBlockKSmem == 64 ? 3 : (kBlockKSmem == 32 ? 2 : 1);
  using SmemLayoutAtomdKVSTG = decltype(composition(
      Swizzle<kSwizzle, 3, 3>{},
      Layout<Shape<Int<8>, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>{}));

  using SmemLayoutAtomdKV =
      std::conditional_t<Use_TMA, SmemLayoutAtomdKVTMA, SmemLayoutAtomdKVSTG>;
  using SmemLayoutdKV = decltype(tile_to_shape(
      SmemLayoutAtomdKV{},
      select<1, 2>(TileShape_MNK{})));
  using SmemLayoutdKVt = decltype(cute::composition(
      SmemLayoutdKV{},
      make_layout(
          make_shape(get<2>(TileShape_MNK{}), get<1>(TileShape_MNK{})),
          make_stride(decltype(get<1>(TileShape_MNK{})){}, _1{}))));

  using SmemCopyAtomdKV = Copy_Atom<
      std::conditional_t<
          ArchTag::kMinComputeCapability >= 90,
          std::conditional_t<
              !dKV_swapAB,
              cute::SM90_U32x4_STSM_N,
              cute::SM90_U16x8_STSM_T>,
          AutoVectorizingCopyWithAssumedAlignment<128>>,
      Element>;

  static constexpr size_t SmemAlignmentdKV =
      ArchTag::kMinComputeCapability >= 90
      ? cutlass::detail::alignment_for_swizzle(SmemLayoutdKV{})
      : 128;
  static_assert(SmemAlignmentdKV >= 128, "Require at least 128B alignment");

  struct TensorStorage : cute::aligned_struct<SmemAlignmentdKV> {
    cute::
        array_aligned<Element, cute::cosize_v<SmemLayoutdKV>, SmemAlignmentdKV>
            smem_dk;
    cute::
        array_aligned<Element, cute::cosize_v<SmemLayoutdKV>, SmemAlignmentdKV>
            smem_dv;
  };

  using ShapedKV =
      cute::Shape<int32_t, int32_t, int32_t, int32_t>; // (seqlen_k, d, head,
                                                       // batch)
  using StridedKV = cute::Stride<int64_t, _1, int64_t, int64_t>;

  using TMA_dKV = std::conditional_t<
      Use_TMA,
      decltype(make_tma_copy(
          GmemTiledCopydKVTMA{},
          make_tensor(
              make_gmem_ptr(static_cast<Element*>(nullptr)),
              ShapedKV{},
              StridedKV{}),
          SmemLayoutdKVTMA{},
          select<1, 2>(TileShape_MNK{}),
          _1{})), // no mcast for dKV
      std::nullptr_t>;

  // Host side kernel arguments
  struct Arguments {
    Element* ptr_dK;
    ShapedKV const shape_dK;
    StridedKV const stride_dK;
    Element* ptr_dV;
    StridedKV const stride_dV;
    int const num_heads_q;
    int const* seq_offsets;
  };

  // Device side kernel params
  struct Params {
    Element* ptr_dK;
    ShapedKV const shape_dK;
    StridedKV const stride_dK;
    Element* ptr_dV;
    StridedKV const stride_dV;
    TMA_dKV tma_store_dK, tma_store_dV;
    int const* seq_offsets = nullptr;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    Tensor mdK =
        make_tensor(make_gmem_ptr(args.ptr_dK), args.shape_dK, args.stride_dK);
    Tensor mdV =
        make_tensor(make_gmem_ptr(args.ptr_dV), args.shape_dK, args.stride_dV);
    TMA_dKV tma_store_dK = [&] {
      if constexpr (Use_TMA) {
        return make_tma_copy(
            GmemTiledCopydKVTMA{},
            mdK,
            SmemLayoutdKVTMA{},
            select<1, 2>(TileShape_MNK{}),
            _1{}); // no mcast for dKV
      } else {
        return nullptr;
      }
    }();
    TMA_dKV tma_store_dV = [&] {
      if constexpr (Use_TMA) {
        return make_tma_copy(
            GmemTiledCopydKVTMA{},
            mdV,
            SmemLayoutdKVTMA{},
            select<1, 2>(TileShape_MNK{}),
            _1{}); // no mcast for dKV
      } else {
        return nullptr;
      }
    }();
    return {
        args.ptr_dK,
        args.shape_dK,
        args.stride_dK,
        args.ptr_dV,
        args.stride_dV,
        tma_store_dK,
        tma_store_dV,
        args.seq_offsets};
  }

  /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best
  /// performance
  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& params) {
    if constexpr (Use_TMA) {
      cute::prefetch_tma_descriptor(params.tma_store_dK.get_tma_descriptor());
      cute::prefetch_tma_descriptor(params.tma_store_dV.get_tma_descriptor());
    }
  }

  template <typename SharedStorage, typename FrgTensorO, typename TiledMma>
  CUTLASS_DEVICE void store(
      Params const& params,
      FrgTensorO const& tdKrdK,
      FrgTensorO const& tdVrdV,
      SharedStorage& shared_storage,
      TiledMma tiled_mma,
      int thread_idx,
      cute::tuple<int32_t, int32_t, int32_t> const& block_coord) {
    auto [n_block, bidh, bidb] = block_coord;
    Tensor sdK = cute::as_position_independent_swizzle_tensor(make_tensor(
        make_smem_ptr(shared_storage.tensors.epilogue.smem_dk.data()),
        SmemLayoutdKV{}));
    Tensor sdV = cute::as_position_independent_swizzle_tensor(make_tensor(
        make_smem_ptr(shared_storage.tensors.epilogue.smem_dv.data()),
        SmemLayoutdKV{}));
    Tensor sdKt = cute::as_position_independent_swizzle_tensor(make_tensor(
        make_smem_ptr(shared_storage.tensors.epilogue.smem_dk.data()),
        SmemLayoutdKVt{}));
    Tensor sdVt = cute::as_position_independent_swizzle_tensor(make_tensor(
        make_smem_ptr(shared_storage.tensors.epilogue.smem_dv.data()),
        SmemLayoutdKVt{}));
    auto smem_tiled_copy_dKV = make_tiled_copy_C(SmemCopyAtomdKV{}, tiled_mma);
    auto smem_thr_copy_dKV = smem_tiled_copy_dKV.get_thread_slice(thread_idx);

    Tensor tdVrdV_out = make_tensor_like<Element>(tdVrdV);
    hstu::convert_type_out(tdVrdV, tdVrdV_out);
    Tensor tdKrdK_out = make_tensor_like<Element>(tdKrdK);
    hstu::convert_type_out(tdKrdK, tdKrdK_out);
    Tensor taccdKrdK = smem_thr_copy_dKV.retile_S(
        tdKrdK_out); // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor taccdVrdV = smem_thr_copy_dKV.retile_S(
        tdVrdV_out); // ((Atom,AtomNum), MMA_M, MMA_N)
    // if (blockIdx.x == 0 && threadIdx.x == 128) { print(smem_thr_copy_dKV);
    // print(sdK); printf("\n"); print(sdKt); printf("\n"); }
    Tensor taccdKsdK =
        smem_thr_copy_dKV.partition_D(cute::conditional_return<!dKV_swapAB>(
            sdK, sdKt)); // ((Atom,AtomNum),PIPE_M,PIPE_N)
    Tensor taccdVsdV =
        smem_thr_copy_dKV.partition_D(cute::conditional_return<!dKV_swapAB>(
            sdV, sdVt)); // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // Make sure all WGs have finished reading K and V
    hstu::named_barrier_sync(
        NumEpilogueThreads,
        cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
    cute::copy(smem_tiled_copy_dKV, taccdVrdV, taccdVsdV);
    cute::copy(smem_tiled_copy_dKV, taccdKrdK, taccdKsdK);
    if constexpr (Use_TMA) {
      cutlass::arch::fence_view_async_shared(); // ensure smem writes are
                                                // visible to TMA
      cutlass::arch::NamedBarrier::arrive(
          NumEpilogueThreads + cutlass::NumThreadsPerWarp,
          cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);

      Tensor mdK = params.tma_store_dK.get_tma_tensor(params.shape_dK);
      Tensor mdV = params.tma_store_dV.get_tma_tensor(params.shape_dK);
      Tensor gdK = local_tile(
          mdK(_, _, bidh, bidb),
          select<1, 2>(TileShape_MNK{}),
          make_coord(n_block, _0{})); // (M, K)
      Tensor gdV = local_tile(
          mdV(_, _, bidh, bidb),
          select<1, 2>(TileShape_MNK{}),
          make_coord(n_block, _0{})); // (M, K)
      auto block_tma_dK = params.tma_store_dK.get_slice(_0{});
      auto block_tma_dV = params.tma_store_dV.get_slice(_0{});
      Tensor tdKgdK = block_tma_dK.partition_D(gdK); // (TMA, TMA_M, TMA_K)
      Tensor tdKsdK = block_tma_dK.partition_S(sdK); // (TMA, TMA_M, TMA_K)
      Tensor tdVgdV = block_tma_dV.partition_D(gdV); // (TMA, TMA_M, TMA_K)
      Tensor tdVsdV = block_tma_dV.partition_S(sdV); // (TMA, TMA_M, TMA_K)
      int warp_idx_sync =
          __shfl_sync(0xffffffff, thread_idx / cutlass::NumThreadsPerWarp, 0);
      if (warp_idx_sync ==
          NumEpilogueThreads / cutlass::NumThreadsPerWarp - 1) {
        cutlass::arch::NamedBarrier::sync(
            NumEpilogueThreads + cutlass::NumThreadsPerWarp,
            cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
        if (cute::elect_one_sync()) {
          cute::copy(params.tma_store_dV, tdVsdV, tdVgdV);
          cute::copy(params.tma_store_dK, tdKsdK, tdKgdK);
          tma_store_arrive();
        }
      }
      tma_store_wait<0>();
      // // Tell warp 0 that smem_k and smem_v are ready
      // cutlass::arch::NamedBarrier::arrive(NumEpilogueThreads +
      // cutlass::NumThreadsPerWarp,
      // static_cast<uint32_t>(BwdNamedBarriers::KVEmpty) /*id*/);

    } else {
      hstu::named_barrier_sync(
          NumEpilogueThreads,
          cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
      static constexpr int kBlockN = get<1>(TileShape_MNK{});
      hstu::SeqlenInfo<Jagged, kBlockN> seqlen_info{
          bidb, size<0>(params.shape_dK), params.seq_offsets};
      Tensor mdK = make_tensor(
          make_gmem_ptr(params.ptr_dK), params.shape_dK, params.stride_dK)(
          _, _, bidh, !Jagged ? bidb : 0);
      Tensor gdK = local_tile(
          cute::domain_offset(make_coord(seqlen_info.offset, _0{}), mdK),
          select<1, 2>(TileShape_MNK{}),
          make_coord(n_block, _0{})); // (M, K)
      Tensor mdV = make_tensor(
          make_gmem_ptr(params.ptr_dV), params.shape_dK, params.stride_dV)(
          _, _, bidh, !Jagged ? bidb : 0);
      Tensor gdV = local_tile(
          cute::domain_offset(make_coord(seqlen_info.offset, _0{}), mdV),
          select<1, 2>(TileShape_MNK{}),
          make_coord(n_block, _0{})); // (M, K)

      GmemTiledCopydKV gmem_tiled_copy_dKV;
      auto gmem_thr_copy_dKV = gmem_tiled_copy_dKV.get_thread_slice(thread_idx);
      Tensor tdKVgdV = gmem_thr_copy_dKV.partition_D(gdV);
      Tensor tdKVsdV =
          gmem_thr_copy_dKV.partition_S(sdV); // (TMA, TMA_M, TMA_K)
      Tensor tdKVgdK = gmem_thr_copy_dKV.partition_D(gdK);
      Tensor tdKVsdK =
          gmem_thr_copy_dKV.partition_S(sdK); // (TMA, TMA_M, TMA_K)
      Tensor tdKVrdV = make_fragment_like(tdKVgdV);
      Tensor tdKVrdK = make_fragment_like(tdKVgdK);
      Tensor cdKV = cute::make_identity_tensor(
          select<1, 2>(TileShape_MNK{})); // (BLK_M,BLK_K) -> (blk_m,blk_k)
      // Repeat the partitioning with identity layouts
      Tensor tdKVcdKV = gmem_thr_copy_dKV.partition_D(cdKV);
      Tensor tdKVpdKV = make_tensor<bool>(make_shape(size<2>(tdKVgdV)));
#pragma unroll
      for (int k = 0; k < size(tdKVpdKV); ++k) {
        tdKVpdKV(k) = get<1>(tdKVcdKV(_0{}, _0{}, k)) < get<1>(params.shape_dK);
      }
      // Need to check OOB when reading from smem if kBlockN isn't evenly tiled
      static constexpr bool EvenN =
          kBlockN % CUTE_STATIC_V(size<0>(GmemLayoutAtom{})) == 0;
      hstu::copy<
          /*Is_even_MN=*/EvenN,
          /*Is_even_K=*/true,
          /*Clear_OOB_MN=*/false>(
          gmem_tiled_copy_dKV, tdKVsdV, tdKVrdV, tdKVcdKV, tdKVpdKV, kBlockN);
      hstu::copy<
          /*Is_even_MN=*/EvenN,
          /*Is_even_K=*/true,
          /*Clear_OOB_MN=*/false>(
          gmem_tiled_copy_dKV, tdKVsdK, tdKVrdK, tdKVcdKV, tdKVpdKV, kBlockN);
      // // Tell warp 0 that smem_k and smem_v are ready
      // cutlass::arch::fence_view_async_shared(); // ensure smem reads are done
      // before next TMA to smem_k/v
      // hstu::named_barrier_arrive(NumEpilogueThreads +
      // cutlass::NumThreadsPerWarp,
      // static_cast<uint32_t>(BwdNamedBarriers::KVEmpty) /*id*/); Construct
      // identity layout for gdKV Clear_OOB_K must be false since we don't want
      // to write zeros to gmem
      hstu::copy<
          /*Is_even_MN=*/false,
          /*Is_even_K=*/false,
          /*Clear_OOB_MN=*/false,
          /*Clear_OOB_K=*/false>(
          gmem_tiled_copy_dKV,
          tdKVrdV,
          tdKVgdV,
          tdKVcdKV,
          tdKVpdKV,
          std::min(seqlen_info.seqlen - n_block * kBlockN, kBlockN));
      hstu::copy<
          /*Is_even_MN=*/false,
          /*Is_even_K=*/false,
          /*Clear_OOB_MN=*/false,
          /*Clear_OOB_K=*/false>(
          gmem_tiled_copy_dKV,
          tdKVrdK,
          tdKVgdK,
          tdKVcdKV,
          tdKVpdKV,
          std::min(seqlen_info.seqlen - n_block * kBlockN, kBlockN));
    }
  }

  CUTLASS_DEVICE void store_tail() {
    // if constexpr (Use_TMA) { tma_store_wait<0>(); }
  }

  // Write 0 to dK and dV
  CUTLASS_DEVICE void store_zero(
      Params const& params,
      int thread_idx,
      cute::tuple<int32_t, int32_t, int32_t> const& block_coord) {
    static constexpr int kBlockN = get<1>(TileShape_MNK{});
    auto [n_block, bidh, bidb] = block_coord;
    hstu::SeqlenInfo<Jagged, kBlockN> seqlen_info{
        bidb, size<0>(params.shape_dK), params.seq_offsets};
    Tensor mdK = make_tensor(
        make_gmem_ptr(params.ptr_dK), params.shape_dK, params.stride_dK)(
        _, _, bidh, !Jagged ? bidb : 0);
    Tensor gdK = local_tile(
        cute::domain_offset(make_coord(seqlen_info.offset, _0{}), mdK),
        select<1, 2>(TileShape_MNK{}),
        make_coord(n_block, _0{})); // (M, K)
    Tensor mdV = make_tensor(
        make_gmem_ptr(params.ptr_dV), params.shape_dK, params.stride_dV)(
        _, _, bidh, !Jagged ? bidb : 0);
    Tensor gdV = local_tile(
        cute::domain_offset(make_coord(seqlen_info.offset, _0{}), mdV),
        select<1, 2>(TileShape_MNK{}),
        make_coord(n_block, _0{})); // (M, K)

    GmemTiledCopydKV gmem_tiled_copy_dKV;
    auto gmem_thr_copy_dKV = gmem_tiled_copy_dKV.get_thread_slice(thread_idx);
    Tensor tdKVgdK = gmem_thr_copy_dKV.partition_D(gdK);
    Tensor tdKVgdV = gmem_thr_copy_dKV.partition_D(gdV);
    Tensor tdKVrdKV = make_fragment_like(tdKVgdK);
    clear(tdKVrdKV);
    // Construct identity layout for gdKV
    Tensor cdKV = cute::make_identity_tensor(
        select<1, 2>(TileShape_MNK{})); // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // Repeat the partitioning with identity layouts
    Tensor tdKVcdKV = gmem_thr_copy_dKV.partition_D(cdKV);
    Tensor tdKVpdKV = make_tensor<bool>(make_shape(size<2>(tdKVgdK)));
#pragma unroll
    for (int k = 0; k < size(tdKVpdKV); ++k) {
      tdKVpdKV(k) = get<1>(tdKVcdKV(_0{}, _0{}, k)) < get<1>(params.shape_dK);
    }
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    hstu::copy<
        /*Is_even_MN=*/false,
        /*Is_even_K=*/false,
        /*Clear_OOB_MN=*/false,
        /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_dKV,
        tdKVrdKV,
        tdKVgdK,
        tdKVcdKV,
        tdKVpdKV,
        seqlen_info.seqlen - n_block * kBlockN);
    hstu::copy<
        /*Is_even_MN=*/false,
        /*Is_even_K=*/false,
        /*Clear_OOB_MN=*/false,
        /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_dKV,
        tdKVrdKV,
        tdKVgdV,
        tdKVcdKV,
        tdKVpdKV,
        seqlen_info.seqlen - n_block * kBlockN);
  }
};

} // namespace hstu
