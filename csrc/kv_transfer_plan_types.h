// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>

#include "kv_transfer_types.h"

// Shared native descriptors for blocked KV transfers. These are
// backend-agnostic value types that describe geometry and per-kernel-group
// invariants without depending on any vendor runtime.

// __host__ __device__ under CUDA/HIP so kernels can call the inline helpers;
// otherwise keep the header toolchain-agnostic for the common C++ extension.
#if defined(__CUDACC__) || defined(__HIPCC__)
  #define LMC_TRANSFER_PLAN_HD __host__ __device__
#else
  #define LMC_TRANSFER_PLAN_HD
#endif

struct PageBufferShapeDesc {
  int kv_size;       // 1 or 2
  int nl;            // num layers
  int nb;            // num blocks
  int bs;            // block size
  int nh;            // num heads
  int hs;            // head size
  int element_size;  // bytes (1 or 2)
  // Physical per-block stride in source-dtype element units, used by
  // formats whose dim-0 is the block axis to step over padding bytes
  // (e.g. DeepSeek V4 compressor / indexer caches sharing a vLLM KV
  // pool with larger attn groups, whose rows are padded up to the
  // pool's max row width). 0 means "unset — fall back to the
  // format-specific tight stride".
  //
  // CONTRACT: pass ``tensor.stride(0)`` verbatim. PyTorch stride
  // semantics already absorb every inner-dim extent (including
  // ``kv_size``), so DO NOT pre-multiply by any inner dim.
  //
  // Honoured today only by NL_X_NB_BS_HS (per-layer [NB, BS, HS],
  // MLA). NL_X_NB_TWO_BS_NH_HS is restricted to the tight form
  // upstream and leaves this field at 0; all other formats either
  // pack non-block info into dim-0 or do not support dim-0 padding,
  // and ignore this field.
  int block_stride_elems;
  // Scale-region width in bytes of one *token's* row for blocked-scale
  // (``NL_X_NB_BSV_BSS``) pages, whose per-block layout is
  // ``[BS x value_bytes][BS x scale_bytes]`` with the two planes segregated.
  //
  // This is the ONE number that varies across quantized DeepSeek MLA caches
  // and that no tensor shape reveals: the trailing axis of the registration is
  // the *whole* record (132/68 for the DSA indexer, 584/528/352/288 for the
  // MLA main KV), so the split has to be supplied out of band. vLLM itself
  // infers the layout from bytes-per-token the same way.
  //
  // Known records (vLLM ``flashmla_sparse.py`` / ``fused_compress_quant_cache``):
  //   584 -> 576 + 8   (V4 fp8_ds_mla; H100/SM90)
  //   528 -> 512 + 16  (V4.1 fp8_ds_mla MXFP8; SM100+)
  //   352 -> 320 + 32  (V3.2 nvfp4_ds_mla; SM100+)
  //   288 -> 256 + 32  (V4.1 nvfp4_ds_mla compressed cache; SM100+)
  //   132 -> 128 + 4   (DSA indexer fp8)
  //    68 ->  64 + 4   (DSA indexer mxfp4)
  //
  // 0 means "unset — fall back to the historical 4-byte indexer scale", which
  // keeps pre-existing indexer callers byte-identical. Default-initialised
  // because the pybind constructor value-initialises nothing else: a caller
  // that sets every other field but not this one must still get 0, not an
  // indeterminate stack value.
  int scale_bytes = 0;

  template <typename ScalarType>
  LMC_TRANSFER_PLAN_HD inline size_t scalars_per_head() const {
    return hs * element_size / sizeof(ScalarType);
  }

  template <typename ScalarType>
  LMC_TRANSFER_PLAN_HD inline size_t scalars_per_token() const {
    return nh * hs * element_size / sizeof(ScalarType);
  }

  // Per (K or V) block step along dim-0, expressed in ``ScalarType``
  // element units (the kernel's working dtype, e.g. uint4 / uint32_t /
  // uint16_t). Returns the tight ``bs * nh * hs`` by default, or the
  // physical ``block_stride_elems`` when dim-0 carries padding (today
  // only NL_X_NB_BS_HS, see ``block_stride_elems`` above). Every
  // ``calculate_engine_global_offset`` branch uses this as the dim-0
  // step, so honouring padding here propagates to all formats without
  // per-branch changes.
  template <typename ScalarType>
  LMC_TRANSFER_PLAN_HD inline size_t scalars_per_block() const {
    const size_t elems = block_stride_elems > 0
                             ? static_cast<size_t>(block_stride_elems)
                             : static_cast<size_t>(bs) * nh * hs;
    return elems * element_size / sizeof(ScalarType);
  }
};

// Per-kernel-group invariants, resolved once on the Python side.
struct KernelGroupSpec {
  uintptr_t paged_buffer_ptrs;                // device ptr-array base address
  std::vector<int64_t> lmcache_objects_ptrs;  // temp GPU buffer ptr per slot
  PageBufferShapeDesc shape_desc;
  int lmcache_chunk_size;
  EngineKVFormat engine_kv_format;
  uintptr_t block_ids_base;  // device int64* base; sliced via block_ids_offset
  int64_t block_ids_capacity;  // total int64 elements behind block_ids_base;
                               // bounds-checks each slice in the executor
};
