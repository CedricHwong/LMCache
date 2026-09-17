# SPDX-License-Identifier: Apache-2.0
"""vLLM KV cache discovery.

Detection is declaration-driven: vLLM resolves a single physical KV cache
layout once in the engine core (``KVCacheLayout``, see
``vllm/vllm/v1/kv_cache_layout.py``) and publishes it on ``CacheConfig`` /
``KVCacheConfig``; the adapter forwards it as the ``kv_layout`` hint (see
``lmcache.integration.vllm.utils``). That declaration is the single source of
truth for which :class:`EngineKVFormat` a per-layer cache has. Byte geometry
(sizes, strides) is used only to *assert* the declaration matches the
registered tensors, never to pick the format, and every fallback path warns.
"""

# mypy: disable-error-code="union-attr"
# Standard
from typing import Optional

# Third Party
import torch

# First Party
from lmcache import torch_device_type
from lmcache.logging import init_logger
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.kv_format.detectors.base import (
    EngineDetector,
    measure_list_depth_until_tensor,
)
from lmcache.v1.gpu_connector.kv_format.types import (
    KV_LAYOUT_NAMES,
    DiscoverableKVCache,
    LayoutHints,
)
import lmcache.lmcache_native as lmcache_native

logger = init_logger(__name__)

# vLLM ``KVCacheLayout`` members whose head axis sits *outside* the per-block
# token run (``LHBNC = [L, H, B, N, C]``, ``BHLNC = [B, H, L, N, C]``). A
# (layer, block) is then fragmented into per-head slabs, while LMCache's
# transfer kernels address one contiguous run per (layer, block). These can
# never be transferred: they must fail fast, never be silently mis-read.
# Note ``LBHNC = [L, B, H, N, C]`` (the identity) is *not* heads-outermost
# and maps to HND / ``NL_X_NB_NH_BS_CS``.
_HEADS_OUTERMOST_LAYOUTS = frozenset(("LHBNC", "BHLNC"))

# vLLM ``KVCacheLayout`` blocks-first members: ONE fused ``[B, L, ...]``
# buffer per cache group whose per-layer 4-D views carry the block step in
# stride(0). Any non-4-D per-layer structure contradicts the declaration.
_BLOCKS_FIRST_LAYOUTS = frozenset(("BLHNC", "BLNHC"))


def resolve_vllm_kv_layout(
    layout_hints: LayoutHints, cpu_attention_backend: bool
) -> str:
    """Resolve the layout that governs detection, from vLLM's declaration.

    The resolved ``kv_layout`` hint (translated from vLLM's ``KVCacheLayout``)
    is the single source of truth for the per-layer byte order. When the hint
    is missing -- a legacy registration or an MP-server process without a
    vLLM config -- fall back to the structural default but **warn**: byte
    sniffing cannot tell BLHNC from BLNHC and may silently mis-read a
    blocks-first cache.

    Args:
        layout_hints: Registration hints; ``kv_layout`` is read from it.
        cpu_attention_backend: Whether the host is a CPU attention backend
            (legacy fallback only).

    Returns:
        The resolved LMCache layout name (``NHD`` / ``HND`` / ``BLHNC`` /
        ``BLNHC``).

    Raises:
        NotImplementedError: If the hint names a heads-outermost vLLM layout
            LMCache cannot transfer.
        ValueError: If the hint names an unknown layout.
    """
    kv_layout = layout_hints.get("kv_layout")
    if kv_layout is None:
        # Registrations predating layout hints relied on the CPU backend's HND
        # allocation and the NHD default everywhere else.
        legacy = "HND" if cpu_attention_backend else "NHD"
        logger.warning(
            "vLLM declared no resolved KV cache layout; falling back to "
            "structural sniffing with the legacy default (%s). A blocks-first "
            "cache (BLHNC/BLNHC) registered without a declared layout cannot "
            "be told apart from its layer-compact twin and may be mis-read.",
            legacy,
        )
        return legacy
    if kv_layout in _HEADS_OUTERMOST_LAYOUTS:
        raise NotImplementedError(
            f"vLLM declared KV cache layout {kv_layout!r}, which LMCache "
            "cannot transfer: the head axis is outer to the per-block token "
            "run, so each (layer, block) is fragmented per head. Select a "
            "transferable layout (LBNHC/LBHNC/BLHNC/BLNHC, e.g. via "
            "VLLM_KV_CACHE_LAYOUT) instead."
        )
    if kv_layout not in KV_LAYOUT_NAMES:
        raise ValueError(
            f"kv_layout hint {kv_layout!r} is not a layout LMCache supports; "
            f"expected one of {', '.join(KV_LAYOUT_NAMES)}."
        )
    return kv_layout


class VLLM_Detector(EngineDetector):
    engine_type = EngineType.VLLM

    def discover(
        self,
        kv_caches: DiscoverableKVCache,
        layout_hints: LayoutHints,
    ) -> "tuple[Optional[lmcache_native.EngineKVFormat], DiscoverableKVCache]":
        kv_layout = resolve_vllm_kv_layout(
            layout_hints, cpu_attention_backend=torch_device_type == "cpu"
        )
        # HND (heads before block size) covers LBHNC's per-layer [B, H, N, C]
        # view and BLHNC's fused views; NHD covers LBNHC / BLNHC likewise.
        is_hnd = kv_layout in ("HND", "BLHNC")

        # Fused K/V is the only rank-4 vLLM layout, so its raw rank
        # identifies it unambiguously (a 5-D split would collide with
        # flash-infer when num_heads == 2). The two middle axes are NH/BS
        # (HND) or BS/NH (NHD) -- indistinguishable from the shape alone, so the
        # declared kv_layout decides (the declaration is the truth; the rank is
        # the assertion that the bytes match it). The tensor is kept raw: the
        # trailing axis is the per-head content size (e.g. 2 * head_size, K/V
        # packed, or an MLA/indexer record). Blocks-first views (BLHNC / BLNHC)
        # have the same per-layer shape and differ only in stride(0), which
        # resolve_block_stride_and_log_layout reads.
        if (
            isinstance(kv_caches, list)
            and kv_caches
            and isinstance(kv_caches[0], torch.Tensor)
            and kv_caches[0].dim() == 4
        ):
            if is_hnd:
                return lmcache_native.EngineKVFormat.NL_X_NB_NH_BS_CS, kv_caches
            return lmcache_native.EngineKVFormat.NL_X_NB_BS_NH_CS, kv_caches

        # Everything below is a non-4-D registration (legacy per-layer 5-D
        # split, 3-D MLA/indexer, fused cross-layer, ...). A blocks-first
        # declaration implies one fused buffer per group, so any other shape is
        # a contradiction: fail fast instead of silently mis-reading.
        if kv_layout in _BLOCKS_FIRST_LAYOUTS:
            raise ValueError(
                f"vLLM declared KV cache layout {kv_layout!r}, which registers "
                "one fused [B, L, ...] buffer per cache group, but detection "
                "observed a per-layer registration whose tensors are not "
                "rank-4. The layout declaration and the registered shapes "
                "contradict each other; refusing to mis-read."
            )

        list_depth, tensor_ndim, first_tensor = measure_list_depth_until_tensor(
            kv_caches
        )

        if list_depth == 0:
            return lmcache_native.EngineKVFormat.NB_NL_TWO_BS_NH_HS, kv_caches
        # vLLM-RBLN: HND with a singleton between heads and block tokens that
        # its attention backend requires. Always HND, so the hint is not read.
        if (
            list_depth == 1
            and tensor_ndim == 6
            and first_tensor.shape[0] == 2
            and first_tensor.shape[3] == 1
        ):
            return lmcache_native.EngineKVFormat.NL_X_TWO_NB_NH_ONE_BS_HS, kv_caches
        # Legacy per-layer 5-D split K/V. No nightly vLLM registration produces
        # these (nightly always emits 4-D per-layer views), so they are reached
        # only from pre-layer-views vLLM under a layer-compact declaration; the
        # declared layout still decides HND/NHD byte order.
        if list_depth == 1 and tensor_ndim == 5:
            if first_tensor.shape[0] == 2:  # K/V axis first
                if is_hnd:
                    return lmcache_native.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS, kv_caches
                return lmcache_native.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS, kv_caches
            if first_tensor.shape[1] == 2:  # num_blocks first
                if is_hnd:
                    return lmcache_native.EngineKVFormat.NL_X_NB_TWO_NH_BS_HS, kv_caches
                return lmcache_native.EngineKVFormat.NL_X_NB_TWO_BS_NH_HS, kv_caches
        # Legacy per-layer 3-D MLA (or DSA indexer) cache. This branch is dead
        # for nightly vLLM -- whose MLA latent and indexer caches are 4-D views
        # handled above -- and survives only for pre-layer-views vLLM. Its
        # format does not depend on the layout declaration (single-head state
        # rows), so the byte shape is the assertion and the hint is ignored.
        if list_depth == 1 and tensor_ndim == 3:  # MLA (or DSA indexer cache)
            if first_tensor.dtype == torch.uint8 and int(first_tensor.shape[-1]) == 132:
                return lmcache_native.EngineKVFormat.NL_X_NB_BSV_BSS, kv_caches
            return lmcache_native.EngineKVFormat.NL_X_NB_BS_HS, kv_caches
        if list_depth == 2 and tensor_ndim == 4 and len(kv_caches[0]) == 2:
            return lmcache_native.EngineKVFormat.NL_X_TWO_X_NB_BS_NH_HS, kv_caches
        return None, kv_caches
