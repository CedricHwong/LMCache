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

One cache is *not* transferable even though its bytes look like a valid pool:
the DeepSeek-V4.1 compressor circular-buffer ring (``CircularBufferSpec``).
Its per-layer tensor is shape-indistinguishable from a compressed-MLA pool, so
detection alone can never tell it apart -- only the vLLM *spec* declares it
non-cacheable (``prefix_cacheable=False``, ``uses_slot_mapping=False``). The
recognition predicate :func:`is_circular_buffer_ring_spec` below is the
single version-tolerant place that reads that declaration; the vLLM group
builders use it to exclude the ring group from LMCache KV groups up-front (see
``lmcache.integration.vllm.kv_cache_groups``) so it is never silently stored
as if it were request-reusable prefix KV.
"""

# mypy: disable-error-code="union-attr"
# Standard
from typing import Any, Optional

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
from lmcache.v1.gpu_connector.kv_format.specs.nl_x_nb_bsv_bss import (
    blocked_scale_bytes,
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


def _is_circular_buffer_ring_class(spec: Any) -> bool:
    """Return whether a vLLM KV cache spec is a ``CircularBufferSpec``.

    Checked by class name so this module stays importable without vLLM (the
    same convention the vLLM group builder uses for sliding-window / Mamba
    specs). ``CircularBufferSpec`` overrides ``prefix_cacheable`` to always
    return ``False`` and ``uses_slot_mapping`` to ``False``, so the class
    identity *is* the declaration; seeing the class but a truthy flag would be
    a contradiction this module cannot guess around, so it fails closed.
    """
    if not any(cls.__name__ == "CircularBufferSpec" for cls in type(spec).__mro__):
        return False
    prefix_cacheable = getattr(spec, "prefix_cacheable", None)
    uses_slot_mapping = getattr(spec, "uses_slot_mapping", None)
    if prefix_cacheable is True or uses_slot_mapping is True:
        raise ValueError(
            "vLLM CircularBufferSpec must declare prefix_cacheable=False and "
            "uses_slot_mapping=False; got prefix_cacheable="
            f"{prefix_cacheable!r}, uses_slot_mapping={uses_slot_mapping!r}. "
            "Refusing to guess whether the compressor ring is cacheable."
        )
    return True


def is_circular_buffer_ring_spec(spec: Any) -> bool:
    """Return whether a vLLM KV cache spec is the compressor circular-buffer ring.

    The ring is DeepSeek-V4.1's per-request scratch: one block per request
    holds the raw ``[kv, score]`` rows of the token group still being
    compressed (``CircularBufferSpec``, block size = the ring capacity). vLLM
    itself declares it ``prefix_cacheable=False`` and ``uses_slot_mapping=
    False`` -- its content is a deterministic function of neither the prefix nor
    the position alone (speculative drafts are written then rolled back, and
    the ring slots are addressed as ``block * capacity + pos % capacity``), so
    it is never reusable prefix KV and must be excluded from LMCache KV groups.
    Its bytes are shape-indistinguishable from a compressed-MLA pool, which is
    why detection cannot -- and must not -- classify it from the tensors.

    Handles a ``UniformTypeKVCacheSpecs`` wrapper too: a group whose every
    leaf is the ring is a ring group. A mixed wrapper (ring beside regular
    pools) must not count, or the whole group would be wrongly excluded.

    Args:
        spec: A vLLM KV cache spec (leaf or ``UniformTypeKVCacheSpecs``),
            or ``None``.

    Returns:
        ``True`` only for a compressor ring spec.

    Raises:
        ValueError: If a ``CircularBufferSpec`` declares itself cacheable.
    """
    if spec is None:
        return False
    leaves = getattr(spec, "kv_cache_specs", None)
    if isinstance(leaves, dict):
        return bool(leaves) and all(
            _is_circular_buffer_ring_class(leaf) for leaf in leaves.values()
        )
    return _is_circular_buffer_ring_class(spec)


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
        ValueError: If the hint names a heads-outermost vLLM layout LMCache
            cannot transfer, or any other unknown layout. ``detect_format``
            documents ``ValueError`` as the failure mode of this layer; the
            adapter boundary (``translate_vllm_kv_cache_layout``) already
            raises ``NotImplementedError`` for heads-outermost names before a
            hint can reach here.
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
        raise ValueError(
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
            t0 = kv_caches[0]
            # A *quantized* DeepSeek MLA page does not lay its tokens out
            # token-by-token: per block it stores every token's value bytes
            # first and every token's scale bytes after ([BS x vals][BS x
            # scales]). Only NL_X_NB_BSV_BSS addresses that segregation, and
            # only a singleton head axis can be laid out that way (one plane).
            # The registration exposes just the whole record width, so the
            # value/scale split is looked up by width; a record that is not a
            # known blocked-scale one falls through to the content-size formats
            # unchanged.
            record = int(t0.shape[-1])
            scale = blocked_scale_bytes(record)
            singleton_head = t0.shape[1] == 1 or t0.shape[2] == 1
            if scale > 0 and singleton_head:
                logger.info(
                    "vLLM registered a %d B/token blocked-scale DeepSeek MLA "
                    "page (%d value + %d scale bytes, segregated per block); "
                    "selecting NL_X_NB_BSV_BSS -- the content-size formats "
                    "would address it token-major and silently corrupt it.",
                    record,
                    record - scale,
                    scale,
                )
                return lmcache_native.EngineKVFormat.NL_X_NB_BSV_BSS, kv_caches
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
