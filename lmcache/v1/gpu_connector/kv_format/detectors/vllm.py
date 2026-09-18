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

Some caches are *not* transferable even though their bytes look like a valid
pool: the engine's per-request scratch. The DeepSeek-V4.1 compressor ring
(``CircularBufferSpec``) is the one that appears in production today, but vLLM
declares several others the same way -- ``KpoolTailSpec`` (one-block circular
scratch for a kpool indexer's raw tail, structurally an ordinary
``SlidingWindowSpec``) and the HiSparse pools. Their per-layer tensors are
shape-indistinguishable from a real, reusable pool, so detection alone can
never tell them apart -- only the vLLM *spec* declares them non-cacheable
(``prefix_cacheable=False``). The recognition predicate
:func:`is_non_prefix_cacheable_spec` below is the single version-tolerant place
that reads that declaration; the vLLM group builders use it to exclude such
groups from LMCache KV groups up-front (see
``lmcache.integration.vllm.kv_cache_groups``) so they are never silently stored
as if they were request-reusable prefix KV. It keys on the declaration rather
than the class name precisely so that a spec LMCache has never heard of is
still excluded.
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


# vLLM spec class names that are per-request scratch *by definition*, i.e.
# vLLM hard-codes ``prefix_cacheable = False`` for them.  Kept only as a
# contradiction guard: the exclusion decision itself is read from the
# declaration (so an unknown future spec is covered), but if one of these ever
# declares itself cacheable the two signals disagree and LMCache refuses to
# guess.  ``KpoolTailSpec`` extends ``SlidingWindowSpec``, so it is
# structurally an ordinary sliding-window group -- the name list is what lets
# that case be caught rather than silently stored.
_KNOWN_SCRATCH_SPEC_CLASSES = frozenset(
    {
        "CircularBufferSpec",
        "KpoolTailSpec",
        "HiSparseHotSpec",
        "HiSparseResidentSpec",
    }
)


def _is_known_scratch_class(spec: Any) -> bool:
    """Return whether *spec*'s class is a known per-request scratch spec.

    Args:
        spec: A vLLM KV cache spec leaf.

    Returns:
        ``True`` if any class in the MRO is one of
        :data:`_KNOWN_SCRATCH_SPEC_CLASSES`.
    """
    return any(
        cls.__name__ in _KNOWN_SCRATCH_SPEC_CLASSES for cls in type(spec).__mro__
    )


def _declares_non_prefix_cacheable(spec: Any) -> bool:
    """Return whether a leaf spec declares itself out of prefix caching.

    The decision is read from the engine's own declaration --
    ``KVCacheSpec.prefix_cacheable`` -- never from the class name or the byte
    shape.  This predicate used to match the class name ``CircularBufferSpec``
    (so the module could stay importable without vLLM), which silently missed
    every *other* per-request scratch spec: vLLM also declares
    ``KpoolTailSpec`` (one-block circular scratch for a kpool indexer's raw
    tail), ``HiSparseHotSpec`` and ``HiSparseResidentSpec`` as
    ``prefix_cacheable=False``.  ``KpoolTailSpec`` extends ``SlidingWindowSpec``,
    so it is structurally an ordinary sliding-window group and would have been
    stored as if it were reusable prefix KV -- poisoning the cache across
    requests.  Keying on the declaration covers all of them and cannot be
    outrun by the next spec vLLM adds.  It also matches what vLLM itself does
    to pick its own storeable groups (``KVCacheConfig.
    prefix_cacheable_group_ids`` filters on exactly this property).

    Args:
        spec: A vLLM KV cache spec leaf.

    Returns:
        ``True`` when the spec declares ``prefix_cacheable=False``.  ``False``
        when it declares ``True`` or does not declare the attribute at all (a
        legacy vLLM, or a non-``KVCacheSpec`` object): an undeclared spec is
        kept, because nothing marks it as per-request scratch.

    Raises:
        ValueError: If the attribute is present but not a ``bool``.  A
            non-boolean declaration cannot be interpreted, and guessing either
            way would silently store or drop a group.
    """
    prefix_cacheable = getattr(spec, "prefix_cacheable", None)
    if prefix_cacheable is None:
        return False
    if not isinstance(prefix_cacheable, bool):
        raise ValueError(
            "vLLM KV cache spec declares a non-boolean prefix_cacheable="
            f"{prefix_cacheable!r} of type {type(prefix_cacheable).__name__}; "
            "LMCache cannot tell whether the group is reusable prefix KV and "
            "will not guess."
        )
    # Contradiction guard. The *decision* is the declaration, so a class
    # LMCache has never heard of cannot be mis-excluded -- but for a class that
    # is per-request scratch by definition, a truthy flag is self-contradictory
    # and storing it would poison the cache. Refuse rather than pick a side.
    if prefix_cacheable and _is_known_scratch_class(spec):
        uses_slot_mapping = getattr(spec, "uses_slot_mapping", None)
        raise ValueError(
            f"vLLM {type(spec).__name__} is per-request scratch and must "
            "declare prefix_cacheable=False"
            f"; got prefix_cacheable={prefix_cacheable!r}, "
            f"uses_slot_mapping={uses_slot_mapping!r}. Refusing to guess "
            "whether the group is reusable prefix KV."
        )
    return not prefix_cacheable


def is_non_prefix_cacheable_spec(spec: Any) -> bool:
    """Return whether a vLLM KV cache spec must be excluded from prefix caching.

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

    Concerned specs are the engine's per-request scratch -- content that is a
    deterministic function of neither the prefix nor the position alone, so it
    can never be reused as prefix KV.  vLLM declares each of them
    ``prefix_cacheable=False``: the DeepSeek-V4.1 compressor ring
    (``CircularBufferSpec``; its slots are addressed as
    ``block * capacity + pos % capacity`` and a speculative write is rolled
    back on rejection), a kpool indexer's raw tail (``KpoolTailSpec``) and the
    HiSparse pools.  Their bytes are shape-indistinguishable from a real pool,
    which is why detection cannot -- and must not -- classify them from the
    tensors.

    Handles a ``UniformTypeKVCacheSpecs`` wrapper too: a group whose every
    leaf is non-cacheable is a scratch group.  A mixed wrapper (scratch beside
    regular pools) must not count, or the whole group would be wrongly
    excluded.

    Args:
        spec: A vLLM KV cache spec (leaf or ``UniformTypeKVCacheSpecs``),
            or ``None``.

    Returns:
        ``True`` when the spec (or every leaf of a wrapper) declares itself
        non-prefix-cacheable.

    Raises:
        ValueError: If a leaf declares a non-boolean ``prefix_cacheable``.
    """
    if spec is None:
        return False
    leaves = getattr(spec, "kv_cache_specs", None)
    if isinstance(leaves, dict):
        return bool(leaves) and all(
            _declares_non_prefix_cacheable(leaf) for leaf in leaves.values()
        )
    return _declares_non_prefix_cacheable(spec)


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
