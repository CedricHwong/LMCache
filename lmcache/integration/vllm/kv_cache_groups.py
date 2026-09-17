# SPDX-License-Identifier: Apache-2.0
"""Build LMCache engine group infos from vLLM KV cache group metadata."""

# Future
from __future__ import annotations

# Standard
from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.gpu_connector.utils import LayoutHints

# First Party
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.group_view import EngineGroupInfo

logger = init_logger(__name__)


def _first_leaf_spec(spec: Any) -> Any:
    """Return *a* leaf spec behind a ``UniformTypeKVCacheSpecs`` wrapper.

    The wrapper holds one spec per layer. vLLM's ``is_uniform_type`` only
    requires one shared registry base class *and* one block size
    (``kv_cache_interface.py:1229-1240``) -- it does **not** require the
    leaves to agree on ``tokens_per_state``, so production G8's 8 MLA leaves
    mix ``{2, 1}``. Use this helper only where *any* leaf stands in for all
    (e.g. the shared ``block_size``); field readers that must be per-layer
    (``resolve_tokens_per_state``, ``read_v41_spec_fields``, and
    ``create_engine_group_infos_from_vllm``'s engine-group path) resolve
    through :func:`_leaf_specs` instead. Every other spec is its own leaf.
    """
    inner = getattr(spec, "kv_cache_specs", None)
    if isinstance(inner, dict) and inner:
        return next(iter(inner.values()))
    return spec


def _leaf_specs(spec: Any) -> list[Any]:
    """Return every leaf spec behind a possibly-wrapped KV cache spec.

    ``UniformTypeKVCacheSpecs`` holds one spec per layer, so all of its
    ``kv_cache_specs`` values are returned (vLLM does not assert they agree --
    see :func:`_first_leaf_spec`); every other spec is its own leaf. Order is
    the wrapper's insertion order.
    """
    inner = getattr(spec, "kv_cache_specs", None)
    if isinstance(inner, dict):
        return list(inner.values())
    return [spec]


def _is_attention_spec(spec: Any) -> bool:
    """Return whether the KV cache spec is a vLLM attention spec.

    Checked by class name so this module stays importable without vLLM. A
    ``UniformTypeKVCacheSpecs`` wrapper is unwrapped to *every* leaf: the
    group counts as attention only when all of its leaves are (it never
    derives from ``AttentionSpec`` itself).
    """
    return all(
        any(cls.__name__ == "AttentionSpec" for cls in type(leaf).__mro__)
        for leaf in _leaf_specs(spec)
    )


def resolve_tokens_per_state(spec: Any) -> int:
    """Resolve the tokens covered by one stored state, as a positive int.

    Reads vLLM nightly's ``AttentionSpec.tokens_per_state`` first and falls
    back to the removed ``MLAAttentionSpec.compress_ratio`` so vLLM v0.21.0
    keeps working. The value does **not** change how many tokens an engine
    block id spans (that stays ``block_size`` — see
    :func:`get_tokens_per_block`); it describes the *state* count inside a
    block (``block_size // tokens_per_state``), which LMCache re-derives from
    the registered tensors as ``slots_per_block``.

    A ``Fraction`` -- several states per token, e.g. Whisper block pooling --
    cannot be represented by LMCache's integer ``tokens_per_block //
    slots_per_block`` compression model, so it raises instead of silently
    truncating. Non-positive values carry no per-state width and resolve to
    the uncompressed default: ``0`` is vLLM's sliding-window marker
    (``DeepseekV41`` layers with ``compress_ratio=0``), ``-1`` the
    ``MambaSpec`` sentinel.

    Args:
        spec: A vLLM KV cache spec (leaf or ``UniformTypeKVCacheSpecs``
            wrapper). A wrapper is unwrapped to its **first** leaf: vLLM does
            not guarantee the leaves agree on this field, so this single-value
            reader cannot represent a mixed group (e.g. production G8's
            ``{2, 1}``) -- pass an explicit leaf for per-layer resolution, as
            :func:`read_v41_spec_fields` and
            :func:`create_engine_group_infos_from_vllm` do.

    Returns:
        The positive integer tokens per stored state; ``1`` when the spec
        declares no compression.

    Raises:
        ValueError: If the declared value is a ``Fraction``.
    """
    leaf = _first_leaf_spec(spec)
    tokens_per_state = getattr(leaf, "tokens_per_state", None)
    if tokens_per_state is None:
        tokens_per_state = getattr(leaf, "compress_ratio", None)
    if tokens_per_state is None:
        return 1
    if isinstance(tokens_per_state, Fraction):
        raise ValueError(
            "fractional tokens_per_state "
            f"{tokens_per_state!r} is not representable by LMCache's integer "
            "compression model (tokens per block / slots per block)"
        )
    if not isinstance(tokens_per_state, int):
        return 1
    return tokens_per_state if tokens_per_state > 0 else 1


def _read_spec_fields(leaf: Any) -> dict[str, Any]:
    """Fault-tolerant read of the V4.1 spec fields from a **single** leaf spec.

    Returns exactly the eight ``EngineGroupInfo`` tail fields added for the
    V4.1 schema, with the contract defaults wherever the leaf -- or the whole
    vLLM version -- does not declare them:

    ``tokens_per_state=1``, ``cache_role="sparse"``,
    ``state_content_bytes=None``, ``block_stride_alignment=None``,
    ``is_index_group_leader=False``, ``prefix_cacheable=True``,
    ``page_size_padded=None``, ``num_head_slots=None``.

    Every read goes through ``getattr`` with a default (never bare attribute
    access), so neither a nightly/older field layout nor a missing attribute
    can raise ``AttributeError``. ``cache_role`` is normalized from the
    ``SparseCacheRole`` enum to its ``str`` value; ``prefix_cacheable`` is the
    leaf's own value/property (``False`` for e.g. ``CircularBufferSpec``,
    ``True`` otherwise).

    Args:
        leaf: A single vLLM KV cache spec leaf (never a
            ``UniformTypeKVCacheSpecs`` wrapper).

    Returns:
        An eight-key dict safe to pass as ``EngineGroupInfo(**fields)``.
    """
    cache_role = getattr(leaf, "cache_role", None)
    if cache_role is not None:
        cache_role = getattr(cache_role, "value", cache_role)
    prefix_cacheable = getattr(leaf, "prefix_cacheable", None)
    if prefix_cacheable is None:
        prefix_cacheable = True
    return {
        "tokens_per_state": resolve_tokens_per_state(leaf),
        "cache_role": cache_role if cache_role else "sparse",
        "state_content_bytes": getattr(leaf, "state_content_bytes", None),
        "block_stride_alignment": getattr(leaf, "block_stride_alignment", None),
        "is_index_group_leader": bool(getattr(leaf, "is_index_group_leader", False)),
        "prefix_cacheable": bool(prefix_cacheable),
        "page_size_padded": getattr(leaf, "page_size_padded", None),
        "num_head_slots": getattr(leaf, "num_head_slots", None),
    }


def read_v41_spec_fields(spec: Any) -> dict[str, Any]:
    """Fault-tolerant read of the vLLM V4.1 spec fields LMCache carries.

    Delegates to :func:`_read_spec_fields` per leaf and returns the shared
    values. A --- possibly ``UniformTypeKVCacheSpecs``-wrapped --- spec whose
    leaves all agree yields one dict ready to spread as
    ``EngineGroupInfo(**fields)``.

    A wrapper whose leaves disagree on any of the eight fields (production G8
    mixes ``tokens_per_state`` in ``{2, 1}`` because vLLM's ``is_uniform_type``
    only checks a shared registry base class + one block size,
    ``kv_cache_interface.py:1229-1240``) has **no single value**, so instead of
    silently broadcasting one leaf this reader raises ``ValueError``. The
    engine-group builder resolves per layer
    (:func:`create_engine_group_infos_from_vllm` reads each layer's own leaf,
    and the identity split re-derives each LMCache group's fields from exactly
    its layers' leaves), so the mixed case is never asked to collapse here.

    Args:
        spec: A vLLM KV cache spec (leaf or ``UniformTypeKVCacheSpecs``).

    Returns:
        An eight-key dict safe to pass as ``EngineGroupInfo(**fields)``.

    Raises:
        ValueError: If a wrapped group's leaves differ on any of the eight
            fields (cannot be represented by one value).
    """
    leaves = _leaf_specs(spec)
    if len(leaves) == 1:
        return _read_spec_fields(leaves[0])
    per_leaf = [_read_spec_fields(leaf) for leaf in leaves]
    for key in per_leaf[0]:
        values = {fields[key] for fields in per_leaf}
        if len(values) > 1:
            raise ValueError(
                f"{key!r} differs across the leaves of this "
                "UniformTypeKVCacheSpecs group "
                f"({sorted(values, key=str)!r}); the wrapper has no single "
                "value. vLLM only requires a shared base class and block size "
                "to form the group (not this field), so resolve per layer "
                "(create_engine_group_infos_from_vllm reads each leaf) instead "
                "of asking one dict for the whole group."
            )
    return per_leaf[0]


def get_group_prefix_cacheable(kv_cache_config: Any) -> list[bool]:
    """Return whether each KV cache group participates in prefix caching.

    Read from ``group.kv_cache_spec.prefix_cacheable`` (a vLLM property that
    is ``all(...)`` over the leaves of a ``UniformTypeKVCacheSpecs`` wrapper),
    so a non-cacheable group -- e.g. the V4.1 compressor ring -- is excluded
    from the hit-length alignment. This is the connector-side half of the
    single source of truth for that alignment: the block spans come from
    :func:`get_tokens_per_block` and the lcm is computed by
    :func:`lmcache.v1.multiprocess.group_view.lcm_cacheable_block_tokens`,
    which applies exactly this ``prefix_cacheable`` filter. When vLLM
    provides no group metadata, preserve the legacy single-group rule
    (cacheable).

    Every read goes through ``getattr`` with a default, so a nightly/older
    field layout without ``prefix_cacheable`` counts as cacheable instead of
    raising.

    Args:
        kv_cache_config: vLLM's resolved KV cache group configuration; may be
            ``None`` when the engine supplied no metadata.

    Returns:
        One boolean per KV cache group, in engine group order (``[True]`` for
        the legacy single-group case).
    """
    groups = (
        getattr(kv_cache_config, "kv_cache_groups", ()) or ()
        if kv_cache_config is not None
        else ()
    )
    if not groups:
        return [True]
    return [
        bool(getattr(getattr(group, "kv_cache_spec", None), "prefix_cacheable", True))
        for group in groups
    ]


def get_tokens_per_block(kv_cache_spec: Any, dcp_size: int) -> int:
    """Global tokens covered by one block id of ``kv_cache_spec``.

    The returned value lives in the scheduler's *token* coordinate space: one
    engine block id always covers ``block_size`` tokens, exactly how vLLM
    sizes the block table (``max_num_blocks_per_req = cdiv(max_len,
    block_size)``) regardless of how many states a block stores. So
    ``tokens_per_state`` does not change the number returned -- what it
    changes is the number of stored states inside each block
    (``block_size // tokens_per_state``), which LMCache detects on the tensor
    side as ``slots_per_block`` (the block tensor's batch dimension). The
    relationship is:

        ``tokens_per_block // slots_per_block == tokens_per_state``

    i.e. the compression factor is re-derived downstream from the
    block-id-to-token span and the physical slot count, and must **not** be
    folded into ``tokens_per_block`` here (LMCache's ``compress_ratio =
    tokens_per_block // slots_per_block`` path would then double-count it).

    What this function does with ``tokens_per_state`` is validate it: a
    ``Fraction`` is unrepresentable in the integer compression model and a
    non-divisor would make vLLM's ``get_num_kernel_states`` drop the tail
    tokens, so both fail closed instead of corrupting later arithmetic. The
    check runs on **every** leaf: a ``UniformTypeKVCacheSpecs`` wrapper may
    mix ``tokens_per_state`` values across its leaves (vLLM ``is_uniform_type``
    does not constrain it), and each one must divide ``block_size``.

    Attention blocks span ``block_size * dcp_size`` tokens under DCP
    (vLLM's ``resolve_kv_cache_block_sizes`` rule); recurrent state is
    replicated, not sharded, and stays at ``block_size``.

    Args:
        kv_cache_spec: vLLM KV cache spec (leaf or ``UniformTypeKVCacheSpecs``).
        dcp_size: Decode context parallel size.

    Returns:
        Logical tokens per engine block id.

    Raises:
        ValueError: If any leaf declares a fractional ``tokens_per_state``, or
            an integer one that does not divide ``block_size``.
    """
    block_size = kv_cache_spec.block_size
    for leaf in _leaf_specs(kv_cache_spec):
        tokens_per_state = resolve_tokens_per_state(leaf)
        if tokens_per_state > 1 and block_size % tokens_per_state != 0:
            raise ValueError(
                f"tokens_per_state {tokens_per_state} does not divide block_size "
                f"{block_size}; vLLM's get_num_kernel_states would drop the tail "
                "tokens"
            )
    if dcp_size <= 1:
        return block_size
    if _is_attention_spec(kv_cache_spec):
        return block_size * dcp_size
    return block_size


def _is_sliding_window_spec(spec: Any) -> bool:
    """Return whether the KV cache spec is a vLLM sliding-window spec.

    Checked by class name so this module stays importable without vLLM.
    Subclasses such as ``SlidingWindowMLASpec`` count.
    """
    return any(cls.__name__ == "SlidingWindowSpec" for cls in type(spec).__mro__)


def _is_cachable_mamba_spec(spec: Any) -> bool:
    """Return whether the spec is a snapshotting Mamba/linear-attention spec.

    Align mode snapshots only the last block; all mode snapshots every block
    boundary. Either way a restore consumes only the last matched block's
    page, so both behave like a cross-chunk sliding window of one block.
    Checked by class name (like :func:`_is_sliding_window_spec`) so this
    module stays importable without vLLM.
    """
    return any(cls.__name__ == "MambaSpec" for cls in type(spec).__mro__) and getattr(
        spec, "mamba_cache_mode", "none"
    ) in ("align", "all")


def _resolve_per_layer_sw_sizes(
    vllm_groups: Sequence[Any],
    layer_to_idx: Mapping[str, int],
    num_layers: int,
) -> list[int]:
    """Resolve the sliding window size in tokens for each registered KV tensor.

    Will resolve -1 for non-sliding-window layers.

    Args:
        vllm_groups: vLLM ``KVCacheGroupSpec`` instances.
        layer_to_idx: Layer name to registered tensor index mapping.
        num_layers: Number of registered KV tensors.

    Returns:
        A list of length ``num_layers`` mapping each registered tensor index
        to its cross-chunk window size in tokens: the sliding window for
        sliding-window attention, the engine-side block size for align-mode
        Mamba/linear layers (a one-block window), or ``-1`` for full-attention
        layers.
    """
    per_layer_sw_size = [-1] * num_layers
    for group in vllm_groups:
        spec = getattr(group, "kv_cache_spec", None)
        if spec is None:
            continue
        # ``UniformTypeKVCacheSpecs`` carries per-layer specs in
        # ``kv_cache_specs``; other specs apply to all of the group's layers.
        per_layer_specs = getattr(spec, "kv_cache_specs", None)
        for name in group.layer_names:
            layer_spec = per_layer_specs[name] if per_layer_specs else spec
            if _is_sliding_window_spec(layer_spec):
                per_layer_sw_size[layer_to_idx[name]] = layer_spec.sliding_window
            elif _is_cachable_mamba_spec(layer_spec):
                per_layer_sw_size[layer_to_idx[name]] = layer_spec.block_size
    return per_layer_sw_size


#: Reserved layer-name prefix for CacheBlend fused-aux page pools:
#: ``cb.aux_pool.<tokens_per_block>[.<label>]`` — the first suffix part is
#: the logical block size; the optional label distinguishes multiple pools
#: sharing one block size (they then share an engine group).
CB_AUX_POOL_LAYER_PREFIX = "cb.aux_pool."


def cb_aux_pool_entries(kv_caches) -> "list[tuple[str, int]]":
    """CacheBlend fused-aux pool entries among the registered tensors.

    Presence-gated: any layer name starting with
    :data:`CB_AUX_POOL_LAYER_PREFIX` is a connector-owned aux page pool
    served by a synthetic engine group (one group per block size).

    Args:
        kv_caches: Registered tensors keyed by layer name.

    Returns:
        ``(layer_name, tokens_per_block)`` per aux pool, in registration
        order; empty for models without one.

    Raises:
        ValueError: If a marker name's block-size suffix does not parse.
    """
    entries: list[tuple[str, int]] = []
    for name in kv_caches:
        if not name.startswith(CB_AUX_POOL_LAYER_PREFIX):
            continue
        suffix = name[len(CB_AUX_POOL_LAYER_PREFIX) :].split(".", 1)[0]
        try:
            tokens_per_block = int(suffix)
        except ValueError as exc:
            raise ValueError(
                f"aux pool layer name {name!r}: block-size suffix "
                f"{suffix!r} is not an integer"
            ) from exc
        if tokens_per_block <= 0:
            raise ValueError(
                f"aux pool layer name {name!r}: tokens_per_block must be "
                f"positive, got {tokens_per_block}"
            )
        entries.append((name, tokens_per_block))
    return entries


def _resolve_per_layer_recurrent(
    vllm_groups: Sequence[Any],
    layer_to_idx: Mapping[str, int],
    num_layers: int,
) -> list[bool]:
    """Resolve whether each registered KV tensor holds recurrent state pages.

    Args:
        vllm_groups: vLLM ``KVCacheGroupSpec`` instances.
        layer_to_idx: Layer name to registered tensor index mapping.
        num_layers: Number of registered KV tensors.

    Returns:
        A list of length ``num_layers``: ``True`` for Mamba/linear-attention
        layers in a snapshotting cache mode (see :func:`_is_cachable_mamba_spec`),
        ``False`` for attention layers.
    """
    per_layer_recurrent = [False] * num_layers
    for group in vllm_groups:
        spec = getattr(group, "kv_cache_spec", None)
        if spec is None:
            continue
        per_layer_specs = getattr(spec, "kv_cache_specs", None)
        for name in group.layer_names:
            layer_spec = per_layer_specs[name] if per_layer_specs else spec
            if _is_cachable_mamba_spec(layer_spec):
                per_layer_recurrent[layer_to_idx[name]] = True
    return per_layer_recurrent


def _merge_layer_recurrent(per_layer_recurrent: list[bool], indices: list[int]) -> bool:
    """Merge the per-layer recurrent-state flags of one LMCache group.

    Args:
        per_layer_recurrent: Recurrent-state flag per registered tensor index.
        indices: Registered tensor indices of the group's layers.

    Returns:
        The group's common flag.

    Raises:
        ValueError: If the group mixes recurrent and attention layers (vLLM
            groups layers by KV cache spec, so a mix indicates inconsistent
            metadata).
    """
    flags = {per_layer_recurrent[idx] for idx in indices}
    if len(flags) != 1:
        raise ValueError(
            f"Layers with indices {indices} mix recurrent-state and attention "
            "layers in one group. This should not happen because vLLM only "
            "groups layers with the same KV cache spec."
        )
    return flags.pop()


def _merge_layer_sw_sizes(per_layer_sw_size: list[int], indices: list[int]) -> int:
    """Merge the per-layer sliding window sizes of one LMCache group.

    Args:
        per_layer_sw_size: Sliding window size per registered tensor index.
        indices: Registered tensor indices of the group's layers.

    Returns:
        The group's common sliding window size in tokens, or ``-1`` when the
        layers are not sliding-window attention.

    Raises:
        ValueError: If the layers have different non-negative sliding window sizes.
    """
    sw_sizes = {per_layer_sw_size[idx] for idx in indices}
    if len(sw_sizes) != 1:
        raise ValueError(
            f"Layers with indices {indices} have different sliding window sizes "
            f"{sw_sizes}, but they are in the same group. This should "
            "not happen because vLLM should only group layers with the same "
            "KV cache spec, but got inconsistent metadata or registered tensors."
        )
    return sw_sizes.pop()


def _merge_v41_fields(
    per_layer_v41_fields: Mapping[int, Mapping[str, Any]],
    indices: Sequence[int],
) -> dict[str, Any]:
    """Merge per-layer V4.1 schema fields into one EngineGroupInfo-valued dict.

    One LMCache ``EngineGroupInfo`` carries a single value per V4.1 field, so
    the layers it wraps must agree on every one of the eight fields -- the
    caller's identity split has already separated different physical shapes, so
    in practice the leaves are uniform here. A disagreement (e.g. two leaves
    with identical tensor identity but different ``tokens_per_state``) means
    one value cannot represent the group, and fails loudly instead of silently
    broadcasting one layer's value.

    Args:
        per_layer_v41_fields: V4.1 field dict per registered tensor index
            (:func:`read_v41_spec_fields` result, keyed by layer index).
        indices: Registered tensor indices of one LMCache group.

    Returns:
        The merged eight-key dict, or ``{}`` when none of the group's layers
        have fields (aux pools, non-hybrid single group, old vLLM) so the
        caller falls through to ``EngineGroupInfo``'s contract defaults.

    Raises:
        ValueError: If the group's layers disagree on any V4.1 field.
    """
    present = [
        per_layer_v41_fields[idx] for idx in indices if idx in per_layer_v41_fields
    ]
    if not present:
        return {}
    merged: dict[str, Any] = {}
    for key in present[0]:
        values = {fields[key] for fields in present}
        if len(values) != 1:
            raise ValueError(
                f"Layers with indices {indices} have different V4.1 schema "
                f"field {key!r} values ({sorted(values, key=str)!r}), but they "
                "share one LMCache group. vLLM's UniformTypeKVCacheSpecs only "
                "requires a shared registry base class and block size, so a "
                "mixed-value group cannot be represented by one "
                "EngineGroupInfo."
            )
        merged[key] = next(iter(values))
    return merged


def create_engine_group_infos_from_vllm(
    kv_cache_config: Any,
    kv_caches: Mapping[str, Any],
    layout_hints: "LayoutHints | None" = None,
    dcp_size: int = 1,
) -> list[EngineGroupInfo]:
    """Build the LMCache engine group infos from vLLM metadata and registered tensors.

    This is the single entry point for the vLLM -> LMCache conversion. It reads
    the vLLM-specific fields (``KVCacheConfig.kv_cache_groups`` and
    ``KVCacheGroupSpec.layer_names`` from the v1 KV cache interface), maps each
    engine KV cache group's layer names to registered tensor indices, then
    splits the layers by physical transfer identity using the real tensors (via
    the shared :func:`lmcache.v1.kv_layer_groups.group_layers_by_identity`).
    vLLM-specific field access is intentionally confined to this function.

    V4.1 schema fields are resolved **per registered layer** from that layer's
    own leaf spec: a ``UniformTypeKVCacheSpecs`` wrapper may mix values across
    its leaves (``is_uniform_type`` only checks registry base class + block
    size; production G8 mixes ``tokens_per_state`` in ``{2, 1}``), so reading a
    single "first leaf" per engine group would broadcast one layer's truth to
    every identity split of the group. The per-layer fields are re-merged per
    LMCache group after the identity split (:func:`_merge_v41_fields`), so each
    ``EngineGroupInfo`` carries exactly its own layers' values.

    Args:
        kv_cache_config: vLLM ``KVCacheConfig`` describing the engine KV cache
            groups (or ``None`` / no groups, which yields a single-group spec).
        kv_caches: Registered KV tensors keyed by layer name, in registration
            order. Keys provide the layer-name -> tensor-index mapping; values
            are inspected for physical shape and dtype.
        layout_hints: Optional engine-provided layout hints forwarded to format
            detection (e.g. ``NHD``/``HND`` and compression metadata).
        dcp_size: Decode context parallel size.

    Note:
        Under DCP each attention group's ``tokens_per_block`` is scaled by
        ``dcp_size`` to stay in the scheduler's coordinate space; its ratio
        to the physical slot count is what sizes each rank's memory object.
        Mamba groups are replicated per rank and stay unscaled.

    Returns:
        The list of ``EngineGroupInfo`` in protocol order, i.e. the LMCache group
        order used by store/retrieve block IDs.
    """
    # First Party
    from lmcache.utils import EngineType
    from lmcache.v1.gpu_connector.utils import (
        normalize_and_discover_per_layer_formats,
    )
    from lmcache.v1.kv_layer_groups import (
        EXCLUDED_ENGINE_GROUP,
        group_layers_by_identity,
    )

    # vLLM-specific field access (confined to this function): map each
    # registered KV tensor to its vLLM engine KV cache group index. vLLM places
    # every registered layer in exactly one group; layers in different groups
    # have disjoint block-id spaces and must not share an LMCache group. ``None``
    # means a single (non-hybrid) group, i.e. every layer shares one block-id
    # space.
    per_layer_discoverable_kv_caches = list(kv_caches.values())
    layer_to_idx = {name: idx for idx, name in enumerate(kv_caches.keys())}
    vllm_groups = (
        getattr(kv_cache_config, "kv_cache_groups", ()) or ()
        if kv_cache_config is not None
        else ()
    )

    # Compressor circular-buffer ring groups (DeepSeek-V4.1 compressor state
    # scratch) are excluded from every LMCache group. One block per request
    # holds the raw ``[kv, score]`` rows of the token group still being
    # compressed; vLLM declares them ``prefix_cacheable=False`` /
    # ``uses_slot_mapping=False``, the ring slots are addressed as
    # ``block * capacity + pos % capacity`` (overwritten each step), and a
    # speculative write is rolled back on rejection - so the content is never a
    # deterministic function of the prompt prefix and can never be reused as
    # prefix KV. Their bytes are shape-indistinguishable from a compressed-MLA
    # pool, so *detection* cannot identify them; the vLLM spec is the only
    # authoritative signal, read here. Excluding the group up-front keeps its
    # tensors out of every LMCache kernel group, so they are never silently
    # stored as if they were request-reusable KV.
    from lmcache.v1.gpu_connector.kv_format.detectors.vllm import (
        is_circular_buffer_ring_spec,
    )

    ring_group_ids = frozenset(
        gid
        for gid, group in enumerate(vllm_groups)
        if is_circular_buffer_ring_spec(getattr(group, "kv_cache_spec", None))
    )
    if ring_group_ids:
        logger.warning(
            "Excluding %d compressor circular-buffer ring KV group(s) "
            "(engine group id(s) %s): per-request compressor scratch, never "
            "cached.",
            len(ring_group_ids),
            ", ".join(str(gid) for gid in sorted(ring_group_ids)),
        )
    # Preserve the original engine group ids (the engine's per-request block-id
    # lists are keyed by them), so only skip the ring groups.
    cacheable_vllm_groups = [
        (gid, group)
        for gid, group in enumerate(vllm_groups)
        if gid not in ring_group_ids
    ]

    layer_index_groups = [
        [layer_to_idx[name] for name in group.layer_names]
        for _, group in cacheable_vllm_groups
    ]

    # CacheBlend fused-aux (presence-gated): the pool joins detection as
    # its own group so its rank-3 layout is classified independently.
    aux_entries = cb_aux_pool_entries(kv_caches)
    layer_index_groups += [[layer_to_idx[name]] for name, _ in aux_entries]
    normalized_kv_caches, engine_kv_formats = normalize_and_discover_per_layer_formats(
        per_layer_discoverable_kv_caches,
        layer_index_groups,
        EngineType.VLLM,
        layout_hints,
    )
    num_layers = len(engine_kv_formats)
    # Layers absent from every engine group's ``layer_names`` are cross-layer
    # KV-sharing layers (e.g. google/gemma-4-E4B-it): vLLM aliases them to a
    # target owner's KV tensor, so the owner's group already covers them. Tag
    # them EXCLUDED_ENGINE_GROUP so they form no group of their own (a
    # wrong-block-size group would corrupt the per-group block-id counts).
    per_layer_group_idx: list[int] | None = None
    group_tokens_per_block: dict[int, int] = {}
    # V4.1 schema fields per registered layer, resolved from that layer's own
    # leaf spec. vLLM does *not* assert these are identical across one
    # ``UniformTypeKVCacheSpecs``'s leaves (``MLAAttentionSpec.merge`` does, but
    # the group-formation path ``UniformTypeKVCacheSpecs.is_uniform_type`` only
    # checks a shared registry base class + one block size, kv_cache_utils.py
    # :2013 / kv_cache_interface.py :1229-1240), so production G8 mixes
    # ``tokens_per_state`` in {2, 1} across its 8 MLA leaves. Resolving per
    # layer keeps that truth; each LMCache group re-merges exactly its own
    # layers' fields after the identity split (_merge_v41_fields).
    per_layer_v41_fields: dict[int, dict[str, Any]] = {}
    per_layer_sw_size = [-1] * num_layers
    per_layer_recurrent = [False] * num_layers
    if vllm_groups:
        per_layer_group_idx = [EXCLUDED_ENGINE_GROUP] * num_layers
        for engine_group_id, group in cacheable_vllm_groups:
            # The spec's block_size is the logical tokens covered by one of
            # this group's paged chunks (block IDs); the physical slot count
            # per chunk is discovered later from the registered tensors.
            # Under DCP the two diverge (see get_tokens_per_block).
            group_tokens_per_block[engine_group_id] = get_tokens_per_block(
                group.kv_cache_spec, dcp_size
            )
            # A UniformTypeKVCacheSpecs wrapper holds one leaf spec per layer;
            # read each layer's OWN leaf (never a single "first leaf") so a
            # mixed-tokens_per_state group keeps per-layer truth.
            per_layer_specs = getattr(group.kv_cache_spec, "kv_cache_specs", None)
            for name in group.layer_names:
                layer_idx = layer_to_idx[name]
                layer_spec = (
                    per_layer_specs[name] if per_layer_specs else group.kv_cache_spec
                )
                per_layer_v41_fields[layer_idx] = read_v41_spec_fields(layer_spec)
                per_layer_group_idx[layer_idx] = engine_group_id
        per_layer_sw_size = _resolve_per_layer_sw_sizes(
            vllm_groups, layer_to_idx, num_layers
        )
        per_layer_recurrent = _resolve_per_layer_recurrent(
            vllm_groups, layer_to_idx, num_layers
        )

    # Aux pools form synthetic engine groups after the vLLM groups (an
    # unassigned marker layer would fall to EXCLUDED_ENGINE_GROUP and never
    # store), bucketed by tokens_per_block: pools sharing a block size share
    # one engine group — and thereby one kernel group when their tensor
    # identities also match. Tags are 1-based per bucket.
    aux_group_tags: dict[int, int] = {}
    if aux_entries:
        if per_layer_group_idx is None:
            # Non-hybrid engine config: all real layers share group 0.
            per_layer_group_idx = [0] * num_layers
        next_group_id = len(vllm_groups) if vllm_groups else 1
        group_by_tpb: dict[int, int] = {}
        for name, tokens_per_block in aux_entries:
            group_id = group_by_tpb.get(tokens_per_block)
            if group_id is None:
                group_id = next_group_id
                next_group_id += 1
                group_by_tpb[tokens_per_block] = group_id
                group_tokens_per_block[group_id] = tokens_per_block
                aux_group_tags[group_id] = len(group_by_tpb)
            per_layer_group_idx[layer_to_idx[name]] = group_id

    # Within one vLLM engine group, layers can have different hidden dimensions
    # (e.g. a different head count), which require different GPU copy kernels.
    # ``group_layers_by_identity`` splits each engine group further by physical
    # transfer identity (kv_size, num_heads, head_size, block_size, dtype), so
    # every resulting LMCache group can be served by a single copy kernel. It is
    # the shared, engine-neutral primitive the server reuses to reproduce the
    # same grouping from the registered tensors.
    return [
        EngineGroupInfo(
            engine_group_id=identity.engine_group_idx,
            layer_indices=tuple(indices),
            tokens_per_block=group_tokens_per_block.get(identity.engine_group_idx, 0),
            sw_size_tokens=_merge_layer_sw_sizes(per_layer_sw_size, indices),
            # Connector-private pools bucket by tag under
            # --separate-object-groups, after the regular groups.
            extra_object_group_tag=aux_group_tags.get(identity.engine_group_idx, 0),
            recurrent_state=_merge_layer_recurrent(per_layer_recurrent, indices),
            # V4.1 schema fields (_merge_v41_fields), merged from the group's
            # own layers' leaf specs; groups without per-layer fields (aux
            # pools, non-hybrid single group, old vLLM) get no kwargs and fall
            # through to the struct's contract defaults.
            **_merge_v41_fields(per_layer_v41_fields, indices),
        )
        for identity, indices in group_layers_by_identity(
            normalized_kv_caches,
            engine_kv_formats,
            per_layer_group_idx,
        )
    ]
