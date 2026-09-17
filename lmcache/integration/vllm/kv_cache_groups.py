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
    """Return the leaf spec behind a ``UniformTypeKVCacheSpecs`` wrapper.

    The wrapper holds one spec per layer; ``UniformTypeKVCacheSpecs`` only
    groups same-typed layers (vLLM ``is_uniform_type`` requires one shared
    registry base class *and* one block size), so the first leaf is
    representative for the layer-type-derived fields this module reads.
    Every other spec is its own leaf.
    """
    inner = getattr(spec, "kv_cache_specs", None)
    if isinstance(inner, dict) and inner:
        return next(iter(inner.values()))
    return spec


def _is_attention_spec(spec: Any) -> bool:
    """Return whether the KV cache spec is a vLLM attention spec.

    Checked by class name so this module stays importable without vLLM.
    ``UniformTypeKVCacheSpecs`` is unwrapped first (same-typed layers, one
    leaf suffices); it does not derive from ``AttentionSpec`` itself.
    """
    spec = _first_leaf_spec(spec)
    return any(cls.__name__ == "AttentionSpec" for cls in type(spec).__mro__)


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
            wrapper; the wrapper is unwrapped to its first leaf).

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


def read_v41_spec_fields(spec: Any) -> dict[str, Any]:
    """Fault-tolerant read of the vLLM V4.1 spec fields LMCache carries.

    Returns exactly the eight ``EngineGroupInfo`` tail fields added for the
    V4.1 schema, with the contract defaults wherever the spec -- or the whole
    vLLM version -- does not declare them, so the caller can spread the result
    as ``EngineGroupInfo(**fields)`` without losing msgspec wire compatibility:

    ``tokens_per_state=1``, ``cache_role="sparse"``,
    ``state_content_bytes=None``, ``block_stride_alignment=None``,
    ``is_index_group_leader=False``, ``prefix_cacheable=True``,
    ``page_size_padded=None``, ``num_head_slots=None``.

    Every read goes through ``getattr`` with a default (never bare attribute
    access), and a ``UniformTypeKVCacheSpecs`` wrapper is unwrapped to its
    first leaf, so neither a nightly/older field layout nor a missing
    attribute can raise ``AttributeError``. ``cache_role`` is normalized from
    the ``SparseCacheRole`` enum to its ``str`` value; ``prefix_cacheable``
    prefers the wrapper's property (which ANDs its leaves) when present.

    Args:
        spec: A vLLM KV cache spec (leaf or ``UniformTypeKVCacheSpecs``).

    Returns:
        An eight-key dict safe to pass as ``EngineGroupInfo(**fields)``.
    """
    leaf = _first_leaf_spec(spec)
    cache_role = getattr(leaf, "cache_role", None)
    if cache_role is not None:
        cache_role = getattr(cache_role, "value", cache_role)
    prefix_cacheable = getattr(spec, "prefix_cacheable", None)
    if prefix_cacheable is None:
        prefix_cacheable = getattr(leaf, "prefix_cacheable", True)
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
    tokens, so both fail closed instead of corrupting later arithmetic.

    Attention blocks span ``block_size * dcp_size`` tokens under DCP
    (vLLM's ``resolve_kv_cache_block_sizes`` rule); recurrent state is
    replicated, not sharded, and stays at ``block_size``.

    Args:
        kv_cache_spec: vLLM KV cache spec (leaf or ``UniformTypeKVCacheSpecs``).
        dcp_size: Decode context parallel size.

    Returns:
        Logical tokens per engine block id.

    Raises:
        ValueError: If the spec declares a fractional ``tokens_per_state``, or
            an integer one that does not divide ``block_size``.
    """
    block_size = kv_cache_spec.block_size
    tokens_per_state = resolve_tokens_per_state(kv_cache_spec)
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

    layer_index_groups = [
        [layer_to_idx[name] for name in group.layer_names] for group in vllm_groups
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
    # V4.1 schema fields per engine group, keyed by engine group id; the
    # vLLM merge asserts these are identical across a group's layers
    # (``MLAAttentionSpec.merge``), so one read per engine group suffices.
    per_engine_v41_fields: dict[int, dict[str, Any]] = {}
    per_layer_sw_size = [-1] * num_layers
    per_layer_recurrent = [False] * num_layers
    if vllm_groups:
        per_layer_group_idx = [EXCLUDED_ENGINE_GROUP] * num_layers
        for engine_group_id, group in enumerate(vllm_groups):
            # The spec's block_size is the logical tokens covered by one of
            # this group's paged chunks (block IDs); the physical slot count
            # per chunk is discovered later from the registered tensors.
            # Under DCP the two diverge (see get_tokens_per_block).
            group_tokens_per_block[engine_group_id] = get_tokens_per_block(
                group.kv_cache_spec, dcp_size
            )
            per_engine_v41_fields[engine_group_id] = read_v41_spec_fields(
                group.kv_cache_spec
            )
            for name in group.layer_names:
                per_layer_group_idx[layer_to_idx[name]] = engine_group_id
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
            # V4.1 schema fields (read_v41_spec_fields) keyed by engine group
            # id; engine groups without a spec (aux pools, non-hybrid single
            # group, old vLLM) get no kwargs and fall through to the struct's
            # contract defaults.
            **per_engine_v41_fields.get(identity.engine_group_idx, {}),
        )
        for identity, indices in group_layers_by_identity(
            normalized_kv_caches,
            engine_kv_formats,
            per_layer_group_idx,
        )
    ]
