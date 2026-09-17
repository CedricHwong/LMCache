# SPDX-License-Identifier: Apache-2.0
"""LMCache's engine-neutral description of a serving engine's KV cache groups.

An *engine group* is one distinct paged-block address space exposed by the
serving engine (e.g. one of vLLM's hybrid KV cache groups): block IDs are only
meaningful within a single group, and layers from different groups must never be
merged into one LMCache KV group. Engine group ids are assumed dense and
consecutive starting from 0.

LMCache's neutral KV cache spec is simply a ``list[EngineGroupInfo]`` (passed as
a ``Sequence[EngineGroupInfo]`` where only order matters). The group order is
the protocol-visible LMCache group order used by store/retrieve block IDs. An
empty list means a single non-hybrid group (the default for engines that do not
report KV cache group metadata). Engine-specific conversion belongs in the
corresponding ``lmcache.integration.<engine>`` package, not here.
"""

# Standard
from collections.abc import Mapping, Sequence
from typing import cast

# Third Party
import msgspec


class EngineGroupInfo(msgspec.Struct, frozen=True):
    """One LMCache KV group: layers of one engine group that share a copy kernel.

    Carries the layer indices and which engine group they belong to. Several
    ``EngineGroupInfo`` instances may share the same ``engine_group_id`` when
    one engine group is split by physical transfer identity (e.g. differing
    hidden dims). A ``list[EngineGroupInfo]`` is carried verbatim in the
    ``REGISTER_KV_CACHE`` IPC payload; the message queue handles
    encoding/decoding.
    """

    engine_group_id: int
    """Engine group these layers live in (one distinct paged-block address
    space). Selects which request block-id list applies. Dense from 0."""

    layer_indices: tuple[int, ...] = ()
    """Registered KV tensor indices assigned to this group."""

    tokens_per_block: int = 0
    """Logical tokens covered by one paged chunk (one engine block ID) of
    this engine group, as declared by the engine's KV cache spec
    (``kv_cache_spec.block_size`` for vLLM). ``0`` means the engine did not
    report it; consumers then fall back to the physical slot count detected
    from the registered tensors (i.e. the group is treated as
    uncompressed)."""

    sw_size_tokens: int = -1
    """Sliding window size in tokens for the layers of this group.
    ``-1`` means the layers are not sliding-window attention."""

    extra_object_group_tag: int = 0
    """Connector-private extra-group tag under ``--separate-object-groups``:
    ``0`` = a regular group; ``> 0`` = an extra group (e.g. the CacheBlend
    fused-aux pool) bucketed by tag, after the regular groups. Defaulted
    field: wire-compatible with old payloads."""

    recurrent_state: bool = False
    """Pages hold recurrent state snapshots (Mamba/GDN) rather than attention
    KV; the one-block window reflects restore semantics and blend full-window
    forcing must not widen it. Defaulted field: wire-compatible."""

    # ------------------------------------------------------------------
    # DeepSeek V4.1 metadata (phase-1 freeze contract: appended LAST, all
    # defaulted so msgspec.Struct IPC stays wire-compatible both ways).
    #   * new consumers reading old payloads: missing keys fall back to
    #     these defaults (object/map encoding);
    #   * old consumers reading new payloads: without forbid_unknown_fields
    #     msgspec ignores the extra keys.
    #   * appending (never inserting) keeps every positional construction
    #     site (tests, qringbuffer, bench client) positionally valid, and
    #     keeps the "fields with defaults follow fields without" msgspec rule.
    # ------------------------------------------------------------------
    tokens_per_state: int = 1
    """Tokens covered by one stored state for this group — DeepSeek V4.1
    sparse-MLA ``tokens_per_state`` (nightly vLLM renamed
    ``MLAAttentionSpec.compress_ratio`` to this): ``0`` = sliding window,
    ``1`` / ``2`` = compressed. ``1`` = one token per state (also the fallback
    when the engine does not report it).

    This is a *state* (token) quantity and is distinct from
    ``tokens_per_block``: the latter is the logical token span of one paged
    chunk (one engine block ID) and is what :func:`slice_block_ids_per_group`
    consumes. ``tokens_per_state`` must never be fed into
    ``group_tokens_per_block`` in place of ``tokens_per_block`` — the two
    have different units and mixing them corrupts per-group block-id counts.
    Defaulted field: wire-compatible."""

    cache_role: str = "sparse"
    """Role of this group's cache in the V4.1 architecture: ``"sparse"``
    (regular sparse-KV group) or ``"indexer"`` (indexer/summary cache).
    Mirrors vLLM's ``SparseCacheRole`` (str enum, values ``"sparse"`` /
    ``"indexer"``). Defaults to ``"sparse"`` because the V4.1 indexer spec
    does *not* set ``cache_role`` (only V3.2 does); the default keeps
    indexer groups indistinguishable the same way the engine leaves them.
    Defaulted field: wire-compatible."""

    state_content_bytes: int | None = None
    """Per ``(head slot, stored state)`` cell size in bytes when the page is
    packed (nightly ``AttentionSpec.state_content_bytes``); ``None`` means
    dense K/V content and consumers fall back to tensor-detected sizes.
    Defaulted field: wire-compatible."""

    block_stride_alignment: int | None = None
    """Required byte alignment of the distance between consecutive blocks of
    this cache. In block-major layouts that distance is the whole block, so
    the allocator rounds the block up to it (e.g. DeepGEMM paged kernels
    address pages as ``base + page * stride`` and need it 512B-aligned).
    ``None`` = no alignment reported. Defaulted field: wire-compatible."""

    is_index_group_leader: bool = False
    """Whether this group is the leader that publishes index (summary) blocks
    shared by other groups. Defaults to ``False`` because the V4.1 indexer
    spec does *not* set it (only V3.2 does). Defaulted field:
    wire-compatible."""

    prefix_cacheable: bool = True
    """Whether this group's pages participate in prefix caching, mirroring
    nightly ``KVCacheSpec.prefix_cacheable``. ``True`` = cacheable (the
    engine default). Defaulted field: wire-compatible."""

    page_size_padded: int | None = None
    """Padded page size in bytes when the engine rounds the page up to an
    alignment (nightly ``AttentionSpec.page_size_padded``); ``None`` =
    unpadded page. Defaulted field: wire-compatible."""

    num_head_slots: int | None = None
    """Logical head count ``H`` of the ``[B, H, N, C]`` page when packing
    diverges from one slot per KV head (nightly
    ``AttentionSpec.num_head_slots``); ``None`` = one slot per KV head.
    Defaulted field: wire-compatible."""


def num_engine_groups(groups: Sequence[EngineGroupInfo]) -> int:
    """Return the number of engine groups (block-id lists per transfer request).

    Engine group ids are assumed dense and consecutive from 0.

    Args:
        groups: The LMCache KV groups, in protocol order.

    Returns:
        ``max(engine_group_id) + 1``, or ``1`` for an empty ``groups`` (single
        non-hybrid group).
    """
    if not groups:
        return 1
    return max(group.engine_group_id for group in groups) + 1


def num_engine_group_infos(groups: Sequence[EngineGroupInfo]) -> int:
    """Return the number of LMCache KV groups visible to transfer requests.

    Args:
        groups: The LMCache KV groups, in protocol order.

    Returns:
        ``len(groups)``, or ``1`` for an empty ``groups`` (single non-hybrid
        group).
    """
    if not groups:
        return 1
    return len(groups)


def _engine_group_id_per_view(
    groups: Sequence[EngineGroupInfo],
) -> tuple[int, ...]:
    """Return, per LMCache group, the engine group it draws block IDs from.

    Args:
        groups: The LMCache KV groups, in protocol order.

    Returns:
        A tuple whose length equals the number of LMCache groups (i.e.
        :func:`num_engine_group_infos`); element ``i`` is the engine group id
        that LMCache group ``i`` reads block IDs from. ``(0,)`` for an empty
        ``groups`` (single non-hybrid group).
    """
    if not groups:
        return (0,)
    return tuple(group.engine_group_id for group in groups)


def engine_group_layer_indices(
    groups: Sequence[EngineGroupInfo],
) -> list[list[int]]:
    """Return each engine group's layer indices, ordered by engine group id.

    Several ``EngineGroupInfo`` may share one ``engine_group_id``; their
    ``layer_indices`` are unioned into that group's entry.

    Args:
        groups: The LMCache KV groups, in protocol order.

    Returns:
        One sorted ``list[int]`` of layer indices per engine group, indexed by
        engine group id (dense from 0). Empty when ``groups`` is empty (a single
        non-hybrid group with no per-group split).
    """
    if not groups:
        return []
    num_groups = max(group.engine_group_id for group in groups) + 1
    per_group: list[list[int]] = [[] for _ in range(num_groups)]
    for group in groups:
        per_group[group.engine_group_id].extend(group.layer_indices)
    return [sorted(indices) for indices in per_group]


def expand_engine_block_ids(
    groups: Sequence[EngineGroupInfo],
    engine_side_block_ids: Sequence[Sequence[int]] | Sequence[int],
) -> list[list[int]]:
    """Expand the engine-side block id list to the list per LMCache kernel group.

    The serving engine reports block IDs per engine group. LMCache transfer
    requests are indexed by LMCache KV group, so each LMCache group reuses the
    block IDs from its source engine group.

    Args:
        groups: The LMCache KV groups, in protocol order.
        engine_side_block_ids: Block IDs indexed by engine group id, i.e. one
            inner ``list[int]`` per engine group (element ``g`` is engine group
            ``g``'s block list).

    Returns:
        Block IDs re-indexed by LMCache group order: one inner list per LMCache
        group, copied from that group's source engine group.
    """
    # Back-compat: older vLLM connectors emit a flat Sequence[int] for the
    # single (non-hybrid) engine group instead of one inner list per group.
    # Normalize both shapes to a concrete list[list[int]] so downstream
    # indexing is unambiguous for both runtime and mypy.
    if not engine_side_block_ids or isinstance(engine_side_block_ids[0], int):
        per_group: Sequence[Sequence[int]] = [
            cast("Sequence[int]", engine_side_block_ids)
        ]
    else:
        per_group = cast("Sequence[Sequence[int]]", engine_side_block_ids)
    return [
        list(per_group[engine_group_id])
        for engine_group_id in _engine_group_id_per_view(groups)
    ]


def slice_block_ids_per_group(
    allocated_block_ids: Mapping[int, Sequence[int]],
    group_tokens_per_block: Sequence[int],
    start_token_idx: int,
    end_token_idx: int,
) -> list[list[int]]:
    """Slice each engine group's block IDs for a token range.

    The range is given in tokens — the only unit shared by every engine
    group. A group whose paged chunks each cover ``tokens_per_block`` tokens
    holds one block ID per ``tokens_per_block`` tokens, so the range is
    divided by that group's ``tokens_per_block``. Example: over the same 256
    tokens, a tokens_per_block-64 group gets 4 IDs while a
    tokens_per_block-256 group gets 1.

    Args:
        allocated_block_ids: Block IDs keyed by engine group id; a missing group
            yields an empty list.
        group_tokens_per_block: Each group's tokens-per-paged-chunk, in
            engine-group order. Every value must be positive and divide both
            range endpoints.
        start_token_idx: Range start token index, inclusive.
        end_token_idx: Range end token index, exclusive.

    Returns:
        One block-ID list per engine group, in engine-group order.

    Raises:
        ValueError: If the range does not align to a group's chunk boundary.
    """
    sliced: list[list[int]] = []
    for engine_group_idx, tokens_per_block in enumerate(group_tokens_per_block):
        if start_token_idx % tokens_per_block != 0 or (
            end_token_idx % tokens_per_block != 0
        ):
            raise ValueError(
                f"token range [{start_token_idx}, {end_token_idx}) does not "
                f"align to group {engine_group_idx} tokens_per_block "
                f"{tokens_per_block}"
            )
        group_block_ids = allocated_block_ids.get(engine_group_idx, [])
        sliced.append(
            list(
                group_block_ids[
                    start_token_idx // tokens_per_block : end_token_idx
                    // tokens_per_block
                ]
            )
        )
    return sliced


def get_engine_group_indices(
    groups: Sequence[EngineGroupInfo],
    num_registered_layers: int,
) -> list[int] | None:
    """Return the engine group index for each registered KV tensor.

    Args:
        groups: The LMCache KV groups, in protocol order.
        num_registered_layers: Number of KV tensors registered with the server,
            i.e. the length of the per-layer mapping to produce.

    Returns:
        A list of length ``num_registered_layers`` mapping each registered
        tensor index to its engine group id, or ``None`` when there is no group
        metadata (empty ``groups`` or zero layers) so callers fall back to
        single-group behavior. Registered tensors not referenced by any group
        are marked with ``EXCLUDED_ENGINE_GROUP`` (cross-layer KV-sharing layers
        whose KV lives in their target owner's blocks); downstream grouping
        skips them.

    Raises:
        ValueError: If a group references a layer index outside
            ``[0, num_registered_layers)``.
    """
    # First Party
    from lmcache.v1.kv_layer_groups import EXCLUDED_ENGINE_GROUP

    if not groups or num_registered_layers == 0:
        return None

    # Default to "excluded": layers no group references are intentionally left
    # out of grouping (e.g. KV-sharing layers aliasing a target owner's cache).
    per_layer_engine_group_idx = [EXCLUDED_ENGINE_GROUP] * num_registered_layers

    for group in groups:
        for layer_idx in group.layer_indices:
            if layer_idx < 0 or layer_idx >= num_registered_layers:
                raise ValueError(
                    f"Layer index {layer_idx} is outside registered layer "
                    f"range [0, {num_registered_layers})"
                )
            per_layer_engine_group_idx[layer_idx] = group.engine_group_id

    return per_layer_engine_group_idx
