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
from dataclasses import dataclass
from math import lcm
from typing import Protocol, cast

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


def engine_group_ids_per_view(
    groups: Sequence[EngineGroupInfo],
) -> tuple[int, ...]:
    """Return, per LMCache group, the engine group it draws block IDs from.

    This is the **only** bridge between the two index spaces of this module:

    * the LMCache protocol order (``EngineGroupInfo`` list position), used by
      :attr:`GroupBlockMask.group_idx` and the runtime kernel-group order; and
    * the engine group id (``EngineGroupInfo.engine_group_id``), used to key
      the engine-side block-id lists that :func:`slice_block_ids_per_group`
      slices and that :func:`expand_engine_block_ids` re-indexes.

    The mapping is many-to-one: several LMCache groups may share one
    ``engine_group_id`` when one engine group is split by transfer identity,
    so ``GroupBlockMask.group_idx`` must never be used as an engine group id.

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


def _engine_group_id_per_view(
    groups: Sequence[EngineGroupInfo],
) -> tuple[int, ...]:
    """Back-compat alias for :func:`engine_group_ids_per_view`.

    Args:
        groups: The LMCache KV groups, in protocol order.

    Returns:
        See :func:`engine_group_ids_per_view`.
    """
    return engine_group_ids_per_view(groups)


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
        for engine_group_id in engine_group_ids_per_view(groups)
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


# ------------------------------------------------------------------ #
#  Per-group block masks (engine-neutral)                             #
# ------------------------------------------------------------------ #
#
# V4.1 serves a model whose KV cache groups have *different* engine block
# sizes (production: 8 x 32, 1 x 64, 1 x 8), so a "mask" cannot be one
# boolean array shared by every group: each group's mask lives on its own
# engine-block grid (``tokens_per_block`` tokens per grid cell).  This
# section mirrors the Mooncake external-store coordinator's per-group masks
# (``store/coordinator.py``: ``load_mask`` / ``store_mask`` / ``lookup_mask``
# / ``_reachable_masks``), which in turn reuse the engine's own
# ``SingleTypeKVCacheManager.reachable_block_mask`` for each group's spec.
#
# LMCache's lookup doubles as prefetch (the hit length is *the* bound for the
# keys submitted and read-locked), unlike vLLM's "query -> converge -> trim"
# flow where a later shrink is harmless.  The one-shot-aligned-length helpers
# below therefore compute the conservative model-wide hit *before* anything is
# submitted, and never trim after the fact (see
# :func:`align_group_servable_length`).
#
# Every function here is duck-typed over group descriptors that expose
# ``tokens_per_block`` / ``sw_size_tokens`` / ``tokens_per_state`` /
# ``prefix_cacheable`` / ``recurrent_state`` — both
# :class:`EngineGroupInfo` and the runtime ``KernelGroupInfo`` satisfy them.
#
# Masks are keyed by protocol order (``GroupBlockMask.group_idx``), which is
# the **LMCache group** index space — not the engine group id space. The two
# differ whenever one engine group is split into several LMCache groups (V4.1
# production G8: four LMCache groups share ``engine_group_id=8``), so a mask's
# engine-side block-id list must be selected through
# :func:`engine_group_ids_per_view` (or :func:`group_engine_group_id` on a
# single descriptor), never by using ``group_idx`` as an engine group id.


class MaskGroup(Protocol):
    """Structural type for a per-group mask input (mypy-only structural match).

    Satisfied structurally by both :class:`EngineGroupInfo` (the protocol-
    visible group view) and the runtime ``KernelGroupInfo`` (dataclass
    instance attributes), so the mask computation is engine-neutral and
    shared by the connector and the server.  Not intended for ``isinstance``
    checks.

    The engine group a descriptor draws block IDs from is deliberately *not*
    part of this structural match: the two classes name it differently
    (``EngineGroupInfo.engine_group_id`` vs ``KernelGroupInfo.engine_group_idx``).
    :func:`group_engine_group_id` reads either name and is what
    :class:`GroupBlockMask` records, so mask consumers never have to guess.
    """

    tokens_per_block: int
    sw_size_tokens: int
    tokens_per_state: int
    prefix_cacheable: bool
    recurrent_state: bool


def group_engine_group_id(group: MaskGroup) -> int | None:
    """Return the engine group id a mask-group descriptor draws block IDs from.

    The engine group id selects which engine-side block-id list (the space of
    :func:`slice_block_ids_per_group` and of the ``STORE``/``RETRIEVE``
    protocol *before* :func:`expand_engine_block_ids`) the group's blocks index
    into.  It is a *different* index space from the LMCache protocol order that
    :attr:`GroupBlockMask.group_idx` uses, and several LMCache groups may share
    one engine group id.

    Descriptors name it differently: :class:`EngineGroupInfo` exposes
    ``engine_group_id`` and the runtime ``KernelGroupInfo`` exposes
    ``engine_group_idx``; either is accepted.

    Args:
        group: One mask group descriptor.

    Returns:
        The engine group id, or ``None`` when the descriptor reports neither
        attribute (an engine without group metadata — the single non-hybrid
        engine group ``0``).  ``None`` is never a valid engine group id, so
        callers must treat it as "block IDs are not group-addressable" rather
        than falling back to ``group_idx``.
    """
    value = getattr(group, "engine_group_id", None)
    if value is None:
        value = getattr(group, "engine_group_idx", None)
    if value is None:
        return None
    return int(value)


@dataclass(frozen=True, slots=True)
class GroupBlockMask:
    """One LMCache group's per-block reachability over a token range.

    Covers engine blocks ``[start_block, end_block)`` of this group, i.e.
    tokens ``[start_block * tokens_per_block, end_block * tokens_per_block)``.
    ``blocks`` is a bool per block (``True`` = reachable / to store / to
    load), or ``None`` as the *all-True sentinel* — mirroring Mooncake's
    ``reachable_block_mask`` returning ``None`` for full attention, so dense
    groups never materialize a long all-True array.
    """

    group_idx: int
    """LMCache group index in protocol order (the ``EngineGroupInfo``
    list position). This is the index into the mask list itself, **not** an
    engine group id: :attr:`engine_group_id` names the engine-side block-id
    list this mask's block indices belong to. The two coincide only when every
    LMCache group has its own engine group (no split)."""
    tokens_per_block: int
    """Logical tokens covered by one engine block of this group (its own
    block grid)."""
    start_block: int
    """First covered engine block index, inclusive."""
    end_block: int
    """Last covered engine block index, exclusive."""
    blocks: tuple[bool, ...] | None
    """Per-block flag over ``[start_block, end_block)``; ``None`` = all True
    (full-attention sentinel)."""
    engine_group_id: int | None = None
    """Engine group id whose block-id list these block indices index into
    (the :func:`slice_block_ids_per_group` space). A mask consumer that has
    engine-side block tables indexed by engine group id must select
    ``engine_side_block_ids[self.engine_group_id]`` — using
    ``self.group_idx`` is only correct when the two spaces happen to be equal.
    ``None`` when the source descriptor reports no engine group id (single
    non-hybrid group); the default keeps positional construction
    back-compatible."""

    @property
    def block_count(self) -> int:
        """Number of covered blocks (``end_block - start_block``)."""
        return self.end_block - self.start_block

    def is_reachable(self, block_index: int) -> bool:
        """Whether one engine block of this group is reachable.

        Args:
            block_index: An engine block index within
                ``[start_block, end_block)``.

        Returns:
            ``True`` if the block is reachable; ``True`` for every block when
            ``blocks`` is the ``None`` (all-True) sentinel.
        """
        if self.blocks is None:
            return True
        return self.blocks[block_index - self.start_block]


def lcm_block_tokens(groups: Sequence[MaskGroup]) -> int:
    """Cross-group block-size common multiple (Mooncake's ``lcm_block_size``).

    vLLM's coordinator asserts every ``prefix_cacheable`` group's block size
    divides the scheduler block size and uses that as the single mask
    alignment (``store/coordinator.py:100-104,112``).  LMCache computes the
    least common multiple directly over the groups that participate in prefix
    caching.

    Args:
        groups: The LMCache KV groups, in protocol order.

    Returns:
        ``lcm(tokens_per_block)`` over ``prefix_cacheable`` groups with a
        positive reported ``tokens_per_block``.

    Raises:
        ValueError: If no participating group reports a positive
            ``tokens_per_block`` (a mask requires a concrete block grid).
    """
    block_sizes = [
        group.tokens_per_block
        for group in groups
        if group.prefix_cacheable and group.tokens_per_block > 0
    ]
    if not block_sizes:
        raise ValueError(
            "cannot compute the cross-group block lcm: no prefix_cacheable "
            "group reports a positive tokens_per_block"
        )
    result = 1
    for block_size in block_sizes:
        result = lcm(result, block_size)
    return result


def align_lookup_length(groups: Sequence[MaskGroup], length: int) -> int:
    """Floor a candidate length to the cross-group mask alignment.

    Mirrors Mooncake's ``align_lookup_length``
    (``store/coordinator.py:121-127``): only lengths aligned to the cross-group
    block common multiple are valid hit/store boundaries.

    Args:
        groups: The LMCache KV groups, in protocol order.
        length: Candidate length in tokens.

    Returns:
        ``length // alignment * alignment`` where ``alignment`` is
        :func:`lcm_block_tokens` of ``groups``.
    """
    alignment = lcm_block_tokens(groups)
    return length // alignment * alignment


def _window_tail_mask(
    block_tokens: int,
    window_tokens: int,
    start_block: int,
    end_block: int,
    segment_tokens: int,
    *,
    use_eagle: bool = False,
) -> list[bool] | None:
    """Sliding-window per-block reachability.

    Faithful translation of vLLM's
    ``SlidingWindowManager.reachable_block_mask`` with ``retention_interval
    = None`` (``single_type_kv_cache_manager.py:1067-1091``): a block is
    reachable iff it can be the tail of a hit at some *segment-aligned*
    boundary.  Concretely, only the last ``need`` blocks before each segment
    boundary can serve an aligned hit, so the interior of a segment is masked
    out.

    Args:
        block_tokens: This group's ``tokens_per_block`` (its block grid).
        window_tokens: The group's sliding window in tokens (0 for a
            one-block mamba-style window).
        start_block: First covered block index, inclusive.
        end_block: Last covered block index, exclusive.
        segment_tokens: Commit-boundary spacing in tokens (LMCache chunk size
            or the cross-group lcm).
        use_eagle: Whether to reserve one extra block for the EAGLE peek
            (Mooncake's ``shift`` / ``need + 1``).

    Returns:
        A per-block bool list of length ``end_block - start_block``, or
        ``None`` when every block is reachable (window covers the whole
        segment, or the segment spacing is not a multiple of ``block_tokens``
        so the mask cannot be exact and we fall back to dense — never drop).

    Raises:
        ValueError: If ``block_tokens`` or ``segment_tokens`` is not positive,
            or the block range is inverted.
    """
    if block_tokens <= 0:
        raise ValueError(f"block_tokens must be positive, got {block_tokens}")
    if segment_tokens <= 0:
        raise ValueError(f"segment_tokens must be positive, got {segment_tokens}")
    if end_block < start_block:
        raise ValueError(f"block range [{start_block}, {end_block}) is inverted")
    if segment_tokens % block_tokens != 0:
        # Sub-block segment: the mask is block-granular and cannot represent
        # the alignment exactly. Dense fallback never drops a reachable block.
        return None
    per_segment = segment_tokens // block_tokens
    # vLLM's ``_contiguous_blocks_for_hit`` (single_type_kv_cache_manager.py:
    # 956-966) is ``cdiv(window_size - 1, block_size)`` -- NOT ``ceil(w / b)``
    # -- plus one when EAGLE peeks a block past the matched run.  The offset
    # matters when ``window % block == 1`` (e.g. block 32 / window 33: vLLM
    # only needs 1 contiguous block, a naive ceil needs 2); mirroring it keeps
    # LMCache's retained block set identical to the local prefix cache's.
    need = -((window_tokens - 1) // -block_tokens)
    if use_eagle:
        # Reserve one extra contiguous block for the EAGLE peek.
        need += 1
    if need >= per_segment:
        # The window (plus peek) spans the whole segment: every block can be
        # the tail of some aligned hit.
        return None
    shift = 1 if use_eagle else 0
    return [
        # ``i >= shift`` mirrors vLLM: with an EAGLE peek the block at
        # i == 0 is deliberately excluded (the peek shifts the matched
        # run's right edge one block past the boundary).
        i >= shift and (i - shift) % per_segment >= per_segment - need
        for i in range(start_block, end_block)
    ]


def _group_block_range(
    tokens_per_block: int, token_len: int, start_token: int
) -> tuple[int, int]:
    """Block range ``[start_block, end_block)`` of one group for a token span.

    Args:
        tokens_per_block: The group's block grid (must be positive).
        token_len: Exclusive end of the token range.
        start_token: Inclusive start of the token range.

    Returns:
        ``(start_block, end_block)`` with ``start_block =
        min(end_block, max(0, ceil(start_token / tpb)))`` and ``end_block =
        token_len // tpb`` — mirroring Mooncake's
        ``_reachable_masks`` (``store/coordinator.py:290-293``).

    Raises:
        ValueError: If ``tokens_per_block`` is not positive or ``token_len``
            is negative.
    """
    if tokens_per_block <= 0:
        raise ValueError(f"tokens_per_block must be positive, got {tokens_per_block}")
    if token_len < 0:
        raise ValueError(f"token_len must be >= 0, got {token_len}")
    end_block = token_len // tokens_per_block
    start_block = min(
        end_block, max(0, (start_token + tokens_per_block - 1) // tokens_per_block)
    )
    return start_block, end_block


def _mask_for_group(
    group: MaskGroup,
    group_idx: int,
    token_len: int,
    *,
    start_token: int,
    segment_tokens: int,
    exclude_non_cacheable: bool,
    exclude_recurrent: bool,
    use_eagle: bool,
) -> GroupBlockMask:
    """Build one group's :class:`GroupBlockMask` over the token range.

    Applies the exclusion rules first (Mooncake's ``_verify_and_split`` skip
    for non-cacheable groups and the ``store_mask`` mamba exclude), then the
    sliding-window tail mask, then the full-attention all-True sentinel.

    Args:
        group: One group descriptor (``EngineGroupInfo`` or ``KernelGroupInfo``).
        group_idx: LMCache group index (protocol order).
        token_len: Exclusive end of the token range.
        start_token: Inclusive start of the token range.
        segment_tokens: Commit-boundary spacing for the window tail mask.
        exclude_non_cacheable: Drop ``prefix_cacheable == False`` groups
            (always all-False).
        exclude_recurrent: Drop ``recurrent_state`` groups (store-side mamba
            exclude; all-False).
        use_eagle: Reserve the EAGLE peek block in the window tail mask.

    Returns:
        The group's block mask.  Excluded groups yield an explicit all-False
        mask so callers can round-trip the group count.  The mask records the
        descriptor's engine group id (see :func:`group_engine_group_id`) so
        consumers pair it with the right engine-side block-id list.
    """
    tpb = group.tokens_per_block
    start_block, end_block = _group_block_range(tpb, token_len, start_token)
    block_count = end_block - start_block
    engine_group_id = group_engine_group_id(group)

    if exclude_non_cacheable and not group.prefix_cacheable:
        return GroupBlockMask(
            group_idx,
            tpb,
            start_block,
            end_block,
            (False,) * block_count,
            engine_group_id=engine_group_id,
        )
    if exclude_recurrent and group.recurrent_state:
        return GroupBlockMask(
            group_idx,
            tpb,
            start_block,
            end_block,
            (False,) * block_count,
            engine_group_id=engine_group_id,
        )
    window_tokens = group.sw_size_tokens
    if window_tokens >= 0:
        blocks = _window_tail_mask(
            tpb,
            window_tokens,
            start_block,
            end_block,
            segment_tokens,
            use_eagle=use_eagle,
        )
        return GroupBlockMask(
            group_idx,
            tpb,
            start_block,
            end_block,
            None if blocks is None else tuple(blocks),
            engine_group_id=engine_group_id,
        )
    # Full attention: all-True sentinel.
    return GroupBlockMask(
        group_idx,
        tpb,
        start_block,
        end_block,
        None,
        engine_group_id=engine_group_id,
    )


def compute_group_store_masks(
    groups: Sequence[MaskGroup],
    token_len: int,
    start_token: int = 0,
    *,
    segment_tokens: int,
    exclude_non_cacheable: bool = True,
    exclude_recurrent: bool = False,
    use_eagle: bool = False,
) -> list[GroupBlockMask]:
    """Per-group store masks for the suffix ``[start_token, token_len)``.

    Mirrors Mooncake's ``store_mask`` (``store/coordinator.py:221-252``):
    which blocks *after* ``start_token`` each group should persist so a future
    aligned cache hit can consume them.  The all-True sentinel means "store
    every block of the range" (full attention).

    Args:
        groups: The LMCache KV groups, in protocol order.
        token_len: Exclusive end of the store range in tokens.
        start_token: Inclusive start of the store range in tokens (suffix
            store).
        segment_tokens: Commit-boundary spacing (LMCache chunk size for the
            transfer downsample grid, or the cross-group lcm).
        exclude_non_cacheable: Drop ``prefix_cacheable == False`` groups
            (per-request scratch is never shareable; Mooncake skips them in
            ``_verify_and_split_kv_cache_groups``).
        exclude_recurrent: Drop ``recurrent_state`` groups (Mooncake's
            ``exclude_mamba=True``; all-False).  LMCache transfers recurrent
            blocks from the engine's live block table each step, so the
            connector artifact Mooncake guards against does not apply — keep
            this False unless a caller wants the store-side exclusion.
        use_eagle: Reserve the EAGLE peek block in the window tail mask.

    Returns:
        One :class:`GroupBlockMask` per LMCache group, in protocol order; each
        carries the descriptor's ``engine_group_id`` (see
        :func:`group_engine_group_id`) so its block indices can be paired with
        the right engine-side block-id list.

    Raises:
        ValueError: If ``token_len`` is not aligned to the cross-group lcm of
            the participating groups (see :func:`lcm_block_tokens`), or any
            group lacks a positive ``tokens_per_block``.
    """
    if token_len < 0:
        raise ValueError(f"token_len must be >= 0, got {token_len}")
    # A store range that crosses a group-boundary misaligns the mask grid;
    # fail closed instead of silently dropping blocks.
    alignment = lcm_block_tokens(groups)
    if token_len % alignment != 0:
        raise ValueError(
            f"token_len {token_len} must be a multiple of the cross-group "
            f"block lcm {alignment} for a per-group store mask"
        )
    masks: list[GroupBlockMask] = []
    for idx, group in enumerate(groups):
        masks.append(
            _mask_for_group(
                group,
                idx,
                token_len,
                start_token=start_token,
                segment_tokens=segment_tokens,
                exclude_non_cacheable=exclude_non_cacheable,
                exclude_recurrent=exclude_recurrent,
                use_eagle=use_eagle,
            )
        )
    return masks


def compute_group_lookup_masks(
    groups: Sequence[MaskGroup],
    token_len: int,
    *,
    segment_tokens: int | None = None,
    exclude_non_cacheable: bool = True,
) -> list[GroupBlockMask]:
    """Per-group lookup masks (valid aligned hit boundaries).

    Mirrors Mooncake's ``lookup_mask`` (``store/coordinator.py:254-269``):
    which per-group blocks can be hit/store boundaries under each group's
    rule.  ``None`` (all-True) means every block of the range is a candidate.

    Args:
        groups: The LMCache KV groups, in protocol order.
        token_len: Length in tokens to cover (must be aligned to the
            cross-group lcm).
        segment_tokens: Commit-boundary spacing; defaults to the cross-group
            lcm (Mooncake passes ``lcm_block_size`` as
            ``alignment_tokens`` to ``reachable_block_mask``).
        exclude_non_cacheable: Drop ``prefix_cacheable == False`` groups.

    Returns:
        One :class:`GroupBlockMask` per LMCache group, in protocol order; each
        carries the descriptor's ``engine_group_id`` (see
        :func:`group_engine_group_id`) so its block indices can be paired with
        the right engine-side block-id list.

    Raises:
        ValueError: If ``token_len`` is not aligned to the cross-group lcm.
    """
    alignment = lcm_block_tokens(groups)
    if token_len % alignment != 0:
        raise ValueError(
            f"token_len {token_len} must be a multiple of the cross-group "
            f"block lcm {alignment} for a per-group lookup mask"
        )
    used_segment = segment_tokens if segment_tokens is not None else alignment
    masks: list[GroupBlockMask] = []
    for idx, group in enumerate(groups):
        masks.append(
            _mask_for_group(
                group,
                idx,
                token_len,
                start_token=0,
                segment_tokens=used_segment,
                exclude_non_cacheable=exclude_non_cacheable,
                exclude_recurrent=False,
                use_eagle=False,
            )
        )
    return masks


def compute_group_load_masks(
    groups: Sequence[MaskGroup],
    hit_length_tokens: int,
    *,
    segment_tokens: int,
    exclude_non_cacheable: bool = True,
) -> list[GroupBlockMask]:
    """Per-group load masks for a model-wide hit of ``hit_length_tokens``.

    Mirrors Mooncake's ``load_mask`` (``store/coordinator.py:198-219``): the
    blocks each group must fill locally to serve the hit.  The presence
    decision (which chunks actually hit) is made separately by LMCache's
    folded lookup; this function only expands the *reachable block set* of the
    retained prefix ``[0, hit_length_tokens)`` for each group.

    Args:
        groups: The LMCache KV groups, in protocol order.
        hit_length_tokens: Model-wide hit length in tokens (already aligned).
        segment_tokens: Commit-boundary spacing for the window tail mask.
        exclude_non_cacheable: Drop ``prefix_cacheable == False`` groups.

    Returns:
        One :class:`GroupBlockMask` per LMCache group, in protocol order; each
        carries the descriptor's ``engine_group_id`` (see
        :func:`group_engine_group_id`) so its block indices can be paired with
        the right engine-side block-id list.

    Raises:
        ValueError: If ``hit_length_tokens`` is not aligned to the cross-group
            lcm.
    """
    alignment = lcm_block_tokens(groups)
    if hit_length_tokens % alignment != 0:
        raise ValueError(
            f"hit_length_tokens {hit_length_tokens} must be a multiple of the "
            f"cross-group block lcm {alignment}"
        )
    masks: list[GroupBlockMask] = []
    for idx, group in enumerate(groups):
        masks.append(
            _mask_for_group(
                group,
                idx,
                hit_length_tokens,
                start_token=0,
                segment_tokens=segment_tokens,
                exclude_non_cacheable=exclude_non_cacheable,
                exclude_recurrent=False,
                use_eagle=False,
            )
        )
    return masks


def align_group_servable_length(groups: Sequence[MaskGroup], candidate: int) -> int:
    """One-shot conservative model-wide hit length (never trimmed after).

    This is the heart of the "lookup doubles as prefetch" difference: vLLM
    converges the hit length group-by-group and *then* trims, which LMCache
    cannot do because a shrink after prefetch wastes bandwidth and misaligns
    the read-lock set.  Instead everything is floored here, before any key is
    submitted:

    1. each ``prefix_cacheable`` group with ``tokens_per_state > 1`` can only
       serve a prefix whose length is a multiple of its ``tokens_per_state``
       (a compressed state exists at ``(pos + 1) % tokens_per_state == 0``);
    2. the model-wide length is the minimum over those per-group reusable
       lengths, then floored again to the cross-group block lcm (Mooncake's
       ``align_lookup_length``).

    Args:
        groups: The LMCache KV groups, in protocol order.
        candidate: The raw candidate hit length in tokens (e.g. the length
            matched by the folded presence bitmaps).

    Returns:
        The conservative aligned hit length, ``<= candidate``.

    Raises:
        ValueError: If no group participates in prefix caching.
    """
    lcm_tokens = lcm_block_tokens(groups)
    servable = candidate
    for group in groups:
        if not group.prefix_cacheable:
            continue
        tokens_per_state = group.tokens_per_state
        if tokens_per_state > 1:
            servable = min(servable, candidate // tokens_per_state * tokens_per_state)
    return servable // lcm_tokens * lcm_tokens
