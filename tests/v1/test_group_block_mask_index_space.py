# SPDX-License-Identifier: Apache-2.0
"""Regression: a mask's index space == the block-id space it actually uses.

``GroupBlockMask.group_idx`` is the *LMCache protocol order* — the position of a
group in the ``EngineGroupInfo`` list. The engine-side block-id lists (the input
of ``slice_block_ids_per_group`` / the ``STORE``/``RETRIEVE`` payload before
``expand_engine_block_ids``) are keyed by ``EngineGroupInfo.engine_group_id`` — a
**different** space. The mapping is many-to-one: one engine group split by
transfer identity becomes several LMCache groups that all share one
``engine_group_id`` (DeepSeek-V4.1 production: one engine group splits into four
LMCache groups).

These tests pin the public contract that fixes that trap:

* :func:`engine_group_ids_per_view` is the LMCache-group -> engine-group bridge;
* :class:`GroupBlockMask` records its ``engine_group_id`` (via public
  :func:`group_engine_group_id`) so a consumer selects the right engine-side
  block list instead of reusing ``group_idx`` as an engine group id.
"""

# Standard
from dataclasses import dataclass

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.kv_layer_groups import KVLayerGroupsManager
from lmcache.v1.multiprocess.group_view import (
    EngineGroupInfo,
    MaskGroup,
    compute_group_load_masks,
    compute_group_lookup_masks,
    compute_group_store_masks,
    engine_group_ids_per_view,
    expand_engine_block_ids,
    group_engine_group_id,
    num_engine_group_infos,
    num_engine_groups,
)
import lmcache.lmcache_native as lmcache_native


def _eg(
    engine_group_id: int,
    tokens_per_block: int,
    layer_indices: tuple[int, ...],
    *,
    sw_size_tokens: int = -1,
    tokens_per_state: int = 1,
) -> EngineGroupInfo:
    """One LMCache group descriptor with the mask-relevant fields set."""
    return EngineGroupInfo(
        engine_group_id=engine_group_id,
        layer_indices=layer_indices,
        tokens_per_block=tokens_per_block,
        sw_size_tokens=sw_size_tokens,
        tokens_per_state=tokens_per_state,
    )


def _split_engine_group_view() -> list[EngineGroupInfo]:
    """3 engine groups split into 5 LMCache groups, interleaved.

    Engine group 0 is split into LMCache groups 0 and 2; engine group 1 into
    LMCache groups 1 and 4; engine group 2 stays whole as LMCache group 3. The
    interleaving makes ``group_idx == engine_group_id`` false for every group
    but the first, which is the geometry that exposes the bug.
    """
    return [
        _eg(0, 64, (0,), tokens_per_state=2),
        _eg(1, 32, (1,)),
        _eg(0, 64, (2,)),
        _eg(2, 8, (3,)),
        _eg(1, 32, (4,)),
    ]


def _engine_side_block_ids() -> list[list[int]]:
    """Engine-side block lists, keyed by engine group id (0, 1, 2)."""
    return [[100, 101], [200, 201], [300, 301]]


def test_engine_group_ids_per_view_is_the_lmcache_to_engine_bridge() -> None:
    """The bridge returns one engine group id per LMCache group (many-to-one)."""
    groups = _split_engine_group_view()
    assert num_engine_groups(groups) == 3
    assert num_engine_group_infos(groups) == 5
    assert engine_group_ids_per_view(groups) == (0, 1, 0, 2, 1)


def test_all_mask_kinds_record_their_engine_group_id() -> None:
    """store/lookup/load masks all carry the descriptor's engine group id."""
    groups = _split_engine_group_view()
    engine_ids = list(engine_group_ids_per_view(groups))
    expected = [
        compute_group_store_masks(groups, 64, segment_tokens=64),
        compute_group_lookup_masks(groups, 64),
        compute_group_load_masks(groups, 64, segment_tokens=64),
    ]
    for masks in expected:
        assert len(masks) == len(groups)
        assert [mask.group_idx for mask in masks] == list(range(len(groups)))
        assert [mask.engine_group_id for mask in masks] == engine_ids


def test_masks_pair_with_expanded_block_ids_not_group_idx() -> None:
    """A mask's blocks index ``engine_side[engine_group_id]``, never ``group_idx``.

    This is the invariant under test: the mask list and the LMCache-ordered
    block-id list produced by ``expand_engine_block_ids`` are both in protocol
    order, and element ``i`` of each is the same group. Using ``group_idx`` as
    an engine group id breaks that pairing.
    """
    groups = _split_engine_group_view()
    engine_side = _engine_side_block_ids()
    expanded = expand_engine_block_ids(groups, engine_side)
    assert expanded == [[100, 101], [200, 201], [100, 101], [300, 301], [200, 201]]

    masks = compute_group_load_masks(groups, 64, segment_tokens=64)
    for group_idx, mask in enumerate(masks):
        assert mask.engine_group_id is not None
        # The block list this mask must index == the LMCache-order expansion.
        assert expanded[group_idx] == engine_side[mask.engine_group_id]

    # Concretely wrong before the fix: LMCache group 2 draws engine group 0's
    # blocks (100, 101) on a 64-token grid, but group_idx == 2 addresses engine
    # group 2, whose blocks are (300, 301) on an 8-token grid.
    split_mask = masks[2]
    assert split_mask.engine_group_id == 0
    assert split_mask.tokens_per_block == 64
    assert engine_side[split_mask.group_idx] == [300, 301]
    assert expanded[split_mask.group_idx] == [100, 101]

    # LMCache groups 3 and 4 have no engine group at index 3/4 at all: reusing
    # group_idx is an out-of-range access, not a silently wrong list.
    assert len(engine_side) == 3
    with pytest.raises(IndexError):
        _ = engine_side[masks[3].group_idx]


def test_group_engine_group_id_reads_both_descriptor_names() -> None:
    """The helper accepts EngineGroupInfo and KernelGroupInfo spellings."""

    @dataclass
    class _RuntimeLike:
        engine_group_idx: int

    @dataclass
    class _NoEngineGroup:
        tokens_per_block: int = 32
        sw_size_tokens: int = -1
        tokens_per_state: int = 1
        prefix_cacheable: bool = True
        recurrent_state: bool = False

    assert group_engine_group_id(_eg(4, 64, (0,))) == 4
    assert group_engine_group_id(_RuntimeLike(engine_group_idx=2)) == 2
    # Never falls back to an index: "no engine group id" stays distinguishable.
    assert group_engine_group_id(_NoEngineGroup()) is None


def test_single_non_hybrid_group_back_compat() -> None:
    """An empty group view keeps the legacy single engine group 0 semantics."""
    assert engine_group_ids_per_view(()) == (0,)
    assert num_engine_groups(()) == 1
    assert num_engine_group_infos(()) == 1
    assert expand_engine_block_ids((), [7, 8, 9]) == [[7, 8, 9]]


def test_runtime_manager_masks_use_kernel_engine_group_idx() -> None:
    """Kernel-group masks carry ``KernelGroupInfo.engine_group_idx``.

    Two LMCache kernel groups come from one engine group (block 256, different
    physical state counts), so their masks share ``engine_group_id == 0`` while
    their ``group_idx`` values differ.
    """
    tensors = [
        torch.randn(2, 8, 64, 1, 64, dtype=torch.float16),
        torch.randn(2, 8, 2, 1, 64, dtype=torch.float16),
    ]
    manager = KVLayerGroupsManager(
        tensors,
        engine_kv_formats=[lmcache_native.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS] * 2,
        engine_group_infos=[
            EngineGroupInfo(0, (0,), tokens_per_block=256),
            EngineGroupInfo(0, (1,), tokens_per_block=256),
        ],
    )
    masks = manager.compute_group_load_masks(256, segment_tokens=256)
    assert len(masks) == 2
    for group_idx, mask in enumerate(masks):
        assert mask.group_idx == group_idx
        kernel_group: MaskGroup = manager.kernel_groups[group_idx]
        assert mask.engine_group_id == group_engine_group_id(kernel_group)
    # Both LMCache groups read engine group 0's block list; group_idx is not it.
    assert {mask.engine_group_id for mask in masks} == {0}
    assert [mask.group_idx for mask in masks] == [0, 1]
