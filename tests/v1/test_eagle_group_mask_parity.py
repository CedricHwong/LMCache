# SPDX-License-Identifier: Apache-2.0
"""EAGLE block-drop parity between vLLM's scheduler and LMCache's mask grids.

vLLM's scheduler keys two decisions on ``use_eagle_block_drop()`` -- the
reachable-block set (``KVCacheManager(use_eagle=...)``) and the trustable
cacheable position (``last_cache_position -= block_size``). LMCache reproduces
the reachable-block set in :func:`_window_tail_mask`, but that helper's
``use_eagle`` flag was never reachable from the production entry points, so the
two sides disagreed about which blocks a hit may consume.

These tests pin the disagreement that the missing wiring caused, and pin its
*direction*: the EAGLE mask is a strict superset of the non-EAGLE one, so
LMCache under-claims. A future change that made it under-claim would be safe
but lossy; one that made it over-claim would let LMCache serve blocks the
engine considers pruned, so the superset assertion is the safety property.
"""

# Standard
import dataclasses

# Third Party
import msgspec.structs
import pytest
import torch

# First Party
from lmcache.integration.vllm.kv_cache_groups import (
    create_engine_group_infos_from_vllm,
)
from lmcache.v1.kv_layer_groups import KVLayerGroupsManager
from lmcache.v1.multiprocess.group_view import (
    EngineGroupInfo,
    compute_group_load_masks,
    compute_group_lookup_masks,
    compute_group_store_masks,
)

# Minimal vLLM doubles for ``create_engine_group_infos_from_vllm``. Unit tests
# must run without vLLM installed, so the group/config shapes are reproduced
# locally; ``is_eagle_group`` is a defaulted field because it is newer than the
# oldest supported vLLM build.
@dataclasses.dataclass
class MockKVCacheSpec:
    """A non-sliding attention spec: only ``block_size`` is read."""

    block_size: int


@dataclasses.dataclass
class MockKVCacheGroup:
    """One vLLM ``KVCacheGroupSpec`` double."""

    layer_names: list[str]
    kv_cache_spec: object
    is_eagle_group: bool = False


@dataclasses.dataclass
class MockKVCacheConfig:
    """One vLLM ``KVCacheConfig`` double."""

    kv_cache_groups: list[MockKVCacheGroup]


@dataclasses.dataclass
class MockKVCacheGroupNoEagle:
    """A group double predating ``is_eagle_group``."""

    layer_names: list[str]
    kv_cache_spec: object


# The production V4.1 kernel-group geometry: 8 SWA groups (block 32, window
# 128), two block-64 full-attention MLA groups (tps=2), one non-cacheable
# block-8 ring. Reached through the public mask entry points.
TOKENS = 1024
STORE_SEGMENT = 256


def _prod_groups() -> list[EngineGroupInfo]:
    """The production group layout as ``EngineGroupInfo`` descriptors."""
    groups = [
        EngineGroupInfo(
            engine_group_id=i,
            layer_indices=(),
            tokens_per_block=32,
            sw_size_tokens=128,
        )
        for i in range(8)
    ]
    groups.append(
        EngineGroupInfo(
            engine_group_id=8,
            layer_indices=(),
            tokens_per_block=64,
            tokens_per_state=2,
        )
    )
    groups.append(
        EngineGroupInfo(
            engine_group_id=8,
            layer_indices=(),
            tokens_per_block=64,
            tokens_per_state=2,
        )
    )
    groups.append(
        EngineGroupInfo(
            engine_group_id=9,
            layer_indices=(),
            tokens_per_block=8,
            prefix_cacheable=False,
        )
    )
    return groups


def _with_eagle(groups: list[EngineGroupInfo]) -> list[EngineGroupInfo]:
    """Return the same descriptors with ``is_eagle_group`` set."""
    return [msgspec.structs.replace(g, is_eagle_group=True) for g in groups]


def _reachable(mask) -> list[int]:
    """Return the reachable block indices of one mask, in protocol order."""
    return [
        mask.start_block + i
        for i in range(mask.block_count)
        if mask.is_reachable(mask.start_block + i)
    ]


def test_is_eagle_group_defaults_off_for_wire_compatibility():
    """A registration that predates the field must read as no-block-drop."""
    group = EngineGroupInfo(engine_group_id=0, layer_indices=(), tokens_per_block=32)
    assert group.is_eagle_group is False


@pytest.mark.parametrize(
    ("name", "entry"),
    [
        ("lookup", lambda gs: compute_group_lookup_masks(gs, TOKENS)),
        (
            "store",
            lambda gs: compute_group_store_masks(
                gs, TOKENS, segment_tokens=STORE_SEGMENT
            ),
        ),
        (
            "load",
            lambda gs: compute_group_load_masks(
                gs, TOKENS, segment_tokens=STORE_SEGMENT
            ),
        ),
    ],
)
def test_eagle_mask_never_drops_a_reachable_block(name, entry):
    """The EAGLE mask must be a superset of the non-EAGLE mask.

    This is the safety property: if EAGLE ever masked *out* a block the
    non-EAGLE grid kept, LMCache would claim a hit covering a block the engine
    considers pruned.
    """
    plain = entry(_prod_groups())
    eagle = entry(_with_eagle(_prod_groups()))

    assert len(plain) == len(eagle)
    for mask_plain, mask_eagle in zip(plain, eagle):
        dropped = set(_reachable(mask_plain)) - set(_reachable(mask_eagle))
        assert not dropped, (
            f"{name}: EAGLE mask dropped blocks {sorted(dropped)} that the "
            f"non-EAGLE grid considered reachable"
        )


def test_lookup_grid_is_dense_and_eagle_invariant():
    """The lookup grid degenerates to all-True, so EAGLE cannot change it.

    ``compute_group_lookup_masks`` defaults ``segment_tokens`` to the
    cross-group lcm (64), where the window ``need`` (4 blocks for block 32 /
    window 128) already spans every segment. The window mask therefore returns
    its dense "never drop" sentinel and the EAGLE shift has nothing to act on.
    """
    plain = compute_group_lookup_masks(_prod_groups(), TOKENS)
    eagle = compute_group_lookup_masks(_with_eagle(_prod_groups()), TOKENS)
    for mask_plain, mask_eagle in zip(plain, eagle):
        assert _reachable(mask_plain) == _reachable(mask_eagle)


def test_eagle_widens_the_store_grid_only_for_the_swa_groups():
    """Pin where the divergence lands and how big it is.

    Only the sliding-window groups (block 32, window 128) are affected; the
    full-attention block-64 groups use the all-True sentinel and the
    non-cacheable ring is excluded outright. At a 256-token commit segment the
    EAGLE grid adds the first block of each segment: 8, 16 and 24 over 1024
    tokens.
    """
    plain = compute_group_store_masks(
        _prod_groups(), TOKENS, segment_tokens=STORE_SEGMENT
    )
    eagle = compute_group_store_masks(
        _with_eagle(_prod_groups()), TOKENS, segment_tokens=STORE_SEGMENT
    )

    added_per_group: dict[int, list[int]] = {}
    for mask_plain, mask_eagle in zip(plain, eagle):
        added = sorted(
            set(_reachable(mask_eagle)) - set(_reachable(mask_plain))
        )
        added_per_group[mask_plain.group_idx] = added

    # Groups 0..7 are the SWA groups.
    for idx in range(8):
        assert added_per_group[idx] == [8, 16, 24], (
            f"SWA group {idx}: unexpected EAGLE divergence "
            f"{added_per_group[idx]}"
        )
    # Groups 8 and 9 are full-attention; group 10 is the excluded ring.
    for idx in (8, 9, 10):
        assert added_per_group[idx] == [], (
            f"group {idx} should be EAGLE-invariant, got {added_per_group[idx]}"
        )


def test_load_grid_matches_the_store_grid_divergence():
    """Store and load must widen together, or a hit would fill what was never stored."""
    plain_store = compute_group_store_masks(
        _prod_groups(), TOKENS, segment_tokens=STORE_SEGMENT
    )
    plain_load = compute_group_load_masks(
        _prod_groups(), TOKENS, segment_tokens=STORE_SEGMENT
    )
    eagle_store = compute_group_store_masks(
        _with_eagle(_prod_groups()), TOKENS, segment_tokens=STORE_SEGMENT
    )
    eagle_load = compute_group_load_masks(
        _with_eagle(_prod_groups()), TOKENS, segment_tokens=STORE_SEGMENT
    )
    for a, b, c, d in zip(plain_store, plain_load, eagle_store, eagle_load):
        assert set(_reachable(c)) - set(_reachable(a)) == set(_reachable(d)) - set(
            _reachable(b)
        )


def test_manager_propagates_eagle_into_kernel_groups():
    """``KVLayerGroupsManager`` must carry the flag onto every kernel group."""
    # Standard
    import lmcache.lmcache_native as lmcache_native

    tensors = [torch.zeros(2, 32, 16, 8, 64, dtype=torch.bfloat16) for _ in range(2)]
    formats = [lmcache_native.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS] * 2
    infos = [
        EngineGroupInfo(
            engine_group_id=0,
            layer_indices=(0,),
            tokens_per_block=16,
        ),
        EngineGroupInfo(
            engine_group_id=1,
            layer_indices=(1,),
            tokens_per_block=16,
        ),
    ]
    eager = [msgspec.structs.replace(g, is_eagle_group=True) for g in infos]

    plain = KVLayerGroupsManager(tensors, formats, engine_group_infos=infos)
    with_eagle = KVLayerGroupsManager(
        tensors, formats, engine_group_infos=eager
    )

    assert all(not g.is_eagle_group for g in plain.kernel_groups)
    assert all(g.is_eagle_group for g in with_eagle.kernel_groups)


def test_connector_reads_the_per_group_eagle_flag():
    """The flag comes from vLLM's per-group bit, not a model-wide switch.

    vLLM annotates ``KVCacheGroupSpec.is_eagle_group`` on the specific groups
    holding the speculative layers, and only when
    ``use_eagle_block_drop()`` holds. Stamping a model-wide value would widen
    every group's mask, so the per-group read is what keeps parity exact.
    """
    caches = {
        "layer.0": torch.zeros(2, 32, 16, 8, 64, dtype=torch.bfloat16),
        "layer.1": torch.zeros(2, 32, 16, 8, 64, dtype=torch.bfloat16),
    }
    config = MockKVCacheConfig(
        kv_cache_groups=[
            MockKVCacheGroup(["layer.0"], MockKVCacheSpec(block_size=16)),
            MockKVCacheGroup(
                ["layer.1"], MockKVCacheSpec(block_size=16), is_eagle_group=True
            ),
        ]
    )

    spec = create_engine_group_infos_from_vllm(config, caches)

    flags = {g.engine_group_id: g.is_eagle_group for g in spec}
    assert flags == {0: False, 1: True}, flags


def test_connector_reads_a_missing_eagle_field_as_no_drop():
    """A vLLM build without ``is_eagle_group`` must read as "no drop"."""
    caches = {"layer.0": torch.zeros(2, 32, 16, 8, 64, dtype=torch.bfloat16)}
    config = MockKVCacheConfig(
        kv_cache_groups=[
            MockKVCacheGroupNoEagle(["layer.0"], MockKVCacheSpec(block_size=16))
        ]
    )

    spec = create_engine_group_infos_from_vllm(config, caches)

    assert spec and all(g.is_eagle_group is False for g in spec)
