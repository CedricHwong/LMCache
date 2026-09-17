# SPDX-License-Identifier: Apache-2.0
"""Per-group store/load mask primitives (V4.1 masks-per-group).

V4.1 serves KV cache groups with *different* engine block sizes (production:
8 x 32, 1 x 64, 1 x 8), so a single shared block-grid mask is not meaningful.
These tests cover the engine-neutral per-group mask computation in
``lmcache.v1.multiprocess.group_view`` and its ``KVLayerGroupsManager``
wiring, mirroring the Mooncake external-store coordinator's per-group masks.
"""

# Standard
from collections.abc import Sequence

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.multiprocess.group_view import (
    EngineGroupInfo,
    GroupBlockMask,
    _group_block_range,
    _window_tail_mask,
    align_group_servable_length,
    align_lookup_length,
    compute_group_load_masks,
    compute_group_lookup_masks,
    compute_group_store_masks,
    lcm_block_tokens,
)
from lmcache.v1.kv_layer_groups import KVLayerGroupsManager
import lmcache.lmcache_native as lmcache_native


def _eg(
    engine_group_id: int,
    tokens_per_block: int,
    *,
    sw_size_tokens: int = -1,
    tokens_per_state: int = 1,
    prefix_cacheable: bool = True,
    recurrent_state: bool = False,
    layer_indices: tuple[int, ...] = (),
) -> EngineGroupInfo:
    """One EngineGroupInfo with mask-relevant fields set explicitly."""
    return EngineGroupInfo(
        engine_group_id=engine_group_id,
        layer_indices=layer_indices,
        tokens_per_block=tokens_per_block,
        sw_size_tokens=sw_size_tokens,
        tokens_per_state=tokens_per_state,
        prefix_cacheable=prefix_cacheable,
        recurrent_state=recurrent_state,
    )


def _prod_groups() -> list[EngineGroupInfo]:
    """The production V4.1 group layout, expanded to LMCache kernel groups.

    The 8 engine SWA groups become 8 LMCache groups; the single block-64
    engine group (G8) holds the main MLA KV *and* the indexer K cache, which
    split into two LMCache kernel groups; the block-8 ring (G9) is one group.
    Total LMCache groups = 8 + 2 + 1 = 11 (vLLM engine groups = 10).
    """
    groups: list[EngineGroupInfo] = [
        # 8 SWA groups: block 32, window 128 tokens, tps=1.
        _eg(i, 32, sw_size_tokens=128, layer_indices=(i,))
        for i in range(8)
    ]
    # G8 = main MLA (compressed, tps=2) + indexer (compressed, tps=2),
    # both block 64, full attention.
    groups.append(_eg(8, 64, tokens_per_state=2, layer_indices=(8,)))
    groups.append(_eg(8, 64, tokens_per_state=2, layer_indices=(9,)))
    # G9 = compressor ring: block 8, NOT prefix cacheable.
    groups.append(_eg(9, 8, prefix_cacheable=False, layer_indices=(10,)))
    return groups


class TestGroupBlockMask:
    def test_sentinel_none_means_all_true(self):
        mask = GroupBlockMask(0, 32, 0, 8, None)
        assert mask.block_count == 8
        assert all(mask.is_reachable(i) for i in range(8))

    def test_explicit_mask_and_out_of_range_index(self):
        mask = GroupBlockMask(1, 32, 2, 5, (True, False, True))
        assert mask.block_count == 3
        assert mask.is_reachable(2) is True
        assert mask.is_reachable(3) is False
        with pytest.raises(IndexError):
            mask.is_reachable(6)


class TestLcmAndAlignment:
    def test_lcm_of_prod_groups(self):
        groups = _prod_groups()
        assert lcm_block_tokens(groups) == 64

    def test_non_cacheable_excluded_from_lcm(self):
        # Only the 64-block group is cacheable here.
        groups = [
            _eg(0, 64),
            _eg(1, 8, prefix_cacheable=False),
            _eg(2, 32, prefix_cacheable=False),
        ]
        assert lcm_block_tokens(groups) == 64

    def test_lcm_raises_without_cacheable_group(self):
        with pytest.raises(ValueError, match="no prefix_cacheable group"):
            lcm_block_tokens([_eg(0, 8, prefix_cacheable=False)])

    def test_align_lookup_length_floors_to_lcm(self):
        groups = _prod_groups()
        assert align_lookup_length(groups, 200) == 192
        assert align_lookup_length(groups, 192) == 192
        assert align_lookup_length(groups, 63) == 0


class TestWindowTailMask:
    """Faithful mirror of SlidingWindowManager.reachable_block_mask."""

    def test_window_spans_segment_returns_dense(self):
        # window 128 >= segment 64: every block reachable.
        assert _window_tail_mask(32, 128, 0, 8, 64) is None

    def test_sub_chunk_tail_pattern(self):
        # block 32, window 128, chunk 256: keep the last 4 of 8 blocks.
        mask = _window_tail_mask(32, 128, 0, 8, 256)
        assert mask == [False, False, False, False, True, True, True, True]

    def test_small_window_tail(self):
        # block 32, window 64, chunk 256: keep the last 2 of 8 blocks.
        mask = _window_tail_mask(32, 64, 0, 8, 256)
        assert mask == [False] * 6 + [True] * 2

    def test_eagle_reserves_one_block(self):
        # window 64 -> need 2, eagle -> need 3; shift=1: the run's right edge
        # sits on the boundary block, and block i=0 is excluded by the
        # ``i >= shift`` guard, so only i in {6, 7} is reachable.
        mask = _window_tail_mask(32, 64, 0, 8, 256, use_eagle=True)
        assert mask == [False, False, False, False, False, False, True, True]

    def test_non_divisible_segment_falls_back_dense(self):
        # segment 100 is not a multiple of block 32 -> cannot be exact.
        assert _window_tail_mask(32, 64, 0, 8, 100) is None

    def test_invalid_args(self):
        with pytest.raises(ValueError, match="block_tokens"):
            _window_tail_mask(0, 64, 0, 8, 256)
        with pytest.raises(ValueError, match="segment_tokens"):
            _window_tail_mask(32, 64, 0, 8, 0)
        with pytest.raises(ValueError, match="inverted"):
            _window_tail_mask(32, 64, 6, 2, 256)

    def test_window_one_more_than_block_multiple(self):
        # vLLM's _contiguous_blocks_for_hit is cdiv(window-1, block), not
        # ceil(window/block): block 32 / window 33 needs only 1 contiguous
        # block, so under a 128-token segment the tail keeps 1 block, not 2.
        mask = _window_tail_mask(32, 33, 0, 16, 128)
        assert mask == [False, False, False, True] * 4

    def test_window_equals_block(self):
        # window 32 / block 32: cdiv(31, 32) == 1 tail block per segment.
        mask = _window_tail_mask(32, 32, 0, 16, 128)
        assert mask == [False, False, False, True] * 4

    def test_one_token_window_never_reachable(self):
        # window 1 (one-token "mamba-style" window glue): need = cdiv(0, 32)
        # = 0, and 0 < per_segment so an explicit all-False mask is returned.
        mask = _window_tail_mask(32, 1, 0, 16, 128)
        assert mask == [False] * 16

    def test_zero_window_all_false(self):
        # window 0: no contiguous block needed -> no block can be a tail.
        assert _window_tail_mask(32, 0, 0, 8, 128) == [False] * 8


class TestBlockRange:
    def test_full_range(self):
        assert _group_block_range(32, 256, 0) == (0, 8)

    def test_suffix_from_mid_range(self):
        assert _group_block_range(32, 256, 128) == (4, 8)

    def test_start_beyond_end_clamps(self):
        assert _group_block_range(32, 256, 300) == (8, 8)

    def test_start_ceils_to_block(self):
        # start_token 129 -> ceil(129/32) = 5.
        assert _group_block_range(32, 256, 129) == (5, 8)

    def test_invalid_tpb(self):
        with pytest.raises(ValueError, match="tokens_per_block"):
            _group_block_range(0, 256, 0)


class TestComputeStoreMasks:
    def test_prod_ten_groups(self):
        groups = _prod_groups()
        masks = compute_group_store_masks(groups, 256, segment_tokens=256)
        assert len(masks) == 11
        # 8 SWA groups: 8 blocks each, keep the last 4 (window 128 of 256).
        for m in masks[:8]:
            assert m.tokens_per_block == 32
            assert m.block_count == 8
            assert m.blocks == (False, False, False, False, True, True, True, True)
        # G8 groups: full attention -> all-True sentinel.
        for m in masks[8:10]:
            assert m.tokens_per_block == 64
            assert m.block_count == 4
            assert m.blocks is None
        # G9 ring: not cacheable -> all-False, 32 blocks.
        ring = masks[10]
        assert ring.tokens_per_block == 8
        assert ring.block_count == 32
        assert ring.blocks == (False,) * 32

    def test_exclude_recurrent_all_false(self):
        groups = [
            _eg(0, 64, recurrent_state=True),
            _eg(1, 64),
        ]
        masks = compute_group_store_masks(
            groups, 128, segment_tokens=128, exclude_recurrent=True
        )
        assert masks[0].blocks == (False, False)
        assert masks[1].blocks is None
        # Without the exclusion the recurrent group stores normally.
        masks = compute_group_store_masks(groups, 128, segment_tokens=128)
        assert masks[0].blocks is None

    def test_suffix_store_starts_at_start_token(self):
        groups = [_eg(0, 64), _eg(1, 32, sw_size_tokens=64)]
        masks = compute_group_store_masks(
            groups, 256, start_token=128, segment_tokens=256
        )
        # full attention: [2, 4)
        assert (masks[0].start_block, masks[0].end_block) == (2, 4)
        # SWA: [4, 8), tail pattern (window 64 = 2 blocks of the 4).
        assert (masks[1].start_block, masks[1].end_block) == (4, 8)
        assert masks[1].blocks == (False, False, True, True)

    def test_unaligned_token_len_raises(self):
        groups = _prod_groups()
        with pytest.raises(ValueError, match="multiple of the cross-group block lcm"):
            compute_group_store_masks(groups, 100, segment_tokens=64)


class TestComputeLookupMasks:
    def test_default_segment_is_lcm(self):
        groups = _prod_groups()
        masks = compute_group_lookup_masks(groups, 256)
        # SWA window 128 >= lcm segment 64 -> all-True.
        for m in masks[:8]:
            assert m.blocks is None
        # ring excluded.
        assert masks[10].blocks == (False,) * 32

    def test_chunk_segment_keeps_swa_tail(self):
        groups = [_eg(0, 32, sw_size_tokens=128)]
        masks = compute_group_lookup_masks(groups, 256, segment_tokens=256)
        assert masks[0].blocks == (False, False, False, False, True, True, True, True)


class TestComputeLoadMasks:
    def test_prod_hit(self):
        groups = _prod_groups()
        masks = compute_group_load_masks(groups, 256, segment_tokens=256)
        # Load mask mirrors the reachable set of the retained prefix.
        for m in masks[:8]:
            assert m.blocks == (False, False, False, False, True, True, True, True)
        for m in masks[8:10]:
            assert m.blocks is None
        assert masks[10].blocks == (False,) * 32


class TestAlignServableLength:
    def test_tps_and_lcm_floors(self):
        groups = _prod_groups()
        # 130 tokens: compressed groups (tps=2) floor to 130; lcm 64 -> 128.
        assert align_group_servable_length(groups, 130) == 128
        # 129 -> compressed floor to 128 -> lcm 128.
        assert align_group_servable_length(groups, 129) == 128
        # 300 -> tps floor to 300 -> lcm floor to 256.
        assert align_group_servable_length(groups, 300) == 256
        # An already-perfect candidate stays put.
        assert align_group_servable_length(groups, 256) == 256

    def test_non_cacheable_excluded_from_alignment(self):
        # Only the ring is non-cacheable; it does not constrain the length.
        groups = [
            _eg(0, 64, tokens_per_state=1),
            _eg(1, 8, tokens_per_state=1, prefix_cacheable=False),
        ]
        assert align_group_servable_length(groups, 100) == 64


def _build_manager(
    tensors: list[torch.Tensor],
    engine_group_infos: Sequence[EngineGroupInfo] = (),
) -> KVLayerGroupsManager:
    return KVLayerGroupsManager(
        tensors,
        engine_kv_formats=[lmcache_native.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS]
        * len(tensors),
        engine_group_infos=engine_group_infos,
    )


class TestKernelGroupWiring:
    def test_engine_fields_flow_into_kernel_group(self):
        tensors = [
            torch.randn(2, 8, 64, 1, 64, dtype=torch.float16),  # compressed: bs=64? no
            torch.randn(2, 8, 32, 1, 64, dtype=torch.float16),
        ]
        # bs is the tensor's dim-2 (block slot count): 64 and 32.
        manager = _build_manager(
            tensors,
            engine_group_infos=[
                EngineGroupInfo(0, (0,), tokens_per_block=64, tokens_per_state=1),
                EngineGroupInfo(1, (1,), tokens_per_block=64, tokens_per_state=2),
            ],
        )
        by = {g.engine_group_idx: g for g in manager.kernel_groups}
        assert by[0].state_tokens == 1
        assert by[0].prefix_cacheable is True
        assert by[1].state_tokens == 2
        assert by[1].prefix_cacheable is True

        # MaskGroup structural match: manager exposes kernel groups as inputs.
        assert isinstance(manager.mask_group_inputs(), list)
        assert manager.lcm_block_tokens() == 64
        # compressed group (tps=2) constrains the servable length.
        assert manager.align_group_servable_length(130) == 128

    def test_compressed_geometry_mismatch_raises(self):
        tensors = [torch.randn(2, 8, 64, 1, 64, dtype=torch.float16)]
        with pytest.raises(ValueError, match="tokens_per_state=2"):
            _build_manager(
                tensors,
                engine_group_infos=[
                    EngineGroupInfo(0, (0,), tokens_per_block=64, tokens_per_state=2),
                ],
            )

    def test_default_one_tps_skips_crosscheck(self):
        # Old-V4-style declaration (default tokens_per_state=1) with compressed
        # geometry is not flagged: the value is indistinguishable from never
        # reported, so geometry is trusted.
        tensors = [
            torch.randn(2, 8, 64, 1, 64, dtype=torch.float16),  # ratio 4
            torch.randn(2, 8, 2, 1, 64, dtype=torch.float16),  # ratio 128
        ]
        manager = _build_manager(
            tensors,
            engine_group_infos=[
                EngineGroupInfo(0, (0,), tokens_per_block=256),
                EngineGroupInfo(0, (1,), tokens_per_block=256),
            ],
        )
        by = {g.layer_indices[0]: g for g in manager.kernel_groups}
        assert by[0].tokens_per_block // by[0].slots_per_block == 4
        assert by[1].tokens_per_block // by[1].slots_per_block == 128

    def test_manager_scope_masks_use_engine_group_info(self):
        # EngineGroupInfo and KernelGroupInfo both drive the same mask math.
        infos = [
            EngineGroupInfo(0, (0,), tokens_per_block=32, sw_size_tokens=128),
            EngineGroupInfo(1, (1,), tokens_per_block=64, tokens_per_state=2),
            EngineGroupInfo(2, (2,), tokens_per_block=8, prefix_cacheable=False),
        ]
        tensors = [
            torch.randn(2, 8, 32, 1, 64, dtype=torch.float16),
            torch.randn(2, 8, 32, 1, 64, dtype=torch.float16),
            torch.randn(2, 8, 8, 1, 64, dtype=torch.float16),
        ]
        manager = _build_manager(tensors, engine_group_infos=infos)

        # group_view-level (EngineGroupInfo) vs manager-level (KernelGroupInfo)
        # must agree about the same geometry.
        from_view = compute_group_store_masks(infos, 256, segment_tokens=256)
        from_manager = manager.compute_group_store_masks(256)
        assert len(from_view) == len(from_manager) == 3
        assert from_view[0].blocks == from_manager[0].blocks
        assert from_manager[0].blocks is not None
        assert from_manager[1].blocks is None  # full attention
        assert from_manager[2].blocks == (False,) * (256 // 8)
        assert manager.lcm_block_tokens() == 64
