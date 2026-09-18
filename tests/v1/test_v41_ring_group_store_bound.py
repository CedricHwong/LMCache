# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the DeepSeek-V4.1 compressor-ring store bound.

vLLM exposes the V4.1 compressor as a *non-prefix-cacheable* KV cache group
with ``tokens_per_block == 8`` that never holds more than a single block,
because it is a fixed-size recurrent ring rather than a prefix of the
sequence. ``GetStoreMetadata`` bounds the storable prefix by the
least-covered engine group, so an unfiltered ``min`` over every group
collapses to ``1 block * 8 tokens == 8`` -- below any usable LMCache chunk
size -- and silently disables storing for *every* request.

These tests pin the ``prefix_cacheable`` filter that mirrors
``lmcache.v1.multiprocess.group_view.lcm_cacheable_block_tokens``, which the
hit-length alignment already applies.
"""

# Standard
from dataclasses import dataclass
from unittest.mock import patch

# Third Party
import pytest

pytest.importorskip("vllm", reason="MP connector imports vLLM at module top")

# Third Party
from vllm.v1.utils import ConstantList  # noqa: E402

# First Party
from lmcache.integration.vllm.lmcache_mp_metadata import (  # noqa: E402
    LMCacheMPRequestTracker,
    LMCacheMPRequestMetadata,
)

# The geometry measured on the real H100 TP4 bring-up of
# DeepSeek-V4.1-Flash: eight sliding-window groups (32), the MLA group (64)
# and the compressor ring (8).
V41_GROUP_TOKENS_PER_BLOCK = [32] * 8 + [64, 8]
V41_GROUP_PREFIX_CACHEABLE = [True] * 8 + [True, False]
V41_RING_GROUP = 9

CHUNK = 256


@dataclass
class _FakeSamplingParams:
    extra_args: dict[str, object] | None = None


class _FakeRequest:
    """Duck-typed vLLM Request carrying only what the tracker reads."""

    def __init__(self, prompt_token_ids: list[int]):
        self.request_id = "req-ring"
        self.resumable = False
        self.cache_salt = ""
        self.prompt_token_ids = list(prompt_token_ids)
        self.all_token_ids = ConstantList(list(prompt_token_ids))
        self.mm_features = []
        self.sampling_params = _FakeSamplingParams()
        self.block_hashes: list = []


def _build_tracker(num_tokens: int, num_prompt_blocks: int) -> LMCacheMPRequestTracker:
    """Return a tracker holding vLLM's real V4.1 block allocation.

    ``num_prompt_blocks`` blocks of prompt are allocated for every cacheable
    group; the ring keeps exactly the one block it always has.
    """
    with (
        patch(
            "lmcache.integration.vllm.lmcache_mp_metadata"
            ".extract_request_configs_from_request",
            return_value=None,
        ),
        patch(
            "lmcache.integration.vllm.lmcache_mp_metadata.extract_mm_features",
            return_value=([], []),
        ),
    ):
        tracker = LMCacheMPRequestTracker(_FakeRequest(list(range(num_tokens))))

    for group_idx, tokens_per_block in enumerate(V41_GROUP_TOKENS_PER_BLOCK):
        num_blocks = 1 if group_idx == V41_RING_GROUP else num_prompt_blocks
        tracker.allocated_block_ids[group_idx] = list(
            range(group_idx * 1000, group_idx * 1000 + num_blocks)
        )
        assert num_blocks * tokens_per_block >= 0
    tracker.increase_num_scheduled_tokens(num_tokens)
    return tracker


def test_ring_group_does_not_collapse_the_store_bound():
    """A 1218-token request still yields store metadata on real V4.1 geometry.

    Without the ``prefix_cacheable`` filter this returns ``None``: the ring
    contributes ``1 * 8 == 8`` tokens to the ``min`` and ``8 // 256 == 0``.
    """
    tracker = _build_tracker(num_tokens=1218, num_prompt_blocks=39)

    metadata = LMCacheMPRequestMetadata.GetStoreMetadata(
        tracker,
        CHUNK,
        V41_GROUP_TOKENS_PER_BLOCK,
        V41_GROUP_PREFIX_CACHEABLE,
    )

    assert metadata is not None
    assert metadata.direction == "STORE"
    # 39 blocks * 32 tokens == 1248 tokens coverable, so 4 whole chunks.
    assert metadata.op.end - metadata.op.start == 4 * CHUNK


def test_unfiltered_min_reproduces_the_ring_deadlock():
    """Document the pre-fix behavior: treating the ring as cacheable kills stores.

    This is the exact failure observed in production-homomorphic bring-up --
    ``allocated=8, computed=1218, min_avail=8, chunk=256, num_chunks=0`` -- so
    the test fails loudly if the filter is ever dropped again.
    """
    tracker = _build_tracker(num_tokens=1218, num_prompt_blocks=39)

    assert (
        LMCacheMPRequestMetadata.GetStoreMetadata(
            tracker,
            CHUNK,
            V41_GROUP_TOKENS_PER_BLOCK,
            [True] * len(V41_GROUP_TOKENS_PER_BLOCK),
        )
        is None
    )


def test_none_means_all_cacheable_for_backwards_compatibility():
    """Omitting the filter preserves the legacy single-group rule."""
    tracker = _build_tracker(num_tokens=1218, num_prompt_blocks=39)

    assert (
        LMCacheMPRequestMetadata.GetStoreMetadata(
            tracker,
            CHUNK,
            V41_GROUP_TOKENS_PER_BLOCK,
        )
        is None
    ), "ring is still counted when the caller supplies no cacheability data"


def test_non_cacheable_group_alone_yields_nothing_to_store():
    """If every group is non-cacheable there is no prefix to store."""
    tracker = _build_tracker(num_tokens=1218, num_prompt_blocks=39)

    assert (
        LMCacheMPRequestMetadata.GetStoreMetadata(
            tracker,
            CHUNK,
            V41_GROUP_TOKENS_PER_BLOCK,
            [False] * len(V41_GROUP_TOKENS_PER_BLOCK),
        )
        is None
    )


def test_mismatched_cacheable_length_fails_fast():
    """A geometry/filter disagreement must not be silently ignored."""
    tracker = _build_tracker(num_tokens=1218, num_prompt_blocks=39)

    with pytest.raises(ValueError, match="engine groups"):
        LMCacheMPRequestMetadata.GetStoreMetadata(
            tracker,
            CHUNK,
            V41_GROUP_TOKENS_PER_BLOCK,
            [True, False],
        )


def test_store_is_chunk_aligned_and_never_exceeds_allocated_tokens():
    """The emitted range stays inside every cacheable group's allocation."""
    # 8 blocks * 32 tokens == 256 == exactly one chunk; smaller spans
    # legitimately store nothing, which the separate test above covers.
    for num_blocks in (8, 39, 64):
        num_tokens = num_blocks * 32
        tracker = _build_tracker(num_tokens, num_blocks)
        metadata = LMCacheMPRequestMetadata.GetStoreMetadata(
            tracker,
            CHUNK,
            V41_GROUP_TOKENS_PER_BLOCK,
            V41_GROUP_PREFIX_CACHEABLE,
        )
        assert metadata is not None
        span = metadata.op.end - metadata.op.start
        assert span % CHUNK == 0
        assert span <= num_blocks * 32


def test_unlimited_offload_tokens_does_not_change_the_bound():
    """``max_offload_tokens`` can only tighten, never widen, the bound."""
    tracker = _build_tracker(num_tokens=1218, num_prompt_blocks=39)
    tracker.max_offload_tokens = 1

    assert (
        LMCacheMPRequestMetadata.GetStoreMetadata(
            tracker,
            CHUNK,
            V41_GROUP_TOKENS_PER_BLOCK,
            V41_GROUP_PREFIX_CACHEABLE,
        )
        is None
    )


class _RecordingCacheable(list):
    """A real list that records which engine groups were consulted."""

    def __init__(self, values: list[bool]):
        super().__init__(values)
        self.indexed: list[int] = []

    def __getitem__(self, index):
        self.indexed.append(index)
        return super().__getitem__(index)


def test_every_engine_group_is_consulted_for_cacheability():
    """Guard that the filter is indexed once per engine group."""
    tracker = _build_tracker(num_tokens=1218, num_prompt_blocks=39)
    cacheable = _RecordingCacheable(V41_GROUP_PREFIX_CACHEABLE)

    metadata = LMCacheMPRequestMetadata.GetStoreMetadata(
        tracker,
        CHUNK,
        V41_GROUP_TOKENS_PER_BLOCK,
        cacheable,
    )

    assert metadata is not None
    assert set(cacheable.indexed) == set(range(len(V41_GROUP_TOKENS_PER_BLOCK)))
