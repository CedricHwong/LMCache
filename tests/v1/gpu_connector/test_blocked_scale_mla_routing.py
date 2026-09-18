# SPDX-License-Identifier: Apache-2.0
"""Blocked-scale MLA routing: the quantized DeepSeek MLA page.

A quantized DeepSeek MLA cache does not lay its tokens out token-by-token. Per
block it stores *every* token's value bytes first, then *every* token's scale
bytes::

    [ BS x value_bytes ][ BS x scale_bytes ]

Only ``NL_X_NB_BSV_BSS`` addresses that segregation (the transfer kernels copy
the two planes separately). Before this routing existed, a nightly vLLM
registration of such a page -- rank-4 ``[NB, 1, BS, W]`` with ``W`` in
{584, 528, 352} -- was judged "fused K/V, HND" and handed to
``NL_X_NB_NH_BS_CS``, whose addressing is token-major. That is silently wrong:
the kernel accepts the launch, the guard passes, and roughly the value plane of
every token after the first is read from the wrong offset. It is invisible to a
same-page store/retrieve round-trip, because an erroneous mapping is its own
inverse there.

The registration exposes only the whole record width, so the value/scale split
comes from the record-width table in
:mod:`~lmcache.v1.gpu_connector.kv_format.specs.nl_x_nb_bsv_bss`.
"""

# Standard
from typing import Any

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.kv_format import detect_format
from lmcache.v1.gpu_connector.kv_format.specs.nl_x_nb_bsv_bss import (
    BLOCKED_SCALE_RECORDS,
    blocked_scale_bytes,
)
from lmcache.v1.gpu_connector.kv_format.specs.registry import get_spec
import lmcache.lmcache_native as lmcache_native

# Per-token record width -> (value bytes, scale bytes), from vLLM:
#   * models/deepseek_v4/common/ops/fused_compress_quant_cache.py -- the V4
#     page is ``[bs*576)`` data plus ``[bs*576, +bs*8)`` UE8M0 scales.
#   * v1/attention/backends/mla/flashmla_sparse.py -- V4.1 is 512 e4m3 + 16
#     ue8m0 (MXFP8), nvfp4 is 320 value + 32 scale.
#   * the DSA indexer rows are 128 + 4 (fp8) and 64 + 4 (mxfp4).
_EXPECTED_SPLIT: dict[int, tuple[int, int]] = {
    584: (576, 8),
    528: (512, 16),
    352: (320, 32),
    132: (128, 4),
    68: (64, 4),
}

# A plain fused-K/V content-size page (4 real heads): NOT blocked-scale, so it
# must keep resolving to the token-major content-size format.
_PLAIN_CS_RECORD = 640


def _hnd_hints(tokens_per_block: int = 64) -> dict[str, Any]:
    """Return HND layout hints as the vLLM adapter publishes them."""
    return {
        "kv_layout": "HND",
        "num_kv_heads": 1,
        "tokens_per_block": tokens_per_block,
        "head_dim": 512,
    }


class TestBlockedScaleRecordTable:
    """The record-width -> scale-bytes table is exact and total."""

    @pytest.mark.parametrize(("record", "split"), sorted(_EXPECTED_SPLIT.items()))
    def test_known_records_decompose_exactly(
        self, record: int, split: tuple[int, int]
    ) -> None:
        """Every known record yields its documented (value, scale) split."""
        vals, scale = split
        assert record == vals + scale
        assert blocked_scale_bytes(record) == scale
        assert BLOCKED_SCALE_RECORDS[record] == scale

    @pytest.mark.parametrize("record", [0, 1, 512, 640, 999, 1024])
    def test_unknown_records_are_not_blocked_scale(self, record: int) -> None:
        """An unrecognised width reports 0, so it cannot be routed here."""
        assert blocked_scale_bytes(record) == 0

    def test_scale_regions_are_whole_4_byte_units(self) -> None:
        """The kernels pin 4-byte transfer units; every scale region fits."""
        for scale in BLOCKED_SCALE_RECORDS.values():
            assert scale > 0
            assert scale % 4 == 0


class TestBlockedScaleDetection:
    """A quantized MLA page must not be read token-major."""

    @pytest.mark.parametrize("record", sorted(_EXPECTED_SPLIT))
    def test_singleton_head_record_routes_to_blocked_scale(
        self, record: int
    ) -> None:
        """Rank-4 ``[NB, 1, BS, W]`` with a known record routes to BSV_BSS."""
        cache = torch.zeros((8, 1, 64, record), dtype=torch.uint8)
        fmt, _ = detect_format([cache], EngineType.VLLM, layout_hints=_hnd_hints())
        assert fmt == lmcache_native.EngineKVFormat.NL_X_NB_BSV_BSS

    def test_nhd_singleton_head_also_routes_to_blocked_scale(self) -> None:
        """``[NB, BS, 1, W]`` is the same single plane; it routes the same way."""
        cache = torch.zeros((8, 64, 1, 584), dtype=torch.uint8)
        hints = dict(_hnd_hints(), kv_layout="NHD")
        fmt, _ = detect_format([cache], EngineType.VLLM, layout_hints=hints)
        assert fmt == lmcache_native.EngineKVFormat.NL_X_NB_BSV_BSS

    @pytest.mark.parametrize("num_heads", [2, 4, 8])
    def test_multi_head_content_size_page_is_untouched(self, num_heads: int) -> None:
        """A real multi-head fused page keeps the content-size format."""
        cache = torch.zeros((8, num_heads, 64, _PLAIN_CS_RECORD), dtype=torch.uint8)
        hints = dict(_hnd_hints(), num_kv_heads=num_heads)
        fmt, _ = detect_format([cache], EngineType.VLLM, layout_hints=hints)
        assert fmt == lmcache_native.EngineKVFormat.NL_X_NB_NH_BS_CS

    def test_multi_head_page_with_record_width_is_not_blocked_scale(self) -> None:
        """A singleton head axis is required; 4 heads of 584 B is not this layout."""
        cache = torch.zeros((8, 4, 64, 584), dtype=torch.uint8)
        hints = dict(_hnd_hints(), num_kv_heads=4)
        fmt, _ = detect_format([cache], EngineType.VLLM, layout_hints=hints)
        assert fmt != lmcache_native.EngineKVFormat.NL_X_NB_BSV_BSS


class TestBlockedScaleSpecGeometry:
    """The spec reads both accepted registrations without changing meaning."""

    def _spec(self, shape: tuple[int, ...]) -> Any:
        cache = torch.zeros(shape, dtype=torch.uint8)
        return get_spec([cache], lmcache_native.EngineKVFormat.NL_X_NB_BSV_BSS)

    @pytest.mark.parametrize(
        ("shape", "num_blocks", "block_size", "record"),
        [
            ((8, 1, 64, 584), 8, 64, 584),
            ((8, 64, 1, 584), 8, 64, 584),
            ((4, 1, 32, 132), 4, 32, 132),
        ],
    )
    def test_axes_resolve_for_every_registration(
        self,
        shape: tuple[int, ...],
        num_blocks: int,
        block_size: int,
        record: int,
    ) -> None:
        """The block and record axes are located whatever the rank."""
        spec = self._spec(shape)
        assert spec.num_blocks() == num_blocks
        assert spec.block_size() == block_size
        assert spec.head_size() == record
        assert spec.hidden_dim() == record
        assert spec.num_heads() == 1
        assert spec.scale_bytes() == blocked_scale_bytes(record)
        assert spec.vals_bytes() == record - blocked_scale_bytes(record)
        assert spec.page_buffer_size() == num_blocks * block_size

    def test_blocked_scale_row_geometry_matches_the_table(self) -> None:
        """``blocked_scale_row_geometry`` is the split the kernels apply."""
        for record, split in _EXPECTED_SPLIT.items():
            assert self._spec((8, 1, 64, record)).blocked_scale_row_geometry() == split

    def test_multi_head_rank_four_is_rejected(self) -> None:
        """Blocked-scale pages are single-plane; a real head axis is a mistake."""
        with pytest.raises(ValueError, match="singleton head axis"):
            self._spec((8, 4, 64, 584)).block_size()

    def test_non_page_rank_is_rejected(self) -> None:
        """Anything other than 3-D/4-D is not a blocked-scale page."""
        with pytest.raises(ValueError, match="3-D|4-D"):
            self._spec((8, 64)).block_size()

    def test_unknown_record_width_has_no_scale_region(self) -> None:
        """An unknown width must not silently claim a scale region."""
        with pytest.raises(ValueError):
            self._spec((8, 1, 64, 640)).vals_bytes()
