# SPDX-License-Identifier: Apache-2.0
"""DSA indexer K row identification & geometry (phase-2 indexer-boundary).

Covers the V4.1 indexer-specific read-correctness surface:

- ``utils.is_indexer_k_row`` / ``utils.resolve_indexer_row_geometry``: the
  evaluable structural criterion (``dtype == uint8`` + fixed row width) that
  identifies the V4.1 indexer K cache without ``cache_role`` /
  ``is_index_group_leader`` (which the V4.1 indexer spec leaves at the MLA
  defaults).
- ``NL_X_NB_BSV_BSS_Spec``: the per-token (values, scale) byte split of the
  ``[BS x vals][BS x scale]`` page, generalised beyond the hardcoded fp8
  ``128 + 4`` record to also cover the MXFP4 ``64 + 4`` (68 B) path.

Environment: no vLLM, no GPU. ``lmcache.lmcache_native`` is provided by the
pure-Python shim (``scripts/v41_stub_lmcache_native.py``) via
``scripts/v41_test_env.sh stub`` (or the real ``.so`` in native mode);
``torch`` is only used to build synthetic tensor shapes.
"""

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.gpu_connector.kv_format import describe_shape, get_spec
from lmcache.v1.gpu_connector.kv_format.specs.nl_x_nb_bsv_bss import (
    BLOCKED_SCALE_RECORDS,
)
from lmcache.v1.gpu_connector.utils import (
    is_indexer_k_row,
    resolve_indexer_row_geometry,
)
import lmcache.lmcache_native as lmcache_native

BSV_BSS = lmcache_native.EngineKVFormat.NL_X_NB_BSV_BSS


def _t(*shape: int, dtype: torch.dtype = torch.uint8) -> torch.Tensor:
    return torch.zeros(shape, dtype=dtype)


# --------------------------------------------------------------------------
# utils.is_indexer_k_row / resolve_indexer_row_geometry
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("row_bytes", "expected"),
    [
        # fp8 DSA indexer (SM90/H100): 128 value bytes + 4-byte fp32 scale.
        (132, (128, 4)),
        # MXFP4 DSA indexer (Blackwell): 64 packed value bytes + 4 ue8m0 scales.
        (68, (64, 4)),
    ],
)
def test_resolve_indexer_row_geometry_quant_paths(row_bytes, expected):
    """The two canonical indexer widths decompose into (values, scale)."""
    assert is_indexer_k_row(row_bytes, torch.uint8) is True
    assert resolve_indexer_row_geometry(row_bytes, torch.uint8) == expected


@pytest.mark.parametrize(
    ("row_bytes", "dtype"),
    [
        # Main fp8_ds_mla record (SWA and compressed-KV layers, SM90).
        (584, torch.uint8),
        # SM100 MXFP8 sliding-window record.
        (528, torch.uint8),
        # NVFP4 compressed record.
        (288, torch.uint8),
        # A plain 128-wide uint8 KV row is not an indexer record.
        (128, torch.uint8),
        # A row that only *width*-matches is not an indexer record unless it is
        # on the uint8 quantized plane.
        (132, torch.float16),
        (132, torch.bfloat16),
        (68, torch.float32),
    ],
)
def test_resolve_indexer_row_geometry_rejects_neighbours(row_bytes, dtype):
    """Model-neighbour records are never mis-identified as indexer rows."""
    assert is_indexer_k_row(row_bytes, dtype) is False
    assert resolve_indexer_row_geometry(row_bytes, dtype) is None


def test_resolve_indexer_row_geometry_non_canonical_widths():
    """Widths outside {132, 68} (even 4-byte ones) are not indexer rows."""
    for width in (4, 16, 100, 136, 516, 640):
        assert is_indexer_k_row(width, torch.uint8) is False
        assert resolve_indexer_row_geometry(width, torch.uint8) is None


# --------------------------------------------------------------------------
# NL_X_NB_BSV_BSS_Spec geometry accessors
# --------------------------------------------------------------------------


def test_bsv_bss_spec_fp8_row():
    """[NB, 32, 132] uint8 decomposes into 128 value + 4 scale bytes."""
    kv_caches = [_t(7, 32, 132) for _ in range(2)]
    spec = get_spec(kv_caches, BSV_BSS)
    assert spec.head_size() == 132
    assert spec.hidden_dim() == 132
    assert spec.indexer_row_width() == 132
    assert spec.vals_bytes() == 128
    assert spec.scale_bytes() == 4
    assert spec.blocked_scale_row_geometry() == (128, 4)


def test_bsv_bss_spec_mxfp4_row():
    """[NB, 64, 68] uint8 decomposes into 64 value + 4 scale bytes."""
    kv_caches = [_t(7, 64, 68) for _ in range(2)]
    spec = get_spec(kv_caches, BSV_BSS)
    assert spec.head_size() == 68
    assert spec.indexer_row_width() == 68
    assert spec.vals_bytes() == 64
    assert spec.scale_bytes() == 4
    assert spec.blocked_scale_row_geometry() == (64, 4)


def test_bsv_bss_split_is_record_derived_not_a_fixed_4_bytes():
    """The value/scale split comes from the record, not from "the last 4 bytes".

    This format's kernels used to hardcode a 4-byte scale, and the spec mirrored
    that as "any %4 row decomposes into ``(W-4, 4)``". That is wrong for the
    quantized MLA main KV: vLLM's 584 B ``fp8_ds_mla`` V4 page is 576 value + 8
    scale, not 580 + 4; the 528 B MXFP8 page carries 16 B and the 352/288 B
    nvfp4 pages 32 B of scale. On an
    H100, a 584 B page routed through the token-major content-size format lost
    62.5% of its bytes with no error; splitting it as ``(580, 4)`` here would
    lose the same bytes a different way. So the split is looked up by record
    width, and a width with no entry is refused rather than guessed at.
    """
    for width, scale in sorted(BLOCKED_SCALE_RECORDS.items()):
        spec = get_spec([_t(7, 32, width)], BSV_BSS)
        assert spec.scale_bytes() == scale
        assert spec.vals_bytes() == width - scale
        assert spec.blocked_scale_row_geometry() == (width - scale, scale)


def test_bsv_bss_unknown_row_width_is_refused_not_guessed():
    """A width with no record-table entry must not be given a 4-byte split."""
    assert is_indexer_k_row(136, torch.uint8) is False
    spec = get_spec([_t(7, 32, 136)], BSV_BSS)
    assert spec.scale_bytes() == 0
    with pytest.raises(ValueError, match="not a segregated value/scale row"):
        spec.vals_bytes()


@pytest.mark.parametrize("width", [0, 3, 4, 5, 6, 136, 580])
def test_bsv_bss_spec_rejects_unknown_row_width(width):
    """An unroutable width raises; it must never be silently truncated."""
    spec = get_spec([_t(7, 32, width)], BSV_BSS)
    assert spec.scale_bytes() == 0
    with pytest.raises(ValueError, match="not a segregated value/scale row"):
        spec.vals_bytes()


def test_bsv_bss_spec_shares_mla_geometry():
    """The indexer spec reports MLA-level geometry (one plane, block / nb)."""
    kv_caches = [_t(6, 32, 132), _t(6, 32, 132)]
    spec = get_spec(kv_caches, BSV_BSS)
    assert spec.is_mla is True
    assert spec.is_layer_list is True
    assert spec.num_layers() == 2
    assert spec.num_blocks() == 6
    assert spec.block_size() == 32
    assert spec.kv_size() == 1
    assert spec.num_heads() == 1


def test_bsv_bss_describe_shape():
    """The format renders its segregated value/scale body symbolically."""
    assert describe_shape(BSV_BSS) == "NL x [NB, BSxVALS, BSxSCALES]"
