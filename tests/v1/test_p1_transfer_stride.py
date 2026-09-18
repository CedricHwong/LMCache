# SPDX-License-Identifier: Apache-2.0
"""P1-7: the transfer paths must carry the physical per-block stride.

``gather_paged_kv_to_cpu`` / ``scatter_cpu_to_paged_kv`` build a
``PageBufferShapeDesc`` that the multi-layer transfer kernel addresses
with.  When the engine's KV pool is dim-0 padded -- e.g. a compressed
DeepSeek-V4.1 compressor/indexer cache sharing a row width with a larger
attention group -- the kernel's tight-stride fallback (``block_stride_elems
<= 0``) reconstructs a stride that is *smaller* than the real per-block step,
so it walks into the padding: the gather reads the wrong bytes and the scatter
writes them back to the wrong places.  Both paths must therefore pass the
resolved stride.

The regression this pins is easy to miss because an unpadded pool is
unaffected (the resolver legitimately returns ``None``), which is exactly the
geometry the pre-existing CPU tests use.  So rather than compare bytes on an
unpadded pool, these tests spy on ``make_page_buffer_shape_desc`` and assert
the argument is supplied and matches the resolver's answer.
"""

# Standard
from typing import Any

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector import utils as U
from lmcache.v1.multiprocess.transfer_context.base import (
    gather_paged_kv_to_cpu,
    scatter_cpu_to_paged_kv,
)

NB, NH, BS, HS, NL = 16, 4, 128, 64, 3
HINTS = {"kv_layout": "HND"}


def _caches() -> dict[str, torch.Tensor]:
    """Per-layer blocks-first tensors as registered: ``[NB, NH, BS, 2 * HS]``."""
    torch.manual_seed(0)
    return {
        f"layer_{i}": torch.randn(NB, NH, BS, 2 * HS) for i in range(NL)
    }


class _DescSpy:
    """Records every ``make_page_buffer_shape_desc`` call's kwargs."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return _real_desc(*args, **kwargs)


_real_desc = U.make_page_buffer_shape_desc


@pytest.fixture()
def desc_spy(monkeypatch: pytest.MonkeyPatch) -> _DescSpy:
    """Patch the resolver's consumer, which the transfer paths import lazily."""
    spy = _DescSpy()
    monkeypatch.setattr(U, "make_page_buffer_shape_desc", spy, raising=True)
    return spy


def test_gather_passes_block_stride_elems(desc_spy: _DescSpy) -> None:
    """The gather path supplies ``block_stride_elems`` to the shape desc."""
    caches = _caches()
    gather_paged_kv_to_cpu(
        caches, [0, 1, 2, 3], blocks_per_chunk=2, layout_hints=HINTS
    )

    assert desc_spy.calls, "gather did not build a shape desc"
    for kwargs in desc_spy.calls:
        assert "block_stride_elems" in kwargs, (
            "gather omitted block_stride_elems; a padded pool would be read "
            "with the tight stride and walk into the padding"
        )


def test_scatter_passes_block_stride_elems(desc_spy: _DescSpy) -> None:
    """The scatter path supplies ``block_stride_elems`` to the shape desc."""
    caches = _caches()
    chunks = gather_paged_kv_to_cpu(
        caches, [0, 1, 2, 3], blocks_per_chunk=2, layout_hints=HINTS
    )
    desc_spy.calls.clear()

    scatter_cpu_to_paged_kv(
        caches, [0, 1, 2, 3], chunks, blocks_per_chunk=2, layout_hints=HINTS
    )

    assert desc_spy.calls, "scatter did not build a shape desc"
    for kwargs in desc_spy.calls:
        assert "block_stride_elems" in kwargs, (
            "scatter omitted block_stride_elems; a padded pool would be "
            "written with the tight stride and corrupt neighbouring rows"
        )


def test_stride_matches_the_resolver(desc_spy: _DescSpy) -> None:
    """The supplied stride must equal what the shared resolver returns.

    Passing a *wrong* stride would be no better than omitting it, so this
    pins the value rather than mere presence.
    """
    caches = _caches()
    fmt, normalized = U.normalize_kv_and_discover_format(
        list(caches.values()), EngineType.VLLM, layout_hints=HINTS
    )
    expected = U.resolve_block_stride_and_log_layout(
        normalized, fmt, layer_idx=0, group_idx=0
    )

    gather_paged_kv_to_cpu(
        caches, [0, 1, 2, 3], blocks_per_chunk=2, layout_hints=HINTS
    )

    assert desc_spy.calls
    for kwargs in desc_spy.calls:
        assert kwargs.get("block_stride_elems") == expected
