# SPDX-License-Identifier: Apache-2.0
"""Regression: BLHNC/CS single-head (NH=1) kernel round trip.

DeepSeek-V4.1's fused-K/V KV cache registers a BLHNC ``[NB, NH, BS, CS]``
pool (``EngineKVFormat.NL_X_NB_NH_BS_CS``) with a single KV head
(``NH == 1``) and content width ``CS == 2 * HS`` (K/V packed).

The C++ transfer branch ``page_buffer_offset`` for that format
(``csrc/cuda/mem_kernels.cu``) decomposes token/head with

    hs2        = 2 * head_size
    num_heads  = scalars_per_token / hs2

where ``spec.head_size()`` for ``NL_X_NB_NH_BS_CS`` returns the *full* fused
content width ``CS`` (``specs/nl_x_nb_nh_bs_cs.py``).  The two are
incompatible: ``num_heads = (NH * CS) / (2 * CS) = NH / 2``, i.e. ``0`` for
V4.1's ``NH == 1`` and ``1`` for ``NH == 2``.  A ``num_heads == 0`` doubles
the in-block token stride (``o * 2 * CS`` instead of ``o * CS``), so the D2H
staging no longer equals a plain torch gather, and the round trip corrupts the
pool.

The legacy ``test_blocks_first_kernel_roundtrip.py`` masked this by manually
passing ``CS // 2`` (so that ``2 * (CS // 2) == CS`` reproduces the true
per-head width).  The tests here deliberately use the *production* path:
``head_size = spec.head_size() == CS`` and the real per-block step
``stride(0)``.  They are RED on the current C++ and GREEN once the branch
decomposes by the true per-head width.  An ``NH == 2`` contrast case is kept
to show why the bug stayed hidden so long: even with the broken ``head_size``
an NH=2 whole-block D2H->H2D round trip restores identical bytes (the broken
addressing is still a per-block bijection), so a round-trip-only suite stays
green; only an explicit ordering check against a torch gather reveals it.
"""

# Third Party
import pytest
import torch

# First Party
from lmcache import device_ops
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector import utils as U
from lmcache.v1.gpu_connector.kv_format import detect_format
import lmcache.lmcache_native as lmcache_native

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

NB, NL = 6, 3  # few blocks/layers suffice: the corruption is per-token
# V4.1 production geometry (8xH100/SM90): fused K/V content width 584
# (576 latent + 8 rope), BLHNC, KV block size 64, single KV head.
PROD_CONTENT, PROD_BLOCK_SIZE = 584, 64


def make_pool(
    num_blocks: int,
    num_layers: int,
    num_heads: int,
    block_size: int,
    content: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """BLHNC blocks-first pool ``[NB, NL, NH, BS, CS]`` plus spare dim-0 rows.

    The spare rows keep the *pre-fix* reads/writes inside the allocation: with
    ``num_heads == 0`` the CS branch overruns the last layer-block by up to
    ``(BS - 1) * 2 * CS`` scalars, which would otherwise fault the CUDA
    context and turn the regression into a crash instead of a clean failure.
    """
    step = num_layers * num_heads * block_size * content
    # Largest pre-fix scalar index for layer (NL - 1): the layer base plus the
    # per-token step doubling: (NB-1)*step + (BS-1)*(2*CS) + (CS-1).
    max_idx = (num_layers - 1) * num_heads * block_size * content + (
        num_blocks - 1
    ) * step + (block_size - 1) * 2 * content + (content - 1)
    storage_rows = max_idx // step + 2
    arch = torch.arange(
        storage_rows * step, dtype=torch.float32, device=device
    ).reshape(storage_rows, num_layers, num_heads, block_size, content)
    buf = arch[:num_blocks]
    return buf, [buf[:, layer] for layer in range(num_layers)]


def torch_gather(
    buf: torch.Tensor,
    slots: list[int],
    num_layers: int,
    num_heads: int,
    block_size: int,
    content: int,
) -> torch.Tensor:
    """Reference ``[1, NL, T, NH*CS]``: token-major, heads flattened."""
    flat = torch.as_tensor(slots, dtype=torch.long)
    blocks, offsets = flat // block_size, flat % block_size
    out = torch.stack(
        [
            buf[b, :, :, o].reshape(num_layers, num_heads * content)
            for b, o in zip(blocks.tolist(), offsets.tolist(), strict=True)
        ],
        dim=1,
    )
    return out.unsqueeze(0)  # [1, NL, T, NH*CS]


def _slot_list(blocks: list[int], block_size: int) -> list[int]:
    """Slots across several blocks and in-block offsets, always including
    offset 0 (the one offset the pre-fix mapping happens to get right) and
    strictly interior/edge offsets (which the pre-fix mapping corrupts)."""
    probes = [0, 1, block_size // 2, block_size - 2, block_size - 1]
    return [b * block_size + o for b in blocks for o in probes if o < block_size]


def _assert_equal(a: torch.Tensor, b: torch.Tensor, msg: str) -> None:
    if not torch.equal(a, b):
        n = int((a != b).sum())
        raise AssertionError(
            f"{msg} -- {n}/{a.numel()} scalars differ"
            f" (max|a-b|={float((a - b).abs().max())})"
        )


# ======================================================================
# CPU: pin the derivation (docs/phase1/08-test-cpp-roundtrip.md) and the
# format-level facts that steer the GPU tests below.
# ======================================================================

def _buggy_num_heads(num_heads: int, content: int) -> int:
    """Pre-fix CS branch arithmetic: ``hs2 = 2 * head_size`` with
    ``head_size == spec.head_size() == content``."""
    scalars_per_token = num_heads * content
    hs2 = 2 * content
    return scalars_per_token // hs2


def test_cs_num_heads_derivation_nh1_vs_nh2():
    """Pin the arithmetic that makes NH=1 catastrophic and NH=2 survivable.

    This only re-derives the (pre-fix) C++ formula; it is documentation, not a
    correctness check of the running kernel.  ``num_heads == 0`` (NH=1) breaks
    every per-block index; ``num_heads == 1`` (NH=2) still covers the whole
    ``[NH][BS][CS]`` plane as a bijection, which is why a whole-block round
    trip can hide the bug there.
    """
    # V4.1 production: NH=1, CS=584 -> SPT=584, hs2=1168 -> num_heads = 0.
    assert _buggy_num_heads(1, 584) == 0
    # Downscaled equivalent.
    assert _buggy_num_heads(1, 64) == 0
    # NH=2: SPT=32, hs2=32 -> num_heads = 1 (non-zero: per-block bijection).
    assert _buggy_num_heads(2, 16) == 1
    # In-block token step: buggy o*hs2 (= o*2*CS) vs true o*CS -> doubled for
    # NH=1; the old test's CS//2 head_size restores hs2 == CS (the true width).
    hs2 = 2 * 584
    assert hs2 == 1168
    # old-test coddling: hs2 = 2 * (CS//2) = CS reproduces the true width.
    assert 2 * (584 // 2) == 584


def test_cs_spec_head_size_is_full_width_on_cpu():
    """Pin fact: for NL_X_NB_NH_BS_CS, spec.head_size() returns the full
    content width CS (=> the kernel must NOT halve it, nor double it)."""
    buf, views = make_pool(NB, NL, 1, PROD_BLOCK_SIZE, PROD_CONTENT, "cpu")
    assert buf.shape == (NB, NL, 1, PROD_BLOCK_SIZE, PROD_CONTENT)
    fmt, kv = detect_format(views, EngineType.VLLM, {"kv_layout": "BLHNC"})
    assert fmt == lmcache_native.EngineKVFormat.NL_X_NB_NH_BS_CS
    assert U.get_num_heads(kv, fmt) == 1
    assert U.get_block_size(kv, fmt) == PROD_BLOCK_SIZE
    # The full K/V-packed content width -- what the connector forwards to the
    # C++ kernel as ``head_size``.
    assert U.get_head_size(kv, fmt) == PROD_CONTENT
    # Blocks-first: the per-block step spans every layer.
    assert kv[0].stride(0) == buf.stride(0) == NL * 1 * PROD_BLOCK_SIZE * PROD_CONTENT


# ======================================================================
# GPU: the actual regression.  Uses the production path (head_size = CS).
# ======================================================================

@pytest.mark.cuda
@cuda_only
@pytest.mark.parametrize(
    "content,block_size",
    [
        pytest.param(PROD_CONTENT, PROD_BLOCK_SIZE, id="v41-prod-C584-BS64"),
        pytest.param(64, 8, id="scaled-C64-BS8"),
    ],
)
def test_nh1_cs_d2h_normal_path_matches_torch(content: int, block_size: int):
    """NH=1 (V4.1): D2H staging must equal a plain torch gather.

    Transfers through the normal path: ``head_size = spec.head_size()`` (the
    full content width ``CS``, not ``CS // 2``) and ``block_stride =
    stride(0)``.  Must FAIL pre-fix (num_heads == 0 corrupts the in-block
    token step) and PASS once the C++ branch decomposes by the true per-head
    width.
    """
    device = torch.device("cuda:0")
    num_heads = 1
    buf, views = make_pool(NB, NL, num_heads, block_size, content, device)
    fmt, kv = detect_format(views, EngineType.VLLM, {"kv_layout": "BLHNC"})
    assert fmt == lmcache_native.EngineKVFormat.NL_X_NB_NH_BS_CS
    head_size = U.get_head_size(kv, fmt)
    assert head_size == content  # production connector forwards the full width
    assert kv[0].stride(0) == buf.stride(0)
    block_stride = kv[0].stride(0)

    slots = _slot_list([0, 1, 3, 4], block_size)
    staging = torch.zeros(
        1, NL, len(slots), num_heads * content, dtype=torch.float32, device=device
    )
    ptrs = torch.tensor([v.data_ptr() for v in kv], dtype=torch.int64, device=device)
    device_ops.multi_layer_kv_transfer(
        staging,
        ptrs,
        torch.tensor(slots, dtype=torch.int64, device=device),
        device,
        NB * block_size,
        int(lmcache_native.TransferDirection.D2H),
        int(fmt),
        block_size,
        head_size,
        0,
        block_stride,
    )
    torch.cuda.synchronize()
    ref = torch_gather(buf, slots, NL, num_heads, block_size, content)
    _assert_equal(staging, ref, "D2H staging != torch gather (CS num_heads=0 bug)")


@pytest.mark.cuda
@cuda_only
@pytest.mark.parametrize(
    "content,block_size",
    [
        pytest.param(PROD_CONTENT, PROD_BLOCK_SIZE, id="v41-prod-C584-BS64"),
        pytest.param(64, 8, id="scaled-C64-BS8"),
    ],
)
def test_nh1_cs_roundtrip_normal_path_restores_bytes(content: int, block_size: int):
    """NH=1 (V4.1): D2H -> zero the touched blocks -> H2D restores the bytes.

    Unlike the legacy test this uses ``head_size = CS``, so pre-fix the buggy
    H2D writes (and reads) at the doubled step and the pool is not restored;
    the assertion must FAIL pre-fix and PASS post-fix.
    """
    device = torch.device("cuda:0")
    num_heads = 1
    buf, views = make_pool(NB, NL, num_heads, block_size, content, device)
    fmt, kv = detect_format(views, EngineType.VLLM, {"kv_layout": "BLHNC"})
    assert fmt == lmcache_native.EngineKVFormat.NL_X_NB_NH_BS_CS
    head_size = U.get_head_size(kv, fmt)
    block_stride = kv[0].stride(0)

    slots = _slot_list([0, 1, 3, 4], block_size)
    staging = torch.zeros(
        1, NL, len(slots), num_heads * content, dtype=torch.float32, device=device
    )
    ptrs = torch.tensor([v.data_ptr() for v in kv], dtype=torch.int64, device=device)
    slots_t = torch.tensor(slots, dtype=torch.int64, device=device)
    device_ops.multi_layer_kv_transfer(
        staging, ptrs, slots_t, device, NB * block_size,
        int(lmcache_native.TransferDirection.D2H), int(fmt), block_size,
        head_size, 0, block_stride,
    )
    torch.cuda.synchronize()

    original = buf.clone()
    touched = sorted({int(s) // block_size for s in slots})
    for b in touched:
        buf[b, :NL].zero_()
    device_ops.multi_layer_kv_transfer(
        staging, ptrs, slots_t, device, NB * block_size,
        int(lmcache_native.TransferDirection.H2D), int(fmt), block_size,
        head_size, 0, block_stride,
    )
    torch.cuda.synchronize()

    used = sorted({int(s) % block_size for s in slots})
    for b in touched:
        _assert_equal(
            buf[b, :NL, :, used], original[b, :NL, :, used],
            f"H2D did not restore block {b} (CS num_heads=0 bug)",
        )


@pytest.mark.cuda
@cuda_only
def test_nh2_cs_whole_block_roundtrip_stays_green():
    """NH=2 contrast: why the bug survived in NH>1 round-trip suites.

    With the (broken) production ``head_size = CS`` the CS branch derives
    ``num_heads = SPT/(2*CS) = (2*CS)/(2*CS) = 1 >= 1``: every *whole-block*
    D2H -> H2D round trip is still a per-block bijection over
    ``[NH][BS][CS] == [1][BS][2*CS]``, so the pool bytes come back identical.
    A suite that only round-trips whole blocks therefore stays green on NH=2
    even pre-fix -- only an explicit ordering check (the D2H-vs-torch tests
    above) exposes it, and NH=1 (num_heads == 0) breaks even the round trip.
    This test stays green before AND after the fix.
    """
    device = torch.device("cuda:0")
    num_heads, content, block_size = 2, 16, 4
    buf, views = make_pool(NB, NL, num_heads, block_size, content, device)
    fmt, kv = detect_format(views, EngineType.VLLM, {"kv_layout": "BLHNC"})
    assert fmt == lmcache_native.EngineKVFormat.NL_X_NB_NH_BS_CS
    head_size = U.get_head_size(kv, fmt)
    assert head_size == content  # full width -- the broken path still passes
    block_stride = kv[0].stride(0)

    blocks = [1, 3]
    slots = [b * block_size + o for b in blocks for o in range(block_size)]
    staging = torch.zeros(
        1, NL, len(slots), num_heads * content, dtype=torch.float32, device=device
    )
    ptrs = torch.tensor([v.data_ptr() for v in kv], dtype=torch.int64, device=device)
    slots_t = torch.tensor(slots, dtype=torch.int64, device=device)
    device_ops.multi_layer_kv_transfer(
        staging, ptrs, slots_t, device, NB * block_size,
        int(lmcache_native.TransferDirection.D2H), int(fmt), block_size,
        head_size, 0, block_stride,
    )
    torch.cuda.synchronize()

    original = buf.clone()
    for b in blocks:
        buf[b, :NL].zero_()
    device_ops.multi_layer_kv_transfer(
        staging, ptrs, slots_t, device, NB * block_size,
        int(lmcache_native.TransferDirection.H2D), int(fmt), block_size,
        head_size, 0, block_stride,
    )
    torch.cuda.synchronize()

    for b in blocks:
        _assert_equal(
            buf[b, :NL], original[b, :NL],
            f"NH=2 whole-block round trip failed for block {b}",
        )
