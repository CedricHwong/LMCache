# SPDX-License-Identifier: Apache-2.0
"""A failed read batch must not strand the NEXT retrieve of the same request.

``read_prefetched_results`` releases the read locks of every SUCCESSFULLY read
key when any key in the batch fails (``not all_good``) -- deliberate, so a
single-shot read leaves nothing dangling (see test_read_prefetched_not_found).

The CB retrieve is not single-shot. vLLM re-runs it per block-alloc round
(measured up to 6x for one request on minimax_m3), the lookup read-locks the
keys ONCE, and ``_release_scattered_locks`` deliberately leaves unapplied
matches locked "for vLLM's full-alloc follow-up retrieve". So a failed batch
handing those locks back leaves every later attempt reading KEY_NOT_READABLE --
with NO underflow warning, because the count went 1 -> 0 legitimately.

That is exactly minimax_m3's signature: 4,448 read failures and zero "finish
read on non-read-locked key" warnings, uniform across all 8 ranks (556 each),
on 8 distinct per-rank kv_ranks.

Fix: ``release_on_failure=False`` for such callers -- they own the release.
Holding a lock only risks pinning to the TTL; releasing early strands every
later reader.
"""

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey, PrefetchRequestSpec
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.storage_manager import StorageManager
from tests.v1.distributed.utils import should_use_lazy_alloc


def _key(i: int, kv_rank: int = 0) -> ObjectKey:
    return ObjectKey(
        chunk_hash=f"chunk{i}".encode().ljust(32, b"\0"),
        model_name="m",
        kv_rank=kv_rank,
        object_group_id=0,
        cache_salt="",
    )


@pytest.fixture
def storage_manager():
    cfg = StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=128 * 1024 * 1024,
                use_lazy=should_use_lazy_alloc(),
                init_size_in_bytes=64 * 1024 * 1024,
                align_bytes=0x1000,
            ),
            write_ttl_seconds=600,
            read_ttl_seconds=300,
        ),
        eviction_config=EvictionConfig(eviction_policy="LRU"),
    )
    sm = StorageManager(cfg)
    yield sm
    sm.close()


@pytest.fixture
def layout():
    return MemoryLayoutDesc(shapes=[torch.Size([100, 2, 512])], dtypes=[torch.bfloat16])


def test_second_retrieve_survives_a_failed_batch(storage_manager, layout):
    """The m3 shape: one reader per key, read twice. With
    release_on_failure=False the failed batch leaves the reservation intact, so
    the follow-up retrieve of the same request still reads."""
    sm = storage_manager
    good = [_key(i) for i in range(1, 5)]
    absent = _key(99)  # never written -> KEY_NOT_EXIST, fails the batch

    ret = sm.reserve_write(good, layout, mode="new")
    sm.finish_write(list(ret.keys()))

    # The CB lookup reserves the read locks once, for this request.
    handle = sm.submit_prefetch_task(
        PrefetchRequestSpec(good, {0: layout}, num_kv_readers=1)
    )
    assert sm.query_prefetch_status(handle).count_leading_ones() == len(good)

    # Retrieve #1 (partial alloc): batch includes a key that is not readable.
    with sm.read_prefetched_results(good + [absent], release_on_failure=False) as objs:
        assert objs is None, "batch with an absent key must yield None"

    # The reservation must survive it.
    err, _ = sm._l1_manager.unsafe_read([good[0]])[good[0]]
    assert err is L1Error.SUCCESS, (
        f"retrieve #1's failure unpinned a good key ({err}); retrieve #2 of the "
        f"same request will read KEY_NOT_READABLE"
    )

    # Retrieve #2 (full alloc), same request, only the good keys.
    with sm.read_prefetched_results(good, release_on_failure=False) as objs:
        assert objs is not None, "retrieve #2 was stranded by retrieve #1"
        assert len(objs) == len(good)

    # The caller owns the release; it drains cleanly.
    sm.finish_read_prefetched(good)
    err, _ = sm._l1_manager.unsafe_read([good[0]])[good[0]]
    assert err is L1Error.KEY_NOT_READABLE


def test_default_still_releases_on_failure(storage_manager, layout):
    """Single-shot readers keep the old behaviour: a failed batch hands the
    locks back, so nothing dangles (test_read_prefetched_not_found's contract)."""
    sm = storage_manager
    good = [_key(i) for i in range(1, 5)]
    absent = _key(99)
    ret = sm.reserve_write(good, layout, mode="new")
    sm.finish_write(list(ret.keys()))
    handle = sm.submit_prefetch_task(
        PrefetchRequestSpec(good, {0: layout}, num_kv_readers=1)
    )
    assert sm.query_prefetch_status(handle).count_leading_ones() == len(good)

    with sm.read_prefetched_results(good + [absent]) as objs:  # default True
        assert objs is None

    err, _ = sm._l1_manager.unsafe_read([good[0]])[good[0]]
    assert err is L1Error.KEY_NOT_READABLE, (
        "default must still release on failure -- no dangling read locks"
    )


def test_extra_reader_survives_the_failed_batch(storage_manager, layout):
    """Control: an extra reader also survives, so the fix is not load-bearing
    for callers that do reserve per-read. Kept as a cross-check that the
    mechanism is lock accounting, not corruption."""
    sm = storage_manager
    good = [_key(i) for i in range(1, 5)]
    absent = _key(99)

    ret = sm.reserve_write(good, layout, mode="new")
    sm.finish_write(list(ret.keys()))
    handle = sm.submit_prefetch_task(
        PrefetchRequestSpec(good, {0: layout}, num_kv_readers=2)
    )
    assert sm.query_prefetch_status(handle).count_leading_ones() == len(good)

    with sm.read_prefetched_results(good + [absent]) as objs:
        assert objs is None

    with sm.read_prefetched_results(good) as objs:
        assert objs is not None, "second retrieve must still read its own lock"
