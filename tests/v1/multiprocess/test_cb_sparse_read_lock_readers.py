# SPDX-License-Identifier: Apache-2.0
"""The CB sparse leg must reserve one read lock per KV reader.

``PrefetchRequestSpec.num_kv_readers`` is "total read locks to take per key --
one per reader", and each reader's retrieve releases exactly one. The prefix leg
passes it; the sparse leg used to omit it, so the spec defaulted to 1.

That is invisible while the KV cache is sharded -- the sparse leg expands one
key per rank, the dedup keeps them all, and each rank releases its own key's
single lock. Under MLA the KV is stored ONCE, so every rank expands to the SAME
object key, the dedup collapses them to one, and a single lock is handed to
whichever rank retrieves first. The other ``tp-1`` releases then underflow and
unpin the object while those ranks still need it -- exactly what
``IPCCacheServerKey.require_num_kv_readers`` warns about ("under-counting unpins
an object mid-copy"): the remaining ranks read ``KEY_NOT_READABLE``, their
scatter reports ``scatter_ran=False``, and CacheBlend degrades to full recompute.

These drive the real ``BlendModule``, matcher and key expansion; only the storage
manager is a lock-counting fake, so the assertions are on real accounting. The
fake honours ``spec.num_kv_readers`` -- a fake that ignores it cannot see this.
"""

# Standard
from unittest.mock import MagicMock

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import AttnWindowDesc, TrimPolicy
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.modules.blend import BlendModule
from lmcache.v1.multiprocess.session import SessionManager

CHUNK = 256
N_CHUNKS = 4
TP = 8


class _LockCountingStorageManager:
    """Counts read locks per key, honouring ``spec.num_kv_readers``.

    ``finish_read_prefetched`` is the only release; releasing more than is held
    raises, which is the underflow the server logs as "finish read on
    non-read-locked key".
    """

    def __init__(self) -> None:
        self.locks: dict = {}
        self.reserved_per_key: int | None = None

    def submit_prefetch_task(self, spec, external_request_id=None):
        if spec.policy == TrimPolicy.SPARSE:
            self.reserved_per_key = spec.num_kv_readers
            for key in spec.keys:
                self.locks[key] = self.locks.get(key, 0) + spec.num_kv_readers
        handle = MagicMock()
        handle.keys = list(spec.keys)
        handle.l2_orig_indices = []
        return handle

    def query_prefetch_status(self, handle):
        bitmap = MagicMock()
        bitmap.get_indices_list.return_value = list(range(len(handle.keys)))
        return bitmap

    def finish_read_prefetched(self, keys, read_locks: int = 1) -> None:
        for key in keys:
            held = self.locks.get(key, 0)
            if held < read_locks:
                raise AssertionError(
                    f"read-lock underflow on {key}: held={held} release={read_locks}"
                )
            self.locks[key] = held - read_locks

    def outstanding(self) -> int:
        return sum(self.locks.values())


def _make_ctx(storage, desc_world_size: int) -> MagicMock:
    ctx = MagicMock()
    ctx.chunk_size = CHUNK
    ctx.storage_manager = storage
    ctx.event_bus.has_subscribers.return_value = False
    ctx.token_hasher.compute_chunk_hashes.return_value = []  # prefix coverage 0
    ctx.layout_desc_registry.find_group_layout_descs.return_value = {0: MagicMock()}
    ctx.layout_desc_registry.find_attn_desc.return_value = AttnWindowDesc(
        num_chunks_in_sw=[-1], world_size=desc_world_size, group_kinds=("attention",)
    )
    ctx.session_manager = SessionManager(
        hasher=MagicMock(), ttl=600, cleanup_interval=None
    )
    return ctx


def _lookup(blend: BlendModule, request_id: str, world_size: int, readers: int):
    """Register N_CHUNKS fingerprints and run a lookup that finds them all
    shifted, so the sparse leg read-locks every chunk."""
    stored = list(range(1000, 1000 + N_CHUNKS * CHUNK))
    hashes = [f"{request_id}-h{i}".encode() for i in range(N_CHUNKS)]
    assert (
        blend._token_range_matcher.on_new_token_hashes(
            stored, hashes, start_chunk_idx=0, position_offset=0
        )
        == N_CHUNKS
    )
    key = IPCCacheServerKey(
        model_name="m",
        world_size=world_size,
        num_kv_readers=readers,
        worker_id=None,
        token_ids=tuple(list(range(50_000, 50_128)) + stored),
        start=0,
        end=128 + len(stored),
        request_id=request_id,
    )
    result = blend.cb_unified_lookup(key, tp_size=TP)
    assert result is not None
    assert result.prefix_coverage_tokens == 0
    assert len(result.non_prefix_segments) == N_CHUNKS
    return blend._lookup_obj_keys_cache[request_id]


def test_mla_replicated_reserves_one_lock_per_reader():
    """MLA stores the KV once: every rank expands to the same key, the dedup
    collapses them, so the single key must carry one lock per rank."""
    storage = _LockCountingStorageManager()
    # MLA: vLLM passes world_size already divided by tp (tp > world_size).
    blend = BlendModule(_make_ctx(storage, 1), lmcache_driven_transfer=MagicMock())

    cached = _lookup(blend, "req-mla", world_size=1, readers=TP)

    assert len(cached) == N_CHUNKS
    assert all(len(keys) == 1 for keys in cached.values()), (
        "MLA must dedup to a single shared object key per chunk"
    )
    assert storage.reserved_per_key == TP, (
        f"sparse leg reserved {storage.reserved_per_key} lock(s) per key, not "
        f"num_kv_readers={TP}: the 2nd..{TP}th rank's release underflows and "
        f"unpins the object while they still need it"
    )
    assert len(storage.locks) == N_CHUNKS
    assert storage.outstanding() == N_CHUNKS * TP


def test_mla_replicated_survives_one_release_per_rank():
    """The accounting balances: TP per-rank releases, no underflow, nothing left
    pinned."""
    storage = _LockCountingStorageManager()
    blend = BlendModule(_make_ctx(storage, 1), lmcache_driven_transfer=MagicMock())
    cached = _lookup(blend, "req-mla-drain", world_size=1, readers=TP)
    shared = [keys[0] for keys in cached.values()]

    for _rank in range(TP):
        storage.finish_read_prefetched(shared)  # each rank releases its one lock

    assert storage.outstanding() == 0, "locks left pinned after every rank released"
    with pytest.raises(AssertionError, match="underflow"):
        storage.finish_read_prefetched(shared)  # a TP+1'th release must be an error


def test_sharded_kv_keeps_one_lock_per_rank_key():
    """Non-MLA: one key per rank, one lock each -- the fix must not over-reserve
    here (num_kv_readers is 1, so it is a no-op)."""
    storage = _LockCountingStorageManager()
    blend = BlendModule(_make_ctx(storage, TP), lmcache_driven_transfer=MagicMock())

    cached = _lookup(blend, "req-sharded", world_size=TP, readers=1)

    assert all(len(keys) == TP for keys in cached.values())
    assert len({k for keys in cached.values() for k in keys}) == N_CHUNKS * TP
    assert storage.reserved_per_key == 1
    assert storage.outstanding() == N_CHUNKS * TP

    for keys in cached.values():
        storage.finish_read_prefetched(keys)  # each rank releases its own key
    assert storage.outstanding() == 0
