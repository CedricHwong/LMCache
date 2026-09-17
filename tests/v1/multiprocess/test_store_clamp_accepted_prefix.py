# SPDX-License-Identifier: Apache-2.0
"""Tests for clamp_store_key_to_accepted_prefix.

The helper bounds a store op's token range to the accepted (verified) prefix.
With speculative decoding the vLLM-side connector can build store ops whose
``end`` lies past the accepted tokens (rejected draft tokens are scheduled and
written to KV but never verified); without the clamp such an op would raise in
``Session.get_hashes`` and fail the whole store.
"""

# Standard

# Third Party

# First Party
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    clamp_store_key_to_accepted_prefix,
)


def make_key(token_ids: list[int], start: int, end: int) -> IPCCacheServerKey:
    return IPCCacheServerKey.from_token_ids(
        model_name="test",
        world_size=1,
        worker_id=0,
        token_ids=token_ids,
        start=start,
        end=end,
        request_id="req-1",
    )


class TestClampStoreKeyToAcceptedPrefix:
    def test_within_accepted_prefix_unchanged(self):
        key = make_key(list(range(256)), 0, 256)
        assert clamp_store_key_to_accepted_prefix(key, 256) is key

    def test_over_claim_clamped_to_chunk_boundary(self):
        # Accepted prefix is 250 tokens; the store op claims 256. No full
        # 256-token chunk fits inside the accepted prefix, so nothing to store.
        key = make_key(list(range(250)), 0, 256)
        assert clamp_store_key_to_accepted_prefix(key, 256) is None

    def test_over_claim_keeps_inner_full_chunks(self):
        # Accepted prefix is 520 tokens; over-claim to 768.
        # one full chunk [0, 512) survives, the trailing partial does not.
        key = make_key(list(range(520)), 0, 768)
        clamped = clamp_store_key_to_accepted_prefix(key, 256)
        assert clamped is not None
        assert clamped.end == 512

    def test_entirely_beyond_accepted_returns_none(self):
        # No full chunk remains inside the accepted prefix.
        key = make_key(list(range(200)), 0, 512)
        assert clamp_store_key_to_accepted_prefix(key, 256) is None

    def test_end_zero_noop(self):
        key = make_key(list(range(128)), 0, 0)
        assert clamp_store_key_to_accepted_prefix(key, 256) is key
