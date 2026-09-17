# SPDX-License-Identifier: Apache-2.0
"""Regression tests for KV cache byte-layout (format) fingerprint isolation.

The behavioural contract under test:

* the fingerprint of a registration changes if -- and only if -- a dimension
  that changes the KV byte layout changes;
* object keys resolved for two different fingerprints never collide, while
  keys resolved for the same fingerprint do;
* a registration with no format metadata keeps its legacy, un-namespaced
  keys so existing caches stay reachable.
"""

# Standard
from dataclasses import fields, replace
from types import SimpleNamespace

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ipc_key_to_object_keys
from lmcache.v1.kv_format_fingerprint import (
    KV_FORMAT_FINGERPRINT_VERSION,
    KVGroupFormatDescriptor,
    compute_format_fingerprint,
    descriptors_from_registration,
    namespace_object_chunk_hash,
)
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey


def _base_descriptor(**overrides) -> KVGroupFormatDescriptor:
    """A complete fp8_ds_mla-shaped descriptor, with per-field overrides."""
    values = {
        "engine_group_id": 0,
        "dtype": "torch.uint8",
        "engine_kv_format": "NL_X_NB_NH_BS_CS:12",
        "kv_size": 1,
        "num_blocks": 512,
        "slots_per_block": 64,
        "num_heads": 1,
        "head_size": 584,
        "element_size": 1,
        "block_stride_elems": 576 * 8,
        "tokens_per_block": 64,
        "tokens_per_state": 1,
        "sw_size_tokens": -1,
        "recurrent_state": False,
        "cache_role": "sparse",
        "state_content_bytes": 584,
        "block_stride_alignment": 576,
        "page_size_padded": 576 * 64,
        "num_head_slots": 1,
    }
    values.update(overrides)
    return KVGroupFormatDescriptor(**values)


def _grp(shape, **overrides):
    """A KernelGroupInfo-like stub exposing only what the builder reads."""
    base = {
        "shape_desc": SimpleNamespace(
            kv_size=shape.get("kv_size", 1),
            nb=shape.get("nb", 512),
            bs=shape.get("bs", 64),
            nh=shape.get("nh", 1),
            hs=shape.get("hs", 584),
            element_size=shape.get("element_size", 1),
            block_stride_elems=shape.get("block_stride_elems", 576 * 8),
        ),
        "dtype": shape.get("dtype", "torch.uint8"),
        "engine_kv_format": shape.get("engine_kv_format", "NL_X_NB_NH_BS_CS:12"),
        "tokens_per_block": shape.get("tokens_per_block", 64),
        "tokens_per_state": shape.get("tokens_per_state", 1),
        "sw_size_tokens": shape.get("sw_size_tokens", -1),
        "recurrent_state": shape.get("recurrent_state", False),
        "engine_group_idx": shape.get("engine_group_idx", 0),
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _info(**overrides):
    """An EngineGroupInfo-like stub with the byte-layout spec fields."""
    base = {
        "engine_group_id": 0,
        "cache_role": "sparse",
        "state_content_bytes": 584,
        "block_stride_alignment": 576,
        "page_size_padded": 576 * 64,
        "num_head_slots": 1,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _ipc_key(**overrides) -> IPCCacheServerKey:
    """A minimal lookup key; ``worker_id=None`` expands to every rank."""
    values = {
        "model_name": "deepseek-v41-flash",
        "world_size": 8,
        "worker_id": None,
        "token_ids": (1, 2, 3, 4),
        "start": 0,
        "end": 4,
        "request_id": "r0",
        "cache_salt": "",
    }
    values.update(overrides)
    return IPCCacheServerKey(**values)


def _resolved_hashes(fingerprint: str) -> list[bytes]:
    """Resolve one chunk to its object keys under ``fingerprint``."""
    groups = ipc_key_to_object_keys(
        _ipc_key(), [b"\xab" * 32], [0], fingerprint
    )
    return [key.chunk_hash for key in groups[0]]


# --------------------------------------------------------------------------- #
# Fingerprint contract                                                        #
# --------------------------------------------------------------------------- #


def test_fingerprint_is_deterministic_and_versioned():
    """The same descriptors always yield the same versioned fingerprint."""
    descriptor = _base_descriptor()
    first = compute_format_fingerprint([descriptor], world_size=8)
    second = compute_format_fingerprint([descriptor], world_size=8)

    assert first == second
    assert first.startswith(f"{KV_FORMAT_FINGERPRINT_VERSION}-")


def test_fingerprint_is_independent_of_group_enumeration_order():
    """Reordering identical groups must not change the namespace."""
    first = _base_descriptor(engine_group_id=0)
    second = _base_descriptor(engine_group_id=1, head_size=132)

    forward = compute_format_fingerprint([first, second], world_size=8)
    reversed_ = compute_format_fingerprint([second, first], world_size=8)

    assert forward == reversed_


def test_fingerprint_changes_for_every_layout_dimension():
    """Every descriptor field -- and world size -- must move the fingerprint.

    This is the core guarantee: a dimension left out of the fingerprint is a
    dimension on which two incompatible caches could silently alias.
    """
    base = _base_descriptor()
    baseline = compute_format_fingerprint([base], world_size=8)

    for field in fields(KVGroupFormatDescriptor):
        original = getattr(base, field.name)
        if isinstance(original, bool):
            mutated = not original
        elif isinstance(original, int):
            mutated = original + 1
        elif isinstance(original, str):
            mutated = original + "-mutated"
        else:  # optional int fields
            mutated = (original or 0) + 1
        assert mutated != original, f"test cannot mutate {field.name}"
        variant = compute_format_fingerprint(
            [replace(base, **{field.name: mutated})], world_size=8
        )
        assert variant != baseline, (
            f"{field.name} does not affect the fingerprint; a layout change "
            "on that axis would alias"
        )

    assert compute_format_fingerprint([base], world_size=4) != baseline


def test_fingerprint_distinguishes_declared_cache_dtype():
    """The engine's --kv-cache-dtype string strengthens the fingerprint."""
    descriptor = _base_descriptor()
    fp8 = compute_format_fingerprint(
        [descriptor], world_size=8, kv_cache_dtype="fp8_ds_mla"
    )
    bf16 = compute_format_fingerprint(
        [descriptor], world_size=8, kv_cache_dtype="bfloat16"
    )

    assert fp8 != bf16


# --------------------------------------------------------------------------- #
# Descriptor construction from a registration                                 #
# --------------------------------------------------------------------------- #


def test_descriptors_from_registration_maps_engine_and_tensor_fields():
    """The builder merges engine spec fields with tensor-detected geometry."""
    descriptors = descriptors_from_registration(
        [_grp({}, engine_group_idx=3)], [_info(engine_group_id=3, cache_role="indexer")]
    )

    assert len(descriptors) == 1
    descriptor = descriptors[0]
    assert descriptor.engine_group_id == 3
    assert descriptor.cache_role == "indexer"
    assert descriptor.head_size == 584
    assert descriptor.slots_per_block == 64
    assert descriptor.page_size_padded == 576 * 64
    assert descriptor.block_stride_alignment == 576


def test_descriptors_from_registration_rejects_group_count_mismatch():
    """A registration whose halves disagree fails closed."""
    with pytest.raises(ValueError, match="engine group infos"):
        descriptors_from_registration([_grp({})], [_info(), _info()])


def test_descriptors_from_registration_distinguishes_indexer_row_width():
    """Main KV (584 B) and indexer K (132 B) groups fingerprint differently."""
    main = descriptors_from_registration([_grp({})], [_info()])
    indexer = descriptors_from_registration(
        [
            _grp(
                {"hs": 132, "block_stride_elems": 132 * 8},
                engine_group_idx=1,
            )
        ],
        [_info(engine_group_id=1, state_content_bytes=None)],
    )

    assert compute_format_fingerprint(
        main, world_size=8
    ) != compute_format_fingerprint(indexer, world_size=8)


# --------------------------------------------------------------------------- #
# Object-key namespacing                                                      #
# --------------------------------------------------------------------------- #


def test_namespaced_hash_is_identity_without_fingerprint():
    """Legacy callers keep byte-for-byte identical object keys."""
    raw = b"\x01" * 32
    assert namespace_object_chunk_hash(raw, "") == raw


def test_namespaced_hash_preserves_length_and_separates_fingerprints():
    """Namespacing is length-preserving and injective per fingerprint."""
    raw = b"\x02" * 32
    first = namespace_object_chunk_hash(raw, "v1-aaaa")
    second = namespace_object_chunk_hash(raw, "v1-bbbb")

    assert len(first) == len(raw)
    assert first != second
    assert first != raw


def test_different_formats_produce_disjoint_object_keys():
    """The regression: two byte layouts must not share a cache namespace."""
    fp8 = compute_format_fingerprint(
        [_base_descriptor()], world_size=8, kv_cache_dtype="fp8_ds_mla"
    )
    bf16_descriptor = _base_descriptor(
        dtype="torch.bfloat16",
        element_size=2,
        head_size=584,
        state_content_bytes=None,
        page_size_padded=None,
        block_stride_alignment=None,
    )
    bf16 = compute_format_fingerprint(
        [bf16_descriptor], world_size=8, kv_cache_dtype="bfloat16"
    )

    assert fp8 != bf16
    assert set(_resolved_hashes(fp8)).isdisjoint(_resolved_hashes(bf16))


def test_same_format_produces_identical_object_keys():
    """Two runs with the same layout still share their cache."""
    fingerprint = compute_format_fingerprint([_base_descriptor()], world_size=8)

    assert _resolved_hashes(fingerprint) == _resolved_hashes(fingerprint)


def test_no_fingerprint_reproduces_un_namespaced_keys():
    """An un-fingerprinted deployment resolves the raw content hash."""
    resolved = _resolved_hashes("")

    assert len(resolved) == 8  # one key per rank
    assert all(chunk_hash == b"\xab" * 32 for chunk_hash in resolved)


# --------------------------------------------------------------------------- #
# Registration-time namespace bookkeeping                                     #
# --------------------------------------------------------------------------- #


def _layout():
    """A one-shape uint8 layout descriptor for the registry tests."""
    # Third Party
    import torch

    # First Party
    from lmcache.v1.distributed.api import MemoryLayoutDesc

    return MemoryLayoutDesc(shapes=[torch.Size([4])], dtypes=[torch.uint8])


def test_registry_resolves_the_registered_fingerprint():
    """The registry is the server's single source of the namespace token."""
    # First Party
    from lmcache.v1.multiprocess.engine_context import LayoutDescRegistry

    registry = LayoutDescRegistry()
    registry.register("m", 8, _layout(), format_fingerprint="v1-aaaa")

    assert registry.find_format_fingerprint("m", 8) == "v1-aaaa"
    assert registry.find_format_fingerprint("unknown", 8) == ""


def test_registry_rejects_two_layouts_for_one_model():
    """Two byte layouts for one (model, world size) fail closed, not silently."""
    # First Party
    from lmcache.v1.multiprocess.engine_context import LayoutDescRegistry

    registry = LayoutDescRegistry()
    registry.register("m", 8, _layout(), format_fingerprint="v1-aaaa")

    with pytest.raises(ValueError, match="conflicting KV format fingerprints"):
        registry.register("m", 8, _layout(), format_fingerprint="v1-bbbb")
    assert registry.find_format_fingerprint("m", 8) == "v1-aaaa"


def test_registry_accepts_repeated_identical_registrations():
    """Ref-counted re-registration of the same layout keeps the namespace."""
    # First Party
    from lmcache.v1.multiprocess.engine_context import LayoutDescRegistry

    registry = LayoutDescRegistry()
    registry.register("m", 8, _layout(), format_fingerprint="v1-aaaa")
    registry.register("m", 8, _layout(), format_fingerprint="v1-aaaa")

    assert registry.find_format_fingerprint("m", 8) == "v1-aaaa"
