# SPDX-License-Identifier: Apache-2.0
"""Cache-format fingerprint: isolate KV objects whose bytes differ.

Two LMCache deployments can serve the *same model* with the *same tokenizer*
and the *same prompt* while storing byte layouts that are completely
incompatible -- most obviously different ``--kv-cache-dtype`` values
(``fp8_ds_mla`` 584 B/token with 576 B pages vs a plain bf16 row), but also
different ``block_size``, page padding, ``tokens_per_state``, physical
layout permutation, DCP sharding, and so on.

The L2 object key, however, is derived only from ``(model_name, chunk token
hash, kv_rank, object_group_id, cache_salt)``. None of those changes when the
dtype or the page geometry changes, so a request can match an object written
by a deployment with a *different* byte layout. The read then has no size or
stride mismatch to fail on: it silently decodes garbage. That is the failure
mode this module removes.

A *format fingerprint* is a short, stable string that captures every
registered dimension which changes the byte layout. Mixing it into the object
key's chunk hash (:func:`namespace_object_chunk_hash`) gives every layout its
own namespace, so cross-layout hits become impossible by construction while
same-layout hits are unaffected.

The fingerprint is computed from the descriptors LMCache already builds at
``REGISTER_KV_CACHE`` time (:func:`descriptors_from_registration`), so no new
information has to cross the wire and no storage adapter needs to change: a
different chunk hash already produces a different object name in every L2
backend, including the C++ filesystem connector.
"""

# Standard
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any, Sequence
import json

KV_FORMAT_FINGERPRINT_VERSION = "v1"
"""Schema version of the descriptor set.

Bump this whenever :class:`KVGroupFormatDescriptor`'s field set changes, so
old and new fingerprints never alias. It is embedded in the fingerprint
string and in the hashed payload.
"""

_FINGERPRINT_HEX_LEN = 16
"""Hex characters of the SHA-256 digest kept in the fingerprint.

64 bits of digest is far beyond what a per-deployment namespace needs: two
different layouts colliding requires a deliberate 64-bit preimage, and a
collision would mis-isolate one deployment, not open a shared namespace.
"""

_SEPARATOR = b"\x00"
"""Domain separator between the fingerprint and the raw chunk hash."""


@dataclass(frozen=True)
class KVGroupFormatDescriptor:
    """Every byte-layout-determining property of one LMCache KV group.

    One descriptor per kernel group, built from the engine-declared
    :class:`~lmcache.v1.multiprocess.group_view.EngineGroupInfo` and the
    tensor-detected kernel-group geometry. Fields that cannot change the
    bytes -- layer indices, prefix-cacheability, group order -- are
    deliberately absent: including them would only cause spurious misses
    when a model is re-sharded or re-parallelised without changing layout.

    Attributes:
        engine_group_id: Engine paged-block address space this group draws
            block ids from. Block ids are only meaningful within one space.
        dtype: Torch dtype name of the registered KV tensors (``uint8`` for
            packed ``*_ds_mla`` records, ``bfloat16`` / ``float8_e4m3fn``
            for plain rows).
        engine_kv_format: Detected engine KV format token (the
            ``EngineKVFormat`` enum name and value); this encodes the
            physical page permutation and plane split.
        kv_size: Transfer kernel K/V plane count (1 fused, 2 split).
        num_blocks: Physical block count of the group's tensor.
        slots_per_block: Physical state slots in one paged chunk detected
            from the registered tensors (the batch-size axis).
        num_heads: Heads per page.
        head_size: Elements per head row (the packed record width when
            ``state_content_bytes`` is set).
        element_size: Bytes per element of the registered tensors.
        block_stride_elems: Per-block stride in elements, i.e. the padded
            distance between consecutive blocks.
        tokens_per_block: Logical engine tokens covered by one paged chunk.
        tokens_per_state: Tokens covered by one stored state (the V4.1
            compression granularity).
        sw_size_tokens: Sliding-window size in tokens; ``-1`` = full
            attention.
        recurrent_state: Whether pages hold recurrent state snapshots.
        cache_role: Engine-declared cache role (``"sparse"`` / ``"indexer"``).
        state_content_bytes: Per-cell packed content width in bytes, or
            ``None`` for dense K/V content.
        block_stride_alignment: Required byte alignment of the inter-block
            distance, or ``None``.
        page_size_padded: Padded page size in bytes, or ``None`` when
            unpadded.
        num_head_slots: Logical head count of the page when it diverges
            from one slot per KV head, or ``None``.
    """

    engine_group_id: int
    dtype: str
    engine_kv_format: str
    kv_size: int
    num_blocks: int
    slots_per_block: int
    num_heads: int
    head_size: int
    element_size: int
    block_stride_elems: int
    tokens_per_block: int
    tokens_per_state: int
    sw_size_tokens: int
    recurrent_state: bool
    cache_role: str
    state_content_bytes: int | None
    block_stride_alignment: int | None
    page_size_padded: int | None
    num_head_slots: int | None


def _enum_token(value: Any) -> str:
    """Render an engine KV format (or ``None``) as a stable string.

    Args:
        value: An ``EngineKVFormat``-like enum member, ``None``, or any
            object whose ``str`` is stable.

    Returns:
        ``"none"`` for ``None``; ``"<name>:<int>"`` for an integer enum
        member; otherwise ``str(value)``.
    """
    if value is None:
        return "none"
    name = getattr(value, "name", None)
    if name is not None and hasattr(value, "__int__"):
        return f"{name}:{int(value)}"
    return str(value)


def _int_or_none(value: Any) -> int | None:
    """Coerce an optional geometry scalar, preserving ``None``."""
    return None if value is None else int(value)


def descriptors_from_registration(
    kernel_groups: Sequence[Any],
    engine_group_infos: Sequence[Any],
) -> list[KVGroupFormatDescriptor]:
    """Build one format descriptor per LMCache KV group from a registration.

    The kernel groups and the engine group infos are produced by the same
    bucketing on either side of the wire and correspond 1:1 in order (see
    ``KVLayerGroupsManager`` construction), so the two sequences are zipped
    positionally and each pair yields one descriptor.

    Every read is defensive: a field the engine or the tensor detection does
    not expose falls back to a neutral value rather than raising, so a
    slightly older engine still produces a usable (more conservative)
    fingerprint.

    Args:
        kernel_groups: Detected kernel groups (``KernelGroupInfo``-like),
            each exposing ``shape_desc``, ``dtype``, ``engine_kv_format``,
            ``tokens_per_block``, ``engine_group_idx``, ``sw_size_tokens``,
            and ``tokens_per_state``.
        engine_group_infos: Engine-declared group metadata
            (``EngineGroupInfo``-like), in the same order, exposing
            ``page_size_padded``, ``num_head_slots``, ``state_content_bytes``,
            ``block_stride_alignment``, ``cache_role``, and
            ``engine_group_id``.

    Returns:
        One :class:`KVGroupFormatDescriptor` per group, in protocol order.

    Raises:
        ValueError: The two sequences disagree on the number of groups.
    """
    if engine_group_infos and len(engine_group_infos) != len(kernel_groups):
        raise ValueError(
            f"got {len(engine_group_infos)} engine group infos for "
            f"{len(kernel_groups)} kernel groups; expecting one info per group"
        )

    descriptors: list[KVGroupFormatDescriptor] = []
    for group_idx, group in enumerate(kernel_groups):
        shape = group.shape_desc
        info = engine_group_infos[group_idx] if engine_group_infos else None
        cache_role = getattr(info, "cache_role", None) if info is not None else None
        descriptors.append(
            KVGroupFormatDescriptor(
                engine_group_id=int(
                    getattr(info, "engine_group_id", None)
                    if info is not None
                    else getattr(group, "engine_group_idx", 0)
                ),
                dtype=str(getattr(group, "dtype", "")),
                engine_kv_format=_enum_token(getattr(group, "engine_kv_format", None)),
                kv_size=int(shape.kv_size),
                num_blocks=int(shape.nb),
                slots_per_block=int(shape.bs),
                num_heads=int(shape.nh),
                head_size=int(shape.hs),
                element_size=int(shape.element_size),
                block_stride_elems=int(shape.block_stride_elems),
                tokens_per_block=int(getattr(group, "tokens_per_block", 0)),
                tokens_per_state=int(getattr(group, "tokens_per_state", 1)),
                sw_size_tokens=int(getattr(group, "sw_size_tokens", -1)),
                recurrent_state=bool(getattr(group, "recurrent_state", False)),
                cache_role=str(cache_role) if cache_role else "sparse",
                state_content_bytes=_int_or_none(
                    getattr(info, "state_content_bytes", None)
                    if info is not None
                    else None
                ),
                block_stride_alignment=_int_or_none(
                    getattr(info, "block_stride_alignment", None)
                    if info is not None
                    else None
                ),
                page_size_padded=_int_or_none(
                    getattr(info, "page_size_padded", None)
                    if info is not None
                    else None
                ),
                num_head_slots=_int_or_none(
                    getattr(info, "num_head_slots", None) if info is not None else None
                ),
            )
        )
    return descriptors


def resolve_object_key_fingerprint(
    context: Any, model_name: str, world_size: int
) -> str:
    """Read the format fingerprint a server context advertises for a model.

    Every object-key construction site resolves the fingerprint through this
    helper, so a deployment that does not expose one keeps working: a missing
    or non-``str`` accessor means "fingerprint nothing", which is exactly the
    legacy, un-namespaced behavior. That tolerance is what makes the
    integration additive for engines and test doubles that predate the
    accessor.

    Args:
        context: The server context holding the registration. It must expose
            a callable ``format_fingerprint(model_name, world_size) -> str``
            to participate in namespacing.
        model_name: The model name whose registration to read.
        world_size: The world size whose registration to read.

    Returns:
        The registered fingerprint, or ``""`` when the context does not
        provide one.
    """
    accessor = getattr(context, "format_fingerprint", None)
    if not callable(accessor):
        return ""
    value = accessor(model_name, world_size)
    return value if isinstance(value, str) else ""


def compute_format_fingerprint(
    descriptors: Sequence[KVGroupFormatDescriptor],
    *,
    world_size: int,
    kv_cache_dtype: str = "",
) -> str:
    """Hash a set of KV group descriptors into a stable namespace token.

    The fingerprint is order-independent in the sense that it is computed
    over a canonically sorted rendering of the descriptors, so two
    registrations that differ only in group enumeration order produce the
    same fingerprint. Any difference in a descriptor field, in the world
    size, or in the declared cache dtype produces a different fingerprint.

    Args:
        descriptors: The per-group format descriptors of one registration.
        world_size: Tensor/pipeline parallel world size. Included because the
            per-rank shard of the KV cache -- and therefore the bytes one
            rank stores -- depends on it.
        kv_cache_dtype: The engine's resolved ``--kv-cache-dtype`` string when
            the caller knows it (e.g. ``"fp8_ds_mla"``, ``"bfloat16"``). The
            registered tensor dtype alone cannot distinguish two packed record
            families that share a byte width, so supplying the declared string
            makes the fingerprint strictly stronger. Empty when unknown.

    Returns:
        A ``"<version>-<hex>"`` string, e.g. ``"v1-3f9a0c7d21b45e88"``. Empty
        ``descriptors`` is legal and yields the fingerprint of an empty group
        set.
    """
    rendered = sorted(
        (json.dumps(asdict(descriptor), sort_keys=True) for descriptor in descriptors)
    )
    payload = json.dumps(
        {
            "version": KV_FORMAT_FINGERPRINT_VERSION,
            "world_size": int(world_size),
            "kv_cache_dtype": kv_cache_dtype,
            "groups": rendered,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = sha256(payload.encode("utf-8")).hexdigest()[:_FINGERPRINT_HEX_LEN]
    return f"{KV_FORMAT_FINGERPRINT_VERSION}-{digest}"


def namespace_object_chunk_hash(chunk_hash: bytes, format_fingerprint: str) -> bytes:
    """Derive a namespaced object chunk hash from a content chunk hash.

    The returned hash is a keyed digest of the fingerprint and the raw chunk
    hash, truncated back to the input length so downstream consumers that
    assume a fixed digest width keep working. An empty ``format_fingerprint``
    returns ``chunk_hash`` unchanged, which keeps un-fingerprinted (legacy)
    deployments byte-for-byte compatible with their existing caches.

    Args:
        chunk_hash: The raw content chunk hash bytes (token-only).
        format_fingerprint: A fingerprint from
            :func:`compute_format_fingerprint`, or ``""`` to opt out of
            namespacing.

    Returns:
        The namespaced chunk hash bytes: identical to ``chunk_hash`` when
        ``format_fingerprint`` is empty, otherwise a same-length digest that
        differs for every different fingerprint.

    Raises:
        ValueError: ``format_fingerprint`` is not a ``str``.
    """
    if not isinstance(format_fingerprint, str):
        raise ValueError(
            "format_fingerprint must be a str, got "
            f"{type(format_fingerprint).__name__}"
        )
    if not format_fingerprint:
        return chunk_hash
    prefix = format_fingerprint.encode("utf-8")
    digest = sha256(prefix + _SEPARATOR + chunk_hash).digest()
    if not chunk_hash:
        return digest
    return digest[: len(chunk_hash)]
