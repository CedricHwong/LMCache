# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the ``EngineGroupInfo`` structural proto codec.

Phase 1 appended 8 V4.1 fields to
``lmcache.v1.multiprocess.group_view.EngineGroupInfo`` without updating
``common.proto``.  ``proto_codec._compile_message_codec`` zips Python struct
fields with proto fields positionally and requires equal counts, so the whole
multiprocess gRPC path (register/store/retrieve) raised ``TypeError: no
structural codec ...`` at setup.

These tests pin both halves of the contract:

* the proto message and the Python struct stay in field-by-field lockstep, and
* every field survives a real protobuf serialization round trip -- including
  an exact fractional ``tokens_per_state`` and a legacy payload that predates
  the V4.1 tail (which must decode to the Python defaults, not the proto3
  defaults).
"""

# Standard
from fractions import Fraction

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.transport.grpc_impl import proto_codec
from lmcache.v1.multiprocess.transport.grpc_impl._proto_gen import common_pb2


def _compile_codec() -> tuple[
    proto_codec.FieldWriter, proto_codec.FieldReader
]:
    """Compile the structural codec, failing if proto and struct disagree."""
    return proto_codec._compile_message_codec(
        common_pb2.EngineGroupInfo.DESCRIPTOR, EngineGroupInfo
    )


def _round_trip(value: EngineGroupInfo) -> EngineGroupInfo:
    """Write, serialize, parse, and read back one ``EngineGroupInfo``."""
    writer, reader = _compile_codec()
    message = common_pb2.EngineGroupInfo()
    writer(message, value)
    parsed = common_pb2.EngineGroupInfo()
    parsed.ParseFromString(message.SerializeToString())
    return reader(parsed)


def test_proto_fields_match_struct_field_by_field() -> None:
    """The proto message mirrors the Python struct name-for-name, in order."""
    proto_names = [field.name for field in common_pb2.EngineGroupInfo.DESCRIPTOR.fields]
    assert proto_names == list(EngineGroupInfo.__struct_fields__)


def test_structural_codec_compiles_for_engine_group_info() -> None:
    """The exact code path that regressed: compiling the codec must not raise."""
    _compile_codec()


def test_full_v41_group_round_trips() -> None:
    """Every one of the 14 fields survives a serialized round trip."""
    group = EngineGroupInfo(
        0,
        (1, 2, 3),
        tokens_per_block=64,
        sw_size_tokens=128,
        extra_object_group_tag=2,
        recurrent_state=True,
        tokens_per_state=2,
        cache_role="indexer",
        state_content_bytes=584,
        block_stride_alignment=576,
        is_index_group_leader=True,
        prefix_cacheable=False,
        page_size_padded=640,
        num_head_slots=1,
    )
    assert _round_trip(group) == group


def test_default_group_round_trips() -> None:
    """A defaulted group round trips with the ``None`` optionals intact."""
    group = EngineGroupInfo(3)
    assert _round_trip(group) == group


@pytest.mark.parametrize(
    "tokens_per_state", [0, 1, 2, Fraction(3, 2), Fraction(1, 3)]
)
def test_tokens_per_state_round_trips_exactly(
    tokens_per_state: int | Fraction,
) -> None:
    """``tokens_per_state`` keeps its exact value, fractional included.

    A plain ``int64`` would encode ``Fraction(1, 2)`` as ``0``; the
    numerator/denominator carrier must not lose the fraction.
    """
    group = EngineGroupInfo(0, tokens_per_block=64, tokens_per_state=tokens_per_state)
    decoded = _round_trip(group)
    assert decoded.tokens_per_state == tokens_per_state
    assert type(decoded.tokens_per_state) is type(tokens_per_state)


def test_legacy_payload_without_v41_tail_decodes_to_python_defaults() -> None:
    """An old (6-field) payload decodes to the Python contract defaults.

    Reading the raw proto3 defaults instead would turn ``prefix_cacheable``
    into ``False`` and ``cache_role`` into ``""``, silently changing the
    group's caching semantics.
    """
    legacy = common_pb2.EngineGroupInfo()
    legacy.engine_group_id = 0
    legacy.layer_indices.extend([4, 5])
    legacy.tokens_per_block = 32
    legacy.sw_size_tokens = 128
    legacy.extra_object_group_tag = 0
    legacy.recurrent_state = False

    parsed = common_pb2.EngineGroupInfo()
    parsed.ParseFromString(legacy.SerializeToString())
    _, reader = _compile_codec()
    decoded = reader(parsed)

    assert decoded == EngineGroupInfo(0, (4, 5), tokens_per_block=32, sw_size_tokens=128)
    assert decoded.prefix_cacheable is True
    assert decoded.cache_role == "sparse"
    assert decoded.tokens_per_state == 1
    assert decoded.state_content_bytes is None


def test_non_rational_tokens_per_state_fails_closed() -> None:
    """A float must not be silently truncated into the integer wire field."""
    writer, _ = _compile_codec()
    message = common_pb2.EngineGroupInfo()
    group = EngineGroupInfo(0, tokens_per_block=64, tokens_per_state=1.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        writer(message, group)
