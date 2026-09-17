# SPDX-License-Identifier: Apache-2.0
"""Standard-suite gate: ``EngineGroupInfo`` and its protobuf schema in lockstep.

``tests/v1/multiprocess`` is excluded from the standard AGENTS.md test command,
so a Python-struct/proto drift there is invisible to that command even though
it breaks the whole multiprocess gRPC path at codec-compile time (phase 1
appended 8 fields to the struct and forgot ``common.proto``).

This test parses ``common.proto`` textually -- it needs no generated stubs --
and asserts that the wire contract still mirrors the Python struct.  It runs
in the standard suite, so the same drift fails fast outside the ignored
directories.
"""

# Standard
from pathlib import Path
import re

# First Party
from lmcache.v1.multiprocess.group_view import EngineGroupInfo

_PROTO_PATH = (
    Path(__file__).resolve().parents[2]
    / "lmcache/v1/multiprocess/transport/grpc_impl/protos/common.proto"
)
_FIELD_RE = re.compile(
    r"^\s*(?:(optional|repeated)\s+)?([A-Za-z_][\w.]*)\s+"
    r"([A-Za-z_]\w*)\s*=\s*(\d+)\s*;",
    re.MULTILINE,
)


def _proto_message_body(message_name: str) -> str:
    """Return the text between the braces of one ``message`` declaration.

    Args:
        message_name: The protobuf message name to locate.

    Returns:
        The message body (declaration lines only, nested braces included).

    Raises:
        AssertionError: If the message declaration is not found.
    """
    text = _PROTO_PATH.read_text()
    match = re.search(rf"^\s*message\s+{re.escape(message_name)}\s*\{{", text, re.M)
    assert match is not None, f"message {message_name} not found in {_PROTO_PATH}"
    start = match.end()
    depth = 1
    index = start
    while depth:
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        index += 1
    return text[start : index - 1]


def _proto_fields(message_name: str) -> list[tuple[str | None, str, str, int]]:
    """Return ``(label, type, name, number)`` per scalar/message field.

    Args:
        message_name: The protobuf message name to parse.

    Returns:
        One tuple per top-level field declaration, in declaration order.
        ``label`` is ``"optional"``, ``"repeated"``, or ``None`` for a plain
        proto3 field.
    """
    return [
        (label or None, field_type, name, int(number))
        for label, field_type, name, number in _FIELD_RE.findall(
            _proto_message_body(message_name)
        )
    ]


def test_engine_group_info_proto_matches_python_struct() -> None:
    """Proto field names, order, and count equal the Python struct's."""
    names = [name for _, _, name, _ in _proto_fields("EngineGroupInfo")]
    assert names == list(EngineGroupInfo.__struct_fields__)


def test_engine_group_info_field_numbers_are_append_only() -> None:
    """Field numbers stay dense from 1; numbering is never reused."""
    numbers = [number for _, _, _, number in _proto_fields("EngineGroupInfo")]
    assert numbers == list(range(1, len(EngineGroupInfo.__struct_fields__) + 1))


def test_presence_sensitive_v41_fields_declare_proto_presence() -> None:
    """Fields whose Python default differs from proto3's must track presence.

    Without presence, a payload from an older build that predates an appended
    field decodes to the proto3 default instead of the Python one:
    ``prefix_cacheable`` would become ``False`` and ``cache_role`` ``""``.
    """
    fields = {
        name: (label, field_type)
        for label, field_type, name, _ in _proto_fields("EngineGroupInfo")
    }
    assert fields["tokens_per_state"] == (None, "TokensPerState")
    assert fields["cache_role"] == ("optional", "string")
    assert fields["prefix_cacheable"] == ("optional", "bool")
    for name in (
        "state_content_bytes",
        "block_stride_alignment",
        "page_size_padded",
        "num_head_slots",
    ):
        assert fields[name][0] == "optional", name


def test_tokens_per_state_carrier_is_a_lossless_rational() -> None:
    """The ``tokens_per_state`` carrier keeps numerator and denominator."""
    fields = {
        name: (field_type, number)
        for _, field_type, name, number in _proto_fields("TokensPerState")
    }
    assert fields == {"numerator": ("int64", 1), "denominator": ("int64", 2)}
