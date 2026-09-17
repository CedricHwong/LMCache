# SPDX-License-Identifier: Apache-2.0
"""Codecs for protobuf types shared by multiple gRPC services."""

# Standard
from fractions import Fraction

# First Party
from lmcache.v1.multiprocess.custom_types import DeviceIPCWrapper
from lmcache.v1.multiprocess.transport.grpc_impl._proto_gen import common_pb2
from lmcache.v1.multiprocess.transport.grpc_impl.codecs.base import (
    RegisteredMessageCodec,
)

TokensPerStateValue = int | Fraction


def _write_device_ipc_wrapper(
    message: common_pb2.DeviceIpcWrapper, value: DeviceIPCWrapper
) -> None:
    message.pickled_payload = DeviceIPCWrapper.Serialize(value)


def _read_device_ipc_wrapper(
    message: common_pb2.DeviceIpcWrapper,
) -> DeviceIPCWrapper:
    return DeviceIPCWrapper.Deserialize(message.pickled_payload)


def _write_tokens_per_state(
    message: common_pb2.TokensPerState, value: TokensPerStateValue
) -> None:
    """Write an exact rational ``tokens_per_state`` into the wire carrier.

    Args:
        message: The generated ``TokensPerState`` protobuf message.
        value: One group's tokens-per-state, an ``int`` for every
            engine-produced value or an exact ``Fraction`` produced by the
            in-process mask math.

    Raises:
        TypeError: If ``value`` is neither an ``int`` nor a ``Fraction``.
            A plain protobuf ``int64`` would silently truncate a fraction
            (``Fraction(1, 2)`` becomes ``0``), so unsupported types fail
            closed instead of corrupting the group geometry.
    """
    if isinstance(value, Fraction):
        message.numerator = value.numerator
        message.denominator = value.denominator
        return
    if isinstance(value, int):
        message.numerator = value
        message.denominator = 1
        return
    raise TypeError(
        f"tokens_per_state must be int or Fraction, got {type(value).__name__}"
    )


def _read_tokens_per_state(message: common_pb2.TokensPerState) -> TokensPerStateValue:
    """Read ``tokens_per_state`` back, restoring its exact value.

    An absent field (a legacy payload, or a payload written before the V4.1
    tail existed) leaves ``denominator == 0`` and decodes to the Python
    contract default ``1``.

    Args:
        message: The generated ``TokensPerState`` protobuf message.

    Returns:
        The integer value when the denominator is ``1``, otherwise the exact
        ``Fraction``.
    """
    if message.denominator == 0:
        return 1
    if message.denominator == 1:
        return message.numerator
    return Fraction(message.numerator, message.denominator)


def get_message_codecs() -> tuple[
    RegisteredMessageCodec[common_pb2.DeviceIpcWrapper, DeviceIPCWrapper],
    RegisteredMessageCodec[common_pb2.TokensPerState, TokensPerStateValue],
]:
    """Return custom codecs for protobuf messages shared across services.

    Returns:
        Explicit registrations for shared non-structural message types.
    """
    return (
        RegisteredMessageCodec(
            protobuf_type="lmcache.mp.DeviceIpcWrapper",
            python_type=DeviceIPCWrapper,
            writer=_write_device_ipc_wrapper,
            reader=_read_device_ipc_wrapper,
            include_subclasses=True,
        ),
        # The structural compiler keys this codec on the struct's declared
        # field type (``int``); the writer additionally accepts the exact
        # ``Fraction`` values the in-process mask math produces.
        RegisteredMessageCodec[common_pb2.TokensPerState, TokensPerStateValue](
            protobuf_type="lmcache.mp.TokensPerState",
            python_type=int,  # type: ignore[arg-type]
            writer=_write_tokens_per_state,
            reader=_read_tokens_per_state,
        ),
    )
