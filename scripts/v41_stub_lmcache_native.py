# SPDX-License-Identifier: Apache-2.0
"""Pure-Python stand-in for the ``lmcache.lmcache_native`` C++ extension.

Why this exists
---------------
The LMCache ``lmcache.lmcache_native`` module is a compiled C++/pybind11
extension (``csrc/lmcache_native/*``).  On developer machines that cannot
build or install it -- no NVIDIA GPU, or ``NO_GPU_EXT``-hostile toolchains --
``import lmcache.lmcache_native`` fails and every pure-Python test that touches
the KV-format / detection / storage-op layer is blocked.  This module is a
*fidelity-first*, pure-Python shim with the same public surface as the compiled
module, so ``scripts/v41_test_env.sh`` can pre-register it as
``lmcache.lmcache_native`` and run the Python-side logic tests offline.

What is mirrored
----------------
* ``EngineKVFormat`` (17 members, values 0..16 -- identical names/values to the
  pyi and to ``csrc/engine_kv_format.h``) and the ``GPUKVFormat`` alias.
* ``TransferDirection``.
* The five classification predicates ``is_kv_list / is_layer_list /
  is_cross_layer / is_mla / is_kv_second_tuple``, driven by a Python port of the
  C++ ``format_facts`` table in ``csrc/engine_kv_format.h``.
* ``PageBufferShapeDesc`` and ``KernelGroupSpec`` transfer descriptors.
* ``TTLLock``, ``Bitmap``, ``ParallelPatternMatcher``, ``RangePatternMatcher``,
  ``PeriodicEventNotifier``, and ``fold`` / ``unfold`` -- each a faithful port
  of the corresponding ``csrc/lmcache_native/*.cpp`` implementation.

This makes the offline suite cover `tests/v1/lmcache_native/*`,
`tests/v1/gpu_connector/test_*` (detection / spec / classification) and
`tests/v1/distributed/test_bitmap_ops.py` (fold/unfold) without a GPU.

Fidelity notes
--------------
* The predicates raise ``ValueError("Unsupported EngineKVFormat")`` for
  out-of-range formats, matching the C++ ``std::invalid_argument`` (surfaced by
  pybind as ``ValueError``).  Unlike the pybind binding (which rejects bare
  ``int`` arguments with ``TypeError``), the shim also accepts any int-like
  input for convenience; callers in-tree always pass enum members.
* ``scripts/v41_test_env.sh parity`` runs an automated cross-check of this shim
  against a *built* native module (when one exists) over every format, a
  randomized Bitmap battery, fold/unfold and the matchers.
"""

from __future__ import annotations

# Standard
import os
import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Sequence

__all__ = [
    "EngineKVFormat",
    "GPUKVFormat",
    "TransferDirection",
    "PageBufferShapeDesc",
    "KernelGroupSpec",
    "is_kv_list",
    "is_layer_list",
    "is_cross_layer",
    "is_mla",
    "is_kv_second_tuple",
    "fold",
    "unfold",
    "TTLLock",
    "Bitmap",
    "ParallelPatternMatcher",
    "RangePatternMatcher",
    "PeriodicEventNotifier",
]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class EngineKVFormat(IntEnum):
    """Enumeration of different engine KV cache memory layouts.

    Names and integer values mirror ``lmcache/lmcache_native.pyi`` and the
    ``EngineKVFormat`` enum in ``csrc/engine_kv_format.h`` (single source of
    truth shared with the pure-Python fallback).
    """

    NB_NL_TWO_BS_NH_HS = 0
    NL_X_TWO_NB_BS_NH_HS = 1
    NL_X_NB_TWO_BS_NH_HS = 2
    NL_X_NB_BS_HS = 3
    TWO_X_NL_X_NBBS_NH_HS = 4
    NL_X_NBBS_ONE_HS = 5
    NL_X_TWO_NB_NH_BS_HS = 6
    NL_X_NB_TWO_NH_BS_HS = 7
    NB_NL_TWO_NH_BS_HS = 8
    TWO_X_NL_X_NB_BS_NH_HS = 9
    NL_X_NB_NH_BS_TWO_HS = 10
    NL_X_NB_BS_NH_TWO_HS = 11
    NL_X_NB_NH_BS_CS = 12
    NL_X_NB_BS_NH_CS = 13
    NL_X_NB_BSV_BSS = 14
    NL_X_TWO_NB_NH_ONE_BS_HS = 15
    NL_X_TWO_X_NB_BS_NH_HS = 16


# Backward-compat alias for EngineKVFormat (pybind: ``m.attr("GPUKVFormat") =
# m.attr("EngineKVFormat")``).
GPUKVFormat = EngineKVFormat


class TransferDirection(IntEnum):
    """Specifies the direction of a memory transfer."""

    H2D = 0
    D2H = 1


# ---------------------------------------------------------------------------
# Format classification facts (port of csrc/engine_kv_format.h)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FormatFacts:
    """Static layout facts for one format; mirrors ``FormatFacts`` in C++."""

    is_cross_layer: bool = False
    is_kv_list: bool = False
    is_layer_list: bool = False
    is_mla: bool = False
    is_hnd: bool = False
    is_fused_packed: bool = False
    is_two_major: bool = False
    is_pbs_fused: bool = False
    is_kv_second_tuple: bool = False


# Ported verbatim from the C++ ``format_facts`` switch in
# ``csrc/engine_kv_format.h``.  ``False`` means "not declared" (all flags
# default to False; only the true ones are set).
_FORMAT_FACTS: dict[EngineKVFormat, FormatFacts] = {
    EngineKVFormat.NB_NL_TWO_BS_NH_HS: FormatFacts(is_cross_layer=True),
    EngineKVFormat.NL_X_TWO_NB_BS_NH_HS: FormatFacts(
        is_layer_list=True, is_two_major=True
    ),
    EngineKVFormat.NL_X_NB_TWO_BS_NH_HS: FormatFacts(is_layer_list=True),
    EngineKVFormat.NL_X_NB_BS_HS: FormatFacts(is_layer_list=True, is_mla=True),
    EngineKVFormat.TWO_X_NL_X_NBBS_NH_HS: FormatFacts(is_kv_list=True),
    EngineKVFormat.NL_X_NBBS_ONE_HS: FormatFacts(
        is_layer_list=True, is_mla=True, is_pbs_fused=True
    ),
    EngineKVFormat.NL_X_TWO_NB_NH_BS_HS: FormatFacts(
        is_layer_list=True, is_hnd=True, is_two_major=True
    ),
    EngineKVFormat.NL_X_NB_TWO_NH_BS_HS: FormatFacts(
        is_layer_list=True, is_hnd=True
    ),
    EngineKVFormat.NB_NL_TWO_NH_BS_HS: FormatFacts(
        is_cross_layer=True, is_hnd=True
    ),
    EngineKVFormat.TWO_X_NL_X_NB_BS_NH_HS: FormatFacts(is_kv_list=True),
    EngineKVFormat.NL_X_NB_NH_BS_TWO_HS: FormatFacts(
        is_layer_list=True, is_hnd=True, is_fused_packed=True
    ),
    EngineKVFormat.NL_X_NB_BS_NH_TWO_HS: FormatFacts(
        is_layer_list=True, is_fused_packed=True
    ),
    EngineKVFormat.NL_X_NB_NH_BS_CS: FormatFacts(
        is_layer_list=True, is_hnd=True, is_fused_packed=True
    ),
    EngineKVFormat.NL_X_NB_BS_NH_CS: FormatFacts(
        is_layer_list=True, is_fused_packed=True
    ),
    EngineKVFormat.NL_X_NB_BSV_BSS: FormatFacts(
        is_layer_list=True, is_mla=True
    ),
    EngineKVFormat.NL_X_TWO_NB_NH_ONE_BS_HS: FormatFacts(
        is_layer_list=True, is_hnd=True, is_two_major=True
    ),
    EngineKVFormat.NL_X_TWO_X_NB_BS_NH_HS: FormatFacts(
        is_layer_list=True, is_kv_second_tuple=True
    ),
}


def format_facts(fmt: Any) -> FormatFacts:
    """Return the static layout facts for *fmt* (mirror of the C++ helper)."""
    try:
        key = EngineKVFormat(int(fmt))
    except (ValueError, TypeError):
        raise ValueError(f"Unsupported EngineKVFormat") from None
    try:
        return _FORMAT_FACTS[key]
    except KeyError:
        raise ValueError(f"Unsupported EngineKVFormat") from None


def is_kv_list(format: Any) -> bool:
    """Return whether the format stores KV as a list of per-token KV tensors."""
    return format_facts(format).is_kv_list


def is_layer_list(format: Any) -> bool:
    """Return whether the format stores one list entry per layer."""
    return format_facts(format).is_layer_list


def is_cross_layer(format: Any) -> bool:
    """Return whether the format stacks KV from different layers into one tensor."""
    return format_facts(format).is_cross_layer


def is_mla(format: Any) -> bool:
    """Return whether the format is an MLA variant (single latent KV head)."""
    return format_facts(format).is_mla


def is_kv_second_tuple(format: Any) -> bool:
    """Return whether each per-layer list entry is a (K, V) tuple of paged tensors."""
    return format_facts(format).is_kv_second_tuple


# ---------------------------------------------------------------------------
# Transfer descriptors
# ---------------------------------------------------------------------------


class PageBufferShapeDesc:
    """Descriptor for one engine page-buffer layout.

    Mirrors ``csrc/kv_transfer_plan_types.h`` ``struct PageBufferShapeDesc``
    (plain value type; the pybind binding allows extra dynamic attributes, e.g.
    ``dtype``, which plain Python classes also permit).
    """

    def __init__(self) -> None:
        self.kv_size = 0
        self.nl = 0
        self.nb = 0
        self.bs = 0
        self.nh = 0
        self.hs = 0
        self.element_size = 0
        self.block_stride_elems = 0


class KernelGroupSpec:
    """Backend-agnostic descriptor for one blocked-transfer kernel group.

    Mirrors ``csrc/kv_transfer_plan_types.h`` ``struct KernelGroupSpec`` and the
    pybind ``__init__`` in ``csrc/lmcache_native/pybind.cpp``.
    """

    def __init__(
        self,
        paged_buffer_ptrs: int,
        lmcache_objects_ptrs: Sequence[int],
        shape_desc: PageBufferShapeDesc,
        lmcache_chunk_size: int,
        engine_kv_format: int,
        block_ids_base: int,
        block_ids_capacity: int,
    ) -> None:
        self.paged_buffer_ptrs = paged_buffer_ptrs
        self.lmcache_objects_ptrs = list(lmcache_objects_ptrs)
        self.shape_desc = shape_desc
        self.lmcache_chunk_size = lmcache_chunk_size
        self.engine_kv_format = engine_kv_format
        self.block_ids_base = block_ids_base
        self.block_ids_capacity = block_ids_capacity


# ---------------------------------------------------------------------------
# Bitmap (port of csrc/lmcache_native/bitmap.cpp)
# ---------------------------------------------------------------------------


class Bitmap:
    """A bitmap for tracking the state of L2 storage operation results.

    Port of ``csrc/lmcache_native/bitmap.cpp``.  Internally a single Python
    ``int`` is used as the bit store with bit ``i`` == set bit ``i``; this is
    bit-for-bit equivalent to the C++ per-byte storage (LSB-first within each
    byte, index 0 == bit 0).
    """

    def __init__(self, size: int, prefix_bits: int = 0) -> None:
        if size < 0:
            raise ValueError("size must be non-negative")
        self._size = int(size)
        self._bits = 0
        if prefix_bits:
            self.set_range(0, int(prefix_bits))

    @classmethod
    def _from_bits(cls, size: int, bits: int) -> "Bitmap":
        bm = cls(size)
        bm._bits = bits
        return bm

    # ---- mutators ----
    def set(self, index: int) -> None:
        """Set the bit at the specified index to 1."""
        if 0 <= index < self._size:
            self._bits |= 1 << index

    def batched_set(self, indices: Sequence[int]) -> None:
        """Set every bit in *indices* to 1 (positions >= size ignored)."""
        for idx in indices:
            self.set(int(idx))

    def set_range(self, start: int, end: int) -> None:
        """Set every bit in the half-open range ``[start, end)`` to 1.

        ``end`` is clamped to the bitmap size; an empty or out-of-range range is
        a no-op.  Equivalent to the per-bit ``set`` loop (the C++ byte-filling
        is bit-for-bit identical).
        """
        end = min(int(end), self._size)
        start = int(start)
        if start >= end:
            return
        span = end - start
        self._bits |= ((1 << span) - 1) << start

    def clear(self, index: int) -> None:
        """Clear the bit at the specified index to 0."""
        if 0 <= index < self._size:
            self._bits &= ~(1 << index)

    # ---- queries ----
    def test(self, index: int) -> bool:
        """Test the bit at the specified index."""
        return 0 <= index < self._size and bool((self._bits >> index) & 1)

    def popcount(self) -> int:
        """Return the number of bits set to 1 (only bits < size count)."""
        return self._masked().bit_count()

    def count_leading_zeros(self) -> int:
        """Number of leading zeros scanning from index 0 (position of the
        lowest set bit, or ``size`` when the bitmap is empty)."""
        masked = self._masked()
        if masked == 0:
            return self._size
        return (masked & -masked).bit_length() - 1

    def count_leading_ones(self) -> int:
        """Number of leading ones scanning from index 0 (position of the
        lowest clear bit, or ``size`` when all bits are set)."""
        masked = self._masked()
        inverted = (~masked) & self._masked_mask()
        if inverted == 0:
            return self._size
        return (inverted & -inverted).bit_length() - 1

    def highest_set_bit(self) -> int:
        """Index of the highest set bit, or ``-1`` if no bit is set."""
        masked = self._masked()
        if masked == 0:
            return -1
        return masked.bit_length() - 1

    def get_indices_list(self) -> list[int]:
        """Return a list of indices where the bit is set to 1, ascending."""
        out: list[int] = []
        m = self._masked()
        while m:
            low = m & -m
            out.append(low.bit_length() - 1)
            m ^= low
        return out

    def get_indices_set(self) -> set[int]:
        """Return a set of indices where the bit is set to 1."""
        return set(self.get_indices_list())

    def gather(self, items: Sequence[Any]) -> list[Any]:
        """Return elements from *items* at indices where the bit is set to 1."""
        return [items[i] for i in self.get_indices_list() if i < len(items)]

    # ---- operators ----
    def __and__(self, other: "Bitmap") -> "Bitmap":
        size = min(self._size, other._size)
        return Bitmap._from_bits(size, (self._bits & other._bits) & ((1 << size) - 1))

    def __or__(self, other: "Bitmap") -> "Bitmap":
        size = min(self._size, other._size)
        return Bitmap._from_bits(size, (self._bits | other._bits) & ((1 << size) - 1))

    def __invert__(self) -> "Bitmap":
        return Bitmap._from_bits(self._size, (~self._bits) & self._masked_mask())

    def __repr__(self) -> str:
        # Matches the C++ ``to_string``: index-0 == leftmost character.
        return "".join("1" if self.test(i) else "0" for i in range(self._size))

    # ---- helpers ----
    def _masked_mask(self) -> int:
        return (1 << self._size) - 1

    def _masked(self) -> int:
        return self._bits & self._masked_mask()


# ---------------------------------------------------------------------------
# TTLLock (port of csrc/lmcache_native/ttl_lock.cpp)
# ---------------------------------------------------------------------------


class TTLLock:
    """A thread-safe lock with TTL (Time-To-Live) support.

    The lock maintains a counter (lock/unlock) and a TTL: once the TTL expires
    the lock is considered released no matter the counter; ``lock()`` after
    expiry resets the counter to 1.
    """

    def __init__(self, ttl_second: int = 300) -> None:
        self._ttl_ms = int(ttl_second) * 1000
        self._guard = threading.Lock()
        self._counter = 0
        self._expiration_ms = 0

    @staticmethod
    def _now_ms() -> int:
        # Steady clock in milliseconds (matches C++ std::chrono::steady_clock).
        return int(time.monotonic() * 1000)

    def lock(self) -> None:
        """Increment the lock counter and refresh the TTL; if the previous TTL
        expired, reset the counter to 1 first."""
        with self._guard:
            now = self._now_ms()
            if now >= self._expiration_ms:
                self._counter = 1
            else:
                self._counter += 1
            self._expiration_ms = now + self._ttl_ms

    def unlock(self) -> None:
        """Decrement the lock counter (minimum 0)."""
        with self._guard:
            if self._counter > 0:
                self._counter -= 1

    def is_locked(self) -> bool:
        """Return True if the lock is held (counter > 0 and TTL not expired)."""
        with self._guard:
            return (self._counter > 0) and (self._now_ms() < self._expiration_ms)

    def reset(self) -> None:
        """Reset the lock to initial state (counter = 0, TTL expired)."""
        with self._guard:
            self._counter = 0
            self._expiration_ms = 0


# ---------------------------------------------------------------------------
# Pattern matchers (port of csrc/lmcache_native/utils.cpp)
# ---------------------------------------------------------------------------


class ParallelPatternMatcher:
    """Pattern matcher for integer vectors.

    Finds all positions where a pattern occurs in the input data.
    """

    def __init__(self, pattern: list[int]) -> None:
        if not pattern:
            raise ValueError("Pattern cannot be empty")
        self._pattern = list(pattern)

    def match(self, data: list[int]) -> list[int]:
        """Return a sorted list of positions where the pattern starts."""
        pattern = self._pattern
        n = len(pattern)
        if len(data) < n:
            return []
        out = []
        for i in range(len(data) - n + 1):
            if all(data[i + j] == pattern[j] for j in range(n)):
                out.append(i)
        return out


class RangePatternMatcher:
    """Range pattern matcher for integer vectors.

    Finds ranges that start with a start pattern and end with an end pattern;
    when multiple end patterns follow a start, matches the first (minimal
    range).  Port of the C++ greedy left-to-right scan in utils.cpp.
    """

    def __init__(
        self, start_pattern: list[int], end_pattern: list[int]
    ) -> None:
        if not start_pattern:
            raise ValueError("Start pattern cannot be empty")
        if not end_pattern:
            raise ValueError("End pattern cannot be empty")
        self._start = list(start_pattern)
        self._end = list(end_pattern)

    def _matches_at(self, data: list[int], pos: int, pattern: list[int]) -> bool:
        if pos + len(pattern) > len(data):
            return False
        return all(data[pos + k] == pattern[k] for k in range(len(pattern)))

    def match(self, data: list[int]) -> list[tuple[int, int]]:
        """Return (start_pos, end_pos) pairs; end_pos is exclusive."""
        data = list(data)
        n = len(data)
        ranges: list[tuple[int, int]] = []
        if n < len(self._start) + len(self._end):
            return ranges
        i = 0
        while i <= n - len(self._start):
            if self._matches_at(data, i, self._start):
                start_pos = i
                found_end = False
                j = i + len(self._start)
                while j <= n - len(self._end):
                    if self._matches_at(data, j, self._end):
                        ranges.append((start_pos, j + len(self._end)))
                        i = j + len(self._end)
                        found_end = True
                        break
                    j += 1
                if not found_end:
                    i += len(self._start)
            else:
                i += 1
        return ranges


# ---------------------------------------------------------------------------
# fold / unfold (port of csrc/lmcache_native/fold.cpp)
# ---------------------------------------------------------------------------


def fold(
    found: Bitmap,
    num_chunks: int,
    num_ranks: int,
    group_windows: Sequence[int],
) -> Bitmap:
    """Fold per-(group, chunk, rank) presence into servable prefix lengths.

    Returns a ``Bitmap`` of size ``num_chunks``; bit ``j`` is set iff every
    object group can serve a length-``j + 1`` prefix.  (Note: the pyi docstring
    claims ``num_chunks + 1`` bits with bit 0 always set, but the C++ and this
    port return ``num_chunks`` bits indexed by ``prefix_len - 1`` -- see
    ``lmcache/v1/distributed/bitmap_ops/fold.py``, which consumes this encoding
    via ``highest_set_bit() + 1``.)
    """
    group_windows = list(group_windows)
    num_groups = len(group_windows)
    chunk_stride = num_groups * num_ranks
    servable = [True] * num_chunks
    for g, window in enumerate(group_windows):
        eff_window = num_chunks if window <= 0 else window
        gbase = g * num_ranks
        run = 0
        for prefix_len in range(1, num_chunks + 1):
            cbase = (prefix_len - 1) * chunk_stride + gbase
            present = all(found.test(cbase + r) for r in range(num_ranks))
            run = run + 1 if present else 0
            if servable[prefix_len - 1] and run < min(eff_window, prefix_len):
                servable[prefix_len - 1] = False
    out = Bitmap(num_chunks)
    for j, ok in enumerate(servable):
        if ok:
            out.set(j)
    return out


def unfold(
    hit_length: int,
    num_chunks: int,
    num_ranks: int,
    group_windows: Sequence[int],
) -> Bitmap:
    """Expand a model-wide hit length into the per-group retain mask.

    Returns a ``Bitmap`` of size ``num_chunks * len(group_windows) * num_ranks``
    (all kv_ranks of each retained ``(group, chunk)`` set).
    """
    hit_length = min(int(hit_length), num_chunks)
    group_windows = list(group_windows)
    num_groups = len(group_windows)
    chunk_stride = num_groups * num_ranks
    retain = Bitmap(num_chunks * chunk_stride)
    if hit_length <= 0:
        return retain
    for g, window in enumerate(group_windows):
        lo = 0
        if window > 0 and hit_length > window:
            lo = hit_length - window
        gbase = g * num_ranks
        for j in range(lo, hit_length):
            base = j * chunk_stride + gbase
            retain.set_range(base, base + num_ranks)
    return retain


# ---------------------------------------------------------------------------
# PeriodicEventNotifier (port of csrc/lmcache_native/periodic_event_notifier.cpp)
# ---------------------------------------------------------------------------


class PeriodicEventNotifier:
    """Singleton that periodically signals registered file descriptors.

    A background thread writes to every registered fd at a configurable
    interval; eventfds get an 8-byte ``1``, pipes a 1-byte ``1``.
    """

    _instance: "PeriodicEventNotifier | None" = None
    _create_lock = threading.Lock()

    def __init__(self, interval_ms: int, use_eventfd: bool) -> None:
        self._interval_ms = max(1, int(interval_ms))
        self._use_eventfd = bool(use_eventfd)
        self._stop = threading.Event()
        self._fds: set[int] = set()
        self._fds_lock = threading.Lock()
        self._wake = threading.Condition(self._fds_lock)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @staticmethod
    def create(interval_ms: int, use_eventfd: bool) -> None:
        """Create the singleton.  Idempotent -- second call is a no-op."""
        with PeriodicEventNotifier._create_lock:
            if PeriodicEventNotifier._instance is not None:
                return
            PeriodicEventNotifier._instance = PeriodicEventNotifier(
                interval_ms, use_eventfd
            )

    @staticmethod
    def get() -> "PeriodicEventNotifier | None":
        """Return the singleton instance, or None if not created."""
        return PeriodicEventNotifier._instance

    @staticmethod
    def shutdown() -> None:
        """Shut down the singleton and join its thread.  Idempotent."""
        with PeriodicEventNotifier._create_lock:
            inst = PeriodicEventNotifier._instance
            if inst is None:
                return
            inst._stop.set()
            with inst._wake:
                inst._wake.notify_all()
            if inst._thread.is_alive():
                inst._thread.join()
            PeriodicEventNotifier._instance = None

    def register_fd(self, fd: int) -> None:
        """Register a file descriptor for periodic signaling."""
        with self._fds_lock:
            self._fds.add(fd)
            self._wake.notify_all()

    def unregister_fd(self, fd: int) -> None:
        """Unregister a file descriptor.  No-op if not registered."""
        with self._fds_lock:
            self._fds.discard(fd)

    def set_interval_ms(self, interval_ms: int) -> None:
        """Change the notification interval.  Clamped to >= 1ms."""
        self._interval_ms = max(1, int(interval_ms))
        with self._fds_lock:
            self._wake.notify_all()

    # ---- internals ----
    def _signal_fd(self, fd: int) -> None:
        try:
            if self._use_eventfd:
                os.write(fd, (1).to_bytes(8, "little"))
            else:
                os.write(fd, b"\x01")
        except (BlockingIOError, OSError):
            pass

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._fds_lock:
                self._wake.wait_for(
                    lambda: self._stop.is_set() or bool(self._fds)
                )
                if self._stop.is_set():
                    return
            while not self._stop.is_set():
                with self._fds_lock:
                    if not self._fds:
                        break
                    snapshot = list(self._fds)
                for fd in snapshot:
                    self._signal_fd(fd)
                with self._fds_lock:
                    self._wake.wait(timeout=self._interval_ms / 1000.0)


# ---------------------------------------------------------------------------
# Self-test / parity-check entry points (used by scripts/v41_test_env.sh)
# ---------------------------------------------------------------------------


def _run_selfcheck() -> int:
    """Quick sanity check that the shim agrees with the C++ format_facts."""

    expected = {
        0: (False, False, True, False, False),
        1: (True, False, False, False, False),
        2: (True, False, False, False, False),
        3: (True, False, False, True, False),
        4: (False, True, False, False, False),
        5: (True, False, False, True, False),
        6: (True, False, False, False, False),
        7: (True, False, False, False, False),
        8: (False, False, True, False, False),
        9: (False, True, False, False, False),
        10: (True, False, False, False, False),
        11: (True, False, False, False, False),
        12: (True, False, False, False, False),
        13: (True, False, False, False, False),
        14: (True, False, False, True, False),
        15: (True, False, False, False, False),
        16: (True, False, False, False, True),
    }
    for v, want in expected.items():
        f = EngineKVFormat(v)
        got = (
            is_layer_list(f),
            is_kv_list(f),
            is_cross_layer(f),
            is_mla(f),
            is_kv_second_tuple(f),
        )
        if got != want:
            print(f"SELFCHECK FAIL format {f.name}: got {got} want {want}")
            return 1

    # Bitmap spot checks (mirrors tests/v1/lmcache_native/test_bitmap.py).
    b = Bitmap(9, 5)
    assert b.popcount() == 5 and b.count_leading_ones() == 5
    inv = ~b
    assert inv.popcount() == 4
    assert inv.test(5) and inv.test(8) and not inv.test(4)
    print("SELFCHECK OK: 17 formats + Bitmap spot checks passed")
    return 0


def _run_parity(repo_root: str) -> int:
    """Cross-check this shim against a built ``lmcache.lmcache_native``."""
    import random
    import sys

    sys.path.insert(0, os.path.abspath(repo_root))

    try:
        import lmcache.lmcache_native as native  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        print(
            "PARITY SKIPPED: no built lmcache.lmcache_native found on the "
            "repo-root path. Build it first (v41_test_env.sh native)."
        )
        return 0

    errors = 0

    def check(label: str, cond: bool) -> None:
        nonlocal errors
        if not cond:
            errors += 1
            print(f"PARITY FAIL: {label}")

    # Enum parity (name + int value; the pybind and IntEnum members are
    # distinct Python types, so compare (name, int) pairs, not object ids).
    def _naming_values(enum_type) -> list[tuple[str, int]]:
        return [(k, int(v)) for k, v in enum_type.__members__.items()]

    check(
        "EngineKVFormat members",
        _naming_values(EngineKVFormat) == _naming_values(native.EngineKVFormat),
    )
    check(
        "TransferDirection members",
        _naming_values(TransferDirection)
        == _naming_values(native.TransferDirection),
    )

    # Predicate parity over every format.
    preds = (is_layer_list, is_kv_list, is_cross_layer, is_mla, is_kv_second_tuple)
    npreds = (
        native.is_layer_list,
        native.is_kv_list,
        native.is_cross_layer,
        native.is_mla,
        native.is_kv_second_tuple,
    )
    for v in EngineKVFormat.__members__.values():
        p = native.EngineKVFormat(v)
        for shim_pred, nat_pred in zip(preds, npreds):
            if shim_pred(v) != nat_pred(p):
                check(f"predicate on {v.name}", False)

    # Bitmap parity battery.
    rng = random.Random(20260917)
    for _ in range(300):
        size = rng.randint(0, 40)
        nb = native.Bitmap(size)
        sb = Bitmap(size)
        for _ in range(rng.randint(0, 8)):
            i = rng.randint(0, size + 4)
            op = rng.random()
            if op < 0.4:
                nb.set(i)
                sb.set(i)
            elif op < 0.7:
                nb.clear(i)
                sb.clear(i)
            elif op < 0.95:
                end = rng.randint(0, size + 4)
                nb.set_range(i, end)
                sb.set_range(i, end)
            else:
                j = rng.randint(0, size + 2)
                nb.batched_set([i, j])
                sb.batched_set([i, j])
        for attr in (
            "popcount",
            "count_leading_zeros",
            "count_leading_ones",
            "highest_set_bit",
            "get_indices_list",
        ):
            if getattr(nb, attr)() != getattr(sb, attr)():
                check(f"Bitmap.{attr} size={size}", False)
        if str(nb) != str(sb):
            check(f"Bitmap.__repr__ size={size}", False)

    # fold/unfold parity.
    for _ in range(200):
        num_chunks = rng.randint(0, 20)
        num_ranks = rng.randint(1, 3)
        num_groups = rng.randint(1, 3)
        group_windows = [rng.choice([-1, 0, 1, 2, 5]) for _ in range(num_groups)]
        total = num_chunks * num_groups * num_ranks
        found_n = native.Bitmap(total)
        found_s = Bitmap(total)
        for i in range(total):
            if rng.random() < 0.6:
                found_n.set(i)
                found_s.set(i)
        fn = native.fold(found_n, num_chunks, num_ranks, group_windows)
        fs = fold(found_s, num_chunks, num_ranks, group_windows)
        if str(fn) != str(fs):
            check("fold", False)
        hit = rng.randint(0, num_chunks + 2)
        un = native.unfold(hit, num_chunks, num_ranks, group_windows)
        us = unfold(hit, num_chunks, num_ranks, group_windows)
        if str(un) != str(us):
            check("unfold", False)

    if errors == 0:
        print("PARITY OK: shim matches native lmcache.lmcache_native")
    else:
        print(f"PARITY FAILED: {errors} mismatch(es)")
    return 1 if errors else 0


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parity",
        action="store_true",
        help="cross-check this shim against a built lmcache.lmcache_native "
        "(pass the repo root with --repo-root or rely on cwd)",
    )
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()

    if args.parity:
        raise SystemExit(_run_parity(os.path.abspath(args.repo_root)))
    raise SystemExit(_run_selfcheck())
