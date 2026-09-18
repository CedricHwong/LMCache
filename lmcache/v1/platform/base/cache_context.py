# SPDX-License-Identifier: Apache-2.0
"""Abstract base class for platform cache contexts.

Defines the common interface shared by :class:`GPUCacheContext` and
:class:`CPUCacheContext`.  Concrete subclasses provide
device-specific implementations of stream / buffer / copy primitives
while the base class owns layout-agnostic helpers (shape calculation,
status reporting, block-ID staging).
"""

# Future
from __future__ import annotations

# Standard
from abc import ABC, abstractmethod
from fractions import Fraction
from typing import TYPE_CHECKING, Any, ClassVar, Sequence
import array
import math

# Third Party
import torch

# First Party
from lmcache.v1.gpu_connector.utils import (
    get_attention_backend,
    get_concrete_engine_kv_shape_from_shape_desc,
    get_engine_kv_shape_description,
)
from lmcache.v1.multiprocess.group_view import lcm_cacheable_block_tokens
import lmcache.lmcache_native as lmcache_native

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.kv_layer_groups import KVLayerGroupsManager
    from lmcache.v1.platform.ops_types import PageBufferShapeDesc


def _floor_to_multiple(value: int, unit: int | Fraction) -> int:
    """Largest multiple of *unit* not exceeding *value*, computed exactly.

    *unit* is an engine-declared block or state span. vLLM's nightly contract
    types ``tokens_per_state`` as ``int | Fraction``, so this helper avoids
    float arithmetic entirely: ``//`` and ``*`` on an ``int`` and a
    ``Fraction`` are exact, and the result of flooring to a multiple is an
    integer token count, hence the ``int`` coercion.

    Args:
        value: Non-negative token count to floor.
        unit: Positive block or state span in tokens (``int`` or ``Fraction``).

    Returns:
        The largest multiple of *unit* that is ``<= value``.

    Raises:
        ValueError: If *unit* is not positive.
    """
    if unit <= 0:
        raise ValueError(f"alignment unit {unit} must be positive")
    return int(value // unit * unit)



class BaseCacheContext(ABC):
    """Abstract base for GPU and CPU cache contexts.

    Subclasses call :meth:`__init__` after computing the common
    layout parameters and before setting up device-specific state.
    All keyword arguments are required so the contract is explicit.

    Concrete subclasses MUST set :attr:`device_type` to the
    ``torch.device.type`` string they handle (``"cuda"``, ``"cpu"``,
    ...). The platform-agnostic :func:`create_cache_context` factory
    uses this attribute (via the platform registry) to pick the right
    subclass without any ``isinstance`` / ``if-elif`` chain.
    """

    #: ``torch.device.type`` string the subclass handles. Concrete
    #: subclasses MUST override this.
    device_type: ClassVar[str] = ""

    def __init__(
        self,
        *,
        kv_caches: list[torch.Tensor],
        device: torch.device,
        num_layers: int,
        kv_layer_groups_manager: KVLayerGroupsManager,
        block_ids_buffer: torch.Tensor,
        lmcache_tokens_per_chunk: int,
    ) -> None:
        self.kv_caches_ = kv_caches
        self.device_ = device
        self.num_layers_ = num_layers
        self.kv_layer_groups_manager_ = kv_layer_groups_manager
        self.block_ids_buffer_ = block_ids_buffer
        self.lmcache_tokens_per_chunk = lmcache_tokens_per_chunk

    # ------------------------------------------------------------------
    # Abstract -- subclasses MUST implement
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def stream(self) -> Any:
        """Returns the device-specific stream for async operations."""
        ...

    @property
    @abstractmethod
    def cupy_stream(self) -> Any:
        """Returns the cupy ExternalStream wrapping *stream*."""
        ...

    @property
    @abstractmethod
    def max_batch_size(self) -> int:
        """Returns the maximum number of concurrent batches."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Release device-specific resources (GDS staging buffers, etc.)."""
        ...

    @abstractmethod
    def get_kernel_group_kv_pointers(self, kernel_group_idx: int) -> torch.Tensor:
        """Returns the KV-cache pointer tensor for *kernel_group_idx*."""
        ...

    @abstractmethod
    def get_temp_kernel_group_buffer(
        self, batch_idx: int, kernel_group_idx: int
    ) -> torch.Tensor:
        """Returns a typed temp-buffer view for a (batch, kernel-group)
        pair."""
        ...

    @abstractmethod
    def get_temp_object_group_buffer(
        self, batch_idx: int, object_group_idx: int
    ) -> torch.Tensor:
        """Returns a flat uint8 temp-buffer view for a (batch, object-group)
        pair."""
        ...

    @abstractmethod
    def get_kernel_group_shape_dtype(
        self,
        num_tokens: int,
        kernel_group_idx: int,
    ) -> tuple[torch.Size, torch.dtype]:
        """Returns ``(shape, dtype)`` for *kernel_group_idx*."""
        ...

    @abstractmethod
    def cache_size_per_token(self) -> int:
        """Returns cache size per logical token in bytes (all groups)."""
        ...

    # ------------------------------------------------------------------
    # Concrete -- shared implementations
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        """Returns the device where KV-cache tensors live."""
        return self.device_

    @property
    def kv_tensors(self) -> list[torch.Tensor]:
        """Returns the list of per-layer KV cache tensors."""
        return self.kv_caches_

    @property
    def num_layers(self) -> int:
        """Returns the number of layers in the model."""
        return self.num_layers_

    @property
    def num_blocks(self) -> int:
        """Returns the number of blocks in the KV cache.

        Sourced from the kernel groups (one shared block-id space), not a
        representative-format computation.
        """
        return self.kv_layer_groups_manager_.num_blocks

    @property
    def hidden_dim_sizes(self) -> list[int]:
        """Returns hidden dimension sizes per KV layer group."""
        return [
            group.hidden_dim_size
            for group in self.kv_layer_groups_manager.kernel_groups
        ]

    @property
    def kv_layer_groups_manager(self) -> KVLayerGroupsManager:
        """Returns the KV layer groups manager."""
        return self.kv_layer_groups_manager_

    def calculate_num_blocks(self, num_tokens: int, kernel_group_idx: int) -> int:
        """Calculate the number of blocks for *num_tokens* in a kernel
        group."""
        return self.kv_layer_groups_manager.calculate_num_blocks(
            kernel_group_idx, num_tokens
        )

    def get_shape_desc(self, group_idx: int) -> PageBufferShapeDesc:
        """Returns the PageBufferShapeDesc for *group_idx*."""
        return self.kv_layer_groups_manager.get_shape_desc(group_idx)

    def get_engine_kv_format(
        self, kernel_group_idx: int
    ) -> "lmcache_native.EngineKVFormat":
        """Returns the Engine KV format of kernel *kernel_group_idx*.

        Raises:
            ValueError: If the group has no format (a bookkeeping group built by
                ``parse_kvcache_shape_spec`` should never reach the transfer
                path; detection-built groups always carry one).
        """
        groups = self.kv_layer_groups_manager.kernel_groups
        engine_kv_format = groups[kernel_group_idx].engine_kv_format
        if engine_kv_format is None:
            raise ValueError(
                f"kernel group {kernel_group_idx} has no engine_kv_format; a "
                "formatless bookkeeping group reached the transfer path"
            )
        return engine_kv_format

    def engine_kv_formats(self) -> list["lmcache_native.EngineKVFormat"]:
        """Returns the Engine KV format of each kernel group, in group order."""
        num_groups = len(self.kv_layer_groups_manager.kernel_groups)
        return [self.get_engine_kv_format(idx) for idx in range(num_groups)]

    def engine_kv_format_per_layer(
        self,
    ) -> list["lmcache_native.EngineKVFormat | None"]:
        """Returns each layer's Engine KV format, indexed by layer index.

        Formats differ across layers for a mixed-format model. ``None`` marks a
        layer in no kernel group (a cross-layer KV-sharing layer).
        """
        formats: list["lmcache_native.EngineKVFormat | None"] = [
            None
        ] * self.num_layers_
        for kernel_group_idx, group in enumerate(
            self.kv_layer_groups_manager.kernel_groups
        ):
            fmt = self.get_engine_kv_format(kernel_group_idx)
            for layer_idx in group.layer_indices:
                formats[layer_idx] = fmt
        return formats

    def get_slots_per_chunk_in_sw(self, kernel_group_idx: int) -> int:
        """Returns the number of slots per lmcache chunk for D/H
        transfer."""
        return self.kv_layer_groups_manager.get_slots_per_chunk_in_sw(kernel_group_idx)

    def get_kv_buffer_shape(
        self, logical_num_tokens: int, group_idx: int = 0
    ) -> torch.Size:
        """Returns the KV buffer shape for *logical_num_tokens*."""
        group = self.kv_layer_groups_manager.kernel_groups[group_idx]
        compress_ratio = group.tokens_per_block // group.slots_per_block
        if logical_num_tokens % compress_ratio != 0:
            raise ValueError(
                "logical_num_tokens (%d) is not a multiple of "
                "compress_ratio (%d) for group %d"
                % (logical_num_tokens, compress_ratio, group_idx)
            )
        num_slots = logical_num_tokens // compress_ratio
        sd = group.shape_desc
        return torch.Size(
            (sd.kv_size, group.num_layers, num_slots, group.hidden_dim_size)
        )

    def stage_block_ids(
        self, block_ids_per_group: list[list[int]]
    ) -> list[torch.Tensor]:
        """Stage per-group block IDs into the shared staging buffer.

        Returns one non-overlapping view per LMCache group.
        """
        offsets = [0]
        flat: array.array = array.array("q")
        for view_block_ids in block_ids_per_group:
            flat.extend(view_block_ids)
            offsets.append(len(flat))

        total = offsets[-1]
        if total > self.block_ids_buffer_.shape[0]:
            raise ValueError(
                "block ID total %d exceeds the pre-allocated buffer "
                "size %d" % (total, self.block_ids_buffer_.shape[0])
            )
        if total:
            cpu_tensor = torch.frombuffer(flat, dtype=torch.long)
            self.block_ids_buffer_[:total].copy_(cpu_tensor, non_blocking=True)

        return [
            self.block_ids_buffer_[offsets[i] : offsets[i + 1]]
            for i in range(len(block_ids_per_group))
        ]

    # ------------------------------------------------------------------
    # Derived properties (pure helpers)
    # ------------------------------------------------------------------

    @property
    def concrete_engine_kv_shape(self) -> str:
        """Returns the engine KV shape with actual numeric values."""
        group = self.kv_layer_groups_manager.kernel_groups[0]
        engine_kv_format = group.engine_kv_format
        # Detection-built groups always set engine_kv_format; this property
        # is only meaningful for them, so None is unreachable here.
        assert engine_kv_format is not None
        return get_concrete_engine_kv_shape_from_shape_desc(
            group.shape_desc, engine_kv_format
        )

    # ------------------------------------------------------------------
    # Hybrid cross-group alignment (pure helpers)
    # ------------------------------------------------------------------
    #
    # A hybrid engine (DeepSeek V4.1: 8 SWA groups @32 tokens/block, 1
    # MLA group @64, 1 ring-buffer group @8) addresses blocks with
    # *different* token spans per group, so a single "hit length" is only
    # well defined at token positions that are whole blocks in **every**
    # group. vLLM / the Mooncake KV store align on the lcm of the group
    # block sizes; these helpers expose the same alignment to LMCache and
    # translate an aligned token length into per-group whole-block / whole-
    # state physical commitments.

    @property
    def group_tokens_per_blocks(self) -> list[int]:
        """Logical engine tokens per paged block, per kernel group.

        ``KernelGroupInfo.tokens_per_block`` is the scheduler-token span of
        one engine block id (32/64/8 in the V4.1 production layout). The
        group manager already normalizes ``0`` (engine did not report) to
        the physical slot count, so every entry here is positive.

        Returns:
            One entry per kernel group, in kernel-group order.
        """
        return [
            group.tokens_per_block
            for group in self.kv_layer_groups_manager.kernel_groups
        ]

    @property
    def group_slots_per_blocks(self) -> list[int]:
        """Physical slots per paged block, per kernel group.

        ``KernelGroupInfo.slots_per_block`` is the detected per-block slot
        count (``shape_desc.bs``, the states-per-block axis ``N`` of the
        BLHNC layout). For a ``tokens_per_state>1`` group it is
        ``tokens_per_block // tokens_per_state``, so it is *smaller* than
        the logical block span.

        Exposed as a property to match the existing
        ``CPUCacheContext.group_slots_per_blocks`` property (same values,
        same call convention) so the two backends share one interface.

        Returns:
            One entry per kernel group, in kernel-group order.
        """
        return [
            group.slots_per_block
            for group in self.kv_layer_groups_manager.kernel_groups
        ]

    def group_compress_ratios(self) -> list[int]:
        """Returns the derived tokens-per-physical-slot per kernel group.

        ``compress_ratio = tokens_per_block // slots_per_block`` is the
        LMCache-recovered "tokens per stored state": one physical slot
        stores ``compress_ratio`` logical tokens, ``1`` = uncompressed. It
        equals the engine-declared ``tokens_per_state`` whenever the tensor
        detection recovered the slot count correctly.

        Raises:
            ValueError: If a group reports non-positive geometry, or
                ``tokens_per_block`` is not a whole multiple of
                ``slots_per_block`` (a single ratio is then undefined --
                mixed ``tokens_per_state`` layers of one engine group must be
                split into separate kernel/engine groups).
        """
        ratios = []
        for idx, group in enumerate(self.kv_layer_groups_manager.kernel_groups):
            tokens_per_block = group.tokens_per_block
            slots_per_block = group.slots_per_block
            if tokens_per_block <= 0 or slots_per_block <= 0:
                raise ValueError(
                    f"kernel group {idx}: non-positive geometry "
                    f"tokens_per_block={tokens_per_block}, "
                    f"slots_per_block={slots_per_block}"
                )
            if tokens_per_block % slots_per_block != 0:
                raise ValueError(
                    f"kernel group {idx}: tokens_per_block {tokens_per_block} "
                    f"is not a multiple of slots_per_block {slots_per_block}; "
                    "a single group-level compression ratio is undefined. "
                    "Mixed tokens_per_state layers (e.g. L2/L8/L14 cr=2 vs "
                    "L20 cr=1 in one engine group) must be split into "
                    "separate groups at detection time."
                )
            ratios.append(tokens_per_block // slots_per_block)
        return ratios

    def group_lcm_block_size(
        self, include: Sequence[bool] | None = None
    ) -> int:
        """Returns the cross-group prefix alignment in logical tokens.

        The lcm of the selected groups' ``tokens_per_block``, mirroring
        vLLM's ``scheduler_block_size = math.lcm(*group_block_sizes)`` that
        the Mooncake coordinator uses as ``mask_alignment``. A hit length is
        only safely shared across groups when every selected group covers a
        whole number of its own blocks, i.e. the length is a multiple of
        this value.

        ``include`` may be a per-kernel-group boolean mask (in kernel-group
        order) selecting only the groups that participate in prefix caching
        (``prefix_cacheable=True``); ``None`` uses every group. Using a
        non-prefix-cacheable group in the lcm can only over-align (never
        under-align), so including all groups is safe as a default.

        Args:
            include: Optional boolean mask over kernel groups selecting the
                groups that participate in the alignment.

        Returns:
            The lcm alignment in tokens (``1`` when no group is selected).

        Raises:
            ValueError: If *include*'s length differs from the number of
                kernel groups, or a selected group reports a non-positive
                ``tokens_per_block``.
        """
        tokens_per_blocks = self.group_tokens_per_blocks
        if include is not None:
            if len(include) != len(tokens_per_blocks):
                raise ValueError(
                    f"include mask has {len(include)} entries but there are "
                    f"{len(tokens_per_blocks)} kernel groups"
                )
            tokens_per_blocks = [
                tokens_per_block
                for tokens_per_block, keep in zip(tokens_per_blocks, include)
                if keep
            ]
        if not tokens_per_blocks:
            return 1
        # Fail fast *before* delegating: the shared helper ignores a
        # non-positive span, and silently dropping one would shrink the lcm
        # (an under-aligned hit length cuts a block).
        for tokens_per_block in tokens_per_blocks:
            if tokens_per_block <= 0:
                raise ValueError(
                    f"non-positive tokens_per_block {tokens_per_block} in a "
                    "selected group; the cross-group alignment is undefined"
                )
        # Delegate the arithmetic to the single source of truth. This path used
        # to call ``math.lcm`` directly, which made it a third independent lcm
        # implementation alongside ``lcm_cacheable_block_tokens`` (the declared
        # authority, used by the vLLM connector) and ``lcm_block_tokens``.
        # Sharing the helper is what keeps the two from forking silently.
        return lcm_cacheable_block_tokens(
            (tokens_per_block, True) for tokens_per_block in tokens_per_blocks
        )

    def align_hit_length(
        self, hit_length: int, include: Sequence[bool] | None = None
    ) -> int:
        """Floors *hit_length* to the cross-group alignment.

        A candidate hit length that is not a multiple of
        :meth:`group_lcm_block_size` would cut a block (or a compressed
        state) mid-way in at least one group; this returns the largest
        multiple of the alignment not exceeding *hit_length* so every
        selected group serves whole blocks and whole states. Mirrors the
        Mooncake store's ``align_lookup_length``.

        Args:
            hit_length: Candidate hit length in logical tokens.
            include: Optional per-kernel-group boolean mask forwarded to
                :meth:`group_lcm_block_size`.

        Returns:
            ``hit_length // alignment * alignment``; ``0`` when *hit_length*
            is smaller than one alignment unit.

        Raises:
            ValueError: If *hit_length* is negative or the alignment cannot
                be computed (see :meth:`group_lcm_block_size`).
        """
        if hit_length < 0:
            raise ValueError(f"hit_length {hit_length} must be non-negative")
        alignment = self.group_lcm_block_size(include)
        return hit_length // alignment * alignment

    def hit_length_per_group(self, hit_length: int) -> list[int]:
        """Returns how many logical tokens each group can serve for a hit.

        Per group ``g`` this is the largest multiple of ``g``'s
        ``tokens_per_block`` not exceeding the candidate hit length, i.e.
        the independent per-group coverage before any joint trim. The
        minimum across all groups equals :meth:`align_hit_length` because
        that alignment is the lcm of the group token spans. This is the
        "who serves whole blocks" breakdown: a 32-token group can serve a
        96-token candidate (3 blocks) while a 64-token group can only serve
        64 (1 block), so the joint hit must be trimmed to 64.

        Args:
            hit_length: Candidate hit length in logical tokens.

        Returns:
            Per-group servable token counts, in kernel-group order; the
            minimum of these equals :meth:`align_hit_length` for the same
            *hit_length*.

        Raises:
            ValueError: If *hit_length* is negative or group geometry is
                invalid.
        """
        if hit_length < 0:
            raise ValueError(f"hit_length {hit_length} must be non-negative")
        per_group = []
        for tokens_per_block in self.group_tokens_per_blocks:
            if tokens_per_block <= 0:
                raise ValueError(
                    f"non-positive tokens_per_block {tokens_per_block}; "
                    "cannot compute per-group hit coverage"
                )
            per_group.append(hit_length // tokens_per_block * tokens_per_block)
        return per_group

    def hit_length_physical_states(
        self, hit_length: int, include: Sequence[bool] | None = None
    ) -> list[int]:
        """Returns each group's whole physical state count for a hit.

        A ``tokens_per_state>1`` group stores one physical slot per
        ``tokens_per_state`` logical tokens, so only group-boundary tokens
        are valid: serving a length that is not a state boundary would cut a
        stored state in half. The cross-group aligned length is first floored
        to this group's own state boundary, then converted to a state count:
        ``aligned_tokens * slots_per_block // tokens_per_block`` (equivalently
        ``aligned_tokens // tokens_per_state`` whenever the geometry is
        consistent).

        The state floor is applied **per group on the cross-group aligned
        length**, independently of the ``include`` mask used to compute that
        alignment. A group excluded from ``include`` does not constrain the
        alignment, so its own block grid must not shrink its reported count:
        e.g. a ``tokens_per_block=64``, ``tokens_per_state=2`` group outside
        the mask at ``aligned=96`` must report ``96 // 2 = 48`` whole states,
        not ``64 // 2 = 32``. For a group that does participate, the lcm
        already makes ``aligned`` a multiple of its ``tokens_per_block``.
        Applying the state floor when ``tokens_per_state == 1`` is the
        identity.

        Args:
            hit_length: Candidate hit length in logical tokens.
            include: Optional per-kernel-group boolean mask forwarded to
                :meth:`group_lcm_block_size`.

        Returns:
            Per-group whole-state counts, in kernel-group order.

        Raises:
            ValueError: If *hit_length* is negative or group geometry is
                invalid (see :meth:`align_hit_length`).
        """
        if hit_length < 0:
            raise ValueError(f"hit_length {hit_length} must be non-negative")
        aligned = self.align_hit_length(hit_length, include)
        states = []
        for idx, group in enumerate(self.kv_layer_groups_manager.kernel_groups):
            tokens_per_block = group.tokens_per_block
            slots_per_block = group.slots_per_block
            tokens_per_state = group.tokens_per_state
            if tokens_per_block <= 0 or slots_per_block <= 0:
                raise ValueError(
                    f"kernel group {idx}: non-positive geometry "
                    f"tokens_per_block={tokens_per_block}, "
                    f"slots_per_block={slots_per_block}"
                )
            if tokens_per_state <= 0:
                raise ValueError(
                    f"kernel group {idx}: non-positive tokens_per_state "
                    f"{tokens_per_state}"
                )
            # Floor to this group's own *state* boundary. The block grid is
            # deliberately not used here: when the group participates in
            # ``include`` the cross-group lcm already makes ``aligned`` a
            # multiple of its ``tokens_per_block`` (so a block floor would be
            # the identity), and when it does not participate its block grid
            # must not shrink the reported count below the whole states that
            # fit in ``aligned``. Applying the state floor is the identity
            # for ``tokens_per_state == 1``.
            group_aligned = _floor_to_multiple(aligned, tokens_per_state)
            states.append(group_aligned * slots_per_block // tokens_per_block)
        return states

    # ------------------------------------------------------------------
    # Shared report_status
    # ------------------------------------------------------------------

    def _build_group_report_map(self) -> dict[int, int]:
        """Map each kernel-group index to its owning object-group index."""
        return {
            kg_idx: og_idx
            for og_idx, og in enumerate(self.kv_layer_groups_manager.object_groups)
            for kg_idx in og.kernel_group_indices
        }

    def _build_single_group_report(
        self,
        kernel_group_idx: int,
        group: Any,
        group_map: dict[int, int],
    ) -> dict:
        """Build a status dict for a single kernel group.

        Override this in subclasses to inject extra per-group fields
        without duplicating the whole :meth:`report_status` method.
        """
        engine_kv_format = self.get_engine_kv_format(kernel_group_idx)
        return {
            "kernel_group_idx": kernel_group_idx,
            "engine_group_idx": group.engine_group_idx,
            "object_group_idx": group_map.get(kernel_group_idx, 0),
            "num_layers": group.num_layers,
            "layer_indices": list(group.layer_indices),
            "tokens_per_block": group.tokens_per_block,
            "slots_per_block": group.slots_per_block,
            "dtype": str(group.dtype),
            "engine_kv_concrete_shape": (
                get_concrete_engine_kv_shape_from_shape_desc(
                    group.shape_desc, engine_kv_format
                )
            ),
            "is_mla": lmcache_native.is_mla(engine_kv_format),
            "engine_kv_format": engine_kv_format.name,
            "engine_kv_shape": get_engine_kv_shape_description(engine_kv_format),
            "attention_backend": get_attention_backend(engine_kv_format),
        }

    def report_status(self) -> dict:
        """Return this context's KV cache layout metadata."""
        manager = self.kv_layer_groups_manager
        kernel_groups = manager.kernel_groups
        group_map = self._build_group_report_map()

        group_reports = [
            self._build_single_group_report(kernel_group_idx, group, group_map)
            for kernel_group_idx, group in enumerate(kernel_groups)
        ]

        return {
            "num_layers": self.num_layers,
            "num_blocks": self.num_blocks,
            "cache_size_per_token": self.cache_size_per_token(),
            "kernel_groups": group_reports,
        }
