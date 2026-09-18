# SPDX-License-Identifier: Apache-2.0
"""DSA indexer k-cache: ``NL x [NB, BS, W]`` uint8, segregated value/scale rows.

Logically identical to the per-layer MLA layout (one plane, ``num_heads ==
1``), which is what vLLM's ``get_kv_cache_shape`` reports and how the tensor
registers. The PHYSICAL page layout differs (vllm ``indexer_k_norm_rope_store``
in ``models/deepseek_v41/common/ops/indexer_k_store.py``): per block, all
tokens' value bytes first, then all tokens' 4-byte scale bytes —
``[BS x vals][BS x scale]``. Only the transfer kernels care (they address
values and scales separately); every geometry accessor here matches the MLA
spec except the two row-decomposition accessors (``vals_bytes`` /
``scale_bytes``).

The per-token row width ``W`` is a dtype-fixed quantized record derived from
``index_head_dim`` (vLLM ``_indexer_k_cache_head_dim`` in
``models/deepseek_v41/attention.py``), not a learned size:

* FP8 indexer (``indexer_kv_dtype="fp8"``, the default; SM90/H100):
  ``index_head_dim``=128 value bytes + one 4-byte fp32 scale = **132 B/row**.
* MXFP4 indexer (``indexer_kv_dtype="mxfp4"``, Blackwell-only;
  ``_fp32x2_to_fp4x2`` packs 2 values/byte): 64 packed value bytes + 4 ue8m0
  block scales = **68 B/row**.

Both satisfy ``W == vals + scale`` with ``scale == 4`` and ``W % 4 == 0`` --
the exact invariant the device kernels already rely on
(``val_units = scalars_per_token - scale_units`` with
``scale_units = 4 / sizeof(ScalarType)`` in the ``NL_X_NB_BSV_BSS`` branch of
``csrc/cuda/mem_kernels.cu`` and ``csrc/cuda/mp_mem_kernels.cu``), so the C++
transfer is already geometry-generic. The accessors here expose that
decomposition as ``(vals_bytes, scale_bytes)`` so Python callers stop
hardcoding the ``128 + 4`` fp8 record.
"""

# First Party
from lmcache.v1.gpu_connector.kv_format.specs.nl_x_nb_bs_hs import (
    NL_X_NB_BS_HS_Spec,
)
import lmcache.lmcache_native as lmcache_native

# Blocked-scale record widths -> per-token scale bytes.
#
# A blocked-scale page stores one whole record per token and lays every token's
# value bytes first, then every token's scale bytes: ``[BS x vals][BS x scale]``.
# The registration exposes only the *total* record width as its trailing axis,
# so the value/scale split has to come from this table. vLLM infers the layout
# from bytes-per-token the same way ("FlashMLA infers which of these the cache
# holds from its bytes-per-token", ``flashmla_sparse.py``).
#
# Sources (vLLM):
#   * ``models/deepseek_v4/common/ops/fused_compress_quant_cache.py``
#     -> ``[0, bs*576)`` fp8/bf16 data + ``[bs*576, +bs*8)`` UE8M0 scales (V4).
#   * ``v1/attention/backends/mla/flashmla_sparse.py``
#     -> 528 B = 512 ``float8_e4m3`` + 16 ``ue8m0`` (V4.1 MXFP8, SM100+);
#        352 B = 256 ``e2m1`` + 64 ``float8_e4m3`` (RoPE) + 32 scale (nvfp4).
#   * DSA indexer rows (this file's original subject): 132 = 128 + 4,
#     68 = 64 + 4.
#
# Every value is a multiple of 4 so the kernels' 4-byte transfer units stay
# whole; the split is exact, not a heuristic.
BLOCKED_SCALE_RECORDS: dict[int, int] = {
    584: 8,  # DeepSeek V4 MLA main KV, fp8_ds_mla      (SM90/H100)
    528: 16,  # DeepSeek V4.1 MLA main KV, fp8_ds_mla   (SM100+, MXFP8)
    352: 32,  # DeepSeek V3.2/V4.1 MLA main KV, nvfp4   (SM100+)
    132: 4,  # DSA indexer K, fp8
    68: 4,  # DSA indexer K, mxfp4
}


def blocked_scale_bytes(record_bytes: int) -> int:
    """Return the per-token scale width of a blocked-scale record.

    Args:
        record_bytes: Total per-token record width (the registration's
            trailing axis), e.g. ``584`` for the V4 MLA main KV.

    Returns:
        Scale bytes per token, e.g. ``8`` for ``584``. ``0`` means the width is
        not a known blocked-scale record (the caller must not use a
        blocked-scale format for it).

    Notes:
        The record width alone determines the split, because each quantized
        cache format has exactly one value/scale decomposition and vLLM itself
        keys off bytes-per-token.
    """
    return BLOCKED_SCALE_RECORDS.get(int(record_bytes), 0)



class NL_X_NB_BSV_BSS_Spec(NL_X_NB_BS_HS_Spec):
    engine_kv_format = lmcache_native.EngineKVFormat.NL_X_NB_BSV_BSS
    attention_backends = ("vLLM MLA",)
    is_layer_list = True
    is_mla = True

    def _page_axes(self, layer_idx: int = 0) -> tuple[int, int]:
        """Return ``(bs_axis, width_axis)`` of ``layer_idx``'s page tensor.

        The blocked-scale page is logically ``[NB, BS, W]`` (one plane,
        ``num_heads == 1``). Two registrations reach this spec:

        * 3-D ``[NB, BS, W]`` -> ``(1, 2)``.
        * 4-D with the singleton head axis inserted, ``[NB, 1, BS, W]`` (HND)
          -> ``(2, 3)``, or ``[NB, BS, 1, W]`` (NHD) -> ``(1, 3)``.

        Returns:
            ``(bs_axis, width_axis)``, both into ``shape``.

        Raises:
            ValueError: If the tensor is not one of the accepted forms with a
                singleton head axis.
        """
        shape = tuple(self.kv_caches[layer_idx].shape)
        if len(shape) == 3:
            return (1, 2)
        if len(shape) == 4:
            if shape[1] == 1:
                return (2, 3)
            if shape[2] == 1:
                return (1, 3)
            raise ValueError(
                f"NL_X_NB_BSV_BSS expects a singleton head axis in a 4-D page "
                f"(blocked-scale pages are single-plane), got shape {shape} "
                f"(layer {layer_idx})."
            )
        raise ValueError(
            f"NL_X_NB_BSV_BSS expects a 3-D [NB, BS, W] or 4-D [NB, 1, BS, W] "
            f"page, got shape {shape} (layer {layer_idx})."
        )

    def block_size(self, layer_idx: int = 0) -> int:
        """Tokens per block: ``shape[bs_axis]`` (see :meth:`_page_axes`)."""
        return int(self.kv_caches[layer_idx].shape[self._page_axes(layer_idx)[0]])

    def hidden_dim(self, layer_idx: int = 0) -> int:
        """Per-token record width ``W`` in bytes."""
        return self._row_bytes(layer_idx)

    def head_size(self, layer_idx: int = 0) -> int:
        """Per-head content size == ``W`` (``num_heads`` is 1)."""
        return self._row_bytes(layer_idx)

    def num_heads(self, layer_idx: int = 0) -> int:
        """Always 1: a blocked-scale page is single-plane."""
        return 1

    def num_blocks(self) -> int:
        """Blocks in the pool (dim 0 in every accepted form)."""
        return int(self.kv_caches[0].shape[0])

    def page_buffer_size(self) -> int:
        """Token slots per pool: ``num_blocks * block_size``."""
        return self.num_blocks() * self.block_size(0)

    def tokens_per_layer(self) -> int:
        """Token slots in one layer's pool."""
        return self.page_buffer_size()

    def _row_bytes(self, layer_idx: int = 0) -> int:
        """Per-token record width ``W`` (bytes) of ``layer_idx``'s cache.

        The trailing axis carries the whole record in both accepted
        registrations:

        * 3-D ``[NB, BS, W]`` -- the legacy per-layer MLA/indexer path whose
          ``shape[2]`` is ``W`` directly (``132``/``68`` for the DSA indexer).
        * 4-D ``[NB, 1, BS, W]`` (HND) or ``[NB, BS, 1, W]`` (NHD) -- what
          nightly vLLM registers for the quantized MLA main KV. ``num_heads``
          is 1 there, so the page is byte-identical to the 3-D form and ``W``
          is ``shape[-1]``.

        Also what the inherited ``head_size()`` / ``hidden_dim()`` return.
        """
        return int(self.kv_caches[layer_idx].shape[-1])

    def vals_bytes(self, layer_idx: int = 0) -> int:
        """Value bytes per token of the blocked-scale page: ``W - scale_bytes``.

        Validates the device-side invariant the transfer kernels demand
        (``scale_bytes % 4 == 0``, the ``mp_mem_kernels.cu`` host check): the
        ue8m0/e4m3 scale region has to be a whole number of 4-byte transfer
        units.

        Raises:
            ValueError: If the record width is not a known blocked-scale record,
                or the right-hand scale region is empty / not a whole number of
                4-byte units.
        """
        w = self._row_bytes(layer_idx)
        scale = self.scale_bytes(layer_idx)
        if w <= scale or scale <= 0:
            raise ValueError(
                f"NL_X_NB_BSV_BSS record width {w} (layer {layer_idx}) is not a "
                f"segregated value/scale row: it must exceed its {scale}-byte "
                "scale region. Known records: "
                f"{sorted(BLOCKED_SCALE_RECORDS)}."
            )
        if scale % 4 != 0:
            raise ValueError(
                f"NL_X_NB_BSV_BSS scale region {scale} B (layer {layer_idx}) is "
                "not a whole number of 4-byte transfer units."
            )
        return w - scale

    def scale_bytes(self, layer_idx: int = 0) -> int:
        """Scale bytes per token of the blocked-scale page.

        Derived from the record width via :data:`BLOCKED_SCALE_RECORDS`:
        ``8`` for the 584 B V4 MLA record, ``16`` for V4.1's 528 B MXFP8 one,
        ``32`` for nvfp4's 352 B one, and the historical ``4`` for the DSA
        indexer records (``132``/``68``).

        Returns:
            Scale bytes per token; ``0`` when the width is not a known
            blocked-scale record (the caller must then not select this format).
        """
        return blocked_scale_bytes(self._row_bytes(layer_idx))

    def blocked_scale_row_geometry(self, layer_idx: int = 0) -> tuple[int, int]:
        """Return ``(vals_bytes, scale_bytes)`` for ``layer_idx``'s row.

        ``(576, 8)`` for the V4 MLA record, ``(128, 4)`` for the fp8 indexer,
        ``(64, 4)`` for the MXFP4 one -- the value/scale split the transfer
        kernels apply per block.
        """
        return (self.vals_bytes(layer_idx), self.scale_bytes(layer_idx))

    def indexer_row_width(self, layer_idx: int = 0) -> int:
        """Alias of :meth:`_row_bytes`: the packed row byte width ``W``."""
        return self._row_bytes(layer_idx)
