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

# Per-token scale region of the DSA indexer page: one 4-byte unit (a single
# fp32 scale on the fp8 path, or the 4 UE8M0 block scales on the MXFP4 path).
# The device kernels fix it at ``4 / sizeof(ScalarType)`` units regardless of
# ``W`` (``mem_kernels.cu`` / ``mp_mem_kernels.cu`` ``NL_X_NB_BSV_BSS``
# branch).
_SCALE_BYTES = 4


class NL_X_NB_BSV_BSS_Spec(NL_X_NB_BS_HS_Spec):
    engine_kv_format = lmcache_native.EngineKVFormat.NL_X_NB_BSV_BSS
    attention_backends = ("vLLM MLA",)
    is_layer_list = True
    is_mla = True

    def _row_bytes(self, layer_idx: int = 0) -> int:
        """Per-token row width ``W`` (bytes) of ``layer_idx``'s indexer cache.

        Equals the shape's trailing axis: ``132`` (fp8) or ``68`` (mxfp4) for
        the default ``index_head_dim == 128``. Also what the inherited
        ``head_size()`` / ``hidden_dim()`` return.
        """
        return int(self.kv_caches[layer_idx].shape[2])

    def vals_bytes(self, layer_idx: int = 0) -> int:
        """Value bytes per token of the blocked-scale page: ``W - 4``.

        Validates the device-side invariant the transfer kernels demand
        (``W % 4 == 0``, ``mp_mem_kernels.cu`` host check): the fp32/ue8m0
        scale region has to be a whole number of 4-byte transfer units.

        Raises:
            ValueError: If the row width is not greater than the 4-byte scale
                region or is not a whole number of 4-byte units.
        """
        w = self._row_bytes(layer_idx)
        if w <= _SCALE_BYTES or w % 4 != 0:
            raise ValueError(
                f"NL_X_NB_BSV_BSS row width {w} (layer {layer_idx}) is not a "
                "segregated value/scale indexer row: it must exceed the "
                f"{_SCALE_BYTES}-byte scale region and be a whole number of "
                "4-byte units (device kernels pin 4-byte transfer units)."
            )
        return w - _SCALE_BYTES

    def scale_bytes(self, layer_idx: int = 0) -> int:
        """Scale bytes per token of the blocked-scale page (always 4)."""
        return _SCALE_BYTES

    def blocked_scale_row_geometry(self, layer_idx: int = 0) -> tuple[int, int]:
        """Return ``(vals_bytes, scale_bytes)`` for ``layer_idx``'s row.

        ``(128, 4)`` for the fp8 indexer, ``(64, 4)`` for the MXFP4 one --
        the value/scale split the transfer kernels apply per block.
        """
        return (self.vals_bytes(layer_idx), self.scale_bytes(layer_idx))

    def indexer_row_width(self, layer_idx: int = 0) -> int:
        """Alias of :meth:`_row_bytes`: the packed row byte width ``W``."""
        return self._row_bytes(layer_idx)
