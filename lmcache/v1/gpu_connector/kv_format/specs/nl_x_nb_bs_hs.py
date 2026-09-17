# SPDX-License-Identifier: Apache-2.0
"""Per-layer MLA: ``NL x [NB, BS, HS]``.

A ``list[NL]`` of a 3-D tensor; K and V share one latent (``num_heads == 1``).
Used by vLLM MLA and normalized SGLang MP MLA.

Each per-layer entry is ``[NB, BS, HS]``: ``BS == shape[1]`` is both the token
span of one paged block-id (``tokens_per_block``) and one physical slot per
block (``slots_per_block``) -- the uncompressed single-latent geometry shared
by every DeepSeek MLA / sliding-window record (including the nightly 4-D
blocks-first siblings ``NL_X_NB_NH_BS_CS`` / ``NL_X_NB_BS_NH_CS`` that the
V4.1 SWA groups detect to under BLHNC / BLNHC; each defines the same ``BS``
block axis at its own tensor index).  The sliding-window helpers below are
pure functions of this ``block_size`` and implement, at the geometry layer,
the token -> block conversion the sliding-window hit-validity criterion
reduces to (see phase-2 docs/phase2/07-swa-groups.md).
"""

# Each spec indexes ``kv_caches`` (Tensor | nested list) per its format, so the
# ``.shape`` / ``[...]`` access is well-defined though mypy cannot prove it.
# mypy: disable-error-code="union-attr,call-overload"
# Standard
from typing import cast

# Third Party
import torch

# First Party
from lmcache.v1.gpu_connector.kv_format.specs.base import KVFormatSpec
import lmcache.lmcache_native as lmcache_native


class NL_X_NB_BS_HS_Spec(KVFormatSpec):
    engine_kv_format = lmcache_native.EngineKVFormat.NL_X_NB_BS_HS
    attention_backends = ("vLLM MLA / SGLang MLA (MP)",)
    is_layer_list = True
    is_mla = True

    def num_layers(self) -> int:
        return len(self.kv_caches)

    def num_blocks(self) -> int:
        return self.kv_caches[0].shape[0]

    def block_size(self, layer_idx: int = 0) -> int:
        return self.kv_caches[layer_idx].shape[1]

    def page_buffer_size(self) -> int:
        return self.kv_caches[0].shape[0] * self.kv_caches[0].shape[1]

    def kv_size(self) -> int:
        return 1

    def num_heads(self, layer_idx: int = 0) -> int:
        return 1

    def hidden_dim(self, layer_idx: int = 0) -> int:
        return self.kv_caches[layer_idx].shape[2]

    def head_size(self, layer_idx: int = 0) -> int:
        return self.kv_caches[layer_idx].shape[2]

    def tokens_per_layer(self) -> int:
        return self.kv_caches[0].shape[0] * self.kv_caches[0].shape[1]

    def elements_per_layer(self) -> int:
        return self.kv_caches[0].numel()

    def dtype(self, layer_idx: int = 0) -> torch.dtype:
        return self.kv_caches[layer_idx].dtype

    def data_ptrs(self, layer_indices: list[int]) -> list[int]:
        layers = cast(list[torch.Tensor], self.kv_caches)
        return [layers[i].data_ptr() for i in layer_indices]

    def sliding_window_blocks(self, sw_size_tokens: int, layer_idx: int = 0) -> int:
        """Number of this format's paged blocks a sliding window spans.

        ``ceil(sw_size_tokens / block_size)``. For the DeepSeek-V4.1 SWA
        groups the engine declares ``SlidingWindowMLASpec.sliding_window =
        128`` on a ``block_size = 32`` spec (``deepseek_v41/attention.py:522``,
        ``sparse_swa.py:111-133``), so a window of 128 tokens spans 4 blocks
        -- exactly the tail a transferred chunk must retain per
        ``object_group_transfer.downsample_and_stage_block_ids`` (whose
        ``keep_blocks_per_chunk = min(sw, chunk) // block_size``). Pure
        geometry of this format: ``block_size`` is ``shape[1]``, i.e. both
        the token span of one block id and one slot's worth of physical data.

        Args:
            sw_size_tokens: Sliding window size in tokens (``>= 0``; ``0``
                yields 0 blocks).
            layer_idx: Layer whose ``block_size`` is used (per-layer formats
                may differ across layers).

        Returns:
            The number of blocks, rounded up, that ``sw_size_tokens`` tokens
            cover.
        """
        bs = self.block_size(layer_idx)
        return (sw_size_tokens + bs - 1) // bs if sw_size_tokens > 0 else 0

    def sliding_window_aligned(self, sw_size_tokens: int, layer_idx: int = 0) -> bool:
        """Whether a sliding window spans a whole number of this format's blocks.

        The strict validity criterion for a sliding-window group is exact only
        at block granularity: LMCache keeps the tail ``min(sw, chunk)`` tokens
        of each chunk on store and slices block ids on retrieve, and the
        store-side guard ``_validate_block_chunk_size_config``
        (``kv_layer_groups.py:755-758``) requires a sub-chunk window
        (``0 < sw_size_tokens < tokens_per_chunk``) to be a whole multiple of
        ``tokens_per_block``. For this format ``tokens_per_block == block_size``
        (one slot per block, ``is_mla``, uncompressed ``tokens_per_state == 1``),
        so alignment to ``block_size`` keeps every retained block whole and the
        tail cut unambiguous. V4.1 SWA: ``128 % 32 == 0``.

        Args:
            sw_size_tokens: Sliding window size in tokens.
            layer_idx: Layer whose ``block_size`` is used.

        Returns:
            ``True`` when the window ends on a block boundary, ``False``
            otherwise.
        """
        return sw_size_tokens % self.block_size(layer_idx) == 0
