# SPDX-License-Identifier: Apache-2.0
"""阶段二（读对/语义正确性）集成测试：DeepSeek-V4.1 × LMCache 10 组形态。

本文件把阶段二各子任务的语义钉在**生产真实形态**上做集成推演（任务 a–e）：

- a1. per-group 掩码（masks-per-group）：``group_view`` 的 store/lookup/load
      掩码在 8x32 SWA + 1x64 MLA + 1x8 ring 布局下逐组正确；
- a2. lcm 对齐跨组 hit-length（hybrid-hitlength）：``lcm=64`` 是一切跨组
      命中长度的公共边界，``align_hit_length``/``hit_length_per_group``/
      ``hit_length_physical_states`` 逐组整块/整态分解正确；
- a3. tps>1 边界取整：压缩组只在 ``tokens_per_state`` 倍数的长度可服务，
      ``align_group_servable_length`` 先 tps 后 lcm 两级下取整；
- a4. indexer 组边界有效性（indexer-boundary）：块对齐命中下每块全部
      state 槽都有有效 key（``64 % tps == 0`` 闭合完整组）；
- a5. 跨层共享下的组去重（cross-layer-kv）：G8 的 8 叶被 identity 拆成
      4 个同质子组、绝不合并；未被任何组引用的层落 ``EXCLUDED``；
- a6. CircularBufferSpec 显式排除（circular-buffer）：ring 组不成 LMCache
      组、掩码全 False、矛盾声明 fail-closed；
- a7. SWA 命中有效性（swa-groups）：窗口 128 = 4 块、严格 chunk 有效性
      判据、窗口尾掩码与 vLLM ``reachable_block_mask`` 对齐；
- b.  用 duck-typed vLLM spec + 54 个生产同形张量跑真实
      ``create_engine_group_infos_from_vllm``，断言组数/lcm/异质 tps 处理；
- c.  **回归**：阶段一 verify 发现的 G8 混合 tps 缺陷（首 leaf 广播）在
      fix-uniform-tps 后不再出现 —— 8 叶逐层解析、混值 wrapper 读取
      fail-closed 抛 ``ValueError``、逐叶整除校验。

环境要求：**无 vLLM、无 GPU**。

- vLLM KV-cache spec 用 duck-typed stub 类构造（名字/形状按
  ``vllm/v1/kv_cache_interface.py`` 与 ``deepseek_v41/attention.py`` 构造，
  与 ``tests/v1/test_vllm_kv_cache_groups.py`` 既有约定一致），不 import
  任何 ``vllm`` 包。
- ``lmcache.lmcache_native`` 由 ``scripts/v41_test_env.sh`` 的纯 Python shim
  预注册（stub 模式）或真实 ``.so``（native 模式）提供；``torch`` 仅用于
  构造合成张量形状，不触碰 CUDA。
- 本文件依赖 01–08 全部补丁合入后的**合并树**（per-layer V4.1 字段解析、
  掩码原语、ring 排除、indexer 几何、SWA 几何、跨层 identity 拆分）：
  在没有这些补丁的 ``v41-phase1`` 基线上收集/运行会直接 ImportError。

> [假定契约] 生产镜像的数值（54 注册张量、10 个 vLLM 组、sw=128、
> lcm=64、G8 逐叶 tps={2,1}、ring 3 层）来自 docs/phase1/09 与
> docs/phase2/01-08；其中「G8 拆成 4 个 LMCache 子组 ⇒ 合并树 12 组」
> 以本文件的实测为准（docs/07 §5.1 的 11 组按 main/indexer 各 1 组草算，
> 未计入同形状下 cr=2/cr=1 的进一步拆分，见本文件 TestProdTenGroupMorphology）。
"""

# Standard
from dataclasses import dataclass, field
from fractions import Fraction

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.utils import normalize_kv_and_discover_format
from lmcache.integration.vllm.kv_cache_groups import (
    create_engine_group_infos_from_vllm,
    get_tokens_per_block,
    read_v41_spec_fields,
    resolve_tokens_per_state,
)
from lmcache.v1.gpu_connector.kv_format import get_spec
from lmcache.v1.gpu_connector.kv_format.detectors.vllm import (
    is_non_prefix_cacheable_spec,
)
from lmcache.v1.gpu_connector.utils import (
    is_indexer_k_row,
    resolve_indexer_row_geometry,
)
from lmcache.v1.platform.base import cache_context
from lmcache.v1.kv_layer_groups import (
    EXCLUDED_ENGINE_GROUP,
    KVLayerGroupsManager,
    group_layers_by_identity,
)
from lmcache.v1.multiprocess.group_view import (
    EngineGroupInfo,
    align_group_servable_length,
    align_lookup_length,
    compute_group_load_masks,
    compute_group_lookup_masks,
    compute_group_store_masks,
    get_engine_group_indices,
    lcm_block_tokens,
)
import lmcache.lmcache_native as lmcache_native

# 生产形态常量（模型卡 / docs/phase1/09）。
_NUM_LAYERS = 40
_NUM_MTP = 3
_NUM_SWA_TENSORS = _NUM_LAYERS + _NUM_MTP  # 43
_LAYER_NAMES_SWA = [
    f"model.layers.{i}.self_attn.swa_cache" for i in range(_NUM_SWA_TENSORS)
]
_KV_SOURCE_CR = {2: 2, 8: 2, 14: 2, 20: 1}
_SWA_BLOCK = 32
_SWA_WINDOW = 128
_G8_BLOCK = 64
_RING_BLOCK = 8

# --------------------------------------------------------------------------
# duck-typed vLLM KV-cache spec 类（名字需与真实类一致，供按类名识别）
# --------------------------------------------------------------------------


@dataclass
class AttentionSpec:
    """vLLM ``AttentionSpec`` 基类的 duck 形态（按类名识别用）。"""

    block_size: int


@dataclass
class SlidingWindowSpec(AttentionSpec):
    """vLLM ``SlidingWindowSpec``：带 ``sliding_window``。"""

    sliding_window: int


@dataclass
class SlidingWindowMLASpec(SlidingWindowSpec):
    """vLLM ``SlidingWindowMLASpec``：V4.1 SWA 组。"""


@dataclass
class FullAttentionSpec(AttentionSpec):
    """vLLM ``FullAttentionSpec``：普通全注意力（非环非 SWA 对照）。"""


@dataclass
class MLAAttentionSpec(AttentionSpec):
    """vLLM ``MLAAttentionSpec`` 的 V4.1 形状（逐叶子读取用）。"""

    tokens_per_state: int = 1
    state_content_bytes: "int | None" = None
    block_stride_alignment: "int | None" = None
    page_size_padded: "int | None" = None
    num_head_slots: "int | None" = None
    prefix_cacheable: bool = True


@dataclass
class CircularBufferSpec(AttentionSpec):
    """vLLM ``CircularBufferSpec``：compressor 环，显式不可缓存。"""

    prefix_cacheable: bool = False
    uses_slot_mapping: bool = False


@dataclass
class KpoolTailSpec(SlidingWindowSpec):
    """vLLM ``KpoolTailSpec``：kpool indexer 原始尾部，一 block 环形 scratch。

    它**继承 ``SlidingWindowSpec``** —— 结构上就是一个普通 SWA 组，所以按形状
    或按类名白名单都不可能可靠识别；只有 ``prefix_cacheable=False`` 这条声明
    能把它与真正可复用的滑窗组区分开。
    """

    prefix_cacheable: bool = False


@dataclass
class UniformTypeKVCacheSpecs:
    """vLLM ``UniformTypeKVCacheSpecs``：每层一个叶子 spec。"""

    block_size: int
    kv_cache_specs: "dict[str, object]" = field(default_factory=dict)


@dataclass
class MockKVCacheGroup:
    """vLLM ``KVCacheGroupSpec``：层名列表 + 该组的 spec。"""

    layer_names: list[str]
    kv_cache_spec: object


@dataclass
class MockKVCacheConfig:
    """vLLM ``KVCacheConfig``：只带 ``kv_cache_groups``。"""

    kv_cache_groups: list[MockKVCacheGroup]


# --------------------------------------------------------------------------
# 生产 10 组形态构造器
# --------------------------------------------------------------------------


def _main_name(layer: int) -> str:
    """kv-source 主压缩 KV 的注册张量名（``attention.py:423-431`` 风格）。"""
    return f"model.layers.{layer}.self_attn"


def _indexer_name(layer: int) -> str:
    """kv-source indexer K cache 的注册张量名（``.indexer.k_cache``）。"""
    return f"model.layers.{layer}.indexer.k_cache"


def _ring_name(layer: int) -> str:
    """kv-source compressor 状态环的注册张量名。"""
    return f"model.layers.{layer}.compressor.state_cache"


# Production registers BLHNC per-layer views ``[B, H, N, C]`` (see the fixture
# docstring); discovery needs the declaration to pick the block axis.
HINTS = {"kv_layout": "BLHNC"}


def _prod_vllm_config_and_caches() -> tuple[MockKVCacheConfig, dict[str, torch.Tensor]]:
    """构造生产真实形态：10 个 vLLM KV group + 54 个注册张量。

    8 个 ``SlidingWindowMLASpec(block_size=32, sliding_window=128)`` 组按
    ``[i::8]`` 摊放 43 个 SWA 张量（6/6/6/5/5/5/5/5，含 3 个 MTP 层）；G8 =
    ``UniformTypeKVCacheSpecs``（8 个 ``MLAAttentionSpec``，主 KV 与
    indexer-K 各 4，``tokens_per_state`` 混 ``{2,1}``）；G9 =
    ``CircularBufferSpec(block_size=8)``（3 层环）。

    BLHNC 逐层视图 ``[B, H, N, C]``：主 cr=2 ``[·,1,32,584]``、主 cr=1
    ``[·,1,64,584]``、indexer cr=2 ``[·,1,32,132]``、indexer cr=1
    ``[·,1,64,132]``、环 ``[·,1,8,1024]`` fp32 —— 使 identity 把 G8 恰好拆成
    4 个子组（与 docs/phase2/05 §3.1 生产复现一致）。

    返回:
        该形态的 (config, kv_caches)；张量注册顺序即层索引序。
    """
    kv_caches: dict[str, torch.Tensor] = {}
    groups: list[MockKVCacheGroup] = []

    for group_id in range(8):
        names = [
            _LAYER_NAMES_SWA[i] for i in range(_NUM_SWA_TENSORS) if i % 8 == group_id
        ]
        for name in names:
            kv_caches[name] = torch.zeros(4, 1, _SWA_BLOCK, 584, dtype=torch.uint8)
        groups.append(
            MockKVCacheGroup(
                names,
                SlidingWindowMLASpec(block_size=_SWA_BLOCK, sliding_window=_SWA_WINDOW),
            )
        )

    leaf_specs: dict[str, object] = {}
    g8_names: list[str] = []
    for layer, tps in _KV_SOURCE_CR.items():
        main = _main_name(layer)
        states = _G8_BLOCK // tps
        kv_caches[main] = torch.zeros(32, 1, states, 584, dtype=torch.uint8)
        leaf_specs[main] = MLAAttentionSpec(
            block_size=_G8_BLOCK, tokens_per_state=tps, state_content_bytes=584
        )
        g8_names.append(main)
        indexer = _indexer_name(layer)
        kv_caches[indexer] = torch.zeros(32, 1, states, 132, dtype=torch.uint8)
        leaf_specs[indexer] = MLAAttentionSpec(
            block_size=_G8_BLOCK,
            tokens_per_state=tps,
            # indexer 不设 state_content_bytes（宽度由 head_dim 定）。
            block_stride_alignment=4608,  # math.lcm(512, 576)
        )
        g8_names.append(indexer)
    groups.append(
        MockKVCacheGroup(
            g8_names,
            UniformTypeKVCacheSpecs(block_size=_G8_BLOCK, kv_cache_specs=leaf_specs),
        )
    )

    ring_names = [
        _ring_name(layer) for layer in _KV_SOURCE_CR if _KV_SOURCE_CR[layer] > 1
    ]
    for name in ring_names:
        kv_caches[name] = torch.zeros(4, 1, _RING_BLOCK, 1024, dtype=torch.float32)
    groups.append(
        MockKVCacheGroup(ring_names, CircularBufferSpec(block_size=_RING_BLOCK))
    )

    assert len(kv_caches) == _NUM_SWA_TENSORS + 8 + 3, len(kv_caches)
    assert len(groups) == 10, len(groups)
    return MockKVCacheConfig(kv_cache_groups=groups), kv_caches


def _prod_engine_group_infos() -> list[EngineGroupInfo]:
    """跑真实转换链路得到生产形态的 12 个 LMCache 组。"""
    config, kv_caches = _prod_vllm_config_and_caches()
    return create_engine_group_infos_from_vllm(
        config, kv_caches, layout_hints={"kv_layout": "BLHNC"}
    )


# ============================================================================
# 任务 b + c：生产 10 组形态端到端 + G8 混合 tps 回归
# ============================================================================


class TestProdTenGroupMorphology:
    """10 个 vLLM 组 → 12 个 LMCache 组，lcm=64，异质 tps 逐叶正确。"""

    def test_group_count_and_lcm(self) -> None:
        """8 SWA + G8 拆 4 子组 = 12 个 LMCache 组；跨组 lcm = 64。

        环组（G9）被显式排除，不成 LMCache 组；docs/07 §5.1 的「11 组」未计
        入 cr=2/cr=1 在 identity 层的进一步拆分。lcm 只统计 prefix-cacheable
        组（8x32 + 4x64）→ 64。
        """
        infos = _prod_engine_group_infos()
        assert len(infos) == 12
        assert [g.tokens_per_block for g in infos] == [
            32,
            32,
            32,
            32,
            32,
            32,
            32,
            32,
            64,
            64,
            64,
            64,
        ]
        assert lcm_block_tokens(infos) == 64

    def test_g8_mixed_tps_resolved_per_subgroup(self) -> None:
        """**回归**：G8 混合 tps 缺陷不再出现（fix-uniform-tps 逐叶解析）。

        阶段一按「首 leaf 广播」会让 G8 的全部子组都拿到 tps=2 / scb=584 /
        bsa=None。修复后 4 个子组必须各自得到：
            main L2/L8/L14 (cr=2): tps=2, scb=584, bsa=None
            idx  L2/L8/L14 (cr=2): tps=2, scb=None, bsa=4608
            main L20      (cr=1): tps=1, scb=584, bsa=None
            idx  L20      (cr=1): tps=1, scb=None, bsa=4608
        """
        infos = _prod_engine_group_infos()
        g8 = [g for g in infos if g.engine_group_id == 8]
        assert len(g8) == 4
        signature = {
            (g.tokens_per_state, g.state_content_bytes, g.block_stride_alignment)
            for g in g8
        }
        assert signature == {
            (2, 584, None),
            (2, None, 4608),
            (1, 584, None),
            (1, None, 4608),
        }

    def test_swa_groups_resolve_window_128(self) -> None:
        """8 个 SWA 组：block=32、窗口 128、tps=1、prefix-cacheable。"""
        infos = _prod_engine_group_infos()
        swa = [g for g in infos if g.tokens_per_block == 32]
        assert len(swa) == 8
        assert all(g.sw_size_tokens == _SWA_WINDOW for g in swa)
        assert all(g.tokens_per_state == 1 for g in swa)
        assert all(g.prefix_cacheable for g in swa)

    def test_ring_layers_never_form_group(self) -> None:
        """环（G9）3 层被排除：不成组、层索引落 ``EXCLUDED_ENGINE_GROUP``。"""
        infos = _prod_engine_group_infos()
        config, kv_caches = _prod_vllm_config_and_caches()
        num_layers = len(kv_caches)
        per_layer = get_engine_group_indices(infos, num_layers)
        ring_indices = [
            i
            for i, name in enumerate(kv_caches)
            if name.endswith(".compressor.state_cache")
        ]
        cacheable_indices = [i for i in range(num_layers) if i not in ring_indices]
        assert [per_layer[i] for i in ring_indices] == [EXCLUDED_ENGINE_GROUP] * 3
        assert all(per_layer[i] != EXCLUDED_ENGINE_GROUP for i in cacheable_indices)

    def test_prod_manager_round_trip(self) -> None:
        """端到端到 ``KVLayerGroupsManager``：几何与掩码口径与 group_view 一致。

        12 个 kernel 组，``tokens_per_block=[32]*8+[64]*4``、
        ``slots_per_block=[32]*8+[32,32,64,64]``、派生 compress_ratio
        ``[1]*8+[2,2,1,1]``、lcm=64；store 掩码按 chunk=256 网格给 SWA 组
        只留末 128 token（4 块）。
        """
        infos = _prod_engine_group_infos()
        config, kv_caches = _prod_vllm_config_and_caches()
        cacheable = {
            k: v
            for k, v in kv_caches.items()
            if not k.endswith(".compressor.state_cache")
        }
        tensors = list(cacheable.values())
        # Use the formats the real detector produces for this geometry rather
        # than hard-coding one. Hard-coding NL_X_NB_NH_BS_CS made the fixture
        # self-contradictory: every main/ SWA/indexer tensor here is a rank-4
        # [NB, 1, BS, W] page whose width (584 / 132) *is* a known blocked-scale
        # record, so discovery routes all of them to NL_X_NB_BSV_BSS -- a
        # content-size format would address a segregated page token-major. The
        # manager now cross-checks that against the spec's declared
        # state_content_bytes and refuses the mismatch, which is exactly what
        # this hard-coded format was tripping.
        detected_formats = [
            normalize_kv_and_discover_format(
                [t], EngineType.VLLM, layout_hints=HINTS
            )[0]
            for t in tensors
        ]
        manager = KVLayerGroupsManager(
            tensors,
            engine_kv_formats=detected_formats,
            engine_group_infos=infos,
            lmcache_tokens_per_chunk=256,
        )
        kg = manager.kernel_groups
        assert len(kg) == 12
        assert [g.tokens_per_block for g in kg] == [32] * 8 + [64] * 4
        assert [g.slots_per_block for g in kg] == [32] * 8 + [32, 32, 64, 64]
        assert manager.lcm_block_tokens() == 64
        store = manager.compute_group_store_masks(256)
        assert store[0].blocks == (False, False, False, False, True, True, True, True)
        assert store[8].blocks is None  # 主压缩 KV 全注意力
        # [假定契约] 派生 compress_ratio = tpb // spb，逐组恢复真实 tps。
        assert manager.align_group_servable_length(130) == 128


class TestG8MixedTpsReadFailClosed:
    """任务 c：G8 混合 tps 的「读」侧 fail-closed（阶段一 verify 反例）。"""

    def test_mixed_wrapper_read_v41_fields_raises(self) -> None:
        """混 tps 的 ``UniformTypeKVCacheSpecs`` 无单值 → ``ValueError``。

        绝不允许静默广播首叶子（阶段一的行为）。8 字段中任一跨叶不一致即抛。
        """
        config, _ = _prod_vllm_config_and_caches()
        g8_spec = config.kv_cache_groups[8].kv_cache_spec
        with pytest.raises(ValueError, match="tokens_per_state"):
            read_v41_spec_fields(g8_spec)

    def test_get_tokens_per_block_checks_every_leaf(self) -> None:
        """``get_tokens_per_block`` 遍历**所有**叶子校验整除（首叶不豁免）。

        tps 混 {2,3}：首叶 tps=2 可整除 64，但第二叶 tps=3 不行 → 必须抛，
        否则 vLLM 的 ``get_num_kernel_states`` 会丢尾部 token。
        """
        spec = UniformTypeKVCacheSpecs(
            block_size=64,
            kv_cache_specs={
                "a": MLAAttentionSpec(block_size=64, tokens_per_state=2),
                "b": MLAAttentionSpec(block_size=64, tokens_per_state=3),
            },
        )
        with pytest.raises(ValueError, match="does not divide block_size"):
            get_tokens_per_block(spec, 1)

    def test_same_shape_different_tps_merge_raises(self) -> None:
        """同张量形状但声明 tps 不同 → 合并成一个组 → ``ValueError``。

        identity 只按 (hs, N, ...) 拆分；若未来引擎把同组 N 尾 pad 齐，
        单 ``EngineGroupInfo`` 无法表达该异质，必须 fail-closed（docs/01 §2.1）。
        """
        leaf_specs = {
            _main_name(2): MLAAttentionSpec(
                block_size=64, tokens_per_state=2, state_content_bytes=584
            ),
            _main_name(20): MLAAttentionSpec(
                block_size=64, tokens_per_state=1, state_content_bytes=584
            ),
        }
        config = MockKVCacheConfig(
            kv_cache_groups=[
                MockKVCacheGroup(
                    [_main_name(2), _main_name(20)],
                    UniformTypeKVCacheSpecs(block_size=64, kv_cache_specs=leaf_specs),
                )
            ]
        )
        caches = {
            _main_name(2): torch.zeros(32, 1, 32, 584, dtype=torch.uint8),
            _main_name(20): torch.zeros(32, 1, 32, 584, dtype=torch.uint8),
        }
        with pytest.raises(ValueError, match="different V4.1 schema field"):
            create_engine_group_infos_from_vllm(
                config, caches, layout_hints={"kv_layout": "BLHNC"}
            )

    def test_uniform_wrapper_still_resolves(self) -> None:
        """同质 wrapper 不受逐叶解析影响（无回归）。"""
        spec = UniformTypeKVCacheSpecs(
            block_size=64,
            kv_cache_specs={
                "a": MLAAttentionSpec(
                    block_size=64, tokens_per_state=2, state_content_bytes=584
                ),
                "b": MLAAttentionSpec(
                    block_size=64, tokens_per_state=2, state_content_bytes=584
                ),
            },
        )
        fields = read_v41_spec_fields(spec)
        assert fields["tokens_per_state"] == 2
        assert fields["state_content_bytes"] == 584
        assert resolve_tokens_per_state(spec) == 2


# ============================================================================
# 任务 a2/a3：lcm 对齐跨组 hit-length + tps>1 边界取整
# ============================================================================


def _prod_lmcache_groups() -> list[EngineGroupInfo]:
    """12 个 LMCache 组的掩码/命中几何视图（不含环）。

    G8 4 子组：main-cr2 (tpb=64,tps=2)、idx-cr2 (64,2)、main-cr1 (64,1)、
    idx-cr1 (64,1)；8 个 SWA (32,tps=1,sw=128)。与
    ``create_engine_group_infos_from_vllm`` 的实测输出一致。
    """
    groups: list[EngineGroupInfo] = [
        EngineGroupInfo(i, tokens_per_block=32, sw_size_tokens=128) for i in range(8)
    ]
    groups.append(EngineGroupInfo(8, tokens_per_block=64, tokens_per_state=2))
    groups.append(EngineGroupInfo(8, tokens_per_block=64, tokens_per_state=2))
    groups.append(EngineGroupInfo(8, tokens_per_block=64, tokens_per_state=1))
    groups.append(EngineGroupInfo(8, tokens_per_block=64, tokens_per_state=1))
    return groups


class TestLcmAlignedHitLength:
    """lcm=64 是一切跨组命中长度的公共边界。"""

    def test_prod_lcm_excludes_ring(self) -> None:
        """lcm 只统计 prefix-cacheable 组；8/32 组被 G9 排除后仍能算出 64。"""
        groups = _prod_lmcache_groups()
        assert lcm_block_tokens(groups) == 64
        ring = EngineGroupInfo(9, tokens_per_block=8, prefix_cacheable=False)
        with_ring = groups + [ring]
        assert lcm_block_tokens(with_ring) == 64

    def test_align_lookup_length_floors_to_lcm(self) -> None:
        """Mooncake ``align_lookup_length`` 语义：向下取整到 64 的倍数。"""
        groups = _prod_lmcache_groups()
        assert align_lookup_length(groups, 130) == 128
        assert align_lookup_length(groups, 256) == 256
        assert align_lookup_length(groups, 63) == 0

    def test_align_group_servable_length_two_stage_floor(self) -> None:
        """tps>1 组先取整到 state 边界，再取整到 lcm（lookup=prefetch 前置收敛）。"""
        groups = _prod_lmcache_groups()
        # 130 -> tps=2 组地板到 128 -> lcm 地板到 128
        assert align_group_servable_length(groups, 130) == 128
        # 129 -> tps 地板到 128 -> lcm 128
        assert align_group_servable_length(groups, 129) == 128
        # 300 -> tps 地板到 300 -> lcm 地板到 256
        assert align_group_servable_length(groups, 300) == 256
        # 已对齐：保持原值
        assert align_group_servable_length(groups, 256) == 256

    def test_non_cacheable_group_does_not_constrain(self) -> None:
        """非 prefix-cacheable 组不参与收敛（Mooncake ``_verify_and_split`` 跳过）。"""
        groups = [
            EngineGroupInfo(0, tokens_per_block=64, tokens_per_state=1),
            EngineGroupInfo(
                1, tokens_per_block=8, tokens_per_state=1, prefix_cacheable=False
            ),
        ]
        # 100 -> lcm(64)=64；8 组排除后不影响。
        assert align_group_servable_length(groups, 100) == 64
        # 若把 12-block 的非缓存组误计入 lcm，lcm(64,12)=192 -> align(100)=0；
        # 正确排除后 lcm=64 -> align(100)=64，可区分「真排除」与「误参与」。
        groups2 = [
            EngineGroupInfo(0, tokens_per_block=64, tokens_per_state=1),
            EngineGroupInfo(
                1, tokens_per_block=12, tokens_per_state=1, prefix_cacheable=False
            ),
        ]
        assert align_group_servable_length(groups2, 100) == 64


class TestHitLengthPhysicalStates:
    """tps>1 组的整态分解（hit_length_physical_states）。"""

    def _ctx(self, groups: list[EngineGroupInfo]):
        """用生产 kernel 组构造一个只读 cache-context 宿主。

        直接 ``object.__new__`` 具体子类绕过抽象构造（本机无 GPU，CPU context
        无需真实 buffer），挂上带 ``kernel_groups`` 的 manager。
        ``group_slots_per_blocks`` 是 property（CPU 端既有约定），从 manager 的
        kernel 组直读 ``slots_per_block``。
        """
        # Local import：platform.cpu 模块只被本测试触达。
        from lmcache.v1.platform.cpu.cache_context import CPUCacheContext
        from lmcache.v1.kv_layer_groups import KernelGroupInfo

        class _Manager:
            kernel_groups: list[KernelGroupInfo]

        manager = _Manager()
        manager.kernel_groups = []
        for idx, g in enumerate(groups):
            # 按 (tpb, tps) 反推 slots_per_block = tpb // tps（与张量 N 轴一致）。
            sd = lmcache_native.PageBufferShapeDesc()  # type: ignore[attr-defined]
            sd.kv_size = 1
            sd.nl = 1
            sd.nb = 32
            sd.bs = g.tokens_per_block // g.tokens_per_state
            sd.nh = 1
            sd.hs = 584
            sd.element_size = 1
            sd.dtype = torch.uint8
            manager.kernel_groups.append(
                KernelGroupInfo(
                    layer_indices=[idx],
                    shape_desc=sd,
                    dtype=torch.uint8,
                    tokens_per_block=g.tokens_per_block,
                    sw_size_tokens=g.sw_size_tokens,
                    tokens_per_state=g.tokens_per_state,
                )
            )
        ctx = object.__new__(CPUCacheContext)
        ctx.kv_layer_groups_manager_ = manager  # type: ignore[attr-defined]
        return ctx

    def test_derived_compress_ratio_and_lcm(self) -> None:
        """派生 compress_ratio 逐组 = tpb // spb，跨组 lcm = 64。"""
        ctx = self._ctx(_prod_lmcache_groups())
        assert ctx.group_tokens_per_blocks == [32] * 8 + [64] * 4
        assert ctx.group_slots_per_blocks == [32] * 8 + [32, 32, 64, 64]
        assert ctx.group_compress_ratios() == [1] * 8 + [2, 2, 1, 1]
        assert ctx.group_lcm_block_size() == 64

    def test_group_lcm_delegates_to_the_shared_helper(self) -> None:
        """``group_lcm_block_size`` 必须调用共享的权威 lcm helper。

        这是一个**结构性**判别式：两种实现（直接 ``math.lcm`` 与委托）在同
        输入上产生**同一个数值**，所以断言返回值无法发现回归 —— 而第三条独立
        的 lcm 实现正是审计要消除的「静默分叉」来源。因此这里 spy 住
        ``cache_context`` 模块里绑定的 helper 名，断言它确实被调用、且收到的是
        参与组（span, True）序列。若有人把 ``math.lcm`` 写回去，这个测试变红。
        """
        ctx = self._ctx(_prod_lmcache_groups())
        seen: list[list[tuple[int, bool]]] = []

        def spy(group_spans):
            pairs = list(group_spans)
            seen.append(pairs)
            return 64

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                cache_context, "lcm_cacheable_block_tokens", spy, raising=True
            )
            assert ctx.group_lcm_block_size() == 64

        assert seen == [[(32, True)] * 8 + [(64, True)] * 4]

    def test_align_hit_length(self) -> None:
        """``align_hit_length`` 对 96/234/256/63 的裁剪（docs/03 §4）。"""
        ctx = self._ctx(_prod_lmcache_groups())
        assert ctx.align_hit_length(96) == 64
        assert ctx.align_hit_length(234) == 192
        assert ctx.align_hit_length(256) == 256
        assert ctx.align_hit_length(63) == 0

    def test_hit_length_per_group_min_equals_align(self) -> None:
        """逐组整块 floor 的 min == 跨组 align（先逐组后 min 等价于 lcm floor）。"""
        ctx = self._ctx(_prod_lmcache_groups())
        per = ctx.hit_length_per_group(96)
        assert per == [96] * 8 + [64, 64, 64, 64]
        assert min(per) == ctx.align_hit_length(96) == 64

    def test_physical_states_rounding(self) -> None:
        """每个 tps>1 组只提交整态：256 token → cr2 组 128 态、cr1 组 256 态。"""
        ctx = self._ctx(_prod_lmcache_groups())
        states = ctx.hit_length_physical_states(256)
        assert states == [256] * 8 + [128, 128, 256, 256]
        states_192 = ctx.hit_length_physical_states(192)
        assert states_192 == [192] * 8 + [96, 96, 192, 192]

    def test_include_mask_keeps_whole_states_for_excluded_tps2_group(self) -> None:
        """W2：include 只选部分组时，tps>1 组仍按 ``aligned/tps`` 报整态。

        10 组生产几何（8×32/tps=1 + 2×64/tps=2），``include`` 只选第 0 组
        ⇒ lcm=32。``hit=96`` 对齐到 96；tps=2 组（tpb=64, slots=32）的物理
        整态数应为 ``96*32//64 = 48``。修复前该函数先按自身 tpb=64 落格到 64，
        少报成 32（≈ -33%）。
        """
        ctx = self._ctx(_prod_lmcache_groups()[:10])
        include = [True] + [False] * 9
        assert ctx.group_lcm_block_size(include) == 32
        states = ctx.hit_length_physical_states(96, include)
        assert ctx.align_hit_length(96, include) == 96
        assert states[8] == 48
        assert states[9] == 48

    def test_include_mask_aligned_and_unaligned_hits(self) -> None:
        """W2：对齐/不对齐 hit 的整态数（aligned 64→32、96→48、128→64）。"""
        ctx = self._ctx(_prod_lmcache_groups()[:10])
        include = [True] + [False] * 9
        assert ctx.hit_length_physical_states(64, include)[8] == 32
        assert ctx.hit_length_physical_states(96, include)[8] == 48
        assert ctx.hit_length_physical_states(128, include)[8] == 64
        # hit=97 floors to the same 96 alignment -> identical whole states.
        states_96 = ctx.hit_length_physical_states(96, include)
        assert ctx.hit_length_physical_states(97, include) == states_96

    def test_include_mask_leaves_tps1_groups_unchanged(self) -> None:
        """W2 的 tps floor 对 tps=1 组是恒等：整态数 == aligned。"""
        ctx = self._ctx(_prod_lmcache_groups()[:10])
        include = [True] + [False] * 9
        assert ctx.hit_length_physical_states(96, include)[:8] == [96] * 8

    def test_fractional_tokens_per_state_is_exact(self) -> None:
        """Fraction 的 state floor 走精确有理数运算（无浮点误差），且返回 int。"""
        # First Party
        from lmcache.v1.platform.base.cache_context import _floor_to_multiple

        assert _floor_to_multiple(96, Fraction(3, 2)) == 96
        assert _floor_to_multiple(64, Fraction(3, 2)) == 63
        assert isinstance(_floor_to_multiple(64, Fraction(3, 2)), int)
        # float(64 // 1.5) * 1.5 would land at 63.0 only by luck; exactness of
        # the helper is what keeps downstream state counts integral.
        assert _floor_to_multiple(100, Fraction(1, 3)) == 100

    def test_fractional_tokens_per_state_through_helper(self) -> None:
        """Fraction tps 经真实 helper：aligned=64 → floor(3/2)=63 → 41 整态。"""
        ctx = self._ctx(
            [EngineGroupInfo(0, tokens_per_block=64, tokens_per_state=Fraction(3, 2))]
        )
        states = ctx.hit_length_physical_states(64)
        assert states == [41]
        assert all(isinstance(state, int) for state in states)


# ============================================================================
# 任务 a1：per-group 掩码（store / lookup / load）
# ============================================================================


def _prod_mask_input() -> list[EngineGroupInfo]:
    """12 个 LMCache 组 + 一个环镜像组（prefix_cacheable=False）。"""
    groups = _prod_lmcache_groups()
    groups.append(EngineGroupInfo(9, tokens_per_block=8, prefix_cacheable=False))
    return groups


class TestPerGroupStoreMasks:
    def test_store_masks_prod_geometry(self) -> None:
        """store 掩码（chunk=256）：SWA 留末 128 token、主 KV/indexer 全 True、
        环全 False。"""
        groups = _prod_mask_input()
        masks = compute_group_store_masks(groups, 256, segment_tokens=256)
        assert len(masks) == len(groups)
        for m in masks[:8]:
            assert m.blocks == (False, False, False, False, True, True, True, True)
        for m in masks[8:12]:
            assert m.blocks is None  # 全注意力 all-True 哨兵
        ring = masks[12]
        assert ring.blocks == (False,) * (256 // 8)

    def test_suffix_store_block_range(self) -> None:
        """store 从 start_token 起：[4, 8) 块范围、窗口 128 = 4 块恰好覆盖全尾。

        ``need = ceil(128/32) = 4``、段 256 = 8 块，落在尾 4 块内的
        ``[4, 8)`` 全部可达（与 vLLM ``reachable_block_mask`` 一致）。
        """
        groups = [_prod_lmcache_groups()[0]]  # SWA: tpb=32, sw=128
        masks = compute_group_store_masks(
            groups, 256, start_token=128, segment_tokens=256
        )
        assert (masks[0].start_block, masks[0].end_block) == (4, 8)
        assert masks[0].blocks == (True, True, True, True)

    def test_unaligned_token_len_fails_closed(self) -> None:
        """非 lcm 倍数的 store 长度必须抛，绝不静默丢块。"""
        groups = _prod_mask_input()
        with pytest.raises(ValueError, match="multiple of the cross-group block lcm"):
            compute_group_store_masks(groups, 100, segment_tokens=64)


class TestPerGroupLookupMasks:
    def test_lookup_default_segment_is_lcm(self) -> None:
        """lookup 默认段 = lcm(64)：SWA window 128 ≥ 64 → 每块都是合法边界。"""
        groups = _prod_mask_input()
        masks = compute_group_lookup_masks(groups, 256)
        for m in masks[:8]:
            assert m.blocks is None
        for m in masks[8:12]:
            assert m.blocks is None
        assert masks[12].blocks == (False,) * (256 // 8)

    def test_lookup_chunk_segment_keeps_swa_tail(self) -> None:
        """以 chunk 为段时 SWA 掩码保留每 chunk 末 need 块。"""
        groups = [_prod_lmcache_groups()[0]]
        masks = compute_group_lookup_masks(groups, 256, segment_tokens=256)
        assert masks[0].blocks == (False, False, False, False, True, True, True, True)


class TestPerGroupLoadMasks:
    def test_load_masks_for_model_wide_hit(self) -> None:
        """load 掩码 = 保留前缀的按组可达集合：SWA 只填尾块、attention 全填。"""
        groups = _prod_mask_input()
        masks = compute_group_load_masks(groups, 256, segment_tokens=256)
        for m in masks[:8]:
            assert m.blocks == (False, False, False, False, True, True, True, True)
        for m in masks[8:12]:
            assert m.blocks is None
        assert masks[12].blocks == (False,) * 32


class TestGroupBlockMask:
    def test_sentinel_and_block_range(self) -> None:
        """``GroupBlockMask``：None 哨兵 = 全可达，``is_reachable`` 按组内偏移。"""
        from lmcache.v1.multiprocess.group_view import GroupBlockMask

        dense = GroupBlockMask(0, 32, 0, 8, None)
        assert dense.block_count == 8
        assert all(dense.is_reachable(i) for i in range(8))
        swa = GroupBlockMask(
            1, 32, 0, 8, (False, False, False, False, True, True, True, True)
        )
        assert swa.is_reachable(0) is False
        assert swa.is_reachable(4) is True
        with pytest.raises(IndexError):
            swa.is_reachable(8)


# ============================================================================
# 任务 a6：CircularBufferSpec 显式排除
# ============================================================================


class TestCircularBufferRingExclusion:
    def test_ring_predicate(self) -> None:
        """按**声明**判定：环为 True；普通全注意力 / SWA 为 False。"""
        assert (
            is_non_prefix_cacheable_spec(CircularBufferSpec(block_size=8)) is True
        )
        assert (
            is_non_prefix_cacheable_spec(FullAttentionSpec(block_size=16)) is False
        )
        assert is_non_prefix_cacheable_spec(None) is False

    def test_kpool_tail_spec_is_excluded(self) -> None:
        """``KpoolTailSpec`` 必须被排除，尽管它看起来就是个滑窗组。

        这是 P1-6 的核心：旧实现按类名匹配 ``CircularBufferSpec``，于是
        ``KpoolTailSpec``（vLLM 声明 ``prefix_cacheable=False``）会被当成可复用
        前缀 KV 存下来，跨请求污染缓存。同一形状的真正可复用 SWA 组必须**不**
        被排除 —— 二者只能靠声明区分。
        """
        tail = KpoolTailSpec(block_size=8, sliding_window=128)
        assert is_non_prefix_cacheable_spec(tail) is True
        # 同一形状、未声明不可缓存的滑窗组必须保留
        swa = SlidingWindowSpec(block_size=8, sliding_window=128)
        assert is_non_prefix_cacheable_spec(swa) is False

    def test_non_bool_declaration_fails_closed(self) -> None:
        """``prefix_cacheable`` 不是 bool 时无法解读，必须抛。"""

        class Ambiguous(FullAttentionSpec):
            prefix_cacheable = "yes"  # type: ignore[assignment]

        with pytest.raises(ValueError, match="non-boolean prefix_cacheable"):
            is_non_prefix_cacheable_spec(Ambiguous(block_size=8))

    def test_unknown_spec_defers_to_declaration(self) -> None:
        """LMCache 从未见过的 spec 也按声明处理 —— 白名单不会漏掉下一个。"""

        class FutureScratchSpec(SlidingWindowSpec):
            prefix_cacheable = False

        assert (
            is_non_prefix_cacheable_spec(
                FutureScratchSpec(block_size=8, sliding_window=128)
            )
            is True
        )
        all_ring = UniformTypeKVCacheSpecs(
            block_size=8,
            kv_cache_specs={
                "a": CircularBufferSpec(block_size=8),
                "b": CircularBufferSpec(block_size=8),
            },
        )
        assert is_non_prefix_cacheable_spec(all_ring) is True
        mixed = UniformTypeKVCacheSpecs(
            block_size=8,
            kv_cache_specs={
                "a": CircularBufferSpec(block_size=8),
                "b": FullAttentionSpec(block_size=16),
            },
        )
        assert is_non_prefix_cacheable_spec(mixed) is False

    def test_ring_contradiction_fails_closed(self) -> None:
        """声明可缓存的环是矛盾，必须抛，不得静默放行。"""
        bad = CircularBufferSpec(block_size=8, prefix_cacheable=True)
        with pytest.raises(ValueError, match="prefix_cacheable=False"):
            is_non_prefix_cacheable_spec(bad)

    def test_ring_mask_all_false(self) -> None:
        """环的掩码显式全 False（round-trip 组数），调用方可整组跳过。"""
        groups = _prod_mask_input()
        masks = compute_group_store_masks(groups, 256, segment_tokens=256)
        ring = masks[12]
        assert ring.blocks == (False,) * (256 // 8)
        lookup = compute_group_lookup_masks(groups, 256)
        assert lookup[12].blocks == (False,) * 32


# ============================================================================
# 任务 a4：indexer 组边界有效性
# ============================================================================


class TestIndexerBoundaryValidity:
    def test_recognizes_indexer_row_widths(self) -> None:
        """fp8 (132) / mxfp4 (68) indexer 行的可求值识别 + 值/scale 分解。"""
        assert is_indexer_k_row(132, torch.uint8) is True
        assert resolve_indexer_row_geometry(132, torch.uint8) == (128, 4)
        assert is_indexer_k_row(68, torch.uint8) is True
        assert resolve_indexer_row_geometry(68, torch.uint8) == (64, 4)
        # 邻居不误判：主 MLA 584B/uint8、132B/非 uint8。
        assert is_indexer_k_row(584, torch.uint8) is False
        assert is_indexer_k_row(132, torch.float16) is False

    def test_block_boundary_closes_complete_groups(self) -> None:
        """引擎块 64 token、tps∈{1,2} ⇒ 整块内组边界 token 恰好填满所有 state。

        只有 ``(pos+1) % tps == 0`` 的 token 发布 indexer key（
        ``indexer_k_store.py:152-154``）；对一块 ``[b*64, (b+1)*64)``，
        ``64 % tps == 0`` ⇒ 块内 boundary token 数 = ``64 // tps`` =
        ``num_states``，所以满块写完时每个 state 槽都有有效 key —— 命中长度
        保持 ``tokens_per_block``（=64）对齐即 indexer 数据全部有效。
        [假定契约] 该推导承接 docs/02（indexer-boundary）§5；块/组对齐的
        LMCache 记账（tpb // spb == tps）在下一用例断言。
        """
        for tps in (1, 2):
            num_states = 64 // tps
            boundary_tokens = [pos for pos in range(64) if (pos + 1) % tps == 0]
            # 边界 token 位于 k*tps-1（每组末 token 发布该组 key），共 num_states 个。
            assert len(boundary_tokens) == num_states
            assert sorted(boundary_tokens) == [
                k * tps - 1 for k in range(1, num_states + 1)
            ]

    def test_lmcache_group_accounting_matches(self) -> None:
        """indexer kernel 组的 tpb//spb == tps（docs/02 §5 的记账恒等式）。

        G8 idx-cr2 子组：tpb=64、N=32、tps=2；idx-cr1：tpb=64、N=64、tps=1。
        """
        infos = _prod_engine_group_infos()
        indexer = [
            g
            for g in infos
            if g.engine_group_id == 8 and g.block_stride_alignment == 4608
        ]
        assert len(indexer) == 2
        # 每块物理 state 数 = tpb // tps（identity 的 block_size 维）。
        for g in indexer:
            assert g.tokens_per_block // g.tokens_per_state in (64 // 2, 64 // 1)


# ============================================================================
# 任务 a7：SWA 命中有效性
# ============================================================================


class TestSwaHitValidity:
    _SPEC = lmcache_native.EngineKVFormat.NL_X_NB_BS_HS

    def _swa_spec(self) -> object:
        # NL_X_NB_BS_HS: per-layer [NB, BS, HS]；BS=32 与生产 SWA 组一致。
        return get_spec(
            [torch.zeros(8, _SWA_BLOCK, 584, dtype=torch.uint8)], self._SPEC
        )

    def test_window_blocks_geometry(self) -> None:
        """窗口 128 token ÷ block 32 = 4 块；ceil 向上取整。"""
        spec = self._swa_spec()
        assert spec.sliding_window_blocks(128) == 4
        assert spec.sliding_window_blocks(100) == 4
        assert spec.sliding_window_blocks(0) == 0

    def test_window_aligned_boundary(self) -> None:
        """128 % 32 == 0 ⇒ 判据精确；100 % 32 != 0 ⇒ 不该宣称整块对齐。"""
        spec = self._swa_spec()
        assert spec.sliding_window_aligned(128) is True
        assert spec.sliding_window_aligned(100) is False

    def test_strict_chunk_validity_criterion(self) -> None:
        """chunk j 有效 ⟺ 其尾部仍在当前窗口：``(j+1)*C >= L - W``。

        V4.1 下 W=128 < C（chunk）⇒ 任意位置只有最后 ``ceil(W/C)=1`` 个 chunk
        的 SWA 数据有效。对 ``C ∈ {256, 4096}``、``L = n*C, n ∈ {1,2,5}``
        断言有效 chunk 集恒为 ``{n-1}``。
        """
        window = _SWA_WINDOW
        for chunk in (256, 4096):
            for n in (1, 2, 5):
                length = n * chunk
                valid = [j for j in range(n) if (j + 1) * chunk >= length - window]
                assert valid == [n - 1], (chunk, n, valid)

    def test_window_tail_mask_matches_prod(self) -> None:
        """生产几何的窗口尾掩码：block 32 / window 128 / chunk 256 → 留末 4 块。"""
        from lmcache.v1.multiprocess.group_view import _window_tail_mask

        mask = _window_tail_mask(32, 128, 0, 8, 256)
        assert mask == [False, False, False, False, True, True, True, True]
        # 窗口 ≥ 段（lcm 64）⇒ 全可达（None 哨兵），永不丢块。
        assert _window_tail_mask(32, 128, 0, 8, 64) is None


# ============================================================================
# 任务 a5：跨层共享下的组去重
# ============================================================================


class TestCrossLayerGroupDedup:
    _BLHNC = lmcache_native.EngineKVFormat.NL_X_NB_NH_BS_CS

    def test_identity_splits_g8_without_merge(self) -> None:
        """G8 的 8 个张量（main/indexer × cr2/cr1）拆成恰好 4 组、绝不合并。

        拆分维度 = head_size（584 vs 132）与每块 state 数 N（32 vs 64）；
        tps 不同 ⇒ N 必不同，identity 的 block_size 维是其单射代理。
        """
        tensors = [
            torch.zeros(32, 1, 32, 584, dtype=torch.uint8),  # main L2  cr=2
            torch.zeros(32, 1, 32, 584, dtype=torch.uint8),  # main L8  cr=2
            torch.zeros(32, 1, 32, 584, dtype=torch.uint8),  # main L14 cr=2
            torch.zeros(32, 1, 64, 584, dtype=torch.uint8),  # main L20 cr=1
            torch.zeros(32, 1, 32, 132, dtype=torch.uint8),  # idx L2  cr=2
            torch.zeros(32, 1, 32, 132, dtype=torch.uint8),  # idx L8  cr=2
            torch.zeros(32, 1, 32, 132, dtype=torch.uint8),  # idx L14 cr=2
            torch.zeros(32, 1, 64, 132, dtype=torch.uint8),  # idx L20 cr=1
        ]
        groups = group_layers_by_identity(
            tensors,
            [self._BLHNC] * len(tensors),
            per_layer_engine_group_idx=[8] * len(tensors),
        )
        assert len(groups) == 4
        members = {
            tuple(sorted(idxs)): (ident.head_size, ident.block_size, ident.kv_size)
            for ident, idxs in groups
        }
        assert members[(0, 1, 2)] == (584, 32, 1)
        assert members[(3,)] == (584, 64, 1)
        assert members[(4, 5, 6)] == (132, 32, 1)
        assert members[(7,)] == (132, 64, 1)
        # 熔接布局（NL_X_NB_NH_BS_CS）是单平面：kv_size 必须为 1。
        for _hs, _bs, kv in members.values():
            assert kv == 1

    def test_all_registered_cacheable_layers_assigned(self) -> None:
        """生产 51 个可缓存层全部被引用、无一落 EXCLUDED（环除外）。"""
        infos = _prod_engine_group_infos()
        config, kv_caches = _prod_vllm_config_and_caches()
        per_layer = get_engine_group_indices(infos, len(kv_caches))
        ring_indices = {
            i
            for i, name in enumerate(kv_caches)
            if name.endswith(".compressor.state_cache")
        }
        cacheable = {i for i in range(len(kv_caches))} - ring_indices
        assert cacheable and ring_indices
        assert cacheable <= {
            i for i, gid in enumerate(per_layer) if gid != EXCLUDED_ENGINE_GROUP
        }

    def test_orphan_layer_never_forms_group(self) -> None:
        """「已注册但未被任何组引用」的张量（跨层共享别名层）被防御性跳过。"""
        infos = [EngineGroupInfo(0, (0,))]
        per_layer = get_engine_group_indices(infos, 3)
        assert per_layer == [0, EXCLUDED_ENGINE_GROUP, EXCLUDED_ENGINE_GROUP]
