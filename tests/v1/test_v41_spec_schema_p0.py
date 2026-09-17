# SPDX-License-Identifier: Apache-2.0
"""P0 读取逻辑单测：vLLM nightly / v0.21.0 的 KV-cache spec schema。

覆盖对象（对应阶段一 P0 的「读懂」）：

- ``kv_cache_groups.resolve_tokens_per_state``：nightly 的 ``tokens_per_state``
  与旧版 ``compress_ratio`` 的兼容读取、缺省值 ``1``、Fraction 显式报错
  （不得静默取整）。
- ``kv_cache_groups.read_v41_spec_fields``：8 个新增字段的容错读取，字段
  缺失返回契约默认值，绝不抛 ``AttributeError``。
- ``kv_cache_group_edits._declares_slot_compression``：在**两版 spec 形状**上
  都给出正确答案，重点回归——v0.21.0 的 ``storage_block_size`` 是 property
  （恒非 None）不能被误判为压缩。
- ``utils.translate_vllm_kv_cache_layout``：布局名 → LMCache 布局名的映射，
  及不支持布局（heads-outermost）的 fail-fast。

环境要求：**无 vLLM、无 GPU** 即可运行。

- vLLM spec 用 duck-typed stub 对象模拟（见 ``NightlyMLASpec`` /
  ``LegacyMLASpec``），名字/字段形状按 ``vllm/v1/kv_cache_interface.py``
  构造，不 import 任何 ``vllm`` 包。
- ``kv_cache_group_edits`` 在模块层 import vLLM 与 torch；本文件在 import
  之前通过 ``sys.modules`` 注入最小 stand-in（沿用
  ``tests/v1/test_vllm_kv_layout_discovery.py`` 的既有模式）。若环境里真有
  vLLM / torch / 已编译的 ``lmcache_native``，则直接使用真实模块，不注入。
- 所有断言都只做纯 Python 读取，不触碰 CUDA 张量。

真实取值回归（任务要求 c）：V4.1 主 MLA spec
``(num_kv_heads=1, head_size=512, state_content_bytes=584, alignment=576,
tokens_per_state/compress_ratio ∈ {0,1,2})``，断言任何情况下都不出现静默错值。

> 本文件依赖阶段一同期合入的 ``p0-groups``（读者实现）补丁，在合并后树上
> 运行。读语义（malformed 输入→1、uniform 取首 leaf、cache_role 假值→sparse
> 等）按 ``p0-groups`` 实测实现对齐，见 docs/phase1/07-*.md。
"""

# Standard
from dataclasses import dataclass, field
from fractions import Fraction
from types import ModuleType, SimpleNamespace
import sys

# Third Party
import pytest

# --------------------------------------------------------------------------
# no-vLLM 引导：在 import ``kv_cache_group_edits`` 之前，确保 ``vllm`` /
# ``torch`` / ``lmcache_native`` 可解析（真实可用则用真实的）。
# --------------------------------------------------------------------------


def _register_module(module: ModuleType) -> None:
    """把 ``module`` 及其父包注册进 ``sys.modules``（供 import 解析用）。"""
    parts = module.__name__.split(".")
    for i in range(1, len(parts)):
        parent = ".".join(parts[:i])
        sys.modules.setdefault(parent, ModuleType(parent))
    sys.modules[module.__name__] = module


def _stub_torch_if_missing() -> None:
    """torch 缺失时注入最小但足以 import 的 stand-in。

    真实 ``lmcache`` 包的导入链在模块层就会引用 dtype 常量与类型对象
    （``lmcache/utils.py`` 的 ``TORCH_DTYPE_TO_STR_DTYPE`` 与
    ``DiskCacheMetadata`` 注解），所以 stub 需要提供这些名字；具体函数体
    (kernel 等) 不会被本测试触达。若环境里真有 torch 则直接用真实的。
    """
    try:
        import torch  # noqa: F401
    except ImportError:
        stub = ModuleType("torch")

        class _DType:
            """嗅探用假 dtype：只要求可哈希、可身份比较（dict 键用）。"""

        def _dtype_const(name: str) -> _DType:
            return _DType()

        # 真实包 import 路径上出现的 dtype 常量与类型对象。
        for _name in (
            "half",
            "float16",
            "bfloat16",
            "float",
            "float32",
            "double",
            "float64",
            "int8",
            "uint8",
            "int16",
            "int32",
            "int64",
            "bool",
            "uint16",
            "uint32",
            "uint64",
            "complex64",
            "complex128",
            "qint8",
            "quint8",
        ):
            setattr(stub, _name, _dtype_const(_name))
        stub.Size = tuple
        stub.dtype = _DType
        stub.Tensor = object
        stub.device = lambda *args, **kwargs: SimpleNamespace(type="cpu")
        stub.tensor = lambda *args, **kwargs: object()
        stub.frombuffer = lambda *args, **kwargs: object()
        stub._dynamo = ModuleType("torch._dynamo")
        _register_module(stub)


def _stub_vllm_if_missing() -> None:
    """vLLM 缺失时注入 ``vllm.v1.kv_cache_interface`` 的最小 stand-in。

    ``kv_cache_group_edits`` 在模块层从它 import 四个名字；本文件只调用
    ``_declares_slot_compression``（不调用 ``get_kv_cache_spec_kind`` 等），
    所以占位实现足够。
    """
    try:
        import vllm  # noqa: F401
    except ImportError:

        class KVCacheSpecKind(str):  # 只含模块常量用到的成员
            FULL_ATTENTION = "full_attention"
            SLIDING_WINDOW = "sliding_window"
            CHUNKED_LOCAL_ATTENTION = "chunked_local_attention"
            SINK_FULL_ATTENTION = "sink_full_attention"

        stub = ModuleType("vllm.v1.kv_cache_interface")
        stub.KVCacheConfig = object
        stub.KVCacheSpec = object
        stub.KVCacheSpecKind = KVCacheSpecKind
        stub.get_kv_cache_spec_kind = lambda spec: None
        _register_module(stub)


def _stub_gpu_connector_utils_if_missing() -> None:
    """``lmcache_native`` 未编译时，为 ``kv_cache_group_edits`` 的
    ``from lmcache.v1.gpu_connector.utils import LayoutHints`` 注入占位模块。

    只有真实模块 import 失败（原生未编译）才注入；此时该模块本来谁都 import
    不了，注入不会污染任何真实代码路径。
    """
    try:
        from lmcache.v1.gpu_connector.utils import LayoutHints  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        stub = ModuleType("lmcache.v1.gpu_connector.utils")
        stub.LayoutHints = dict  # TypedDict；读取侧只当 Mapping 用
        _register_module(stub)


def _ensure_no_vllm_bootstrap() -> None:
    """只在本环境缺少依赖时注入 stand-in；幂等。"""
    _stub_torch_if_missing()
    _stub_vllm_if_missing()
    _stub_gpu_connector_utils_if_missing()


_ensure_no_vllm_bootstrap()

# First Party
from lmcache.integration.vllm.kv_cache_group_edits import (  # noqa: E402
    _declares_slot_compression,
)
from lmcache.integration.vllm.kv_cache_groups import (  # noqa: E402
    read_v41_spec_fields,
    resolve_tokens_per_state,
)
from lmcache.integration.vllm.utils import translate_vllm_kv_cache_layout  # noqa: E402

# --------------------------------------------------------------------------
# duck-typed spec stub：按真实 vLLM 类形状构造
# --------------------------------------------------------------------------
# nightly（``vllm/vllm/v1/kv_cache_interface.py``）：
#   ``AttentionSpec.tokens_per_state: int | Fraction = 1``（:484-487）、
#   ``state_content_bytes``（:482）、``page_size_padded``（:477）、
#   ``num_head_slots``（:478）；``MLAAttentionSpec`` 的 ``cache_role``
#   （:642）、``is_index_group_leader``（:643）、``storage_block_size``
#   （:644，**dataclass 字段**，默认 None）、``block_stride_alignment``（:646）。
# v0.21.0（``vllm-0.21.0-ref/.../kv_cache_interface.py``）：无新字段，只有
#   ``compress_ratio: int = 1``（:329）与作为 **property** 的
#   ``storage_block_size``（:337-338，恒返回 int、不落实例 __dict__）。
# --------------------------------------------------------------------------


@dataclass
class NightlyMLASpec:
    """nightly ``MLAAttentionSpec`` 的 duck-typed 形状（读取侧用到的字段）。

    ``cache_role`` 默认 ``None`` 表示「未设置」——V4.1 的 main/indexer spec
    都不设（``vllm/models/deepseek_v41/attention.py:1046-1056,1096-1104``），
    读取器应回退到契约默认 ``"sparse"``。frozen=False（普通 dataclass），
    因此实例字段（含 ``storage_block_size`` 默认 None）都落进 ``__dict__``，
    与 nightly 的 dataclass 字段行为一致。
    """

    block_size: int
    num_kv_heads: int = 1
    head_size: int = 512
    tokens_per_state: "int | Fraction" = 1
    state_content_bytes: "int | None" = None
    page_size_padded: "int | None" = None
    num_head_slots: "int | None" = None
    cache_role: object = None
    is_index_group_leader: bool = False
    prefix_cacheable: bool = True
    storage_block_size: "int | None" = None  # dataclass 字段（nightly）
    block_stride_alignment: "int | None" = None
    model_version: "str | None" = None
    alignment: "int | None" = None
    cache_dtype_str: "str | None" = None


@dataclass
class LegacyMLASpec:
    """v0.21.0 ``MLAAttentionSpec`` 的 duck-typed 形状。

    ``compress_ratio: int = 1``（:329）；``storage_block_size`` 是 **property**
    （:337-338，``block_size // compress_ratio``），只挂在类上、永不进实例
    ``__dict__``——这是「v0.21.0 不得被误判为压缩」回归的构造关键。
    """

    block_size: int
    num_kv_heads: int = 1
    head_size: int = 512
    compress_ratio: int = 1
    page_size_padded: "int | None" = None  # v0.21.0 AttentionSpec 本就有
    model_version: "str | None" = None
    alignment: "int | None" = None
    cache_dtype_str: "str | None" = None

    @property
    def storage_block_size(self) -> int:
        return self.block_size // self.compress_ratio


@dataclass
class UniformTypeKVCacheSpecs:
    """nightly ``UniformTypeKVCacheSpecs``：逐层包叶子 spec（读取侧形状）。"""

    block_size: int
    kv_cache_specs: dict[str, object] = field(default_factory=dict)


class DummySparseRole:
    """``SparseCacheRole`` 的迷你版：带 ``.value``，用于测枚举→字符串归一化。"""

    SPARSE = SimpleNamespace(value="sparse")
    INDEXER = SimpleNamespace(value="indexer")


def _round_up_byte_pages(unpadded: int, alignment: int) -> int:
    """镜像 nightly 的 ``_apply_alignment_padding``（``:626-632``）：把未对齐
    页字节上取整到 ``alignment`` 的倍数。真实 spec 在 ``__post_init__`` 里
    把它写进实例字段 ``page_size_padded``。"""
    return ((unpadded + alignment - 1) // alignment) * alignment


def v41_main_mla(compress_ratio: int) -> NightlyMLASpec:
    """构造 V4.1 主 sparse-MLA KV 的真实取值 spec（``attention.py:1046``）。

    ``num_kv_heads=1``（H=1）、``state_content_bytes=584``（SM90 584B/token）、
    ``alignment=576``、``cache_dtype_str="fp8_ds_mla"``、
    ``model_version="deepseek_v4"``。``cache_role`` / ``is_index_group_leader``
    不设置（V4.1 不设，对应事实 5）。

    ``page_size_padded`` 按真实派生算术给出（``round_up(states*584, 576)``）：
    cr=1 → 37440、cr=2 → 19008（docs/02 §4.10 算术；H100 运行时数值 [待验证]）。
    ``cr=0`` 是构造出来的滑窗层（真实 cr=0 层不产 spec，走 ``SlidingWindowMLASpec``
    且 ``tokens_per_state`` 缺省 1），此处不带派生页大小。
    """
    page_size_padded = None
    if compress_ratio > 0:
        states_per_block = 64 // compress_ratio
        page_size_padded = _round_up_byte_pages(states_per_block * 584, 576)
    return NightlyMLASpec(
        block_size=64,
        num_kv_heads=1,
        head_size=512,
        tokens_per_state=compress_ratio,
        state_content_bytes=584,
        page_size_padded=page_size_padded,
        alignment=576,
        cache_dtype_str="fp8_ds_mla",
        model_version="deepseek_v4",
    )


def v41_indexer_mla(compress_ratio: int, sparse_logits: bool = True) -> NightlyMLASpec:
    """构造 V4.1 indexer 的真实取值 spec（``attention.py:1096``）。

    与主 spec 一样 ``num_kv_heads=1`` 但**不设** ``state_content_bytes``
    （indexer 记录宽度由 ``head_dim`` 决定，``attention.py:1096-1104`` 未传）；
    稀疏 logits 时 ``block_stride_alignment=math.lcm(512, 576)=4608``
    （``attention.py:1124-1126``，SM90 下 ``page_alignment=576``）。
    """
    return NightlyMLASpec(
        block_size=64,
        num_kv_heads=1,
        head_size=512,
        tokens_per_state=compress_ratio,
        alignment=576,
        cache_dtype_str="fp8_ds_mla",
        model_version="deepseek_v4",
        block_stride_alignment=4608 if sparse_logits else None,
    )


# ============================================================================
# resolve_tokens_per_state
# ============================================================================


class TestResolveTokensPerState:
    """``resolve_tokens_per_state``：兼容两版字段名、缺省 1、Fraction 报错。"""

    def test_nightly_tokens_per_state_declared(self) -> None:
        """nightly 直接声明 ``tokens_per_state`` 时原样返回。"""
        assert resolve_tokens_per_state(v41_main_mla(2)) == 2
        assert resolve_tokens_per_state(v41_main_mla(1)) == 1

    def test_legacy_compress_ratio_fallback(self) -> None:
        """v0.21.0 无 ``tokens_per_state``，回退读 ``compress_ratio``。"""
        legacy = LegacyMLASpec(block_size=64, compress_ratio=2)
        assert resolve_tokens_per_state(legacy) == 2

    def test_legacy_compress_ratio_one_not_compressed(self) -> None:
        """``compress_ratio=1`` 等价于「未压缩」→ 1。"""
        legacy = LegacyMLASpec(block_size=64, compress_ratio=1)
        assert resolve_tokens_per_state(legacy) == 1

    def test_default_when_undeclared(self) -> None:
        """两版字段都不存在时返回契约缺省 1。"""
        assert resolve_tokens_per_state(SimpleNamespace(block_size=64)) == 1
        assert resolve_tokens_per_state(object()) == 1

    @pytest.mark.parametrize("fraction", [Fraction(1, 4), Fraction(3, 2)])
    def test_fraction_raises(self, fraction: Fraction) -> None:
        """Fraction 必须显式抛 ``ValueError``，不得 ``int()`` 静默取整。"""
        spec = NightlyMLASpec(block_size=64, tokens_per_state=fraction)
        with pytest.raises(ValueError, match="Fraction|fractional|tokens_per_state"):
            resolve_tokens_per_state(spec)

    def test_malformed_type_ignored(self) -> None:
        """非 ``int``/``Fraction`` 的声明（如 ``"2"``）视为未声明 → 1。

        与 p0-groups 实际实现一致（``kv_cache_groups.py`` 打补丁后
        ``resolve_tokens_per_state``：非 int 直接返回 1）；fail-closed 只在
        ``Fraction`` 上触发（任务契约只强制 Fraction 报错）。
        """
        spec = NightlyMLASpec(  # type: ignore[arg-type]
            block_size=64, tokens_per_state="2"
        )
        assert resolve_tokens_per_state(spec) == 1

    def test_mamba_minus_one_sentinel_ignored(self) -> None:
        """MambaSpec 的 ``tokens_per_state=-1`` 哨兵（``:1002``）不代表压缩，
        按未声明处理 → 1。[假定契约] 与参考实现一致（``raw > 0`` 才参与）。"""
        spec = NightlyMLASpec(block_size=64, tokens_per_state=-1)
        assert resolve_tokens_per_state(spec) == 1

    def test_v41_sliding_window_zero_not_silently_wrong(self) -> None:
        """V4.1 ``compress_ratio=0``（纯滑窗）不得解析出 0 或 2。

        ``0`` 会让下游 ``tokens_per_block % tokens_per_state`` 除零；``2``
        会把滑窗层错当成 ratio-2 压缩层。正确语义是每 token 一个 state → 1。
        [假定契约] 参考实现以 ``raw > 0`` 过滤，0 落回缺省 1。
        """
        for spec in (v41_main_mla(0), v41_indexer_mla(0)):
            assert resolve_tokens_per_state(spec) == 1

    def test_uniform_container_uses_first_leaf(self) -> None:
        """``UniformTypeKVCacheSpecs`` 取**首个** leaf（p0-groups 实测实现）。

        vLLM 的 ``is_uniform_type`` 保证同容器叶子同类型（
        ``kv_cache_interface.py:1229-1240``），首 leaf 即可代表；所以
        「首 leaf=2、次 leaf=1」也返回 2，而不是取 max/min。
        """
        spec = UniformTypeKVCacheSpecs(
            block_size=64,
            kv_cache_specs={
                "layer.0": NightlyMLASpec(block_size=64, tokens_per_state=2),
                "layer.1": NightlyMLASpec(block_size=64, tokens_per_state=1),
            },
        )
        assert resolve_tokens_per_state(spec) == 2


# ============================================================================
# read_v41_spec_fields
# ============================================================================


class TestReadV41SpecFields:
    """``read_v41_spec_fields``：8 字段容错读取，缺失用契约默认值。"""

    def test_all_missing_fields_fall_back_to_contract_defaults(self) -> None:
        """空对象 → 8 个字段全部为契约默认值，且不抛 ``AttributeError``。"""
        assert read_v41_spec_fields(object()) == {
            "tokens_per_state": 1,
            "cache_role": "sparse",
            "state_content_bytes": None,
            "block_stride_alignment": None,
            "is_index_group_leader": False,
            "prefix_cacheable": True,
            "page_size_padded": None,
            "num_head_slots": None,
        }

    def test_v41_main_spec_reads_real_values(self) -> None:
        """V4.1 主 spec：读到真实字段值；未设置的角色字段回退 ``sparse``。"""
        fields = read_v41_spec_fields(v41_main_mla(2))
        assert fields["tokens_per_state"] == 2
        assert fields["state_content_bytes"] == 584  # SM90 584B/token
        assert fields["cache_role"] == "sparse"  # V4.1 不设 → 默认 SPARSE
        assert fields["is_index_group_leader"] is False  # V4.1 不设
        assert fields["block_stride_alignment"] is None  # 主组不设
        assert fields["prefix_cacheable"] is True
        assert fields["page_size_padded"] == 19008  # round_up(32*584, 576)
        assert fields["num_head_slots"] is None

    def test_v41_indexer_spec_role_stays_sparse(self) -> None:
        """V4.1 indexer 不设 ``cache_role``/``is_index_group_leader``（事实 5）。

        不能靠 ``cache_role`` 识别 indexer；读取器必须给出默认
        ``sparse``/``False``；indexer 也不设 ``state_content_bytes``
        （宽度由 ``head_dim`` 定，``attention.py:1096-1104`` 未传）。
        """
        spec = v41_indexer_mla(2, sparse_logits=True)
        fields = read_v41_spec_fields(spec)
        assert fields["cache_role"] == "sparse"
        assert fields["is_index_group_leader"] is False
        assert fields["state_content_bytes"] is None
        assert fields["block_stride_alignment"] == 4608  # math.lcm(512, 576)

    def test_enum_role_normalized_to_string(self) -> None:
        """nightly 的 ``cache_role`` 是 ``SparseCacheRole`` 枚举 → 归一成字符串。"""
        spec = v41_main_mla(2)
        spec.cache_role = DummySparseRole.INDEXER  # type: ignore[attr-defined]
        assert read_v41_spec_fields(spec)["cache_role"] == "indexer"

    def test_str_role_passthrough(self) -> None:
        """已经是字符串的 ``cache_role`` 原样保留。"""
        spec = v41_main_mla(1)
        spec.cache_role = "indexer"  # type: ignore[attr-defined]
        assert read_v41_spec_fields(spec)["cache_role"] == "indexer"

    def test_falsy_role_falls_back_to_sparse(self) -> None:
        """空字符串/缺失的角色值按缺省 ``sparse`` 处理（``""`` 是假值）。"""
        spec = v41_main_mla(1)
        spec.cache_role = ""  # type: ignore[attr-defined]
        assert read_v41_spec_fields(spec)["cache_role"] == "sparse"

    def test_legacy_spec_returns_new_field_defaults(self) -> None:
        """v0.21.0 spec：新字段缺失全部走默认；``page_size_padded`` 本就存在。

        [假定契约] ``tokens_per_state`` 经 ``compress_ratio`` 回退得到 2
        （与 ``resolve_tokens_per_state`` 一致）；若 p0-groups 把
        ``read_v41_spec_fields`` 做成纯 raw 读取（缺失即 1），此处需同步改。
        """
        legacy = LegacyMLASpec(block_size=64, compress_ratio=2, page_size_padded=37376)
        fields = read_v41_spec_fields(legacy)
        assert fields["tokens_per_state"] == 2  # compress_ratio 回退
        assert fields["cache_role"] == "sparse"
        assert fields["is_index_group_leader"] is False
        assert fields["prefix_cacheable"] is True
        assert fields["state_content_bytes"] is None
        assert fields["block_stride_alignment"] is None
        assert fields["num_head_slots"] is None

    def test_prefix_cacheable_false_respected(self) -> None:
        """显式 ``prefix_cacheable=False`` 要透传（不可静默变回 True）。"""
        spec = v41_main_mla(1)
        spec.prefix_cacheable = False  # type: ignore[attr-defined]
        assert read_v41_spec_fields(spec)["prefix_cacheable"] is False

    def test_index_group_leader_true_respected(self) -> None:
        """V3.2 会设 ``is_index_group_leader=True``（事实 5），须透传。"""
        spec = v41_main_mla(1)
        spec.is_index_group_leader = True  # type: ignore[attr-defined]
        assert read_v41_spec_fields(spec)["is_index_group_leader"] is True

    def test_declared_page_size_padded_passthrough(self) -> None:
        """对齐填充后 nightly 会把 ``page_size_padded`` 写进实例字段
        （``kv_cache_interface.py:626-632``），读取器须透传实例值而非回退
        None——无论该值是否与派生算术吻合。"""
        spec = v41_main_mla(1)
        spec.page_size_padded = 12345  # type: ignore[attr-defined]  # 显式申明
        fields = read_v41_spec_fields(spec)
        assert fields["page_size_padded"] == 12345
        assert fields["tokens_per_state"] == 1

    def test_object_with_partial_fields_never_raises(self) -> None:
        """只带部分字段的对象也不抛 ``AttributeError``（其余走默认）。"""
        spec = SimpleNamespace(tokens_per_state=2, num_head_slots=1)
        fields = read_v41_spec_fields(spec)
        assert fields["tokens_per_state"] == 2
        assert fields["num_head_slots"] == 1
        assert fields["cache_role"] == "sparse"
        assert fields["state_content_bytes"] is None
        assert fields["block_stride_alignment"] is None


# ============================================================================
# _declares_slot_compression（两版 spec 形状回归）
# ============================================================================


class TestDeclaresSlotCompression:
    """``_declares_slot_compression`` 在两版 spec 上都给出正确判断。"""

    # -- v0.21.0：storage_block_size 是 property，恒非 None，绝不能误判 --

    def test_legacy_full_attention_property_storage_not_compression(self) -> None:
        """v0.21.0 普通 attention：property 型 storage_block_size=block_size，
        不得判为压缩（rev.2 关键回归项，docs/06 §6.2）。"""
        # 用 LegacyMLASpec 形状（property 类属性、不进实例 __dict__）。
        legacy = LegacyMLASpec(block_size=64)
        assert "storage_block_size" not in legacy.__dict__
        assert not _declares_slot_compression(legacy)

    def test_legacy_mla_compress_ratio_one_not_compression(self) -> None:
        """v0.21.0 ``compress_ratio=1`` 的 MLA：property 不触发，ratio=1 也不触发。"""
        legacy = LegacyMLASpec(block_size=64, compress_ratio=1)
        assert "storage_block_size" not in legacy.__dict__
        assert not _declares_slot_compression(legacy)

    def test_legacy_mla_compress_ratio_two_compression(self) -> None:
        """v0.21.0 ``compress_ratio=2``：经 ``compress_ratio`` 回退判为压缩。"""
        legacy = LegacyMLASpec(block_size=64, compress_ratio=2)
        assert _declares_slot_compression(legacy)

    def test_legacy_tq_slot_size(self) -> None:
        """``tq_slot_size>0`` 两版一致判压缩；``0`` 不判。"""
        assert _declares_slot_compression(SimpleNamespace(tq_slot_size=8))
        assert not _declares_slot_compression(SimpleNamespace(tq_slot_size=0))

    # -- nightly：tokens_per_state > 1 必须被识别（V4.1 压缩层不再漏报） --

    def test_nightly_v41_main_tokens_per_state_two_compression(self) -> None:
        """nightly V4.1 主 MLA ``tokens_per_state=2`` 必须判为压缩（关键回归）。"""
        assert _declares_slot_compression(v41_main_mla(2))

    def test_nightly_v41_indexer_tokens_per_state_two_compression(self) -> None:
        """nightly V4.1 indexer ``tokens_per_state=2``：同样判为压缩。"""
        assert _declares_slot_compression(v41_indexer_mla(2))

    def test_nightly_tokens_per_state_one_not_compression(self) -> None:
        """``tokens_per_state=1``（含 storage_block_size=None 字段）不判压缩。"""
        assert not _declares_slot_compression(v41_main_mla(1))

    def test_nightly_storage_block_size_field_triggered(self) -> None:
        """nightly 字段型 ``storage_block_size != None``（GLM-5 kpool）判压缩。"""
        spec = NightlyMLASpec(block_size=64, tokens_per_state=1, storage_block_size=16)
        assert _declares_slot_compression(spec)

    def test_nightly_storage_block_size_none_not_compression(self) -> None:
        """nightly 字段型 ``storage_block_size=None`` = 未声明几何，不判压缩。"""
        spec = NightlyMLASpec(block_size=64, tokens_per_state=1)
        assert "storage_block_size" in spec.__dict__
        assert not _declares_slot_compression(spec)

    def test_nightly_fractional_tokens_per_state_compression(self) -> None:
        """Fraction 且 != 1（Whisper 一个 token 多个 state）也是压缩路径。"""
        spec = NightlyMLASpec(block_size=64, tokens_per_state=Fraction(1, 4))
        assert _declares_slot_compression(spec) is True

    def test_nightly_mamba_minus_one_not_compression(self) -> None:
        """Mamba ``-1`` 哨兵不是压缩信号。"""
        spec = NightlyMLASpec(block_size=64, tokens_per_state=-1)
        assert not _declares_slot_compression(spec)

    def test_block_stride_alignment_alone_not_compression(self) -> None:
        """``block_stride_alignment`` 是对齐元数据，不是压缩；单独出现不判。"""
        spec = NightlyMLASpec(
            block_size=64, tokens_per_state=1, block_stride_alignment=576
        )
        assert not _declares_slot_compression(spec)

    def test_v41_sliding_window_zero_not_compression(self) -> None:
        """V4.1 ``compress_ratio=0``（滑窗层）不得被判成压缩层。"""
        assert not _declares_slot_compression(v41_main_mla(0))
        assert not _declares_slot_compression(v41_indexer_mla(0))


# --------------------------------------------------------------------------
# 布局 → LMCache 布局名映射（EngineKVFormat 选择的前置），不支持布局 fail-fast
# --------------------------------------------------------------------------
# LMCache 只登记 4 个可传布局名（``integration/vllm/utils.py:59-69``）：
#   LBNHC -> NHD、LBHNC -> HND、BLHNC -> BLHNC、BLNHC -> BLNHC；
# heads-outermost（LHBNC/BHLNC）与未知名抛 NotImplementedError。
# 生产 V4.1（H100）解析到 **BLHNC**（docs/04 §1.7），block-outer：
#   per-layer 视图 ``[B,H,N,C]`` 的 stride(0) 是跨层打包整行，detector 据此
#   选 ``NL_X_NB_NH_BS_CS``（docs/04 §2.2 映射行），块步进由
#   ``resolve_block_stride_and_log_layout()`` 运行期补足。
# --------------------------------------------------------------------------


class TestV41LayoutMapping:
    """``translate_vllm_kv_cache_layout``：布局名映射 + 不支持布局 fail-fast。"""

    def test_blhnc_production_layout_maps(self) -> None:
        """生产 V4.1 布局 BLHNC 映射到 LMCache 名 ``BLHNC``（CS 格式族）。"""
        assert translate_vllm_kv_cache_layout("BLHNC") == "BLHNC"

    def test_blnhc_maps(self) -> None:
        assert translate_vllm_kv_cache_layout("BLNHC") == "BLNHC"

    def test_global_aliases_map_to_per_layer_names(self) -> None:
        """layer-outer 别名：LBNHC≈NHD、LBHNC≈HND。"""
        assert translate_vllm_kv_cache_layout("LBNHC") == "NHD"
        assert translate_vllm_kv_cache_layout("LBHNC") == "HND"

    def test_lmcache_names_pass_through(self) -> None:
        for name in ("NHD", "HND"):
            assert translate_vllm_kv_cache_layout(name) == name

    @pytest.mark.parametrize("name", ["LHBNC", "BHLNC"])
    def test_heads_outermost_layouts_fail_fast(self, name: str) -> None:
        """heads-outermost 无 EngineKVFormat 可表达，必须启动即失败。"""
        with pytest.raises(NotImplementedError, match=name):
            translate_vllm_kv_cache_layout(name)

    def test_unknown_layout_fails_fast(self) -> None:
        with pytest.raises(NotImplementedError, match="XYZ"):
            translate_vllm_kv_cache_layout("XYZ")


# --------------------------------------------------------------------------
# 组合回归：V4.1 真实取值，三个压缩比同时过三组读取器（任务 c）
# --------------------------------------------------------------------------


class TestV41RealValuesRegression:
    """V4.1 真实取值（H=1、C=584、compress_ratio∈{0,1,2}）组合回归。

    断言「不出现静默错值」：

    - ``compress_ratio=2``：必须解析成 2 并判为压缩——不能漏报成未压缩
      （字节布局错），也不能静默取整；
    - ``compress_ratio=1``：解析成 1、不判压缩；
    - ``compress_ratio=0``（滑窗）：解析成 1（非 0/2）、不判压缩。
    """

    @staticmethod
    def _expect(compress_ratio: int) -> tuple[int, bool]:
        return {0: (1, False), 1: (1, False), 2: (2, True)}[compress_ratio]

    @pytest.mark.parametrize("compress_ratio", [0, 1, 2])
    def test_main_spec_no_silent_wrong_value(self, compress_ratio: int) -> None:
        resolved, compressed = self._expect(compress_ratio)
        spec = v41_main_mla(compress_ratio)
        assert resolve_tokens_per_state(spec) == resolved
        assert _declares_slot_compression(spec) is compressed
        fields = read_v41_spec_fields(spec)
        assert fields["tokens_per_state"] == resolved
        assert fields["state_content_bytes"] == 584
        assert fields["cache_role"] == "sparse"
        assert fields["is_index_group_leader"] is False
        # 真实派生页大小：cr=2 → 19008、cr=1 → 37440、cr=0（滑窗）→ 未申明。
        expected_page = {2: 19008, 1: 37440, 0: None}[compress_ratio]
        assert fields["page_size_padded"] == expected_page

    @pytest.mark.parametrize("compress_ratio", [0, 1, 2])
    def test_indexer_spec_no_silent_wrong_value(self, compress_ratio: int) -> None:
        resolved, compressed = self._expect(compress_ratio)
        spec = v41_indexer_mla(compress_ratio)
        assert resolve_tokens_per_state(spec) == resolved
        assert _declares_slot_compression(spec) is compressed
        fields = read_v41_spec_fields(spec)
        assert fields["tokens_per_state"] == resolved
        # indexer 不设 state_content_bytes（宽度由 head_dim 定），必须回退 None。
        assert fields["state_content_bytes"] is None
        assert fields["block_stride_alignment"] == 4608  # math.lcm(512, 576)
        assert fields["cache_role"] == "sparse"
