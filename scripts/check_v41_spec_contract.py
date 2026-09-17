#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Assert that LMCache still understands every vLLM KV-cache-spec field it needs.

Why this gate exists
--------------------
LMCache reads vLLM's per-layer ``KVCacheSpec`` objects by attribute name
(``lmcache/integration/vllm/kv_cache_groups.py``,
``kv_cache_group_edits.py``).  vLLM nightly renames and reshapes those fields
freely -- ``MLAAttentionSpec.compress_ratio`` became
``AttentionSpec.tokens_per_state`` in one nightly step -- and a rename is
*silent* for LMCache: ``getattr(spec, "old_name", default)`` keeps returning
the default, so LMCache would keep running with wrong geometry and corrupt KV
interoperability rather than raising.

This script generalises ``scripts/check_v41_gap.sh`` (a hand-written,
non-failing report over 11 hard-coded names) into an assertion-based gate:

1. **Extract** the full member inventory (dataclass fields *and* properties) of
   every tracked KV-cache-spec class from a pinned vLLM
   ``vllm/v1/kv_cache_interface.py``, parsed with ``ast``.
2. **Diff** that inventory against ``scripts/v41_spec_contract.json`` -- the
   reviewed contract.  A member that vLLM added is ``UNREVIEWED_MEMBER``; one
   that vLLM removed is ``REMOVED_MEMBER`` (the ``compress_ratio`` scenario);
   a field that became a property is ``KIND_CHANGED``.
3. **Cover** every contract member whose disposition is ``required`` by a
   word-boundary search over ``lmcache/**/*.py``.  A ``required`` member with
   zero references is ``BLIND_REQUIRED_MEMBER``.

Any of those three findings is a hard failure (exit code 1), so a vLLM bump
cannot land without either updating LMCache or explicitly re-reviewing the
contract.

Dispositions
------------
``required``
    LMCache must name this member.  Blind => fail.
``derived``
    LMCache intentionally does not read this member by name; it re-derives the
    same quantity from the registered KV tensors (byte width, slot count,
    layer views).  Blind => warning only.  Each entry records the reason.
``known_gap``
    A reviewed, still-open gap that is out of scope for the current
    adaptation.  Blind => warning only; recorded in the contract so it stays
    visible instead of being silently classified as ``derived``.
``not_applicable_v41``
    The member exists on a V4.1-reachable class but is never set (or never
    consumed) by DeepSeek-V4.1 on H100/SM90.  Blind => warning only.
``unreviewed``
    Nobody has classified this member yet -- the default for a member that a
    vLLM bump just introduced.  Always a failure, so the bump forces a review.

Scope and limitations
---------------------
* Only the declared *members* (fields/properties) are tracked; method bodies
  are not, so a semantic change that keeps the name (e.g. new arithmetic in
  ``KVCacheSpec.get_num_kernel_states``) is only caught by the
  ``spec_module_sha256`` provenance warning plus the offline tests.
* Coverage is a name-reference check, not a dataflow proof: a member that
  appears only in a comment does not count (comment lines are skipped), but a
  member named in a docstring does.  Its job is to catch a *dropped* read
  (``required`` member referenced nowhere in ``lmcache/**/*.py``), not to prove
  that the reference is the one that matters.  ``required`` is therefore the
  set of members LMCache reads by name on the vLLM-facing surface today; the
  non-tautological drift protection is the contract diff and the
  ``unreviewed`` default.
* Only the classes listed in the contract's ``tracked_classes`` are diffed.
  That set is the DeepSeek-V4.1 group chain documented in
  ``docs/02-v41-spec-field-reference.md`` section 1, not the whole
  ``KVCacheSpec`` hierarchy (Mamba/HiSparse/chunked-local are out of scope).

Usage
-----
::

    # Gate the current tree against the pinned vLLM spec (downloads one file).
    python scripts/check_v41_spec_contract.py

    # Gate against a local vLLM checkout or a single spec file.
    python scripts/check_v41_spec_contract.py --vllm-root /path/to/vllm
    python scripts/check_v41_spec_contract.py \
        --vllm-spec-file /path/to/kv_cache_interface.py

    # Nightly drift alarm: latest main instead of the pin.
    python scripts/check_v41_spec_contract.py \\
        --vllm-spec-file https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/kv_cache_interface.py

    # Re-derive the inventory after a vLLM bump (new members land as
    # ``unreviewed``, which fails the gate until a human classifies them).
    python scripts/check_v41_spec_contract.py \
        --vllm-root /path/to/vllm --update-contract

Environment fallbacks: ``V41_VLLM_SPEC_FILE`` (file or URL) and ``VLLM_SRC``
(a vLLM checkout root).
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence
import argparse
import ast
import hashlib
import json
import os
import re
import sys
import urllib.request

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXIT_OK = 0
EXIT_GATE_FAILED = 1
EXIT_SETUP_ERROR = 2

DEFAULT_SPEC_MODULE = "vllm/v1/kv_cache_interface.py"
DEFAULT_CONTRACT = "scripts/v41_spec_contract.json"

REQUIRED = "required"
DERIVED = "derived"
KNOWN_GAP = "known_gap"
NOT_APPLICABLE = "not_applicable_v41"
UNREVIEWED = "unreviewed"

DISPOSITIONS = (REQUIRED, DERIVED, KNOWN_GAP, NOT_APPLICABLE, UNREVIEWED)

FIELD = "field"
PROPERTY = "property"


class GateError(RuntimeError):
    """Raised when the gate cannot run (bad inputs, unreadable source)."""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Member:
    """One declared member of a vLLM KV-cache-spec class.

    Attributes:
        name: Member identifier, e.g. ``tokens_per_state``.
        kind: ``"field"`` for a dataclass field, ``"property"`` for a property.
        owner: Name of the most-derived class that declares the member.
    """

    name: str
    kind: str
    owner: str


@dataclass(frozen=True)
class Finding:
    """One gate result line.

    Attributes:
        level: ``"fail"``, ``"warn"`` or ``"info"``.
        code: Stable machine-readable code, e.g. ``REMOVED_MEMBER``.
        member: Member the finding is about, or ``"-"`` for global findings.
        message: Human-readable explanation.
    """

    level: str
    code: str
    member: str
    message: str


# ---------------------------------------------------------------------------
# Contract I/O
# ---------------------------------------------------------------------------


def load_contract(path: Path) -> dict:
    """Read and minimally validate the contract JSON.

    Args:
        path: Contract file path.

    Returns:
        The parsed contract mapping.

    Raises:
        GateError: If the file is missing, unreadable, malformed, or lacks the
            ``tracked_classes`` / ``members`` mappings.
    """
    if not path.is_file():
        raise GateError(f"contract not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GateError(f"cannot read contract {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise GateError(f"contract {path} must be a JSON object")
    for key in ("tracked_classes", "members", "pinned_vllm"):
        if key not in data:
            raise GateError(f"contract {path} is missing the {key!r} key")
    members = data["members"]
    if not isinstance(members, dict):
        raise GateError(f"contract {path}: 'members' must be an object")
    unknown = sorted(
        name
        for name, entry in members.items()
        if not isinstance(entry, dict) or entry.get("disposition") not in DISPOSITIONS
    )
    if unknown:
        raise GateError(
            f"contract {path}: members with a missing/unknown disposition: {unknown}"
        )
    return data


def dump_contract(path: Path, contract: dict) -> None:
    """Write the contract back as stable, human-diffable JSON.

    Args:
        path: Destination path.
        contract: Contract mapping to serialise.
    """
    text = json.dumps(contract, indent=2, sort_keys=False, ensure_ascii=False)
    path.write_text(text + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# vLLM source resolution
# ---------------------------------------------------------------------------


def _read_url(url: str) -> str:
    """Fetch a text document over HTTP(S).

    Args:
        url: Absolute URL.

    Returns:
        The decoded body.

    Raises:
        GateError: If the request fails or the body is not decodable UTF-8.
    """
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = response.read()
    except Exception as exc:  # urllib raises a wide family of errors
        raise GateError(f"cannot download {url}: {exc}") from exc
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise GateError(f"downloaded {url} is not UTF-8: {exc}") from exc


def _installed_vllm_root() -> str | None:
    """Return the directory containing the installed ``vllm`` package."""
    try:
        # Third Party
        import vllm  # noqa: F401  (import is the probe)
    except Exception:
        return None
    package_file = getattr(vllm, "__file__", None)
    if not package_file:
        return None
    return os.path.dirname(os.path.dirname(os.path.abspath(package_file)))


def resolve_spec_source(
    spec_file: str | None, vllm_root: str | None, contract: dict
) -> tuple[str, str]:
    """Locate the vLLM spec module and return ``(label, source)``.

    Resolution order: ``--vllm-spec-file`` (path or URL), ``--vllm-root``,
    ``$V41_VLLM_SPEC_FILE``, ``$VLLM_SRC``, the installed ``vllm`` package,
    then the ``pinned_vllm.spec_url`` recorded in the contract.

    Args:
        spec_file: Explicit spec file path or URL, if the user passed one.
        vllm_root: Explicit vLLM checkout root, if the user passed one.
        contract: Parsed contract, used for the pinned URL fallback.

    Returns:
        A ``(label, source)`` pair; ``label`` is what gets printed and recorded.

    Raises:
        GateError: If no source can be resolved or the resolved file is missing.
    """
    candidate = spec_file or os.environ.get("V41_VLLM_SPEC_FILE")
    if candidate:
        if candidate.startswith(("http://", "https://")):
            return candidate, _read_url(candidate)
        path = Path(candidate)
        if not path.is_file():
            raise GateError(f"spec file not found: {path}")
        return str(path), path.read_text(encoding="utf-8")

    root = vllm_root or os.environ.get("VLLM_SRC") or _installed_vllm_root()
    if root:
        path = Path(root) / DEFAULT_SPEC_MODULE
        if not path.is_file():
            raise GateError(
                f"{path} not found; {DEFAULT_SPEC_MODULE} may have moved in this "
                "vLLM revision -- that itself is drift worth investigating"
            )
        return str(path), path.read_text(encoding="utf-8")

    pinned = contract.get("pinned_vllm", {})
    url = pinned.get("spec_url")
    if not url:
        raise GateError(
            "no vLLM source available: pass --vllm-spec-file/--vllm-root, set "
            "V41_VLLM_SPEC_FILE/VLLM_SRC, install vllm, or add "
            "pinned_vllm.spec_url to the contract"
        )
    return url, _read_url(url)


# ---------------------------------------------------------------------------
# vLLM spec extraction
# ---------------------------------------------------------------------------


def _direct_bases(node: ast.ClassDef) -> list[str]:
    """Return the directly named base classes of ``node`` (attribute bases skipped)."""
    names: list[str] = []
    for base in node.bases:
        if isinstance(base, ast.Name):
            names.append(base.id)
        elif isinstance(base, ast.Attribute):
            names.append(base.attr)
    return names


def _declared_members(node: ast.ClassDef) -> list[tuple[str, str]]:
    """Return ``(kind, name)`` for every field and property declared on ``node``.

    Annotated assignments are dataclass fields; functions decorated with
    ``@property`` are properties.  Everything else (methods, nested classes,
    ``ClassVar`` members) is intentionally ignored.
    """
    declared: list[tuple[str, str]] = []
    for statement in node.body:
        if isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target, ast.Name
        ):
            declared.append((FIELD, statement.target.id))
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
            isinstance(decorator, ast.Name) and decorator.id == "property"
            for decorator in statement.decorator_list
        ):
            declared.append((PROPERTY, statement.name))
    return declared


def _mro(name: str, bases_by_class: dict[str, list[str]]) -> list[str]:
    """Return the single-inheritance MRO chain starting at ``name``.

    vLLM's KV-cache specs use single inheritance, so following the first base
    is sufficient and avoids importing vLLM to use ``__mro__``.
    """
    chain: list[str] = []
    current: str | None = name
    while current and current not in chain:
        chain.append(current)
        bases = bases_by_class.get(current, [])
        current = bases[0] if bases else None
    return chain


def _prefer(candidate: Member, current: Member) -> bool:
    """Return whether ``candidate`` should replace ``current`` in the universe.

    A real dataclass field wins over a property of the same name (a derived
    class may re-declare a base property as a field, as ``AttentionSpec`` does
    for ``tokens_per_state``); otherwise the first-seen entry wins.
    """
    return candidate.kind == FIELD and current.kind == PROPERTY


def extract_universe(source: str, tracked_classes: Sequence[str]) -> dict[str, Member]:
    """Extract the effective member universe of the tracked spec classes.

    For each tracked class the MRO is walked from the most-derived class
    upwards, so a member is attributed to the class that actually declares the
    version the derived class sees.

    Args:
        source: Contents of ``vllm/v1/kv_cache_interface.py``.
        tracked_classes: Class names from the contract.

    Returns:
        Mapping of member name to :class:`Member`.

    Raises:
        GateError: If the source does not parse, or a tracked class is absent.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise GateError(f"cannot parse the vLLM spec module: {exc}") from exc

    classes: dict[str, ast.ClassDef] = {}
    bases_by_class: dict[str, list[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            classes[node.name] = node
            bases_by_class[node.name] = _direct_bases(node)

    missing = [name for name in tracked_classes if name not in classes]
    if missing:
        raise GateError(
            "tracked class(es) not found in the vLLM spec module: "
            f"{missing} -- either vLLM renamed them or the contract is stale"
        )

    universe: dict[str, Member] = {}
    for tracked in tracked_classes:
        per_class: dict[str, Member] = {}
        for owner in _mro(tracked, bases_by_class):
            class_node = classes.get(owner)
            if class_node is None:
                continue
            for kind, name in _declared_members(class_node):
                if name not in per_class:
                    per_class[name] = Member(name=name, kind=kind, owner=owner)
        for name, member in per_class.items():
            current = universe.get(name)
            if current is None or _prefer(member, current):
                universe[name] = member
    return universe


# ---------------------------------------------------------------------------
# LMCache coverage scan
# ---------------------------------------------------------------------------


def iter_python_sources(root: Path) -> Iterator[tuple[str, str]]:
    """Yield ``(relative_path, text)`` for every ``.py`` file under ``root``.

    ``__pycache__`` directories are skipped.  Unreadable files are skipped
    rather than aborting: the gate only needs a positive reference, so one
    unreadable file must not become a false failure.

    Args:
        root: Directory to walk (``<repo>/lmcache``).

    Yields:
        Pairs of a repo-relative path and the file's text.
    """
    for directory, subdirs, filenames in os.walk(root):
        subdirs[:] = [name for name in subdirs if name != "__pycache__"]
        for filename in sorted(filenames):
            if not filename.endswith(".py"):
                continue
            path = Path(directory) / filename
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            yield str(path), text


def member_pattern(name: str) -> re.Pattern[str]:
    """Return a word-boundary regex matching the identifier ``name``.

    ``_`` counts as a word character, so ``block_size`` does not match inside
    ``storage_block_size``.  This is the fix for ``check_v41_gap.sh``'s
    ``grep -F`` substring matching, which reported ``head_size_v`` as covered
    because of the unrelated ``_get_head_size_view`` helper.
    """
    return re.compile(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])")


def _line_of(text: str, offset: int) -> int:
    """Return the 1-based line number of a character ``offset`` in ``text``."""
    return text.count("\n", 0, offset) + 1


def scan_coverage(
    root: Path, names: Iterable[str], max_refs: int = 3
) -> dict[str, list[str]]:
    """Find the first few ``path:line`` references for each name under ``root``.

    Comment-only lines are skipped so that a stale ``# TODO: read X`` does not
    count as coverage.  The scan stops early for a name once it is found, which
    keeps the whole gate around a second on the LMCache tree.

    Args:
        root: ``lmcache`` package directory.
        names: Member names to look for.
        max_refs: Maximum number of references to record per name.

    Returns:
        Mapping of name to reference list (empty when the name is blind).
    """
    patterns = [(name, member_pattern(name)) for name in sorted(set(names))]
    references: dict[str, list[str]] = {name: [] for name, _ in patterns}

    pending = [name for name, _ in patterns]

    if not pending:
        return references

    for path, text in iter_python_sources(root):
        still_pending: list[str] = []
        for name in pending:
            pattern = member_pattern(name)
            for match in pattern.finditer(text):
                line_start = text.rfind("\n", 0, match.start()) + 1
                line_end = text.find("\n", match.start())
                line = text[line_start : line_end if line_end != -1 else len(text)]
                if line.lstrip().startswith("#"):
                    continue
                references[name].append(f"{path}:{_line_of(text, match.start())}")
                break
            if len(references[name]) < max_refs:
                still_pending.append(name)
        pending = still_pending
        if not pending:
            break
    return references


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


def check_contract_diff(universe: dict[str, Member], contract: dict) -> list[Finding]:
    """Diff the live vLLM universe against the reviewed contract membership.

    Args:
        universe: Members extracted from the vLLM spec module.
        contract: Parsed contract.

    Returns:
        Findings for added, removed, kind-changed and owner-moved members.
    """
    findings: list[Finding] = []
    contract_members = contract["members"]

    for name in sorted(set(universe) - set(contract_members)):
        member = universe[name]
        findings.append(
            Finding(
                "fail",
                "UNREVIEWED_MEMBER",
                name,
                f"vLLM's {member.owner} declares a {member.kind} that the "
                "contract does not know about; classify it (required/derived/"
                "known_gap/not_applicable_v41) and re-run with --update-contract",
            )
        )

    for name in sorted(set(contract_members) - set(universe)):
        findings.append(
            Finding(
                "fail",
                "REMOVED_MEMBER",
                name,
                "declared in the contract but absent from vLLM -- a removed or "
                "renamed field is exactly the compress_ratio->tokens_per_state "
                "failure mode; update LMCache and the contract",
            )
        )

    for name in sorted(set(universe) & set(contract_members)):
        member = universe[name]
        entry = contract_members[name]
        if entry.get("kind") and entry["kind"] != member.kind:
            findings.append(
                Finding(
                    "fail",
                    "KIND_CHANGED",
                    name,
                    f"contract says {entry['kind']}, vLLM now has a "
                    f"{member.kind} (a field/property flip changes how LMCache "
                    "must read it)",
                )
            )
        if entry.get("owner") and entry["owner"] != member.owner:
            findings.append(
                Finding(
                    "warn",
                    "OWNER_MOVED",
                    name,
                    f"declared on {entry['owner']} in the contract, now on "
                    f"{member.owner}",
                )
            )
    return findings


def check_coverage(
    universe: dict[str, Member], contract: dict, references: dict[str, list[str]]
) -> list[Finding]:
    """Check that every contract member is either referenced or acknowledged.

    Args:
        universe: Members extracted from the vLLM spec module.
        contract: Parsed contract.
        references: Coverage scan output.

    Returns:
        One finding per contract member that vLLM still declares.
    """
    findings: list[Finding] = []
    for name in sorted(contract["members"]):
        if name not in universe:
            continue  # already reported as REMOVED_MEMBER
        entry = contract["members"][name]
        disposition = entry["disposition"]
        refs = references.get(name, [])
        where = refs[0] if refs else "no reference"

        if disposition == UNREVIEWED:
            findings.append(
                Finding(
                    "fail",
                    "UNREVIEWED_MEMBER",
                    name,
                    "disposition is 'unreviewed'; a human must classify it "
                    "before this gate can pass",
                )
            )
            continue
        if disposition == REQUIRED:
            if refs:
                findings.append(
                    Finding("info", "COVERED", name, f"referenced, e.g. {where}")
                )
            else:
                findings.append(
                    Finding(
                        "fail",
                        "BLIND_REQUIRED_MEMBER",
                        name,
                        "contract says LMCache must read this, but lmcache/**/*.py "
                        "never names it -- LMCache is silently reading a default",
                    )
                )
            continue

        reason = entry.get("reason", "")
        if refs:
            findings.append(
                Finding(
                    "info",
                    f"COVERED_{disposition.upper()}",
                    name,
                    f"referenced, e.g. {where}",
                )
            )
        else:
            findings.append(
                Finding(
                    "warn",
                    f"BLIND_{disposition.upper()}",
                    name,
                    f"not referenced by name; reason: {reason}",
                )
            )
    return findings


def check_provenance(
    label: str, source: str, contract: dict, strict_pin: bool
) -> list[Finding]:
    """Compare the loaded spec module against the pinned baseline hash.

    Args:
        label: Source label (path or URL) for the message.
        source: Contents of the loaded spec module.
        contract: Parsed contract.
        strict_pin: When true, a hash mismatch is a failure instead of a warning.

    Returns:
        Zero or one provenance finding.
    """
    pinned = contract.get("pinned_vllm", {})
    expected = pinned.get("spec_module_sha256")
    if not expected:
        return []
    actual = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if actual == expected:
        return [
            Finding(
                "info",
                "PIN_MATCH",
                "-",
                f"spec module matches the pinned {pinned.get('commit', '?')[:12]}",
            )
        ]
    level = "fail" if strict_pin else "warn"
    code = "PIN_MISMATCH_STRICT" if strict_pin else "PIN_MISMATCH"
    message = (
        f"{label} is not the pinned vLLM spec module "
        f"(expected sha256 {expected[:16]}..., got {actual[:16]}...). "
    )
    if strict_pin:
        message += (
            "This job is supposed to run against the pin; the pinned commit is "
            "wrong, force-pushed, or the fetch fell through to another revision."
        )
    else:
        message += (
            "Expected after a vLLM bump: re-review the field reference in "
            "docs/02-v41-spec-field-reference.md, run --update-contract, "
            "classify any new members, then re-pin."
        )
    return [Finding(level, code, "-", message)]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_report(findings: Sequence[Finding], label: str, contract_path: Path) -> None:
    """Print the human-readable gate report.

    Args:
        findings: All findings, in any order.
        label: The vLLM source label.
        contract_path: Contract path, for the header.
    """
    print("=" * 78)
    print(" DeepSeek-V4.1 x vLLM KV-cache-spec contract gate")
    print(f"   vLLM spec : {label}")
    print(f"   contract  : {contract_path}")
    print("=" * 78)

    fails = [f for f in findings if f.level == "fail"]
    warns = [f for f in findings if f.level == "warn"]
    infos = [f for f in findings if f.level == "info"]

    for level, group in (("FAIL", fails), ("WARN", warns), ("INFO", infos)):
        if not group:
            continue
        print(f"\n---- {level} ({len(group)}) ----")
        for finding in sorted(group, key=lambda f: (f.code, f.member)):
            print(f"  [{finding.code}] {finding.member}")
            print(f"      {finding.message}")

    print("\n" + "=" * 78)
    if fails:
        print(f" RESULT: FAIL -- {len(fails)} blocking finding(s).")
    else:
        print(f" RESULT: PASS -- 0 blocking findings, {len(warns)} warning(s).")
    print("=" * 78)


def write_json_report(path: Path, findings: Sequence[Finding], label: str) -> None:
    """Write the machine-readable report used by CI artifacts.

    Args:
        path: Destination path.
        findings: All findings.
        label: The vLLM source label.
    """
    payload = {
        "vllm_spec_source": label,
        "result": "fail" if any(f.level == "fail" for f in findings) else "pass",
        "findings": [
            {
                "level": finding.level,
                "code": finding.code,
                "member": finding.member,
                "message": finding.message,
            }
            for finding in findings
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Contract refresh
# ---------------------------------------------------------------------------


def refresh_contract(contract: dict, universe: dict[str, Member]) -> tuple[dict, int]:
    """Re-derive the contract's member inventory, preserving reviewed entries.

    Existing entries keep their disposition and reason; new members are added as
    ``unreviewed`` (which the gate then fails on until a human classifies them);
    members that vLLM dropped are kept as ``REMOVED_MEMBER`` evidence so the
    removal stays visible in the diff.

    Args:
        contract: Parsed contract, mutated in place.
        universe: Members extracted from the live vLLM spec module.

    Returns:
        A ``(contract, added_count)`` pair.
    """
    members = contract["members"]
    added = 0
    for name in sorted(universe):
        member = universe[name]
        entry = members.get(name)
        if isinstance(entry, dict):
            entry["kind"] = member.kind
            entry["owner"] = member.owner
            continue
        members[name] = {
            "kind": member.kind,
            "owner": member.owner,
            "disposition": UNREVIEWED,
            "reason": "new in this vLLM revision -- classify before merging",
        }
        added += 1
    return contract, added


def update_pin_sha(contract: dict, source: str) -> None:
    """Refresh the pinned spec-module hash from the loaded source.

    Args:
        contract: Parsed contract, mutated in place.
        source: Contents of the loaded spec module.
    """
    pinned = contract.setdefault("pinned_vllm", {})
    pinned["spec_module_sha256"] = hashlib.sha256(source.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="check_v41_spec_contract.py",
        description=(
            "Assert LMCache still reads every vLLM KV-cache-spec field the "
            "DeepSeek-V4.1 adaptation depends on. Offline, no GPU."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--vllm-spec-file",
        default=None,
        help="vLLM kv_cache_interface.py path or URL (default: pinned URL from "
        "the contract; $V41_VLLM_SPEC_FILE also works)",
    )
    parser.add_argument(
        "--vllm-root",
        default=None,
        help="vLLM checkout root containing vllm/v1/kv_cache_interface.py "
        "($VLLM_SRC also works)",
    )
    parser.add_argument(
        "--lmcache-root",
        default=None,
        help="LMCache repo root; coverage scans <root>/lmcache (default: repo root)",
    )
    parser.add_argument(
        "--contract",
        default=None,
        help=f"contract JSON (default: <repo>/{DEFAULT_CONTRACT})",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="also write a machine-readable report to this path",
    )
    parser.add_argument(
        "--strict-pin",
        action="store_true",
        help="fail when the loaded spec module's sha256 differs from the pin",
    )
    parser.add_argument(
        "--update-contract",
        action="store_true",
        help="re-derive the inventory from the loaded source and rewrite the "
        "contract (new members become 'unreviewed'; the gate still runs after)",
    )
    parser.add_argument(
        "--skip-coverage",
        action="store_true",
        help="only diff the contract, do not scan LMCache for references",
    )
    return parser


def _repo_root() -> Path:
    """Return the repository root from this script's location."""
    return Path(__file__).resolve().parent.parent


def main(argv: Sequence[str] | None = None) -> int:
    """Run the gate.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code: 0 pass, 1 gate failure, 2 setup error.
    """
    args = build_parser().parse_args(argv)
    repo_root = _repo_root()
    contract_path = (
        Path(args.contract) if args.contract else repo_root / DEFAULT_CONTRACT
    )
    lmcache_root = Path(args.lmcache_root) if args.lmcache_root else repo_root
    package_dir = lmcache_root / "lmcache"

    try:
        contract = load_contract(contract_path)
        label, source = resolve_spec_source(
            args.vllm_spec_file, args.vllm_root, contract
        )
        tracked = contract["tracked_classes"]
        universe = extract_universe(source, tracked)
    except GateError as exc:
        print(f"SETUP ERROR: {exc}", file=sys.stderr)
        return EXIT_SETUP_ERROR

    if args.update_contract:
        contract, added = refresh_contract(contract, universe)
        update_pin_sha(contract, source)
        dump_contract(contract_path, contract)
        print(
            f"[contract] refreshed {contract_path}: {len(universe)} members, "
            f"{added} new 'unreviewed' entr{'y' if added == 1 else 'ies'}"
        )

    findings: list[Finding] = []
    findings.extend(check_provenance(label, source, contract, args.strict_pin))
    findings.extend(check_contract_diff(universe, contract))

    if args.skip_coverage:
        findings.append(
            Finding("info", "COVERAGE_SKIPPED", "-", "--skip-coverage was passed")
        )
    else:
        if not package_dir.is_dir():
            print(
                f"SETUP ERROR: LMCache package not found at {package_dir}",
                file=sys.stderr,
            )
            return EXIT_SETUP_ERROR
        references = scan_coverage(package_dir, contract["members"])
        findings.extend(check_coverage(universe, contract, references))

    print_report(findings, label, contract_path)
    if args.json_out:
        write_json_report(Path(args.json_out), findings, label)

    return EXIT_GATE_FAILED if any(f.level == "fail" for f in findings) else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
