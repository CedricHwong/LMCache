# SPDX-License-Identifier: Apache-2.0
"""CPU-only lifecycle regressions for the MP abort/retrieve guards.

Covers the LOADING state and the abort completion guards of the three-file
2026-09 abort hotfix (lmcache_mp_metadata, lmcache_mp_connector and
vllm_multi_process_adapter). The patched methods are extracted from the real
sources with the ast module and exec'd in isolation, so no torch, vLLM or GPU
is required -- the connector modules import those at module scope.

End-to-end live coverage stays with the MP connector tests in
tests/v1/test_vllm_mp_adapter.py and tests/v1/test_lmcache_mp_connector.py.
"""

# Standard
from pathlib import Path
import ast
import enum
import logging
import types
import unittest

_VLLM_DIR = Path(__file__).resolve().parents[2] / "lmcache" / "integration" / "vllm"


def _extract(filename, names, extra=None):
    """Exec the named top-level functions of *filename* in a bare namespace."""
    tree = ast.parse((_VLLM_DIR / filename).read_text())
    nodes = [
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    assert len(nodes) == len(names), (names, [n.name for n in nodes])
    for node in nodes:
        node.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            )
        ]
        + nodes,
        type_ignores=[],
    )
    namespace = {"logger": logging.getLogger("test"), **(extra or {})}
    exec(compile(ast.fix_missing_locations(module), filename, "exec"), namespace)
    return {name: namespace[name] for name in names}


class _Future:
    def __init__(self, ready=False, result=True):
        self.ready = ready
        self.value = result

    def query(self):
        return self.ready

    def result(self, timeout):
        return self.value


class _State(enum.Enum):
    WAITING_FOR_LOAD = 1
    LOADING = 2
    READY = 3


_Worker = type(
    "Worker",
    (),
    _extract(
        "vllm_multi_process_adapter.py",
        {
            "get_finished",
            "_process_finished_stores",
            "_update_and_get_finished_store",
            "mark_aborted_retrieves",
        },
    ),
)


def _worker():
    w = _Worker()
    w.dispatcher = None
    w.is_healthy = True
    for key in ("store_futures", "retrieve_futures", "store_events", "retrieve_events"):
        setattr(w, key, {})
    for key in (
        "error_block_ids",
        "_dropped_retrieves",
        "_aborted_retrieves",
        "_returned_finished",
        "finished_stores",
        "previously_finished",
    ):
        setattr(w, key, set())
    w.request_telemetry = types.SimpleNamespace(
        on_request_store_finished=lambda **kw: None
    )
    w.model_name, w.world_size, w.worker_id = "model", 8, 0
    return w


class TestAbortLifecycle(unittest.TestCase):
    def test_abort_engine_before_receive(self):
        w = _worker()
        f = _Future()
        w.retrieve_futures["r"] = (f, [1])
        w.mark_aborted_retrieves({"r"})
        self.assertEqual(w.get_finished({"r"}), (set(), set()))
        f.ready = True
        self.assertEqual(w.get_finished(set()), (set(), {"r"}))
        self.assertEqual(w.get_finished({"r"}), (set(), set()))

    def test_abort_receive_before_engine(self):
        w = _worker()
        w.retrieve_futures["r"] = (_Future(True), [1])
        w.mark_aborted_retrieves({"r"})
        self.assertEqual(w.get_finished(set()), (set(), {"r"}))
        self.assertEqual(w._aborted_retrieves, {"r"})
        self.assertEqual(w.get_finished({"r"}), (set(), set()))
        self.assertEqual(w.get_finished({"r"}), (set(), set()))

    def test_same_step(self):
        w = _worker()
        w.retrieve_futures["r"] = (_Future(True), [1])
        w.mark_aborted_retrieves({"r"})
        self.assertEqual(w.get_finished({"r"}), (set(), {"r"}))

    def test_normal_retrieve_then_store(self):
        w = _worker()
        w.retrieve_futures["r"] = (_Future(True), [1])
        self.assertEqual(w.get_finished(set()), (set(), {"r"}))
        self.assertEqual(w.get_finished({"r"}), ({"r"}, set()))
        self.assertEqual(w.get_finished({"r"}), (set(), set()))

    def test_unhealthy_abort(self):
        w = _worker()
        w.is_healthy = False
        w.retrieve_futures["r"] = (_Future(), [7])
        w.mark_aborted_retrieves({"r"})
        self.assertEqual(w.get_finished({"r"}), (set(), {"r"}))
        self.assertEqual(w.error_block_ids, {7})
        self.assertEqual(w.get_finished({"r"}), (set(), set()))

    def test_dropped_once(self):
        w = _worker()
        w._dropped_retrieves.add("r")
        self.assertEqual(w.get_finished(set()), (set(), {"r"}))
        self.assertEqual(w.get_finished(set()), (set(), set()))

    def test_pending_store(self):
        w = _worker()
        f = _Future()
        w.store_futures["s"] = f
        self.assertEqual(w.get_finished({"s"}), (set(), set()))
        f.ready = True
        self.assertEqual(w.get_finished(set()), ({"s"}, set()))

    def test_scheduler_lifecycle(self):
        funcs = _extract(
            "lmcache_mp_connector.py",
            {"request_finished", "update_connector_output"},
            {"LMCacheMPRequestState": _State},
        )
        connector = type("Connector", (), funcs)()
        connector.lazy_offload = False
        connector._aborted_retrieve_req_ids = set()
        connector.request_trackers = {"r": types.SimpleNamespace(state=_State.LOADING)}
        connector._cleanup_request_tracker = lambda r: connector.request_trackers.pop(r)
        connector.scheduler_adapter = types.SimpleNamespace(
            cleanup_lookup_result=lambda r: None, end_session=lambda r: None
        )
        req = types.SimpleNamespace(request_id="r")
        self.assertEqual(connector.request_finished(req, []), (False, None))
        self.assertEqual(connector._aborted_retrieve_req_ids, {"r"})
        connector.request_trackers["n"] = types.SimpleNamespace(state=_State.LOADING)
        connector.update_connector_output(types.SimpleNamespace(finished_recving={"n"}))
        self.assertEqual(connector.request_trackers["n"].state, _State.READY)
        self.assertEqual(
            connector.request_finished(types.SimpleNamespace(request_id="n"), []),
            (True, None),
        )


if __name__ == "__main__":
    unittest.main()
