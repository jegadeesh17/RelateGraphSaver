"""Integration tests for opening a store read-only for inspection (ADR-0015).

Covers AC-81 to AC-87: a read-only open leaves the file byte-identical, refuses writes,
reads the same data as a writable open, and rejects invalid combinations at construction.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import re
import shutil
import threading
from importlib.metadata import version
from pathlib import Path
from typing import Any, TypedDict

import pytest
import relate
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.graph import END, START, StateGraph

from langgraph_checkpoint_relate import (
    RelateCheckpointError,
    RelateCheckpointNotFoundError,
    RelateConnectionError,
    RelateReadOnlyError,
    RelateSaver,
)
from langgraph_checkpoint_relate.proxy import BatchCommand, DbCommand, RelateThreadProxy

THREAD_ID = "ro-thread"
CONFIG: RunnableConfig = {"configurable": {"thread_id": THREAD_ID, "checkpoint_ns": ""}}


class LoopState(TypedDict):
    counter: int


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _listing(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir())


def _checkpoint(checkpoint_id: str, value: str) -> Any:
    cp: dict[str, Any] = dict(empty_checkpoint())
    cp["id"] = checkpoint_id
    cp["channel_values"] = {"value": value}
    cp["channel_versions"] = {"value": "1.0"}
    return cp


@pytest.fixture
def store(tmp_path: Path) -> Path:
    """A closed file store with two checkpoints, a loop-free lineage and one failed tool call."""
    db = tmp_path / "store" / "ro.rl"
    db.parent.mkdir()
    with RelateSaver(str(db)) as saver:
        first = saver.put(CONFIG, _checkpoint("ro-cp-1", "one"), {"step": 1, "source": "loop"}, {})
        second = saver.put(
            first, _checkpoint("ro-cp-2", "two"), {"step": 2, "source": "loop", "parents": {}}, {}
        )
        ai = AIMessage(
            content="", tool_calls=[{"id": "call-ro-1", "name": "fetch_source", "args": {}}]
        )
        saver.put_writes(second, [("messages", [ai])], task_id="task-ai")
        tool_msg = ToolMessage(content="Error: down", tool_call_id="call-ro-1", status="error")
        saver.put_writes(second, [("messages", [tool_msg])], task_id="task-tool")
    return db


class _RecordingConnection:
    """Fake driver connection that records every call, to prove nothing reaches the driver."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def execute(self, cypher: str, params: Any = None) -> Any:
        self.calls.append(f"execute:{cypher}")
        return self

    def fetchall(self) -> list[Any]:
        return []

    def commit(self) -> None:
        self.calls.append("commit")

    def rollback(self) -> None:
        self.calls.append("rollback")

    def flush(self) -> None:
        self.calls.append("flush")

    def close(self) -> None:
        self.calls.append("close")


class TestReadOnlyOpen:
    def test_open_and_close_leave_file_and_directory_unchanged_ac_81(self, store: Path) -> None:
        """AC-81: opening and closing with no queries changes nothing and adds no sidecar."""
        before_hash, before_listing = _sha256(store), _listing(store.parent)

        saver = RelateSaver(str(store), read_only=True)
        assert saver.read_only is True
        saver.close()

        assert _sha256(store) == before_hash
        assert _listing(store.parent) == before_listing

    def test_reads_leave_file_unchanged_ac_81(self, store: Path) -> None:
        """AC-81: reads and a no-op flush also leave the file and directory unchanged."""
        before_hash, before_listing = _sha256(store), _listing(store.parent)

        with RelateSaver(str(store), read_only=True) as saver:
            assert saver.get_tuple(CONFIG) is not None
            assert len(list(saver.list(CONFIG))) == 2
            saver.flush()

        assert _sha256(store) == before_hash
        assert _listing(store.parent) == before_listing

    def test_missing_path_raises_connection_error_and_creates_nothing_ac_85(
        self, tmp_path: Path
    ) -> None:
        """AC-85: a missing file or missing parent directory is not created."""
        missing_file = tmp_path / "absent.rl"
        with pytest.raises(RelateConnectionError):
            RelateSaver(str(missing_file), read_only=True)
        assert not missing_file.exists()

        missing_dir_file = tmp_path / "no_such_dir" / "absent.rl"
        with pytest.raises(RelateConnectionError):
            RelateSaver(str(missing_dir_file), read_only=True)
        assert not missing_dir_file.parent.exists()
        assert list(tmp_path.iterdir()) == []

    def test_memory_store_rejected_ac_86(self) -> None:
        """AC-86: read_only with ':memory:' raises ValueError (default and explicit path)."""
        with pytest.raises(ValueError):
            RelateSaver(read_only=True)
        with pytest.raises(ValueError):
            RelateSaver(":memory:", read_only=True)
        with pytest.raises(ValueError):
            RelateThreadProxy(":memory:", read_only=True)

    def test_writable_proxy_rejected_ac_86(self, store: Path) -> None:
        """AC-86: read_only=True with a passed writable proxy raises ValueError."""
        proxy = RelateThreadProxy(str(store))
        try:
            with pytest.raises(ValueError):
                RelateSaver(proxy, read_only=True)
            with pytest.raises(ValueError):
                RelateSaver(proxy=proxy, read_only=True)
        finally:
            proxy.close()

    def test_passed_read_only_proxy_sets_saver_state_ac_86(self, store: Path) -> None:
        """A saver built on a read-only proxy is read-only whether or not it says so."""
        proxy = RelateThreadProxy(str(store), read_only=True)
        try:
            assert RelateSaver(proxy).read_only is True
            assert RelateSaver(proxy=proxy).read_only is True
            assert RelateSaver(proxy, read_only=True).read_only is True
        finally:
            proxy.close()

    def test_writable_default_is_not_read_only_ac_87(self, tmp_path: Path) -> None:
        """AC-87: a saver without read_only is writable."""
        with RelateSaver(str(tmp_path / "w.rl")) as saver:
            assert saver.read_only is False


class TestReadOnlyRefusesWrites:
    def test_put_and_put_writes_raise_and_file_unchanged_ac_82(self, store: Path) -> None:
        """AC-82: put and put_writes raise RelateReadOnlyError; nothing reaches the file."""
        before_hash = _sha256(store)
        with RelateSaver(str(store), read_only=True) as saver:
            with pytest.raises(RelateReadOnlyError):
                saver.put(CONFIG, _checkpoint("ro-cp-3", "three"), {"step": 3}, {})
            with pytest.raises(RelateReadOnlyError):
                saver.put_writes(CONFIG, [("messages", ["x"])], task_id="t")
            with pytest.raises(RelateReadOnlyError):
                saver.put_writes(CONFIG, [], task_id="t")
            assert saver.get_tuple(CONFIG) is not None
        assert _sha256(store) == before_hash

    def test_async_writes_raise_from_the_await_ac_82(self, store: Path) -> None:
        """AC-82: aput and aput_writes raise RelateReadOnlyError from the await."""
        before_hash = _sha256(store)

        async def run() -> None:
            with RelateSaver(str(store), read_only=True) as saver:
                with pytest.raises(RelateReadOnlyError):
                    await saver.aput(CONFIG, _checkpoint("ro-cp-3", "three"), {"step": 3}, {})
                with pytest.raises(RelateReadOnlyError):
                    await saver.aput_writes(CONFIG, [("messages", ["x"])], task_id="t")
                assert await saver.aget_tuple(CONFIG) is not None
                assert len([t async for t in saver.alist(CONFIG)]) == 2

        asyncio.run(run())
        assert _sha256(store) == before_hash

    def test_read_only_error_is_a_checkpoint_error_ac_82(self) -> None:
        """RelateReadOnlyError is caught by existing RelateCheckpointError handlers."""
        assert issubclass(RelateReadOnlyError, RelateCheckpointError)

    def test_proxy_write_methods_raise_and_file_unchanged_ac_82(self, store: Path) -> None:
        """AC-82: every proxy write entry point raises before queuing."""
        before_hash = _sha256(store)
        proxy = RelateThreadProxy(str(store), read_only=True)
        try:
            with pytest.raises(RelateReadOnlyError):
                proxy.execute_write("CREATE (n:X {a: 1})")
            with pytest.raises(RelateReadOnlyError):
                proxy.execute("CREATE (n:X {a: 1})")
            with pytest.raises(RelateReadOnlyError):
                proxy.execute_batch([("CREATE (n:X {a: 1})", {})])
            with pytest.raises(RelateReadOnlyError):
                proxy.transaction([("CREATE (n:X {a: 1})", {})])
            proxy.flush()
            assert proxy.execute_read("MATCH (c:Checkpoint) RETURN count(c) AS n")[0]["n"] == 2
        finally:
            proxy.close()
        assert _sha256(store) == before_hash

    def test_queued_write_forms_fail_without_reaching_the_driver_ac_82(self) -> None:
        """AC-82: write-flagged items queued directly are failed and never executed."""
        conn = _RecordingConnection()
        proxy = RelateThreadProxy(
            "fake.rl", connection_factory=lambda _path: conn, read_only=True, auto_init_schema=True
        )
        try:
            cmd_future: concurrent.futures.Future[Any] = concurrent.futures.Future()
            proxy._queue.put(DbCommand("CREATE (n:X)", {}, True, cmd_future))
            batch_future: concurrent.futures.Future[Any] = concurrent.futures.Future()
            proxy._queue.put(BatchCommand([("CREATE (n:X)", {})], batch_future))
            dict_exec: concurrent.futures.Future[Any] = concurrent.futures.Future()
            proxy._queue.put(
                {"type": "execute", "cypher": "CREATE (n:X)", "is_write": True, "future": dict_exec}
            )
            dict_default: concurrent.futures.Future[Any] = concurrent.futures.Future()
            proxy._queue.put({"type": "execute", "cypher": "CREATE (n:X)", "future": dict_default})
            dict_tx: concurrent.futures.Future[Any] = concurrent.futures.Future()
            proxy._queue.put(
                {"type": "transaction", "statements": [("CREATE (n:X)", {})], "future": dict_tx}
            )
            event = threading.Event()
            holder: dict[str, Any] = {}
            proxy._queue.put(("CREATE (n:X)", {}, event, holder))

            for fut in (cmd_future, batch_future, dict_exec, dict_default, dict_tx):
                with pytest.raises(RelateReadOnlyError):
                    fut.result(timeout=5)
            assert event.wait(timeout=5)
            assert isinstance(holder["error"], RelateReadOnlyError)
            assert holder["result"] is None

            # A read still reaches the driver.
            proxy.execute_read("MATCH (n) RETURN n")
        finally:
            proxy.close()

        assert conn.calls == ["execute:MATCH (n) RETURN n", "close"]

    def test_read_only_proxy_never_flushes_or_runs_schema_setup_ac_81(self) -> None:
        """AC-81: no schema statement, no flush on flush() or close()."""
        conn = _RecordingConnection()
        proxy = RelateThreadProxy(
            "fake.rl", connection_factory=lambda _path: conn, read_only=True, wal_enabled=True
        )
        proxy.flush()
        proxy.close()
        assert conn.calls == ["close"]

    def test_custom_factory_is_called_with_the_path_only_ac_86(self) -> None:
        """A connection_factory is still called as factory(path); it must open read-only itself."""
        seen: list[tuple[Any, ...]] = []

        def factory(*args: Any) -> _RecordingConnection:
            seen.append(args)
            return _RecordingConnection()

        proxy = RelateThreadProxy("fake.rl", connection_factory=factory, read_only=True)
        proxy.close()
        assert seen == [("fake.rl",)]


class TestReadOnlyParity:
    def test_reads_equal_a_writable_open_of_a_copy_ac_83(self, store: Path, tmp_path: Path) -> None:
        """AC-83: get_tuple, list, detect_loops, trace_root_cause and fork_in_memory agree."""
        copy = tmp_path / "copy.rl"
        shutil.copy(store, copy)

        with (
            RelateSaver(str(store), read_only=True) as ro,
            RelateSaver(str(copy)) as rw,
        ):
            assert ro.get_tuple(CONFIG) == rw.get_tuple(CONFIG)
            assert list(ro.list(CONFIG)) == list(rw.list(CONFIG))
            assert ro.detect_loops(THREAD_ID, threshold=2) == rw.detect_loops(
                THREAD_ID, threshold=2
            )
            ro_cause = ro.trace_root_cause(THREAD_ID)
            rw_cause = rw.trace_root_cause(THREAD_ID)
            assert ro_cause == rw_cause
            assert ro_cause.root_tool_name == "fetch_source"

            ro_fork = ro.fork_in_memory(THREAD_ID, "ro-cp-2")
            rw_fork = rw.fork_in_memory(THREAD_ID, "ro-cp-2")
            try:
                assert ro_fork.read_only is False
                assert list(ro_fork.list(CONFIG)) == list(rw_fork.list(CONFIG))
                assert ro_fork.trace_root_cause(THREAD_ID) == rw_fork.trace_root_cause(THREAD_ID)
            finally:
                ro_fork.close()
                rw_fork.close()

    def test_detect_loops_on_a_looping_thread_matches_a_writable_open_ac_83(
        self, tmp_path: Path
    ) -> None:
        """AC-83, AC-188: a non-empty loop report is identical read-only and writable (threshold 2)."""

        def bump(state: LoopState) -> dict[str, int]:
            return {"counter": state["counter"] + 1}

        def route(state: LoopState) -> str:
            return "ping" if state["counter"] % 2 == 0 else "pong"

        def done(state: LoopState) -> str:
            return END if state["counter"] >= 6 else route(state)

        workflow = StateGraph(LoopState)
        workflow.add_node("ping", bump)
        workflow.add_node("pong", bump)
        workflow.add_edge(START, "ping")
        workflow.add_conditional_edges("ping", done, {"ping": "ping", "pong": "pong", END: END})
        workflow.add_conditional_edges("pong", done, {"ping": "ping", "pong": "pong", END: END})

        db = tmp_path / "loop.rl"
        thread = "ro-loop-thread"
        config: RunnableConfig = {"configurable": {"thread_id": thread}}
        with RelateSaver(str(db)) as saver:
            workflow.compile(checkpointer=saver).invoke({"counter": 0}, config=config)

        copy = tmp_path / "loop-copy.rl"
        shutil.copy(db, copy)
        with RelateSaver(str(db), read_only=True) as ro, RelateSaver(str(copy)) as rw:
            ro_loops = ro.detect_loops(thread, threshold=2)
            assert ro_loops
            assert ro_loops == rw.detect_loops(thread, threshold=2)

    def test_forensics_and_visualizer_work_on_a_read_only_saver_ac_83(self, store: Path) -> None:
        """AC-83: DAGVisualizer and RelateForensics accept a read-only saver and do not close it."""
        from langgraph_checkpoint_relate import DAGVisualizer, RelateForensics

        before_hash = _sha256(store)
        with RelateSaver(str(store), read_only=True) as saver:
            with DAGVisualizer(saver) as viz:
                assert viz.render_tree(THREAD_ID) is not None
            with RelateForensics(saver) as forensics:
                assert forensics.trace_root_cause(THREAD_ID).root_tool_call_id == "call-ro-1"
            assert saver.get_tuple(CONFIG) is not None
        assert _sha256(store) == before_hash


class TestWritableUnchanged:
    def test_save_close_reopen_round_trip_ac_87(self, tmp_path: Path) -> None:
        """AC-87: a default saver still sets up the schema, saves and persists across reopen."""
        db = tmp_path / "rw.rl"
        with RelateSaver(str(db)) as saver:
            saver.put(CONFIG, _checkpoint("rw-cp-1", "kept"), {"step": 1}, {})
        assert db.stat().st_size > 0

        with RelateSaver(str(db)) as reopened:
            tup = reopened.get_tuple(CONFIG)
            assert tup is not None
            assert tup.checkpoint["channel_values"]["value"] == "kept"
            reopened.put(CONFIG, _checkpoint("rw-cp-2", "more"), {"step": 2}, {})

        with RelateSaver(str(db), read_only=True) as ro:
            assert len(list(ro.list(CONFIG))) == 2

    def test_dependency_floor_is_declared_and_installed_ac_81(self) -> None:
        """The declared floor is at or above 1.0.2, the version read_only was verified on."""
        pyproject = (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text(
            encoding="utf-8"
        )
        match = re.search(r'"relate-db>=(\d+)\.(\d+)\.(\d+)', pyproject)
        assert match is not None
        assert tuple(int(p) for p in match.groups()) >= (1, 0, 2)
        installed = tuple(int(p) for p in version("relate-db").split(".")[:3])
        assert installed >= (1, 0, 2)


class TestReadOnlyEngineBehavior:
    """Behavior of the driver's read-only connection that the design relies on."""

    def test_failed_read_then_rollback_is_harmless_ac_81(self, store: Path) -> None:
        """A failed read triggers the proxy's rollback; later reads work and the file is intact."""
        before_hash = _sha256(store)
        proxy = RelateThreadProxy(str(store), read_only=True)
        try:
            with pytest.raises(RelateCheckpointError):
                proxy.execute_read("THIS IS NOT CYPHER")
            with pytest.raises(RelateCheckpointError):
                proxy.execute_read("MATCH (c:Checkpoint) RETURN c.no_such(")
            rows = proxy.execute_read("MATCH (c:Checkpoint) RETURN count(c) AS n")
            assert rows[0]["n"] == 2
        finally:
            proxy.close()
        assert _sha256(store) == before_hash

    def test_raw_rollback_on_read_only_connection_is_harmless_ac_81(self, store: Path) -> None:
        """Calling rollback directly on a read-only driver connection raises nothing odd."""
        before_hash = _sha256(store)
        conn = relate.connect(database=str(store), read_only=True)
        try:
            with pytest.raises(Exception):  # noqa: B017 - any driver error is acceptable here
                conn.execute("THIS IS NOT CYPHER").fetchall()
            conn.rollback()
            rows = conn.execute("MATCH (c:Checkpoint) RETURN count(c) AS n").fetchall()
            assert rows[0]["n"] == 2
        finally:
            conn.close()
        assert _sha256(store) == before_hash

    def test_old_store_without_labels_or_properties_reads_empty_ac_83(self, tmp_path: Path) -> None:
        """MATCH on labels and properties an older store lacks returns nothing instead of raising."""
        db = tmp_path / "old.rl"
        conn = relate.connect(database=str(db))
        conn.execute("CREATE (n:Unrelated {name: 'x'})")
        conn.commit()
        conn.flush()
        conn.close()
        before_hash = _sha256(db)

        with RelateSaver(str(db), read_only=True) as saver:
            assert saver.get_tuple(CONFIG) is None
            assert list(saver.list(CONFIG)) == []
            # The thread has no checkpoints, so forensics reports "not found" (the same
            # error a writable open gives), not a driver error from the missing labels.
            with pytest.raises(RelateCheckpointNotFoundError):
                saver.detect_loops(THREAD_ID)
            rows = saver.proxy.execute_read(
                "MATCH (t:ToolCall) WHERE t.status = 'error' RETURN t.call_id AS id"
            )
            assert rows == []
            rows = saver.proxy.execute_read(
                "MATCH (c:Checkpoint)-[:INVOKED]->(t:ToolCall) RETURN c.nothing AS n"
            )
            assert rows == []
            with pytest.raises(RelateCheckpointNotFoundError):
                saver.trace_root_cause(THREAD_ID)
        assert _sha256(db) == before_hash

    def test_store_saved_before_tool_calls_reports_no_tool_failure_ac_83(
        self, tmp_path: Path
    ) -> None:
        """A store with checkpoints but no :ToolCall nodes reports no tool failure."""
        db = tmp_path / "pre_tools.rl"
        with RelateSaver(str(db)) as saver:
            saver.put(CONFIG, _checkpoint("old-cp-1", "one"), {"step": 1, "source": "loop"}, {})
        before_hash = _sha256(db)

        with RelateSaver(str(db), read_only=True) as saver:
            report = saver.trace_root_cause(THREAD_ID)
            assert report.root_tool_call_id is None
            assert report.root_tool_name is None
        assert _sha256(db) == before_hash
