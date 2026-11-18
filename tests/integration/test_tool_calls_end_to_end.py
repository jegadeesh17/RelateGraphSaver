"""Cross-feature integration tests for tool call capture, forensics and quality gates.

Covers the wiring between features that the per-module tests exercise only in isolation:
- Static quality gates run exactly as the project defines them (AC-35, AC-36, AC-37, AC-38).
- Tool call capture on the asynchronous path: ainvoke and astream persist ToolCall nodes and
  root-cause forensics names the failing tool (AC-41, AC-43, AC-45, AC-51, AC-52).
- Concurrent async runs that call tools keep every ToolCall node on its own thread (AC-42, AC-53).
- Malformed tool entries written through aput_writes never raise on the async path (AC-47, AC-54).
- fork_in_memory sandboxes a thread that has tool call history (AC-19).
- The benchmark script runs end to end and rejects invalid arguments (AC-56).
"""

# ruff: noqa: E402

from __future__ import annotations

import asyncio
import importlib.util
import subprocess
import sys
from collections.abc import Generator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from examples.tool_agent import (
    OK_URL,
    TOOL_NAME,
    UNREACHABLE_URL,
    create_tool_agent_graph,
    run_tool_agent,
)
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import empty_checkpoint

from langgraph_checkpoint_relate.errors import CheckpointNotFoundError
from langgraph_checkpoint_relate.saver import RelateSaver

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_BENCHMARK_SCRIPT = _PROJECT_ROOT / "benchmarks" / "bench_checkpointers.py"


def _tool_config(thread_id: str, checkpoint_id: str | None = None) -> RunnableConfig:
    configurable: dict[str, Any] = {"thread_id": thread_id, "checkpoint_ns": ""}
    if checkpoint_id is not None:
        configurable["checkpoint_id"] = checkpoint_id
    return {"configurable": configurable}


def _only_call_id(state: dict[str, Any]) -> str:
    """Return the id of the single tool call requested in a run's AI messages."""
    calls = [call for m in state["messages"] if isinstance(m, AIMessage) for call in m.tool_calls]
    assert len(calls) == 1, calls
    return str(calls[0]["id"])


def _tool_call_rows(saver: RelateSaver) -> list[dict[str, Any]]:
    """Return each ToolCall node with the thread of the checkpoint that invoked it."""
    return saver.proxy.execute_read(
        "MATCH (c:Checkpoint)-[:INVOKED]->(tc:ToolCall) "
        "RETURN c.thread_id AS thread_id, c.checkpoint_id AS checkpoint_id, "
        "tc.call_id AS call_id, tc.tool_name AS tool_name, "
        "tc.result_blob AS result_blob, tc.error AS error"
    )


def _count_tool_nodes(saver: RelateSaver) -> int:
    rows = saver.proxy.execute_read("MATCH (tc:ToolCall) RETURN count(tc) AS n")
    return int(rows[0]["n"])


def _run_process(args: list[str], timeout: float = 600.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _output_tail(result: subprocess.CompletedProcess[str], lines: int = 50) -> str:
    combined = (result.stdout + result.stderr).splitlines()
    return "\n".join(combined[-lines:])


def _load_benchmark_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("bench_checkpointers", _BENCHMARK_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def memory_saver() -> Generator[RelateSaver]:
    """A saver backed by a real in-memory Relate database."""
    saver = RelateSaver(db_path=":memory:")
    try:
        yield saver
    finally:
        saver.close()


class TestStaticQualityGates:
    """AC-35 to AC-38: the quality gates pass when run from the repository root."""

    @pytest.mark.slow
    def test_ruff_check_exits_zero_ac_35(self) -> None:
        result = _run_process([sys.executable, "-m", "ruff", "check", "."])
        assert result.returncode == 0, _output_tail(result)

    @pytest.mark.slow
    def test_ruff_format_check_exits_zero_ac_36(self) -> None:
        result = _run_process([sys.executable, "-m", "ruff", "format", "--check", "."])
        assert result.returncode == 0, _output_tail(result)

    @pytest.mark.slow
    def test_mypy_src_and_tests_exits_zero_ac_37(self) -> None:
        result = _run_process([sys.executable, "-m", "mypy", "src", "tests"])
        assert result.returncode == 0, _output_tail(result)

    @pytest.mark.slow
    def test_every_test_module_collects_without_import_errors_ac_38(self) -> None:
        result = _run_process([sys.executable, "-m", "pytest", "--collect-only", "-q", "tests"])
        assert result.returncode == 0, _output_tail(result)


class TestToolCallsOnAsyncPath:
    """Tool calls captured through ainvoke and astream feed forensics unchanged."""

    def test_ainvoke_failing_tool_is_named_by_root_cause_ac_45(
        self, memory_saver: RelateSaver
    ) -> None:
        """AC-41, AC-45, AC-51: aput/aput_writes capture the call and the failure is named."""
        thread_id = "async-fail"
        graph = create_tool_agent_graph(checkpointer=memory_saver)

        state: dict[str, Any] = asyncio.run(
            graph.ainvoke(
                {"messages": [HumanMessage(content=UNREACHABLE_URL)]}, _tool_config(thread_id)
            )
        )
        call_id = _only_call_id(state)
        tool_messages = [m for m in state["messages"] if isinstance(m, ToolMessage)]
        assert [m.status for m in tool_messages] == ["error"]

        rows = _tool_call_rows(memory_saver)
        assert len(rows) == 1
        assert rows[0]["thread_id"] == thread_id
        assert rows[0]["call_id"] == call_id
        assert rows[0]["tool_name"] == TOOL_NAME
        assert "upstream unreachable" in str(rows[0]["error"])

        report = memory_saver.trace_root_cause(thread_id)
        assert report.root_tool_name == TOOL_NAME
        assert report.root_tool_call_id == call_id

    def test_astream_successful_tool_records_result_ac_43_ac_52(
        self, memory_saver: RelateSaver
    ) -> None:
        """AC-43, AC-52: astream yields every node step and the successful call stores a result."""
        thread_id = "async-stream-ok"
        graph = create_tool_agent_graph(checkpointer=memory_saver)

        async def stream() -> list[dict[str, Any]]:
            return [
                chunk
                async for chunk in graph.astream(
                    {"messages": [HumanMessage(content=OK_URL)]},
                    _tool_config(thread_id),
                    stream_mode="updates",
                )
            ]

        chunks = asyncio.run(stream())
        assert [next(iter(chunk)) for chunk in chunks] == ["agent", "tools", "agent"]

        rows = _tool_call_rows(memory_saver)
        assert len(rows) == 1
        assert rows[0]["thread_id"] == thread_id
        assert rows[0]["tool_name"] == TOOL_NAME
        assert rows[0]["result_blob"]
        assert rows[0]["error"] is None

        report = memory_saver.trace_root_cause(thread_id)
        assert report.root_tool_name is None
        assert report.error_message == "No error detected"

    def test_ainvoke_and_run_tool_agent_persist_the_same_call_shape_ac_41(
        self, memory_saver: RelateSaver
    ) -> None:
        """AC-41: the async and sync paths store the same ToolCall fields for the same demo run."""
        graph = create_tool_agent_graph(checkpointer=memory_saver)
        async_state: dict[str, Any] = asyncio.run(
            graph.ainvoke(
                {"messages": [HumanMessage(content=UNREACHABLE_URL)]},
                _tool_config("shape-async"),
            )
        )
        sync_state = run_tool_agent(memory_saver, "shape-sync", fail=True)

        rows = {str(r["thread_id"]): r for r in _tool_call_rows(memory_saver)}
        assert set(rows) == {"shape-async", "shape-sync"}
        assert rows["shape-async"]["call_id"] == _only_call_id(async_state)
        assert rows["shape-sync"]["call_id"] == _only_call_id(sync_state)
        for thread_id in ("shape-async", "shape-sync"):
            assert rows[thread_id]["tool_name"] == TOOL_NAME
            assert "upstream unreachable" in str(rows[thread_id]["error"])
            assert rows[thread_id]["result_blob"] is None

    def test_aput_writes_skips_malformed_tool_entries_ac_47_ac_54(
        self, memory_saver: RelateSaver
    ) -> None:
        """AC-47, AC-54: a malformed entry through aput_writes is skipped without raising."""
        thread_id = "async-malformed"
        checkpoint: Any = empty_checkpoint()
        checkpoint["id"] = "chk-malformed"
        metadata: Any = {"source": "loop", "step": 0, "parents": {}}
        config = asyncio.run(memory_saver.aput(_tool_config(thread_id), checkpoint, metadata, {}))

        good_id = "call_m4_good"
        bad_message = AIMessage(content="", tool_calls=[{"id": "", "name": "x", "args": {}}])
        good_message = AIMessage(
            content="", tool_calls=[{"id": good_id, "name": "lookup", "args": {"k": 1}}]
        )
        writes: list[tuple[str, Any]] = [
            ("messages", [bad_message, {"not": "a message"}, None, good_message]),
            ("counter", 3),
        ]
        result = asyncio.run(memory_saver.aput_writes(config, writes, "task-1"))

        assert result is None
        rows = _tool_call_rows(memory_saver)
        assert [r["call_id"] for r in rows] == [good_id]
        assert rows[0]["checkpoint_id"] == "chk-malformed"
        stored = memory_saver.proxy.execute_read(
            "MATCH (w:Write {checkpoint_id: $cid}) RETURN w.channel AS channel",
            {"cid": "chk-malformed"},
        )
        assert sorted(str(r["channel"]) for r in stored) == ["counter", "messages"]


class TestConcurrentToolRuns:
    """AC-42, AC-53: concurrent async runs that call tools keep their calls separate."""

    def test_concurrent_tool_runs_keep_each_call_on_its_own_thread_ac_42_ac_53(
        self, memory_saver: RelateSaver
    ) -> None:
        thread_ids = [f"conc-{idx}" for idx in range(6)]
        urls = [UNREACHABLE_URL if idx % 2 == 0 else OK_URL for idx in range(len(thread_ids))]
        graph = create_tool_agent_graph(checkpointer=memory_saver)

        async def run_all() -> list[dict[str, Any]]:
            return list(
                await asyncio.gather(
                    *(
                        graph.ainvoke({"messages": [HumanMessage(content=url)]}, _tool_config(tid))
                        for tid, url in zip(thread_ids, urls, strict=True)
                    )
                )
            )

        states = asyncio.run(run_all())
        expected = {
            tid: _only_call_id(state) for tid, state in zip(thread_ids, states, strict=True)
        }

        rows = _tool_call_rows(memory_saver)
        assert len(rows) == len(thread_ids)
        calls_by_thread: dict[str, set[str]] = {}
        for row in rows:
            calls_by_thread.setdefault(str(row["thread_id"]), set()).add(str(row["call_id"]))
        assert calls_by_thread == {tid: {call_id} for tid, call_id in expected.items()}
        assert _count_tool_nodes(memory_saver) == len(thread_ids)

        for idx, tid in enumerate(thread_ids):
            report = memory_saver.trace_root_cause(tid)
            if idx % 2 == 0:
                assert report.root_tool_name == TOOL_NAME
                assert report.root_tool_call_id == expected[tid]
            else:
                assert report.root_tool_name is None
                assert report.root_tool_call_id is None


class TestForkOfToolRun:
    """AC-19: fork_in_memory sandboxes a thread whose history includes tool calls."""

    def test_fork_is_an_isolated_replay_of_the_tool_run_ac_19(self, tmp_path: Path) -> None:
        thread_id = "fork-tools"
        source_file = str(tmp_path / "source.rl")
        with RelateSaver(db_path=source_file) as source:
            run_tool_agent(source, thread_id, fail=True)
            config = _tool_config(thread_id)
            latest = source.get_tuple(config)
            assert latest is not None
            target_id = str(latest.config["configurable"]["checkpoint_id"])
            source_checkpoints = len(list(source.list(config)))
            assert _count_tool_nodes(source) == 1

            fork = source.fork_in_memory(thread_id, target_id)
            try:
                forked = fork.get_tuple(config)
                assert forked is not None
                assert forked.config["configurable"]["checkpoint_id"] == target_id
                assert len(list(fork.list(config))) == source_checkpoints

                child: Any = empty_checkpoint()
                child["id"] = "fork-child"
                child_meta: Any = {"source": "loop", "step": 99, "parents": {}}
                fork.put(_tool_config(thread_id, target_id), child, child_meta, {})

                assert len(list(fork.list(config))) == source_checkpoints + 1
                assert len(list(source.list(config))) == source_checkpoints
                source_latest = source.get_tuple(config)
                assert source_latest is not None
                assert source_latest.config["configurable"]["checkpoint_id"] == target_id
                assert _count_tool_nodes(source) == 1

                # ADR-0010: the fork copies the ToolCall node (and its outcome, because the
                # fork point is the latest checkpoint), so forensics names the same failing tool.
                assert _count_tool_nodes(fork) == 1
                source_report = source.trace_root_cause(thread_id)
                fork_report = fork.trace_root_cause(thread_id)
                assert source_report.root_tool_name == TOOL_NAME
                assert fork_report.root_tool_name == source_report.root_tool_name
                assert fork_report.root_tool_call_id == source_report.root_tool_call_id
            finally:
                fork.close()

    def test_fork_at_unknown_checkpoint_is_rejected_ac_19(self, memory_saver: RelateSaver) -> None:
        run_tool_agent(memory_saver, "fork-invalid", fail=True)
        with pytest.raises(CheckpointNotFoundError):
            memory_saver.fork_in_memory("fork-invalid", "no-such-checkpoint")


class TestBenchmarkScript:
    """AC-56: the benchmark script runs end to end and reports both savers."""

    @pytest.mark.slow
    def test_smoke_run_prints_three_result_columns_ac_56(self, tmp_path: Path) -> None:
        result = _run_process(
            [
                sys.executable,
                str(_BENCHMARK_SCRIPT),
                "--checkpoints",
                "20",
                "--payload-bytes",
                "16",
                "--workdir",
                str(tmp_path),
            ],
            timeout=300.0,
        )
        assert result.returncode == 0, _output_tail(result)

        lines = result.stdout.splitlines()
        assert "- Checkpoints (N): 20" in lines
        header = next(line for line in lines if line.startswith("| Metric |"))
        columns = [cell.strip() for cell in header.strip("|").split("|")]
        assert columns == [
            "Metric",
            "RelateSaver (flush per write, default)",
            "RelateSaver (wal_enabled=False)",
            "SqliteSaver",
        ]

        metric_rows = {
            cells[0]: cells[1:]
            for cells in (
                [cell.strip() for cell in line.strip("|").split("|")]
                for line in lines
                if line.startswith("| ") and not line.startswith(("| Metric", "| :---"))
            )
        }
        for metric in (
            "Put throughput (checkpoints/s)",
            "Get p50 (ms)",
            "Get p95 (ms)",
            "On-disk size (KiB)",
            "Ancestry query, median of 5 (ms)",
        ):
            assert metric in metric_rows, metric
            values = [float(value) for value in metric_rows[metric]]
            assert len(values) == 3
            assert all(value >= 0.0 for value in values)

    @pytest.mark.slow
    def test_run_replaces_only_its_own_files_in_workdir_ac_56(self, tmp_path: Path) -> None:
        # Files whose names merely start with a database name must survive; stale owned
        # files and sidecars are replaced.
        keep = {
            "relate.rl.keep": b"mine",
            "relate-noflush.rl.bak": b"mine too",
            "sqlite.db.notes": b"notes",
            "unrelated.txt": b"unrelated",
        }
        for name, content in keep.items():
            (tmp_path / name).write_bytes(content)
        stale_sidecar = tmp_path / "sqlite.db-journal"
        stale_sidecar.write_bytes(b"stale")

        result = _run_process(
            [
                sys.executable,
                str(_BENCHMARK_SCRIPT),
                "--checkpoints",
                "20",
                "--payload-bytes",
                "16",
                "--workdir",
                str(tmp_path),
            ],
            timeout=300.0,
        )
        # Exit code 0 means all three runs returned the same ancestry count.
        assert result.returncode == 0, _output_tail(result)
        for name, content in keep.items():
            assert (tmp_path / name).read_bytes() == content, name
        assert not stale_sidecar.exists()
        for owned in ("relate.rl", "relate-noflush.rl", "sqlite.db"):
            assert (tmp_path / owned).is_file(), owned

    def test_disk_size_counts_only_exact_files_and_sidecars_ac_56(self, tmp_path: Path) -> None:
        module = _load_benchmark_module()
        (tmp_path / "relate.rl").write_bytes(b"a" * 1024)
        (tmp_path / "relate.rl-wal").write_bytes(b"b" * 2048)
        (tmp_path / "relate.rl.keep").write_bytes(b"c" * 4096)
        (tmp_path / "relate.rl-other").write_bytes(b"d" * 8192)
        assert module.disk_size_kib(tmp_path, "relate.rl") == 3.0

    def test_sqlite_ancestry_query_uses_primary_key_walk_and_counts_ancestors_ac_56(self) -> None:
        import sqlite3

        module = _load_benchmark_module()
        assert "FROM ancestry a CROSS JOIN checkpoints c ON" in module.SQLITE_ANCESTRY_SQL
        connection = sqlite3.connect(":memory:")
        try:
            connection.execute(
                "CREATE TABLE checkpoints (thread_id TEXT, checkpoint_ns TEXT, "
                "checkpoint_id TEXT, parent_checkpoint_id TEXT, "
                "PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id))"
            )
            rows = [("t", "", "c1", None), ("t", "", "c2", "c1"), ("t", "", "c3", "c2")]
            rows.append(("t", "", "b1", "c1"))
            connection.executemany("INSERT INTO checkpoints VALUES (?, ?, ?, ?)", rows)
            row = connection.execute(module.SQLITE_ANCESTRY_SQL, ("t", "c3", "t")).fetchone()
        finally:
            connection.close()
        assert row[0] == 3

    def test_zero_checkpoints_is_rejected_with_usage_error_ac_56(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        module = _load_benchmark_module()
        with pytest.raises(SystemExit) as excinfo:
            module.parse_args(["--checkpoints", "0"])
        assert excinfo.value.code == 2
        assert "--checkpoints must be at least 1" in capsys.readouterr().err
