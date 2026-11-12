"""Integration tests verifying in-memory sandboxed time-travel replay (TimeTravelEngine)."""

# ruff: noqa: E402

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any, TypedDict, cast

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    empty_checkpoint,
)
from langgraph.graph import END, START, StateGraph

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from langgraph_checkpoint_relate.errors import CheckpointNotFoundError
from langgraph_checkpoint_relate.saver import RelateSaver
from langgraph_checkpoint_relate.time_travel import TimeTravelEngine, fork_in_memory


@pytest.fixture
def temp_disk_db() -> Generator[str]:
    """Provide a temporary disk-backed .rl database file path and clean up on teardown."""
    temp_dir = tempfile.mkdtemp(prefix="relate_time_travel_")
    db_file = os.path.join(temp_dir, "timetravel_test.rl")
    yield db_file

    for root, _, files in os.walk(temp_dir, topdown=False):
        for f in files:
            try:
                os.remove(os.path.join(root, f))
            except OSError:
                pass
    try:
        os.rmdir(temp_dir)
    except OSError:
        pass


def _make_checkpoint(
    checkpoint_id: str,
    channel_values: dict[str, Any] | None = None,
    parent_checkpoint_id: str | None = None,
) -> Checkpoint:
    """Helper to create a structured Checkpoint dictionary."""
    base: dict[str, Any] = dict(empty_checkpoint())
    base["id"] = checkpoint_id
    if parent_checkpoint_id is not None:
        base["parent_checkpoint_id"] = parent_checkpoint_id
    if channel_values:
        base["channel_values"] = channel_values
        base["channel_versions"] = dict.fromkeys(channel_values, "1.0")
    return cast(Checkpoint, base)


def _compute_file_sha256(path: str) -> str:
    """Compute SHA-256 hash of a file on disk."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class WorkflowState(TypedDict, total=False):
    """Schema for test workflow state."""

    count: int
    history: list[str]


class TestTimeTravelReplay:
    """Integration suite verifying in-memory sandboxed time-travel replay."""

    def test_fork_in_memory_returns_isolated_sandbox_ac_19(self, temp_disk_db: str) -> None:
        """AC-19: Given durable checkpoints, fork_in_memory returns an in-memory saver with ancestor graph."""
        thread_id = "thread-ac19-replay"
        source_saver = RelateSaver(db_path=temp_disk_db, wal_enabled=True)

        try:
            config: RunnableConfig = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}

            # Step 0: Initial checkpoint
            cp_0 = _make_checkpoint("cp-tt-000", {"count": 10, "history": ["init"]})
            meta_0: CheckpointMetadata = {"step": 0, "source": "input"}
            new_v_0: ChannelVersions = {"count": "1.0", "history": "1.0"}
            source_saver.put(config, cp_0, meta_0, new_v_0)

            # Step 1: Intermediate checkpoint with pending write
            cp_1 = _make_checkpoint(
                "cp-tt-001", {"count": 20, "history": ["init", "step1"]}, "cp-tt-000"
            )
            meta_1: CheckpointMetadata = {"step": 1, "source": "loop"}
            new_v_1: ChannelVersions = {"count": "2.0", "history": "2.0"}
            cfg_1 = source_saver.put(
                {"configurable": {"thread_id": thread_id, "checkpoint_id": "cp-tt-000"}},
                cp_1,
                meta_1,
                new_v_1,
            )
            source_saver.put_writes(
                cfg_1,
                [("feedback", "tool execution pending review")],
                task_id="task-001",
            )

            # Step 2: Final checkpoint on source thread
            cp_2 = _make_checkpoint(
                "cp-tt-002", {"count": 30, "history": ["init", "step1", "step2"]}, "cp-tt-001"
            )
            meta_2: CheckpointMetadata = {"step": 2, "source": "loop"}
            new_v_2: ChannelVersions = {"count": "3.0", "history": "3.0"}
            source_saver.put(
                {"configurable": {"thread_id": thread_id, "checkpoint_id": "cp-tt-001"}},
                cp_2,
                meta_2,
                new_v_2,
            )

            # Replay time-travel targeting cp-tt-001 (step 1)
            sandboxed_saver = source_saver.fork_in_memory(
                thread_id=thread_id, checkpoint_id="cp-tt-001"
            )

            assert sandboxed_saver.db_path == ":memory:"

            # Verify targeted checkpoint state is retrieved with get_tuple()
            latest_tuple = sandboxed_saver.get_tuple({"configurable": {"thread_id": thread_id}})
            assert latest_tuple is not None
            assert latest_tuple.config["configurable"]["checkpoint_id"] == "cp-tt-001"
            assert latest_tuple.checkpoint["channel_values"]["count"] == 20
            assert latest_tuple.checkpoint["channel_values"]["history"] == ["init", "step1"]

            # Verify associated pending writes were cloned into the sandbox
            assert latest_tuple.pending_writes is not None
            assert len(latest_tuple.pending_writes) == 1
            task_id, channel, value = latest_tuple.pending_writes[0]
            assert task_id == "task-001"
            assert channel == "feedback"
            assert value == "tool execution pending review"

            # Verify ancestor checkpoint cp-tt-000 exists in sandbox
            tuple_0 = sandboxed_saver.get_tuple(
                {"configurable": {"thread_id": thread_id, "checkpoint_id": "cp-tt-000"}}
            )
            assert tuple_0 is not None
            assert tuple_0.checkpoint["channel_values"]["count"] == 10

            # Verify subsequent checkpoint cp-tt-002 is NOT in sandbox
            tuple_2 = sandboxed_saver.get_tuple(
                {"configurable": {"thread_id": thread_id, "checkpoint_id": "cp-tt-002"}}
            )
            assert tuple_2 is None

            # Verify list() yields exactly the 2 ancestor checkpoints in reverse-chronological order
            tuples = list(sandboxed_saver.list({"configurable": {"thread_id": thread_id}}))
            assert len(tuples) == 2
            assert tuples[0].config["configurable"]["checkpoint_id"] == "cp-tt-001"
            assert tuples[1].config["configurable"]["checkpoint_id"] == "cp-tt-000"

            sandboxed_saver.close()
        finally:
            source_saver.close()

    def test_sandboxed_mutations_leave_disk_db_bitwise_identical_ac_20(
        self, temp_disk_db: str
    ) -> None:
        """AC-20: Writing new checkpoints in sandboxed session leaves durable checkpoints.rl bitwise identical."""
        thread_id = "thread-ac20-durability"
        source_saver = RelateSaver(db_path=temp_disk_db, wal_enabled=True)

        try:
            config: RunnableConfig = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}

            # Write baseline durable checkpoints
            cp_0 = _make_checkpoint("cp-durable-0", {"value": "baseline-0"})
            source_saver.put(config, cp_0, {"step": 0}, {"value": "1.0"})

            cp_1 = _make_checkpoint("cp-durable-1", {"value": "baseline-1"}, "cp-durable-0")
            source_saver.put(
                {"configurable": {"thread_id": thread_id, "checkpoint_id": "cp-durable-0"}},
                cp_1,
                {"step": 1},
                {"value": "2.0"},
            )

            # Create in-memory sandbox from active source saver
            sandboxed_saver = source_saver.fork_in_memory(
                thread_id=thread_id, checkpoint_id="cp-durable-1"
            )

            # Flush and close source_saver so file handles are released for bitwise hashing
            source_saver.close()

            # Record disk file baseline state immediately after forking
            pre_hash = _compute_file_sha256(temp_disk_db)
            pre_stat = os.stat(temp_disk_db)
            pre_size = pre_stat.st_size
            pre_mtime_ns = pre_stat.st_mtime_ns

            # Perform extensive mutations in the sandbox session
            for i in range(2, 7):
                new_cp = _make_checkpoint(
                    f"cp-sandbox-{i}",
                    {"value": f"speculative-replay-{i}"},
                    f"cp-sandbox-{i - 1}" if i > 2 else "cp-durable-1",
                )
                parent_cid = f"cp-sandbox-{i - 1}" if i > 2 else "cp-durable-1"
                cfg = sandboxed_saver.put(
                    {"configurable": {"thread_id": thread_id, "checkpoint_id": parent_cid}},
                    new_cp,
                    {"step": i},
                    {"value": f"{i}.0"},
                )
                sandboxed_saver.put_writes(
                    cfg,
                    [(f"channel_{i}", f"sandbox write {i}")],
                    task_id=f"sandbox-task-{i}",
                )

            # Verify the in-memory sandbox has all new speculative steps
            sandbox_latest = sandboxed_saver.get_tuple({"configurable": {"thread_id": thread_id}})
            assert sandbox_latest is not None
            assert sandbox_latest.config["configurable"]["checkpoint_id"] == "cp-sandbox-6"
            assert sandbox_latest.checkpoint["channel_values"]["value"] == "speculative-replay-6"

            sandboxed_saver.close()

            # Verify durable disk file is bitwise identical
            post_hash = _compute_file_sha256(temp_disk_db)
            post_stat = os.stat(temp_disk_db)

            assert post_hash == pre_hash, (
                "Durable disk file hash changed during in-memory sandboxing!"
            )
            assert post_stat.st_size == pre_size, (
                "Durable disk file size changed during in-memory sandboxing!"
            )
            assert post_stat.st_mtime_ns == pre_mtime_ns, (
                "Durable disk file mtime changed during in-memory sandboxing!"
            )

            # Re-open source saver and verify durable database is unaffected
            source_saver = RelateSaver(db_path=temp_disk_db, wal_enabled=True)
            source_latest = source_saver.get_tuple({"configurable": {"thread_id": thread_id}})
            assert source_latest is not None
            assert source_latest.config["configurable"]["checkpoint_id"] == "cp-durable-1"
            assert source_latest.checkpoint["channel_values"]["value"] == "baseline-1"
        finally:
            source_saver.close()

    def test_nonexistent_checkpoint_raises_not_found_with_candidates_ac_21(
        self, temp_disk_db: str
    ) -> None:
        """AC-21: fork_in_memory with nonexistent checkpoint_id raises CheckpointNotFoundError listing candidates."""
        thread_id = "thread-ac21-validation"
        source_saver = RelateSaver(db_path=temp_disk_db, wal_enabled=True)

        try:
            config: RunnableConfig = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}

            # Populate 3 checkpoints
            cp_0 = _make_checkpoint("cp-step-000", {"k": 0})
            source_saver.put(config, cp_0, {"step": 0}, {"k": "1.0"})

            cp_1 = _make_checkpoint("cp-step-001", {"k": 1}, "cp-step-000")
            source_saver.put(
                {"configurable": {"thread_id": thread_id, "checkpoint_id": "cp-step-000"}},
                cp_1,
                {"step": 1},
                {"k": "2.0"},
            )

            cp_2 = _make_checkpoint("cp-step-002", {"k": 2}, "cp-step-001")
            source_saver.put(
                {"configurable": {"thread_id": thread_id, "checkpoint_id": "cp-step-001"}},
                cp_2,
                {"step": 2},
                {"k": "3.0"},
            )

            # Attempt to fork from a non-existent checkpoint ID
            nonexistent_id = "cp-nonexistent-999"
            with pytest.raises(CheckpointNotFoundError) as exc_info:
                source_saver.fork_in_memory(thread_id=thread_id, checkpoint_id=nonexistent_id)

            err_msg = str(exc_info.value)
            assert f"Checkpoint '{nonexistent_id}' not found for thread '{thread_id}'" in err_msg
            assert "Available checkpoints: ['cp-step-000', 'cp-step-001', 'cp-step-002']" in err_msg

            # Test with thread containing zero checkpoints
            with pytest.raises(CheckpointNotFoundError) as empty_exc:
                source_saver.fork_in_memory(
                    thread_id="nonexistent-thread-id", checkpoint_id="cp-any"
                )

            empty_msg = str(empty_exc.value)
            assert "Available checkpoints: []" in empty_msg
        finally:
            source_saver.close()

    def test_time_travel_branch_continuation_with_branch_thread_id(self, temp_disk_db: str) -> None:
        """Branch continuation creates isolated thread branch and forked relationship."""
        source_thread = "thread-main-flow"
        branch_thread = "thread-branch-alt"
        source_saver = RelateSaver(db_path=temp_disk_db, wal_enabled=True)

        try:
            config: RunnableConfig = {
                "configurable": {"thread_id": source_thread, "checkpoint_ns": ""}
            }

            cp_0 = _make_checkpoint("cp-main-0", {"branch": "main", "step": 0})
            source_saver.put(config, cp_0, {"step": 0}, {"branch": "1.0", "step": "1.0"})

            cp_1 = _make_checkpoint("cp-main-1", {"branch": "main", "step": 1}, "cp-main-0")
            source_saver.put(
                {"configurable": {"thread_id": source_thread, "checkpoint_id": "cp-main-0"}},
                cp_1,
                {"step": 1},
                {"branch": "2.0", "step": "2.0"},
            )

            # Fork at cp-main-0 with branch_thread_id
            sandboxed = source_saver.fork_in_memory(
                thread_id=source_thread,
                checkpoint_id="cp-main-0",
                branch_thread_id=branch_thread,
            )

            # Continue execution on the new branch in the sandbox
            branch_cp_1 = _make_checkpoint(
                "cp-branch-1", {"branch": "experimental", "step": 1}, "cp-main-0"
            )
            sandboxed.put(
                {"configurable": {"thread_id": branch_thread, "checkpoint_id": "cp-main-0"}},
                branch_cp_1,
                {"step": 1},
                {"branch": "1.0", "step": "1.0"},
            )

            branch_cp_2 = _make_checkpoint(
                "cp-branch-2", {"branch": "experimental", "step": 2}, "cp-branch-1"
            )
            sandboxed.put(
                {"configurable": {"thread_id": branch_thread, "checkpoint_id": "cp-branch-1"}},
                branch_cp_2,
                {"step": 2},
                {"branch": "2.0", "step": "2.0"},
            )

            # Verify branch state in sandbox
            branch_latest = sandboxed.get_tuple({"configurable": {"thread_id": branch_thread}})
            assert branch_latest is not None
            assert branch_latest.config["configurable"]["checkpoint_id"] == "cp-branch-2"
            assert branch_latest.checkpoint["channel_values"]["branch"] == "experimental"

            # Verify FORKED_TO relationship in sandboxed graph
            fork_edges = sandboxed.proxy.execute_read(
                "MATCH (c:Checkpoint {checkpoint_id: 'cp-main-0'})-[:FORKED_TO]->(target) "
                "RETURN labels(target) AS labels, target.checkpoint_id AS cid, target.thread_id AS tid"
            )
            assert len(fork_edges) >= 1

            # Verify source saver is completely untouched
            source_branch_tuple = source_saver.get_tuple(
                {"configurable": {"thread_id": branch_thread}}
            )
            assert source_branch_tuple is None

            sandboxed.close()
        finally:
            source_saver.close()

    def test_time_travel_engine_direct_invocation(self, temp_disk_db: str) -> None:
        """Verify TimeTravelEngine and module-level fork_in_memory function."""
        thread_id = "thread-direct-engine"
        source_saver = RelateSaver(db_path=temp_disk_db, wal_enabled=True)

        try:
            config: RunnableConfig = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
            cp_0 = _make_checkpoint("cp-eng-0", {"num": 100})
            source_saver.put(config, cp_0, {"step": 0}, {"num": "1.0"})

            # Invoke via TimeTravelEngine classmethod/staticmethod
            engine_sandbox = TimeTravelEngine.fork_in_memory(source_saver, thread_id, "cp-eng-0")
            assert engine_sandbox.db_path == ":memory:"
            tup1 = engine_sandbox.get_tuple({"configurable": {"thread_id": thread_id}})
            assert tup1 is not None and tup1.checkpoint["channel_values"]["num"] == 100
            engine_sandbox.close()

            # Invoke via module-level convenience function
            func_sandbox = fork_in_memory(source_saver, thread_id, "cp-eng-0")
            assert func_sandbox.db_path == ":memory:"
            tup2 = func_sandbox.get_tuple({"configurable": {"thread_id": thread_id}})
            assert tup2 is not None and tup2.checkpoint["channel_values"]["num"] == 100
            func_sandbox.close()
        finally:
            source_saver.close()

    def test_langgraph_workflow_time_travel_branching(self, temp_disk_db: str) -> None:
        """Verify LangGraph StateGraph execution, replay branching, and isolated what-if evaluation."""
        source_saver = RelateSaver(db_path=temp_disk_db, wal_enabled=True)

        try:
            # Build simple LangGraph pipeline: node_a -> node_b
            builder = StateGraph(WorkflowState)

            def node_a(state: WorkflowState) -> dict[str, Any]:
                cnt = state.get("count", 0) + 1
                history = list(state.get("history", [])) + ["visited_a"]
                return {"count": cnt, "history": history}

            def node_b(state: WorkflowState) -> dict[str, Any]:
                cnt = state.get("count", 0) * 10
                history = list(state.get("history", [])) + ["visited_b"]
                return {"count": cnt, "history": history}

            builder.add_node("node_a", node_a)
            builder.add_node("node_b", node_b)
            builder.add_edge(START, "node_a")
            builder.add_edge("node_a", "node_b")
            builder.add_edge("node_b", END)

            app = builder.compile(checkpointer=source_saver)

            thread_id = "thread-agent-pipeline"
            run_config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

            # Run initial pipeline
            output = app.invoke({"count": 5, "history": ["start"]}, run_config)
            assert output["count"] == 60  # (5 + 1) * 10
            assert output["history"] == ["start", "visited_a", "visited_b"]

            # Inspect checkpoints in source_saver to locate intermediate state (count == 6)
            checkpoints = list(source_saver.list(run_config))
            intermediate_checkpoint = next(
                (
                    cp
                    for cp in checkpoints
                    if cp.checkpoint.get("channel_values", {}).get("count") == 6
                ),
                None,
            )
            assert intermediate_checkpoint is not None
            intermediate_checkpoint_id = intermediate_checkpoint.config["configurable"][
                "checkpoint_id"
            ]

            # Fork in-memory at the intermediate state (after node_a, before node_b)
            sandboxed_saver = source_saver.fork_in_memory(
                thread_id=thread_id,
                checkpoint_id=intermediate_checkpoint_id,
            )

            # Recompile app with sandboxed checkpointer
            sandboxed_app = builder.compile(checkpointer=sandboxed_saver)

            # Resume execution in sandbox from intermediate checkpoint
            replay_config: RunnableConfig = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_id": intermediate_checkpoint_id,
                }
            }
            # Verify sandboxed state matches intermediate step
            inter_tuple = sandboxed_saver.get_tuple(replay_config)
            assert inter_tuple is not None
            assert inter_tuple.checkpoint["channel_values"]["count"] == 6

            # Resume workflow from intermediate checkpoint in sandbox
            resumed_output = sandboxed_app.invoke(None, replay_config)
            assert resumed_output["count"] == 60
            assert resumed_output["history"] == ["start", "visited_a", "visited_b"]

            # Modify state channel in sandbox and verify branch isolation
            sandboxed_saver.put(
                replay_config,
                _make_checkpoint(
                    "cp-modified",
                    {"count": 100, "history": ["start", "visited_a", "injected_param"]},
                ),
                {"step": 99},
                {"count": "99.0", "history": "99.0"},
            )

            resumed_tuple = sandboxed_saver.get_tuple({"configurable": {"thread_id": thread_id}})
            assert resumed_tuple is not None
            assert resumed_tuple.checkpoint["channel_values"]["count"] == 100

            sandboxed_saver.close()

            # Verify source disk saver state was NOT altered
            final_source_tuple = source_saver.get_tuple(run_config)
            assert final_source_tuple is not None
            assert final_source_tuple.checkpoint["channel_values"]["count"] == 60
        finally:
            source_saver.close()
