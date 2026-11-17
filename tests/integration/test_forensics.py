"""Integration tests for Cypher forensic engine (loop detection and root cause analysis).

Validates:
- Cycle and oscillation detection across node sequences and state transitions (AC-25).
- Clean diagnostic reports for linear, non-cyclical executions (AC-26).
- Backward causal path traversal pinpointing antecedent writes preceding errors (AC-27).
- Nominal diagnostics for healthy workflows with zero errors (AC-28).
- Edge cases and error handling for missing threads and invalid parameters.
"""

from __future__ import annotations

import os
import sys
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any, TypedDict

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.graph import END, START, StateGraph

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
_SRC_DIR = str(Path(__file__).resolve().parents[2] / "src")
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from langgraph_checkpoint_relate.errors import CheckpointNotFoundError  # noqa: E402
from langgraph_checkpoint_relate.forensics import RelateForensics  # noqa: E402
from langgraph_checkpoint_relate.models import LoopReport, RootCauseReport  # noqa: E402
from langgraph_checkpoint_relate.saver import RelateSaver  # noqa: E402


@pytest.fixture
def temp_disk_db() -> Generator[str]:
    """Provide a temporary disk-backed database file path and clean up after the test."""
    temp_dir = tempfile.mkdtemp(prefix="relate_forensics_qa_")
    db_file = os.path.join(temp_dir, "checkpoints.rl")
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


class OscillatingState(TypedDict):
    """Workflow state for simulating agent state oscillation."""

    counter: int
    signal: str
    history: list[str]


class LinearState(TypedDict):
    """Workflow state for standard sequential executions."""

    step_count: int
    data: str


class FailureState(TypedDict):
    """Workflow state for poison-write failure injection."""

    payload: str
    status: str


class TestRelateForensics:
    """Comprehensive test suite for RelateForensics engine and RelateSaver delegation."""

    def test_detect_loops_oscillating_state_ac_25(self, temp_disk_db: str) -> None:
        """AC-25: Verify detect_loops identifies cycle sequence, occurrence count, and range."""

        def node_ping(state: OscillatingState) -> dict[str, Any]:
            return {
                "counter": state["counter"] + 1,
                "signal": "ping",
                "history": state["history"] + ["ping"],
            }

        def node_pong(state: OscillatingState) -> dict[str, Any]:
            return {
                "counter": state["counter"] + 1,
                "signal": "pong",
                "history": state["history"] + ["pong"],
            }

        def route_condition(state: OscillatingState) -> str:
            # Oscillation runs 3 iterations of (ping -> pong) = 6 steps
            if state["counter"] < 6:
                return "ping" if state["signal"] == "pong" else "pong"
            return END

        workflow = StateGraph(OscillatingState)
        workflow.add_node("ping", node_ping)
        workflow.add_node("pong", node_pong)
        workflow.add_edge(START, "ping")
        workflow.add_conditional_edges(
            "ping", route_condition, {"ping": "ping", "pong": "pong", END: END}
        )
        workflow.add_conditional_edges(
            "pong", route_condition, {"ping": "ping", "pong": "pong", END: END}
        )

        thread_id = "thread-oscillation-001"
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        with RelateSaver(db_path=temp_disk_db, wal_enabled=True) as saver:
            app = workflow.compile(checkpointer=saver)
            res = app.invoke({"counter": 0, "signal": "", "history": []}, config=config)
            assert res["counter"] >= 6

            # Check via saver delegation
            reports = saver.detect_loops(thread_id, threshold=2)
            assert len(reports) >= 1

            rep = reports[0]
            assert isinstance(rep, LoopReport)
            assert rep.thread_id == thread_id
            assert rep.cycle_nodes == ["ping", "pong"]
            assert rep.occurrence_count >= 2
            assert isinstance(rep.checkpoint_id_range, tuple)
            assert len(rep.checkpoint_id_range) == 2
            assert rep.checkpoint_id_range[0] != rep.checkpoint_id_range[1]

            # Check via direct RelateForensics instance
            with RelateForensics(saver) as forensics:
                forensics_reports = forensics.detect_loops(thread_id, threshold=2)
                assert len(forensics_reports) == len(reports)
                assert forensics_reports[0].cycle_nodes == ["ping", "pong"]

            # Higher threshold should filter out if cycle count < threshold
            high_reports = saver.detect_loops(thread_id, threshold=10)
            assert high_reports == []

    def test_detect_loops_linear_no_cycles_ac_26(self, temp_disk_db: str) -> None:
        """AC-26: Verify detect_loops returns empty list for linear non-repeating executions."""

        def step_one(state: LinearState) -> dict[str, Any]:
            return {"step_count": state["step_count"] + 1, "data": state["data"] + "->1"}

        def step_two(state: LinearState) -> dict[str, Any]:
            return {"step_count": state["step_count"] + 1, "data": state["data"] + "->2"}

        def step_three(state: LinearState) -> dict[str, Any]:
            return {"step_count": state["step_count"] + 1, "data": state["data"] + "->3"}

        workflow = StateGraph(LinearState)
        workflow.add_node("step_one", step_one)
        workflow.add_node("step_two", step_two)
        workflow.add_node("step_three", step_three)
        workflow.add_edge(START, "step_one")
        workflow.add_edge("step_one", "step_two")
        workflow.add_edge("step_two", "step_three")
        workflow.add_edge("step_three", END)

        thread_id = "thread-linear-001"
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        with RelateSaver(db_path=temp_disk_db, wal_enabled=True) as saver:
            app = workflow.compile(checkpointer=saver)
            app.invoke({"step_count": 0, "data": "start"}, config=config)

            # Execution is strictly linear with no repeating nodes or state oscillation
            reports = saver.detect_loops(thread_id, threshold=2)
            assert reports == []

    def test_trace_root_cause_poison_write_ac_27(self, temp_disk_db: str) -> None:
        """AC-27: Verify trace_root_cause pinpoints offending antecedent write preceding error."""

        def inject_poison(state: FailureState) -> dict[str, Any]:
            # Step 1: Write malformed or poisonous payload into state
            return {"payload": "POISON_TOKEN_MALFORMED", "status": "poisoned"}

        def process_payload(state: FailureState) -> dict[str, Any]:
            # Step 2: Validate payload; explode if poison is detected
            if "POISON" in state["payload"]:
                raise ValueError("Fatal parser exception: encountered poison payload in channel")
            return {"status": "success"}

        workflow = StateGraph(FailureState)
        workflow.add_node("inject_poison", inject_poison)
        workflow.add_node("process_payload", process_payload)
        workflow.add_edge(START, "inject_poison")
        workflow.add_edge("inject_poison", "process_payload")
        workflow.add_edge("process_payload", END)

        thread_id = "thread-failure-001"
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        with RelateSaver(db_path=temp_disk_db, wal_enabled=True) as saver:
            app = workflow.compile(checkpointer=saver)

            # Workflow execution will fail at process_payload step
            with pytest.raises(ValueError, match="Fatal parser exception"):
                app.invoke({"payload": "initial", "status": "pending"}, config=config)

            # Now run root-cause analysis
            report = saver.trace_root_cause(thread_id)
            assert isinstance(report, RootCauseReport)
            assert report.thread_id == thread_id
            assert report.error_checkpoint_id != ""
            assert "Fatal parser exception" in report.error_message
            assert report.root_channel_write is not None
            assert report.causal_path_length >= 1
            assert "Failure at checkpoint" in report.diagnostic_summary

    def test_trace_root_cause_successful_thread_ac_28(self, temp_disk_db: str) -> None:
        """AC-28: Verify trace_root_cause indicates no failure detected on successful thread."""

        def nominal_node(state: LinearState) -> dict[str, Any]:
            return {"step_count": state["step_count"] + 1, "data": "nominal_complete"}

        workflow = StateGraph(LinearState)
        workflow.add_node("nominal", nominal_node)
        workflow.add_edge(START, "nominal")
        workflow.add_edge("nominal", END)

        thread_id = "thread-healthy-001"
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        with RelateSaver(db_path=temp_disk_db, wal_enabled=True) as saver:
            app = workflow.compile(checkpointer=saver)
            app.invoke({"step_count": 0, "data": "init"}, config=config)

            report = saver.trace_root_cause(thread_id)
            assert isinstance(report, RootCauseReport)
            assert report.thread_id == thread_id
            assert report.error_checkpoint_id == ""
            assert report.error_message == "No error detected"
            assert report.root_tool_call_id is None
            assert report.root_channel_write is None
            assert report.causal_path_length == 0
            assert report.diagnostic_summary == "Workflow completed without errors."

    def test_forensics_missing_thread_and_invalid_arguments(self, temp_disk_db: str) -> None:
        """Verify CheckpointNotFoundError for non-existent thread and argument validation."""
        with RelateSaver(db_path=temp_disk_db, wal_enabled=True) as saver:
            # 1. Non-existent thread
            with pytest.raises(
                CheckpointNotFoundError, match="not found or contains no checkpoints"
            ):
                saver.detect_loops("missing-thread-xyz")

            with pytest.raises(
                CheckpointNotFoundError, match="not found or contains no checkpoints"
            ):
                saver.trace_root_cause("missing-thread-xyz")

            # 2. Empty or invalid thread_id
            with pytest.raises(
                CheckpointNotFoundError, match="thread_id must be a non-empty string"
            ):
                saver.detect_loops("")

            with pytest.raises(
                CheckpointNotFoundError, match="thread_id must be a non-empty string"
            ):
                saver.trace_root_cause("")

            # 3. Invalid threshold
            with pytest.raises(ValueError, match="threshold must be an integer >= 1"):
                saver.detect_loops("missing-thread-xyz", threshold=0)

    def test_relate_forensics_file_constructor_and_context_manager(self, temp_disk_db: str) -> None:
        """Verify RelateForensics supports direct path constructor and context management."""
        with RelateSaver(db_path=temp_disk_db, wal_enabled=True) as saver:
            # Create a simple checkpoint so thread exists
            cfg: RunnableConfig = {"configurable": {"thread_id": "test-th-file"}}
            cp = empty_checkpoint()
            saver.put(cfg, cp, {}, {})

        # Open forensics using file path directly
        with RelateForensics(temp_disk_db) as forensics:
            assert isinstance(forensics.saver, RelateSaver)
            loops = forensics.detect_loops("test-th-file", threshold=2)
            assert loops == []
            rc = forensics.trace_root_cause("test-th-file")
            assert rc.error_message == "No error detected"

    def test_detect_loops_multi_node_triplet_cycle(self, temp_disk_db: str) -> None:
        """Verify cycle detection of length-3 repeating node sequence [planner, researcher, critic]."""
        thread_id = "thread-multi-agent-triplet"
        with RelateSaver(db_path=temp_disk_db, wal_enabled=True) as saver:
            # Build 2 repeating cycles of [planner, researcher, critic]
            sequence = ["planner", "researcher", "critic", "planner", "researcher", "critic"]
            prev_cid: str | None = None
            for idx, node_name in enumerate(sequence):
                cid = f"cp-step-{idx}"
                cfg: RunnableConfig = {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_id": cid,
                    }
                }
                cp: Any = dict(empty_checkpoint())
                cp["id"] = cid
                if prev_cid:
                    cp["parent_checkpoint_id"] = prev_cid
                saver.put(cfg, cp, {"node": node_name, "step": idx}, {})  # type: ignore[typeddict-unknown-key]
                prev_cid = cid

            reports = saver.detect_loops(thread_id, threshold=2)
            assert len(reports) == 1
            report = reports[0]
            assert report.cycle_nodes == ["planner", "researcher", "critic"]
            assert report.occurrence_count == 2
            assert report.checkpoint_id_range == ("cp-step-0", "cp-step-5")

    def test_trace_root_cause_tool_call_antecedent(self, temp_disk_db: str) -> None:
        """Verify trace_root_cause pinpoints offending tool call linked via INVOKED edge."""
        thread_id = "thread-tool-fail-001"
        with RelateSaver(db_path=temp_disk_db, wal_enabled=True) as saver:
            # Step 1: Checkpoint with ToolCall that failed
            cid_step1 = "cp-tool-001"
            cfg1: RunnableConfig = {
                "configurable": {"thread_id": thread_id, "checkpoint_id": cid_step1}
            }
            cp1: Any = dict(empty_checkpoint())
            cp1["id"] = cid_step1
            saver.put(cfg1, cp1, {"step": 1, "node": "tool_executor"}, {})  # type: ignore[typeddict-unknown-key]

            # Create tool call and link via INVOKED
            saver.proxy.execute_write(
                "CREATE (tc:ToolCall {call_id: $call_id, tool_name: $name, args_blob: $args, error: $err, created_at: $created_at}) "
                "WITH tc "
                "MATCH (c:Checkpoint {checkpoint_id: $cid}) "
                "MERGE (c)-[:INVOKED]->(tc)",
                {
                    "call_id": "call-search-err-99",
                    "name": "search_api",
                    "args": b"{}",
                    "err": "HTTP 429: Rate limit exceeded on endpoint",
                    "created_at": "2026-11-12T00:00:00Z",
                    "cid": cid_step1,
                },
            )

            # Step 2: Checkpoint downstream where error is recorded
            cid_step2 = "cp-tool-002"
            cfg2: RunnableConfig = {
                "configurable": {"thread_id": thread_id, "checkpoint_id": cid_step2}
            }
            cp2: Any = dict(empty_checkpoint())
            cp2["id"] = cid_step2
            cp2["parent_checkpoint_id"] = cid_step1
            saver.put(cfg2, cp2, {"step": 2, "node": "error_handler"}, {})  # type: ignore[typeddict-unknown-key]

            # Put __error__ write at step 2
            saver.put_writes(
                cfg2,
                [("__error__", "Cascading failure: Tool call search_api failed")],
                task_id="task-fail",
            )

            report = saver.trace_root_cause(thread_id)
            assert report.thread_id == thread_id
            assert report.error_checkpoint_id == cid_step2
            assert report.root_tool_call_id == "call-search-err-99"
            assert report.causal_path_length == 1
            assert "HTTP 429" in report.error_message or "search_api" in report.error_message
            assert "call-search-err-99" in report.diagnostic_summary

    def test_trace_root_cause_names_failing_tool_from_put_writes_ac_45(
        self, temp_disk_db: str
    ) -> None:
        """AC-45: a tool call recorded via put_writes() is reported by tool_name and call_id."""
        thread_id = "thread-tool-name-001"
        call_id = "call-weather-77"
        with RelateSaver(db_path=temp_disk_db, wal_enabled=True) as saver:
            cfg: RunnableConfig = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
            cp: Any = dict(empty_checkpoint())
            cp["id"] = "cp-name-001"
            saved_cfg = saver.put(cfg, cp, {"step": 1, "source": "loop", "parents": {}}, {})
            ai = AIMessage(
                content="",
                tool_calls=[{"id": call_id, "name": "get_weather", "args": {"city": "X"}}],
            )
            saver.put_writes(saved_cfg, [("messages", [ai])], task_id="task-ai")
            tool_msg = ToolMessage(content="upstream down", tool_call_id=call_id, status="error")
            saver.put_writes(saved_cfg, [("messages", [tool_msg])], task_id="task-tool")

            report = saver.trace_root_cause(thread_id)

        assert report.root_tool_call_id == call_id
        assert report.root_tool_name == "get_weather"
        assert f"Root tool invocation: '{call_id}'. Failing tool: 'get_weather'." in (
            report.diagnostic_summary
        )
