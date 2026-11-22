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
import time
from collections.abc import Generator
from pathlib import Path
from typing import Annotated, Any, TypedDict
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

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

    def test_detect_loops_multi_namespace_no_false_positive_ac_182(self, temp_disk_db: str) -> None:
        """AC-182: Verify multi-namespace thread with intervening subgraph does not produce false positive loops."""
        thread_id = "thread-multi-ns-no-false-positive"
        with RelateSaver(db_path=temp_disk_db, wal_enabled=True) as saver:
            # Root graph steps 0 and 1
            cp0: Any = dict(empty_checkpoint())
            cp0["id"] = "cp-root-0"
            cp0["channel_values"] = {"counter": 0}
            cfg0: RunnableConfig = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": "",
                    "checkpoint_id": "cp-root-0",
                }
            }
            saver.put(cfg0, cp0, {"node": "step_a", "step": 0}, {})  # type: ignore[typeddict-unknown-key]

            cp1: Any = dict(empty_checkpoint())
            cp1["id"] = "cp-root-1"
            cp1["parent_checkpoint_id"] = "cp-root-0"
            cp1["channel_values"] = {"counter": 1}
            cfg1: RunnableConfig = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": "",
                    "checkpoint_id": "cp-root-1",
                }
            }
            saver.put(cfg1, cp1, {"node": "step_b", "step": 1}, {})  # type: ignore[typeddict-unknown-key]

            # Intervening subgraph execution at steps 2 and 3 with matching node names (step_a, step_b)
            # In global step sorting, this creates a bogus sequence [step_a, step_b, step_a, step_b].
            cp_s0: Any = dict(empty_checkpoint())
            cp_s0["id"] = "cp-sub-2"
            cp_s0["parent_checkpoint_id"] = "cp-root-1"
            cp_s0["channel_values"] = {"sub_val": 10}
            cfg_s0: RunnableConfig = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": "subgraph:worker",
                    "checkpoint_id": "cp-sub-2",
                }
            }
            saver.put(cfg_s0, cp_s0, {"node": "step_a", "step": 2}, {})  # type: ignore[typeddict-unknown-key]

            cp_s1: Any = dict(empty_checkpoint())
            cp_s1["id"] = "cp-sub-3"
            cp_s1["parent_checkpoint_id"] = "cp-sub-2"
            cp_s1["channel_values"] = {"sub_val": 11}
            cfg_s1: RunnableConfig = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": "subgraph:worker",
                    "checkpoint_id": "cp-sub-3",
                }
            }
            saver.put(cfg_s1, cp_s1, {"node": "step_b", "step": 3}, {})  # type: ignore[typeddict-unknown-key]

            # Root graph resumes execution at steps 4 and 5
            cp2: Any = dict(empty_checkpoint())
            cp2["id"] = "cp-root-4"
            cp2["parent_checkpoint_id"] = "cp-root-1"
            cp2["channel_values"] = {"counter": 2}
            cfg2: RunnableConfig = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": "",
                    "checkpoint_id": "cp-root-4",
                }
            }
            saver.put(cfg2, cp2, {"node": "step_c", "step": 4}, {})  # type: ignore[typeddict-unknown-key]

            cp3: Any = dict(empty_checkpoint())
            cp3["id"] = "cp-root-5"
            cp3["parent_checkpoint_id"] = "cp-root-4"
            cp3["channel_values"] = {"counter": 3}
            cfg3: RunnableConfig = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": "",
                    "checkpoint_id": "cp-root-5",
                }
            }
            saver.put(cfg3, cp3, {"node": "step_d", "step": 5}, {})  # type: ignore[typeddict-unknown-key]

            # Namespace-isolated traversal along NEXT edges avoids cross-namespace step interleaving
            reports = saver.detect_loops(thread_id, threshold=2)
            assert reports == []

    def test_detect_loops_subgraph_genuine_cycle_detected_ac_182(self, temp_disk_db: str) -> None:
        """AC-182: Verify loops genuinely present in a subgraph namespace are detected within that subgraph namespace."""
        thread_id = "thread-subgraph-genuine-loop"
        with RelateSaver(db_path=temp_disk_db, wal_enabled=True) as saver:
            # Root caller step
            cp_root: Any = dict(empty_checkpoint())
            cp_root["id"] = "cp-root-start"
            cp_root["channel_values"] = {"status": "init"}
            cfg_root: RunnableConfig = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": "",
                    "checkpoint_id": "cp-root-start",
                }
            }
            saver.put(cfg_root, cp_root, {"node": "root_caller", "step": 0}, {})  # type: ignore[typeddict-unknown-key]

            # Subgraph performs a repeating loop of [planner, worker] twice
            sub_sequence = ["planner", "worker", "planner", "worker"]
            prev_sub_cid: str | None = None
            for idx, node_name in enumerate(sub_sequence, start=1):
                cid = f"cp-sub-{idx}"
                cp: Any = dict(empty_checkpoint())
                cp["id"] = cid
                cp["parent_checkpoint_id"] = prev_sub_cid if prev_sub_cid else "cp-root-start"
                cp["channel_values"] = {"sub_step": idx}
                cfg: RunnableConfig = {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": "subgraph:task_executor",
                        "checkpoint_id": cid,
                    }
                }
                saver.put(cfg, cp, {"node": node_name, "step": idx}, {})  # type: ignore[typeddict-unknown-key]
                prev_sub_cid = cid

            # Root finishes after subgraph
            cp_finish: Any = dict(empty_checkpoint())
            cp_finish["id"] = "cp-root-end"
            cp_finish["parent_checkpoint_id"] = "cp-root-start"
            cp_finish["channel_values"] = {"status": "done"}
            cfg_finish: RunnableConfig = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": "",
                    "checkpoint_id": "cp-root-end",
                }
            }
            saver.put(cfg_finish, cp_finish, {"node": "root_finisher", "step": 5}, {})  # type: ignore[typeddict-unknown-key]

            reports = saver.detect_loops(thread_id, threshold=2)
            assert len(reports) == 1
            report = reports[0]
            assert report.thread_id == thread_id
            assert report.cycle_nodes == ["planner", "worker"]
            assert report.occurrence_count == 2
            assert report.checkpoint_id_range == ("cp-sub-1", "cp-sub-4")

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

    def _record_failed_tool_call(self, db_path: str, thread_id: str, error_text: str) -> None:
        """Persist one failed tool call (AI request plus error ToolMessage) for a thread."""
        with RelateSaver(db_path=db_path, wal_enabled=True) as saver:
            cfg: RunnableConfig = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
            cp: Any = dict(empty_checkpoint())
            cp["id"] = "cp-suffix-001"
            saved_cfg = saver.put(cfg, cp, {"step": 1, "source": "loop", "parents": {}}, {})
            ai = AIMessage(
                content="",
                tool_calls=[{"id": "call-sfx-1", "name": "fetch_source", "args": {}}],
            )
            saver.put_writes(saved_cfg, [("messages", [ai])], task_id="task-ai")
            tool_msg = ToolMessage(content=error_text, tool_call_id="call-sfx-1", status="error")
            saver.put_writes(saved_cfg, [("messages", [tool_msg])], task_id="task-tool")

    def test_trace_root_cause_drops_trailing_langgraph_suffix_ac_65(
        self, temp_disk_db: str
    ) -> None:
        """AC-65: the displayed error loses the trailing suffix; the stored error is unchanged."""
        stored = "Error: ValueError('boom')\n Please fix your mistakes."
        self._record_failed_tool_call(temp_disk_db, "thread-sfx-trim", stored)

        with RelateSaver(db_path=temp_disk_db, wal_enabled=True) as saver:
            report = saver.trace_root_cause("thread-sfx-trim")
            rows = saver.proxy.execute_read(
                "MATCH (tc:ToolCall {call_id: 'call-sfx-1'}) RETURN tc.error AS error", {}
            )

        assert report.error_message == "Error: ValueError('boom')"
        assert "Please fix your mistakes" not in report.diagnostic_summary
        assert rows[0]["error"] == stored

    def test_trace_root_cause_keeps_error_without_trailing_suffix_ac_66(
        self, temp_disk_db: str
    ) -> None:
        """AC-66: an error that only contains the phrase mid-text is displayed exactly."""
        stored = "Please fix your mistakes. The upstream API rejected the request"
        self._record_failed_tool_call(temp_disk_db, "thread-sfx-keep", stored)

        with RelateSaver(db_path=temp_disk_db, wal_enabled=True) as saver:
            report = saver.trace_root_cause("thread-sfx-keep")

        assert report.error_message == stored

    def test_trace_root_cause_long_next_chain_is_fast_ac_27(self, temp_disk_db: str) -> None:
        """AC-27: root cause on a 300-checkpoint NEXT chain finds the first antecedent quickly."""
        thread_id = "thread-long-chain-001"
        chain_length = 300
        with RelateSaver(db_path=temp_disk_db, wal_enabled=False) as saver:
            config: RunnableConfig = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
            first_id = ""
            for step in range(chain_length):
                cp: Any = dict(empty_checkpoint())
                cp["id"] = f"cp-long-{step:04d}"
                config = saver.put(config, cp, {"step": step, "source": "loop", "parents": {}}, {})
                if step == 0:
                    first_id = config["configurable"]["checkpoint_id"]

            saver.proxy.execute_write(
                "CREATE (tc:ToolCall {call_id: $call_id, tool_name: $name, args_blob: $args, "
                "error: $err, created_at: $created_at}) "
                "WITH tc "
                "MATCH (c:Checkpoint {checkpoint_id: $cid}) "
                "MERGE (c)-[:INVOKED]->(tc)",
                {
                    "call_id": "call-long-chain-1",
                    "name": "slow_api",
                    "args": b"{}",
                    "err": "HTTP 503: upstream unavailable",
                    "created_at": "2026-11-12T00:00:00Z",
                    "cid": first_id,
                },
            )
            first_cfg: RunnableConfig = {
                "configurable": {"thread_id": thread_id, "checkpoint_id": first_id}
            }
            saver.put_writes(
                first_cfg, [("poison", "upstream unavailable")], task_id="task-first-poison"
            )
            saver.put_writes(config, [("__error__", "Chain aborted")], task_id="task-long")

            started = time.perf_counter()
            report = saver.trace_root_cause(thread_id)
            elapsed = time.perf_counter() - started

        assert report.error_checkpoint_id == config["configurable"]["checkpoint_id"]
        assert (
            report.root_channel_write is not None
            and "task-first-poison" in report.root_channel_write
        )
        assert report.causal_path_length == chain_length - 1
        assert elapsed < 1.0


class ProvenanceState(TypedDict):
    """State for the argument provenance graph: a planner-chosen target and a message log."""

    target: str
    messages: Annotated[list[AnyMessage], add_messages]


PLANNED_TARGET = "pkg-alpha-9000"
FETCH_CALL_ID = "call-provenance-1"


@tool
def fetch_package(url: str, retries: int = 1) -> str:
    """Fetch a package; fails for every url unless it ends with 'ok'."""
    if not url.endswith("ok"):
        raise RuntimeError(f"no such package: {url}")
    return f"fetched {url} after {retries} tries"


def _run_provenance_graph(saver: RelateSaver, thread_id: str, target: str) -> None:
    """Run planner -> agent -> tools; the planner writes ``target`` and the tool consumes it."""

    def planner(state: ProvenanceState) -> dict[str, Any]:
        return {"target": target}

    def agent(state: ProvenanceState) -> dict[str, Any]:
        call = {
            "id": FETCH_CALL_ID,
            "name": "fetch_package",
            "args": {"url": state["target"], "retries": 5},
        }
        return {"messages": [AIMessage(content="", tool_calls=[call])]}

    builder = StateGraph(ProvenanceState)
    builder.add_node("planner", planner)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode([fetch_package], handle_tool_errors=True))
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "agent")
    builder.add_edge("agent", "tools")
    builder.add_edge("tools", END)
    graph = builder.compile(checkpointer=saver)
    graph.invoke(
        {"messages": [HumanMessage(content="please look the package up")], "target": ""},
        {"configurable": {"thread_id": thread_id}},
    )


class TestArgumentProvenance:
    """Argument provenance: which step first held a failing tool's argument (ADR-0018)."""

    THREAD = "thread-provenance"

    @staticmethod
    def _failure_checkpoint_id(saver: RelateSaver, thread_id: str) -> str:
        """Return the checkpoint whose writes hold the failing tool's ToolMessage."""
        rows = saver.proxy.execute_read(
            "MATCH (c:Checkpoint {thread_id: $t})-[:HAS_WRITE]->(w:Write) "
            "WHERE w.channel = 'messages' RETURN c.checkpoint_id AS id, c.step AS step, "
            "w.write_blob AS blob ORDER BY c.step",
            {"t": thread_id},
        )
        for row in rows:
            value = saver.serde.loads(bytes(row["blob"]))
            if any(isinstance(m, ToolMessage) for m in value):
                return str(row["id"])
        raise AssertionError("no ToolMessage write found")

    def test_origin_names_node_step_and_hops_of_earliest_write_ac_119(self) -> None:
        """AC-119: the planner's write, not the later model message, is the origin."""
        with RelateSaver(":memory:") as saver:
            _run_provenance_graph(saver, self.THREAD, PLANNED_TARGET)
            failure_cp = self._failure_checkpoint_id(saver, self.THREAD)

            with RelateForensics(saver) as forensics:
                origin = forensics.trace_argument_origin(
                    self.THREAD, failure_cp, {"url": PLANNED_TARGET, "retries": 5}
                )

        assert origin is not None
        assert origin.argument == "url"
        assert origin.value == PLANNED_TARGET
        assert origin.node == "planner"
        # The planner ran from the step 0 checkpoint; the tool ran from the step 2 checkpoint.
        assert origin.step == 0
        assert origin.hops == 2

    def test_origin_is_case_sensitive_ac_119(self) -> None:
        """AC-119: containment is case-sensitive, so a different case finds nothing."""
        with RelateSaver(":memory:") as saver:
            _run_provenance_graph(saver, self.THREAD, PLANNED_TARGET)
            failure_cp = self._failure_checkpoint_id(saver, self.THREAD)

            with RelateForensics(saver) as forensics:
                origin = forensics.trace_argument_origin(
                    self.THREAD, failure_cp, {"url": PLANNED_TARGET.upper()}
                )

        assert origin is None

    @pytest.mark.parametrize(
        "args",
        [
            {"retries": 5, "flag": True, "items": ["pkg-alpha-9000"]},
            {"url": "pk"},
            {"url": "never-written-anywhere"},
            {},
        ],
        ids=["non-string", "shorter-than-3", "untraceable", "empty"],
    )
    def test_untraceable_arguments_return_none_ac_120(self, args: dict[str, Any]) -> None:
        """AC-120: non-string, short and untraceable arguments give no origin."""
        with RelateSaver(":memory:") as saver:
            _run_provenance_graph(saver, self.THREAD, PLANNED_TARGET)
            failure_cp = self._failure_checkpoint_id(saver, self.THREAD)

            with RelateForensics(saver) as forensics:
                assert forensics.trace_argument_origin(self.THREAD, failure_cp, args) is None

    def test_invalid_identifiers_are_rejected_ac_120(self) -> None:
        """AC-120: an empty thread or checkpoint id raises instead of returning a guess."""
        with RelateSaver(":memory:") as saver, RelateForensics(saver) as forensics:
            with pytest.raises(CheckpointNotFoundError):
                forensics.trace_argument_origin("", "cp-1", {"url": "abcdef"})
            with pytest.raises(CheckpointNotFoundError):
                forensics.trace_argument_origin("thread", " ", {"url": "abcdef"})

    def test_trace_root_cause_fills_origin_for_failing_tool_ac_121(self) -> None:
        """AC-121: the report names the argument, node, step and hops from the failing call."""
        with RelateSaver(":memory:") as saver:
            _run_provenance_graph(saver, self.THREAD, PLANNED_TARGET)

            with RelateForensics(saver) as forensics:
                report = forensics.trace_root_cause(self.THREAD)

        assert report.root_tool_call_id == FETCH_CALL_ID
        assert report.origin_argument == "url"
        assert report.origin_node == "planner"
        assert report.origin_step == 0
        assert report.origin_hops == 2
        assert "planner" not in report.diagnostic_summary

    def test_trace_root_cause_leaves_origin_empty_without_failure_ac_121(self) -> None:
        """AC-121: a thread whose tool call succeeds has no origin fields."""
        with RelateSaver(":memory:") as saver:
            _run_provenance_graph(saver, self.THREAD, "pkg-ok")

            with RelateForensics(saver) as forensics:
                report = forensics.trace_root_cause(self.THREAD)

        assert report.error_message == "No error detected"
        assert report.origin_argument is None
        assert report.origin_node is None
        assert report.origin_step is None
        assert report.origin_hops is None

    def test_ac121_root_cause_resolves_the_failure_checkpoint_once(self) -> None:
        """AC-121: one root-cause trace anchors on the failure checkpoint once (ADR-0021)."""
        original = RelateForensics._failure_checkpoint_id
        with RelateSaver(":memory:") as saver:
            _run_provenance_graph(saver, self.THREAD, PLANNED_TARGET)

            with RelateForensics(saver) as forensics:
                with patch.object(
                    RelateForensics,
                    "_failure_checkpoint_id",
                    autospec=True,
                    side_effect=original,
                ) as anchor:
                    report = forensics.trace_root_cause(self.THREAD)

        assert report.root_tool_call_id == FETCH_CALL_ID
        assert report.origin_node == "planner"
        assert anchor.call_count == 1


@tool
def parse_manifest(url: str) -> str:
    """Parse a manifest; always fails."""
    raise RuntimeError(f"cannot parse manifest at {url}")


@tool
def alpha_check(url: str) -> str:
    """Check a package; always fails."""
    raise RuntimeError(f"check failed for {url}")


@tool
def echo_value(url: str) -> str:
    """Echo the url; never fails."""
    return f"echo {url}"


SUMMARY_TOOLS = [fetch_package, parse_manifest, alpha_check, echo_value]


def _run_summary_thread(
    saver: RelateSaver, thread_id: str, call_id: str, tool_name: str, target: str
) -> None:
    """Run planner -> agent -> tools where the agent calls ``tool_name`` with the planned target."""

    def planner(state: ProvenanceState) -> dict[str, Any]:
        return {"target": target}

    def agent(state: ProvenanceState) -> dict[str, Any]:
        call = {"id": call_id, "name": tool_name, "args": {"url": state["target"]}}
        return {"messages": [AIMessage(content="", tool_calls=[call])]}

    builder = StateGraph(ProvenanceState)
    builder.add_node("planner", planner)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(SUMMARY_TOOLS, handle_tool_errors=True))
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "agent")
    builder.add_edge("agent", "tools")
    builder.add_edge("tools", END)
    builder.compile(checkpointer=saver).invoke(
        {"messages": [HumanMessage(content="please look it up")], "target": ""},
        {"configurable": {"thread_id": thread_id}},
    )


class TestFailureSummary:
    """Tool failures summarized across every thread of a store (AC-123)."""

    def test_summary_counts_threads_origins_and_order_ac_123(self) -> None:
        """AC-123: one entry per failing tool, ordered by failures then name, ok tools omitted."""
        with RelateSaver(":memory:") as saver:
            _run_summary_thread(saver, "t-a", "call-a", "fetch_package", "pkg-alpha-9000")
            _run_summary_thread(saver, "t-b", "call-b", "fetch_package", "pkg-beta-9000")
            _run_summary_thread(saver, "t-c", "call-c", "fetch_package", "pkg-ok")
            _run_summary_thread(saver, "t-d", "call-d", "parse_manifest", "manifest-url-1")
            _run_summary_thread(saver, "t-e", "call-e", "alpha_check", "check-target-1")
            _run_summary_thread(saver, "t-f", "call-f", "echo_value", "echo-target-1")

            with RelateForensics(saver) as forensics:
                summary = forensics.failure_summary()

        assert [s.tool_name for s in summary] == ["fetch_package", "alpha_check", "parse_manifest"]
        fetch, alpha, manifest = summary
        assert (fetch.failures, fetch.total_calls) == (2, 3)
        assert fetch.threads == ("t-a", "t-b")
        assert fetch.origin_nodes == ("planner",)
        assert (alpha.failures, alpha.total_calls, alpha.threads) == (1, 1, ("t-e",))
        assert alpha.origin_nodes == ("planner",)
        assert (manifest.failures, manifest.total_calls, manifest.threads) == (1, 1, ("t-d",))
        assert manifest.origin_nodes == ("planner",)

    def test_summary_is_empty_without_failures_ac_123(self) -> None:
        """AC-123: a store whose tool calls all succeed, and an empty store, give no entries."""
        with RelateSaver(":memory:") as saver:
            with RelateForensics(saver) as forensics:
                assert forensics.failure_summary() == []
            _run_summary_thread(saver, "t-ok", "call-ok", "echo_value", "echo-target-1")
            with RelateForensics(saver) as forensics:
                assert forensics.failure_summary() == []

    def test_ac158_failure_summary_counts_calls_linked_through_a_write(self) -> None:
        """AC-158/AC-123: a tool call linked only by (:Write)-[:INVOKED] is summarized too."""
        thread_id, call_id = "t-write", "call-write"
        with RelateSaver(":memory:") as saver:
            _run_summary_thread(saver, thread_id, call_id, "parse_manifest", "manifest-url-1")
            rows = saver.proxy.execute_read(
                "MATCH (c:Checkpoint)-[:INVOKED]->(tc:ToolCall {call_id: $call_id}) "
                "RETURN c.checkpoint_id AS checkpoint_id",
                {"call_id": call_id},
            )
            assert rows, "the run must record the failing call on a checkpoint"
            checkpoint_id = str(rows[0]["checkpoint_id"])

            # Move the link from the checkpoint to one of its writes (the reserved edge).
            saver.proxy.execute_write(
                "MATCH (:Checkpoint)-[r:INVOKED]->(:ToolCall {call_id: $call_id}) DELETE r",
                {"call_id": call_id},
            )
            for write_id, channel in (("w-reserved", "results"), ("w-error", "__error__")):
                saver.proxy.execute_write(
                    "MATCH (c:Checkpoint {checkpoint_id: $cid}) "
                    "CREATE (w:Write {write_id: $write_id, thread_id: $thread_id, "
                    "  checkpoint_ns: '', checkpoint_id: $cid, task_id: 'task-w', "
                    "  channel: $channel, idx: 0}) "
                    "CREATE (c)-[:HAS_WRITE]->(w)",
                    {
                        "cid": checkpoint_id,
                        "write_id": write_id,
                        "thread_id": thread_id,
                        "channel": channel,
                    },
                )
            saver.proxy.execute_write(
                "MATCH (w:Write {write_id: 'w-reserved'}), (tc:ToolCall {call_id: $call_id}) "
                "CREATE (w)-[:INVOKED]->(tc)",
                {"call_id": call_id},
            )

            with RelateForensics(saver) as forensics:
                summary = forensics.failure_summary()
                report = forensics.trace_root_cause(thread_id)

        assert [(s.tool_name, s.failures, s.total_calls, s.threads) for s in summary] == [
            ("parse_manifest", 1, 1, (thread_id,))
        ]
        assert report.root_tool_call_id == call_id
        assert report.root_tool_name == summary[0].tool_name

    def test_ac163_trace_root_cause_detects_write_linked_failed_tool_call(self) -> None:
        """AC-163: a thread whose only failed ToolCall is Write-linked is detected by trace_root_cause."""
        thread_id, call_id = "t-write-fail", "call-write-fail"
        with RelateSaver(":memory:") as saver:
            _run_summary_thread(saver, thread_id, call_id, "parse_manifest", "manifest-url-1")
            rows = saver.proxy.execute_read(
                "MATCH (c:Checkpoint)-[:INVOKED]->(tc:ToolCall {call_id: $call_id}) "
                "RETURN c.checkpoint_id AS checkpoint_id",
                {"call_id": call_id},
            )
            assert rows, "the run must record the failing call on a checkpoint"
            checkpoint_id = str(rows[0]["checkpoint_id"])

            # Move the link from the checkpoint to a Write node (without creating any __error__ write).
            saver.proxy.execute_write(
                "MATCH (:Checkpoint)-[r:INVOKED]->(:ToolCall {call_id: $call_id}) DELETE r",
                {"call_id": call_id},
            )
            saver.proxy.execute_write(
                "MATCH (c:Checkpoint {checkpoint_id: $cid}) "
                "CREATE (w:Write {write_id: 'w-task', thread_id: $thread_id, "
                "  checkpoint_ns: '', checkpoint_id: $cid, task_id: 'task-w', "
                "  channel: 'results', idx: 0}) "
                "CREATE (c)-[:HAS_WRITE]->(w)",
                {
                    "cid": checkpoint_id,
                    "thread_id": thread_id,
                },
            )
            saver.proxy.execute_write(
                "MATCH (w:Write {write_id: 'w-task'}), (tc:ToolCall {call_id: $call_id}) "
                "CREATE (w)-[:INVOKED]->(tc)",
                {"call_id": call_id},
            )

            # Confirm no __error__ write exists for this thread
            error_writes = saver.proxy.execute_read(
                "MATCH (c:Checkpoint {thread_id: $thread_id})-[:HAS_WRITE]->(w:Write) "
                "WHERE w.channel = '__error__' RETURN w",
                {"thread_id": thread_id},
            )
            assert not error_writes

            with RelateForensics(saver) as forensics:
                report = forensics.trace_root_cause(thread_id)
                summary = forensics.failure_summary()

        assert report.root_tool_call_id == call_id
        assert report.root_tool_name == "parse_manifest"
        assert report.error_checkpoint_id
        assert [(s.tool_name, s.failures, s.total_calls, s.threads) for s in summary] == [
            ("parse_manifest", 1, 1, (thread_id,))
        ]
        assert report.root_tool_name == summary[0].tool_name

    def test_ac171_root_cause_names_failing_write_linked_call_not_direct_ok_call(self) -> None:
        """AC-171: a checkpoint with a direct ok tool call and write-linked failing tool call."""
        thread_id, call_id = "t-ac171-fail", "call-ac171-fail"
        with RelateSaver(":memory:") as saver:
            _run_summary_thread(saver, thread_id, call_id, "parse_manifest", "manifest-url-1")
            rows = saver.proxy.execute_read(
                "MATCH (c:Checkpoint)-[:INVOKED]->(tc:ToolCall {call_id: $call_id}) "
                "RETURN c.checkpoint_id AS checkpoint_id",
                {"call_id": call_id},
            )
            assert rows, "the run must record the failing call on a checkpoint"
            checkpoint_id = str(rows[0]["checkpoint_id"])

            # Move the link from the checkpoint to a Write node (without creating any __error__ write).
            saver.proxy.execute_write(
                "MATCH (:Checkpoint)-[r:INVOKED]->(:ToolCall {call_id: $call_id}) DELETE r",
                {"call_id": call_id},
            )
            saver.proxy.execute_write(
                "MATCH (c:Checkpoint {checkpoint_id: $cid}) "
                "CREATE (w:Write {write_id: 'w-ac171', thread_id: $thread_id, "
                "  checkpoint_ns: '', checkpoint_id: $cid, task_id: 'task-w', "
                "  channel: 'results', idx: 0}) "
                "CREATE (c)-[:HAS_WRITE]->(w)",
                {
                    "cid": checkpoint_id,
                    "thread_id": thread_id,
                },
            )
            saver.proxy.execute_write(
                "MATCH (w:Write {write_id: 'w-ac171'}), (tc:ToolCall {call_id: $call_id}) "
                "CREATE (w)-[:INVOKED]->(tc)",
                {"call_id": call_id},
            )
            # Create a successful :ToolCall {call_id: 'call-ok', tool_name: 'ok_tool'} (no error)
            # and link it directly to the checkpoint.
            saver.proxy.execute_write(
                "MATCH (c:Checkpoint {checkpoint_id: $cid}) "
                "CREATE (tc_ok:ToolCall {call_id: 'call-ok', tool_name: 'ok_tool'}) "
                "CREATE (c)-[:INVOKED]->(tc_ok)",
                {"cid": checkpoint_id},
            )

            with RelateForensics(saver) as forensics:
                report = forensics.trace_root_cause(thread_id)

        assert report.root_tool_call_id == call_id
        assert report.root_tool_name == "parse_manifest"

    def test_ac164_failure_summary_fanout_origin_nodes_listed_separately(self) -> None:
        """AC-164: a fan-out step with multiple concurrent writing nodes lists origin nodes separately."""

        class FanoutState(TypedDict):
            target: str
            extra: str
            messages: list[Any]

        def fetcher(state: FanoutState) -> dict[str, Any]:
            return {"target": "manifest-url-1"}

        def parser(state: FanoutState) -> dict[str, Any]:
            return {"extra": "parsed"}

        def agent(state: FanoutState) -> dict[str, Any]:
            call = {"id": "call-fanout", "name": "parse_manifest", "args": {"url": state["target"]}}
            return {"messages": [AIMessage(content="", tool_calls=[call])]}

        builder = StateGraph(FanoutState)
        builder.add_node("fetcher", fetcher)
        builder.add_node("parser", parser)
        builder.add_node("agent", agent)
        builder.add_node("tools", ToolNode(SUMMARY_TOOLS, handle_tool_errors=True))
        builder.add_edge(START, "fetcher")
        builder.add_edge(START, "parser")
        builder.add_edge("fetcher", "agent")
        builder.add_edge("parser", "agent")
        builder.add_edge("agent", "tools")
        builder.add_edge("tools", END)

        with RelateSaver(":memory:") as saver:
            builder.compile(checkpointer=saver).invoke(
                {"messages": [HumanMessage(content="go")], "target": "", "extra": ""},
                {"configurable": {"thread_id": "t-fanout"}},
            )
            with RelateForensics(saver) as forensics:
                summary = forensics.failure_summary()
                report = forensics.trace_root_cause("t-fanout")
                origin = forensics.trace_argument_origin(
                    "t-fanout", report.error_checkpoint_id, {"url": "manifest-url-1"}
                )

        assert len(summary) == 1
        # AC-164/AC-172: only the fetcher node wrote the argument, parser did not write it
        assert summary[0].origin_nodes == ("fetcher",)
        # trace_argument_origin(...).node remains a joined string (unchanged public behavior)
        assert origin is not None
        assert origin.node == "fetcher"
        assert report.origin_node == "fetcher"

    def test_ac164_both_fanout_writers_of_the_argument_are_listed(self) -> None:
        """AC-172: when multiple fan-out nodes write the argument, all are listed in origin nodes."""

        class FanoutState(TypedDict):
            target: str
            extra: str
            messages: list[Any]

        def fetcher(state: FanoutState) -> dict[str, Any]:
            return {"target": "manifest-url-1"}

        def parser(state: FanoutState) -> dict[str, Any]:
            return {"extra": "manifest-url-1"}

        def agent(state: FanoutState) -> dict[str, Any]:
            call = {"id": "call-fanout", "name": "parse_manifest", "args": {"url": state["target"]}}
            return {"messages": [AIMessage(content="", tool_calls=[call])]}

        builder = StateGraph(FanoutState)
        builder.add_node("fetcher", fetcher)
        builder.add_node("parser", parser)
        builder.add_node("agent", agent)
        builder.add_node("tools", ToolNode(SUMMARY_TOOLS, handle_tool_errors=True))
        builder.add_edge(START, "fetcher")
        builder.add_edge(START, "parser")
        builder.add_edge("fetcher", "agent")
        builder.add_edge("parser", "agent")
        builder.add_edge("agent", "tools")
        builder.add_edge("tools", END)

        with RelateSaver(":memory:") as saver:
            builder.compile(checkpointer=saver).invoke(
                {"messages": [HumanMessage(content="go")], "target": "", "extra": ""},
                {"configurable": {"thread_id": "t-fanout-both"}},
            )
            with RelateForensics(saver) as forensics:
                summary = forensics.failure_summary()
                report = forensics.trace_root_cause("t-fanout-both")
                origin = forensics.trace_argument_origin(
                    "t-fanout-both", report.error_checkpoint_id, {"url": "manifest-url-1"}
                )

        assert len(summary) == 1
        assert summary[0].origin_nodes == ("fetcher", "parser")
        assert origin is not None
        assert origin.node == "fetcher, parser"
        assert report.origin_node == "fetcher, parser"

    def test_ac164_failure_summary_origin_nodes_unknown_when_not_found(self) -> None:
        """AC-164: when origin writing nodes cannot be determined, '<unknown>' is reported."""
        thread_id, call_id = "t-unknown-node", "call-unknown-node"
        with RelateSaver(":memory:") as saver:
            _run_summary_thread(saver, thread_id, call_id, "parse_manifest", "manifest-url-1")
            with RelateForensics(saver) as forensics:
                # Monkeypatch _writing_nodes to return empty tuple (unknown)
                forensics._writing_nodes = lambda cid, tid: ()  # type: ignore[assignment]
                summary = forensics.failure_summary()

        assert len(summary) == 1
        assert summary[0].origin_nodes == ("<unknown>",)

    def test_ac181_in_thread_fork_traverses_forked_to_for_root_cause_and_argument_origin(
        self, temp_disk_db: str
    ) -> None:
        """AC-181: Lineage walk traverses across in-thread FORKED_TO edges."""
        thread_id = "thread-ac181-fork"
        with RelateSaver(db_path=temp_disk_db, wal_enabled=True) as saver:
            # Step 0: Root checkpoint with tool call and argument write
            cid_root = "cp-ac181-root"
            cfg_root: RunnableConfig = {
                "configurable": {"thread_id": thread_id, "checkpoint_id": cid_root}
            }
            cp_root: Any = dict(empty_checkpoint())
            cp_root["id"] = cid_root
            saver.put(cfg_root, cp_root, {"step": 0}, {})
            saver.put_writes(cfg_root, [("target", "val-forked-secret")], task_id="task-root")

            saver.proxy.execute_write(
                "CREATE (tc:ToolCall {call_id: $call_id, tool_name: $name, args_blob: $args, "
                "error: $err, created_at: $created_at}) "
                "WITH tc "
                "MATCH (c:Checkpoint {checkpoint_id: $cid}) "
                "MERGE (c)-[:INVOKED]->(tc)",
                {
                    "call_id": "call-root-fail-181",
                    "name": "root_api",
                    "args": b'{"target": "val-forked-secret"}',
                    "err": "Connection reset by peer",
                    "created_at": "2026-11-22T00:00:00Z",
                    "cid": cid_root,
                },
            )

            # Step 1: Linear child cp_main (via NEXT)
            cid_main = "cp-ac181-main"
            cfg_main: RunnableConfig = {
                "configurable": {"thread_id": thread_id, "checkpoint_id": cid_main}
            }
            cp_main: Any = dict(empty_checkpoint())
            cp_main["id"] = cid_main
            cp_main["parent_checkpoint_id"] = cid_root
            saver.put(cfg_main, cp_main, {"step": 1}, {})

            # Step 1: Fork child cp_fork from root (via FORKED_TO in same thread)
            cid_fork = "cp-ac181-fork"
            cfg_fork: RunnableConfig = {
                "configurable": {"thread_id": thread_id, "checkpoint_id": cid_fork}
            }
            cp_fork: Any = dict(empty_checkpoint())
            cp_fork["id"] = cid_fork
            cp_fork["parent_checkpoint_id"] = cid_root
            saver.put(cfg_fork, cp_fork, {"step": 1}, {})

            # Write tool response holding ToolMessage at cp_fork
            tool_msg = ToolMessage(
                content="failed result", tool_call_id="call-root-fail-181", status="error"
            )
            saver.put_writes(cfg_fork, [("messages", [tool_msg])], task_id="task-tool-res")

            # Step 2: Child cp_sub in the fork branch (via NEXT from cp_fork)
            cid_sub = "cp-ac181-sub"
            cfg_sub: RunnableConfig = {
                "configurable": {"thread_id": thread_id, "checkpoint_id": cid_sub}
            }
            cp_sub: Any = dict(empty_checkpoint())
            cp_sub["id"] = cid_sub
            cp_sub["parent_checkpoint_id"] = cid_fork
            saver.put(cfg_sub, cp_sub, {"step": 2}, {})
            saver.put_writes(
                cfg_sub, [("__error__", "Error downstream in branch")], task_id="task-err"
            )

            # Verify FORKED_TO exists between root and fork checkpoint
            fork_rels = saver.proxy.execute_read(
                "MATCH (p:Checkpoint {checkpoint_id: $p_id})-[r:FORKED_TO]->(c:Checkpoint {checkpoint_id: $c_id}) "
                "RETURN count(r) AS count",
                {"p_id": cid_root, "c_id": cid_fork},
            )
            assert fork_rels[0]["count"] == 1

            with RelateForensics(saver) as forensics:
                # trace_argument_origin walks back across NEXT then FORKED_TO to root
                origin = forensics.trace_argument_origin(
                    thread_id, cid_sub, {"target": "val-forked-secret"}
                )
                assert origin is not None
                assert origin.argument == "target"
                assert origin.value == "val-forked-secret"
                assert origin.checkpoint_id == cid_root
                assert origin.hops == 2

                # _failure_checkpoint_id traverses forward across FORKED_TO to cp_fork
                fail_cp = forensics._failure_checkpoint_id(
                    thread_id, cid_root, "call-root-fail-181"
                )
                assert fail_cp == cid_fork

                # Under AC-184, distant tool call at distance 2 is not blamed on the __error__ path;
                # the nearest antecedent write at distance 1 is identified instead.
                report = forensics.trace_root_cause(thread_id)
                assert report.error_checkpoint_id == cid_sub
                assert (
                    report.root_channel_write is not None
                    and "task-tool-res" in report.root_channel_write
                )
                assert report.root_tool_call_id is None
                assert report.causal_path_length == 1

    def test_ac181_cross_thread_branch_isolated_from_lineage_walk(self, temp_disk_db: str) -> None:
        """AC-181: Lineage walk strictly excludes checkpoints belonging to other thread_ids."""
        thread_t1 = "thread-ac181-t1"
        thread_t2 = "thread-ac181-t2"
        with RelateSaver(db_path=temp_disk_db, wal_enabled=True) as saver:
            # Thread 1: root checkpoint
            cid_t1_root = "cp-ac181-t1-root"
            cfg_t1_root: RunnableConfig = {
                "configurable": {"thread_id": thread_t1, "checkpoint_id": cid_t1_root}
            }
            cp_t1_root: Any = dict(empty_checkpoint())
            cp_t1_root["id"] = cid_t1_root
            saver.put(cfg_t1_root, cp_t1_root, {"step": 0}, {})
            saver.put_writes(cfg_t1_root, [("payload", "t1-secret-data")], task_id="task-t1-root")

            # Thread 1: step 1 checkpoint with error
            cid_t1_err = "cp-ac181-t1-err"
            cfg_t1_err: RunnableConfig = {
                "configurable": {"thread_id": thread_t1, "checkpoint_id": cid_t1_err}
            }
            cp_t1_err: Any = dict(empty_checkpoint())
            cp_t1_err["id"] = cid_t1_err
            cp_t1_err["parent_checkpoint_id"] = cid_t1_root
            saver.put(cfg_t1_err, cp_t1_err, {"step": 1}, {})
            saver.put_writes(cfg_t1_err, [("__error__", "t1 failure")], task_id="task-t1-err")

            # Thread 2: branches from Thread 1's root (cross-thread FORKED_TO)
            cid_t2_root = "cp-ac181-t2-root"
            cfg_t2_root: RunnableConfig = {
                "configurable": {"thread_id": thread_t2, "checkpoint_id": cid_t2_root}
            }
            cp_t2_root: Any = dict(empty_checkpoint())
            cp_t2_root["id"] = cid_t2_root
            cp_t2_root["parent_checkpoint_id"] = cid_t1_root
            saver.put(cfg_t2_root, cp_t2_root, {"step": 1}, {})
            tool_msg = ToolMessage(
                content="t2 result", tool_call_id="call-root-fail-181", status="error"
            )
            saver.put_writes(
                cfg_t2_root,
                [
                    ("payload", "t2-secret-data"),
                    ("__error__", "t2 failure"),
                    ("messages", [tool_msg]),
                ],
                task_id="task-t2",
            )

            # Verify cross-thread FORKED_TO edge was created
            cross_rels = saver.proxy.execute_read(
                "MATCH (p:Checkpoint {thread_id: $t1})-[r:FORKED_TO]->(c:Checkpoint {thread_id: $t2}) "
                "RETURN count(r) AS count",
                {"t1": thread_t1, "t2": thread_t2},
            )
            assert cross_rels[0]["count"] == 1

            with RelateForensics(saver) as forensics:
                # Thread 1's lineage walk does NOT find t2's payload
                t1_origin = forensics.trace_argument_origin(
                    thread_t1, cid_t1_err, {"payload": "t2-secret-data"}
                )
                assert t1_origin is None

                # Thread 1 finds its own payload
                t1_own_origin = forensics.trace_argument_origin(
                    thread_t1, cid_t1_err, {"payload": "t1-secret-data"}
                )
                assert t1_own_origin is not None
                assert t1_own_origin.checkpoint_id == cid_t1_root

                # Thread 1's _failure_checkpoint_id does NOT traverse into Thread 2
                fail_cp_t1 = forensics._failure_checkpoint_id(
                    thread_t1, cid_t1_root, "call-root-fail-181"
                )
                assert fail_cp_t1 == cid_t1_root

                # Thread 2's lineage walk does NOT walk backwards across FORKED_TO into Thread 1
                t2_leak_origin = forensics.trace_argument_origin(
                    thread_t2, cid_t2_root, {"payload": "t1-secret-data"}
                )
                assert t2_leak_origin is None

                # Thread 2 finds its own payload
                t2_own_origin = forensics.trace_argument_origin(
                    thread_t2, cid_t2_root, {"payload": "t2-secret-data"}
                )
                assert t2_own_origin is not None
                assert t2_own_origin.checkpoint_id == cid_t2_root

                # Invalid input: empty thread_id raises CheckpointNotFoundError
                with pytest.raises(CheckpointNotFoundError):
                    forensics.trace_argument_origin("", cid_t1_root, {"payload": "any"})

                # Unknown thread raises CheckpointNotFoundError for trace_root_cause
                with pytest.raises(CheckpointNotFoundError):
                    forensics.trace_root_cause("nonexistent-thread-id")


class TestPoisonDetection:
    """Poison pill detection matching channel names exclusively (AC-183, ADR-0029)."""

    @pytest.mark.parametrize(
        "value_text",
        ["default value", "fault diagnosis", "a default fallback with fault logging"],
    )
    def test_ac183_non_poison_channel_with_fault_substring_not_flagged_as_poison(
        self, temp_disk_db: str, value_text: str
    ) -> None:
        """AC-183: Substring matches for 'poison' and 'fault' in serialized write values are not performed.

        A non-poison channel with 'default value' or 'fault diagnosis' is not flagged as
        a poison root cause, allowing upstream tool calls to be properly identified as root cause.
        """
        thread_id = f"thread-ac183-clean-{abs(hash(value_text))}"
        with RelateSaver(db_path=temp_disk_db, wal_enabled=True) as saver:
            # Step 1: Antecedent write on a benign channel containing "default value" or "fault diagnosis"
            cid1 = "cp-ac183-clean-1"
            cfg1: RunnableConfig = {"configurable": {"thread_id": thread_id, "checkpoint_id": cid1}}
            cp1: Any = dict(empty_checkpoint())
            cp1["id"] = cid1
            saver.put(cfg1, cp1, {"step": 1, "node": "config_loader"}, {})  # type: ignore[typeddict-unknown-key]
            saver.put_writes(
                cfg1,
                [("config", value_text)],
                task_id="task-config",
            )

            # Step 2: Intermediate step with an upstream tool invocation (no explicit error)
            cid2 = "cp-ac183-clean-2"
            cfg2: RunnableConfig = {"configurable": {"thread_id": thread_id, "checkpoint_id": cid2}}
            cp2: Any = dict(empty_checkpoint())
            cp2["id"] = cid2
            cp2["parent_checkpoint_id"] = cid1
            saver.put(cfg2, cp2, {"step": 2, "node": "tool_runner"}, {})  # type: ignore[typeddict-unknown-key]

            saver.proxy.execute_write(
                "CREATE (tc:ToolCall {call_id: $call_id, tool_name: $name, args_blob: $args, created_at: $created_at}) "
                "WITH tc "
                "MATCH (c:Checkpoint {checkpoint_id: $cid}) "
                "MERGE (c)-[:INVOKED]->(tc)",
                {
                    "call_id": "call-upstream-tool-ac183",
                    "name": "upstream_tool",
                    "args": b"{}",
                    "created_at": "2026-11-22T00:00:00Z",
                    "cid": cid2,
                },
            )

            # Step 3: Failure checkpoint where error is recorded
            cid3 = "cp-ac183-clean-3"
            cfg3: RunnableConfig = {"configurable": {"thread_id": thread_id, "checkpoint_id": cid3}}
            cp3: Any = dict(empty_checkpoint())
            cp3["id"] = cid3
            cp3["parent_checkpoint_id"] = cid2
            saver.put(cfg3, cp3, {"step": 3, "node": "failing_step"}, {})  # type: ignore[typeddict-unknown-key]
            saver.put_writes(
                cfg3,
                [("__error__", "Failure occurred downstream")],
                task_id="task-err",
            )

            with RelateForensics(saver) as forensics:
                report = forensics.trace_root_cause(thread_id)

            # AC-183: The write with 'default value' or 'fault diagnosis' is NOT flagged as poison.
            # Instead, Priority 3 upstream tool invocation is correctly identified.
            assert report.root_tool_call_id == "call-upstream-tool-ac183"
            assert report.root_tool_name == "upstream_tool"
            assert report.root_channel_write is None

    @pytest.mark.parametrize(
        "channel_name",
        ["poison", "poison_channel", "sys_poison_input", "POISON"],
    )
    def test_ac183_poison_channel_name_flagged_as_poison_root_cause(
        self, temp_disk_db: str, channel_name: str
    ) -> None:
        """AC-183: Poison checks match exclusively against the channel name (case-insensitively).

        A write on a channel named 'poison' (or containing 'poison') is flagged as
        a poison root cause (Priority 2), taking precedence over upstream tool invocations.
        """
        thread_id = f"thread-ac183-poison-{abs(hash(channel_name))}"
        with RelateSaver(db_path=temp_disk_db, wal_enabled=True) as saver:
            # Step 1: Write on a channel whose name contains "poison"
            cid1 = "cp-ac183-p-1"
            cfg1: RunnableConfig = {"configurable": {"thread_id": thread_id, "checkpoint_id": cid1}}
            cp1: Any = dict(empty_checkpoint())
            cp1["id"] = cid1
            saver.put(cfg1, cp1, {"step": 1, "node": "injector"}, {})  # type: ignore[typeddict-unknown-key]
            saver.put_writes(
                cfg1,
                [(channel_name, "arbitrary benign content")],
                task_id="task-poison",
            )

            # Step 2: Intermediate step with an upstream tool invocation (no explicit error)
            cid2 = "cp-ac183-p-2"
            cfg2: RunnableConfig = {"configurable": {"thread_id": thread_id, "checkpoint_id": cid2}}
            cp2: Any = dict(empty_checkpoint())
            cp2["id"] = cid2
            cp2["parent_checkpoint_id"] = cid1
            saver.put(cfg2, cp2, {"step": 2, "node": "tool_runner"}, {})  # type: ignore[typeddict-unknown-key]

            saver.proxy.execute_write(
                "CREATE (tc:ToolCall {call_id: $call_id, tool_name: $name, args_blob: $args, created_at: $created_at}) "
                "WITH tc "
                "MATCH (c:Checkpoint {checkpoint_id: $cid}) "
                "MERGE (c)-[:INVOKED]->(tc)",
                {
                    "call_id": "call-upstream-tool-poison-test",
                    "name": "upstream_tool",
                    "args": b"{}",
                    "created_at": "2026-11-22T00:00:00Z",
                    "cid": cid2,
                },
            )

            # Step 3: Failure checkpoint
            cid3 = "cp-ac183-p-3"
            cfg3: RunnableConfig = {"configurable": {"thread_id": thread_id, "checkpoint_id": cid3}}
            cp3: Any = dict(empty_checkpoint())
            cp3["id"] = cid3
            cp3["parent_checkpoint_id"] = cid2
            saver.put(cfg3, cp3, {"step": 3, "node": "failing_step"}, {})  # type: ignore[typeddict-unknown-key]
            saver.put_writes(
                cfg3,
                [("__error__", "Failure occurred downstream")],
                task_id="task-err",
            )

            with RelateForensics(saver) as forensics:
                report = forensics.trace_root_cause(thread_id)

            # AC-183: The channel whose name contains 'poison' IS flagged as poison root cause (Priority 2),
            # preempting the Priority 3 upstream tool call.
            assert report.root_channel_write is not None
            assert report.root_tool_call_id is None
            assert report.causal_path_length == 2

    def test_ac183_empty_or_whitespace_channel_not_flagged(self, temp_disk_db: str) -> None:
        """AC-183: Empty or whitespace channel names are safely handled and not flagged as poison."""
        thread_id = "thread-ac183-empty-chan"
        with RelateSaver(db_path=temp_disk_db, wal_enabled=True) as saver:
            cid1 = "cp-ac183-empty-1"
            cfg1: RunnableConfig = {"configurable": {"thread_id": thread_id, "checkpoint_id": cid1}}
            cp1: Any = dict(empty_checkpoint())
            cp1["id"] = cid1
            saver.put(cfg1, cp1, {"step": 1, "node": "step1"}, {})  # type: ignore[typeddict-unknown-key]
            saver.put_writes(
                cfg1,
                [("", "empty channel text"), ("   ", "whitespace channel text")],
                task_id="task-empty",
            )

            cid2 = "cp-ac183-empty-2"
            cfg2: RunnableConfig = {"configurable": {"thread_id": thread_id, "checkpoint_id": cid2}}
            cp2: Any = dict(empty_checkpoint())
            cp2["id"] = cid2
            cp2["parent_checkpoint_id"] = cid1
            saver.put(cfg2, cp2, {"step": 2, "node": "step2"}, {})  # type: ignore[typeddict-unknown-key]
            saver.put_writes(
                cfg2,
                [("__error__", "Crash at step 2")],
                task_id="task-err",
            )

            with RelateForensics(saver) as forensics:
                report = forensics.trace_root_cause(thread_id)

            assert report.thread_id == thread_id
            assert report.error_checkpoint_id == cid2

    def test_ac184_tool_call_error_at_distance_1_attributed_on_error_write_path(
        self, temp_disk_db: str
    ) -> None:
        """AC-184: Tool-call error at distance 1 is attributed as root cause on the __error__ path."""
        thread_id = "thread-ac184-dist1"
        call_id = "call-search-dist1"
        with RelateSaver(db_path=temp_disk_db, wal_enabled=True) as saver:
            # Step 1: Initial checkpoint
            cid1 = "cp-ac184-dist1-s1"
            cfg1: RunnableConfig = {"configurable": {"thread_id": thread_id, "checkpoint_id": cid1}}
            cp1: Any = dict(empty_checkpoint())
            cp1["id"] = cid1
            saver.put(cfg1, cp1, {"step": 1, "node": "agent"}, {})  # type: ignore[typeddict-unknown-key]

            # Step 2: Tool execution checkpoint with failed tool call (distance 1 to step 3)
            cid2 = "cp-ac184-dist1-s2"
            cfg2: RunnableConfig = {"configurable": {"thread_id": thread_id, "checkpoint_id": cid2}}
            cp2: Any = dict(empty_checkpoint())
            cp2["id"] = cid2
            cp2["parent_checkpoint_id"] = cid1
            saver.put(cfg2, cp2, {"step": 2, "node": "tool_node"}, {})  # type: ignore[typeddict-unknown-key]
            ai = AIMessage(
                content="",
                tool_calls=[{"id": call_id, "name": "vector_search", "args": {"q": "foo"}}],
            )
            saver.put_writes(cfg2, [("messages", [ai])], task_id="task-ai")
            tool_msg = ToolMessage(
                content="Connection refused", tool_call_id=call_id, status="error"
            )
            saver.put_writes(cfg2, [("messages", [tool_msg])], task_id="task-tool")

            # Step 3: Failure checkpoint terminating with __error__ write
            cid3 = "cp-ac184-dist1-s3"
            cfg3: RunnableConfig = {"configurable": {"thread_id": thread_id, "checkpoint_id": cid3}}
            cp3: Any = dict(empty_checkpoint())
            cp3["id"] = cid3
            cp3["parent_checkpoint_id"] = cid2
            saver.put(cfg3, cp3, {"step": 3, "node": "error_handler"}, {})  # type: ignore[typeddict-unknown-key]
            saver.put_writes(
                cfg3,
                [("__error__", "Cascading failure following tool execution")],
                task_id="task-err",
            )

            with RelateForensics(saver) as forensics:
                report = forensics.trace_root_cause(thread_id)

            assert report.thread_id == thread_id
            assert report.error_checkpoint_id == cid3
            assert report.root_tool_call_id == call_id
            assert report.root_tool_name == "vector_search"
            assert report.causal_path_length == 1
            assert "vector_search" in report.diagnostic_summary

    def test_ac184_tool_call_error_at_distance_2_not_attributed_on_error_write_path(
        self, temp_disk_db: str
    ) -> None:
        """AC-184: Antecedent tool-call error at distance >= 2 is not blamed as root cause on the __error__ path."""
        thread_id = "thread-ac184-dist2"
        call_id = "call-old-distant-err"
        with RelateSaver(db_path=temp_disk_db, wal_enabled=True) as saver:
            # Step 1: Checkpoint with failed tool call (distance 2 to step 3)
            cid1 = "cp-ac184-dist2-s1"
            cfg1: RunnableConfig = {"configurable": {"thread_id": thread_id, "checkpoint_id": cid1}}
            cp1: Any = dict(empty_checkpoint())
            cp1["id"] = cid1
            saver.put(cfg1, cp1, {"step": 1, "node": "old_tool"}, {})  # type: ignore[typeddict-unknown-key]
            ai = AIMessage(
                content="",
                tool_calls=[{"id": call_id, "name": "legacy_lookup", "args": {"k": "v"}}],
            )
            saver.put_writes(cfg1, [("messages", [ai])], task_id="task-ai-1")
            tool_msg = ToolMessage(content="Service timeout", tool_call_id=call_id, status="error")
            saver.put_writes(cfg1, [("messages", [tool_msg])], task_id="task-tool-1")

            # Step 2: Normal recovery checkpoint with regular channel write
            cid2 = "cp-ac184-dist2-s2"
            cfg2: RunnableConfig = {"configurable": {"thread_id": thread_id, "checkpoint_id": cid2}}
            cp2: Any = dict(empty_checkpoint())
            cp2["id"] = cid2
            cp2["parent_checkpoint_id"] = cid1
            saver.put(cfg2, cp2, {"step": 2, "node": "fallback_node"}, {})  # type: ignore[typeddict-unknown-key]
            saver.put_writes(
                cfg2,
                [("messages", [HumanMessage(content="fallback completed successfully")])],
                task_id="task-fallback",
            )

            # Step 3: Unrelated crash checkpoint terminating with __error__ write
            cid3 = "cp-ac184-dist2-s3"
            cfg3: RunnableConfig = {"configurable": {"thread_id": thread_id, "checkpoint_id": cid3}}
            cp3: Any = dict(empty_checkpoint())
            cp3["id"] = cid3
            cp3["parent_checkpoint_id"] = cid2
            saver.put(cfg3, cp3, {"step": 3, "node": "unrelated_crasher"}, {})  # type: ignore[typeddict-unknown-key]
            saver.put_writes(
                cfg3,
                [("__error__", "Out of memory in unrelated crasher")],
                task_id="task-err-3",
            )

            with RelateForensics(saver) as forensics:
                report = forensics.trace_root_cause(thread_id)

            assert report.thread_id == thread_id
            assert report.error_checkpoint_id == cid3
            # The distant tool error at distance 2 must NOT be blamed as root tool call
            assert report.root_tool_call_id != call_id
            assert report.root_tool_call_id is None
            assert report.root_tool_name is None
            # The offending antecedent should be the nearest antecedent write preceding failure (step 2)
            assert report.root_channel_write is not None
            assert report.causal_path_length == 1

    def test_ac184_distant_tool_error_without_intermediate_writes_not_attributed(
        self, temp_disk_db: str
    ) -> None:
        """AC-184: Distant tool error at distance >= 2 without intermediate writes is not blamed on __error__ path."""
        thread_id = "thread-ac184-nowrite"
        call_id = "call-distant-no-write"
        with RelateSaver(db_path=temp_disk_db, wal_enabled=True) as saver:
            # Step 1: Checkpoint with tool error (distance 2 to step 3)
            cid1 = "cp-ac184-nw-s1"
            cfg1: RunnableConfig = {"configurable": {"thread_id": thread_id, "checkpoint_id": cid1}}
            cp1: Any = dict(empty_checkpoint())
            cp1["id"] = cid1
            saver.put(cfg1, cp1, {"step": 1, "node": "tool_step"}, {})  # type: ignore[typeddict-unknown-key]
            ai = AIMessage(
                content="",
                tool_calls=[{"id": call_id, "name": "distant_tool", "args": {}}],
            )
            saver.put_writes(cfg1, [("messages", [ai])], task_id="task-ai")
            tool_msg = ToolMessage(content="Failed", tool_call_id=call_id, status="error")
            saver.put_writes(cfg1, [("messages", [tool_msg])], task_id="task-tool")

            # Step 2: Intermediate checkpoint with no channel writes
            cid2 = "cp-ac184-nw-s2"
            cfg2: RunnableConfig = {"configurable": {"thread_id": thread_id, "checkpoint_id": cid2}}
            cp2: Any = dict(empty_checkpoint())
            cp2["id"] = cid2
            cp2["parent_checkpoint_id"] = cid1
            saver.put(cfg2, cp2, {"step": 2, "node": "noop_step"}, {})  # type: ignore[typeddict-unknown-key]

            # Step 3: Error checkpoint terminating with __error__ write
            cid3 = "cp-ac184-nw-s3"
            cfg3: RunnableConfig = {"configurable": {"thread_id": thread_id, "checkpoint_id": cid3}}
            cp3: Any = dict(empty_checkpoint())
            cp3["id"] = cid3
            cp3["parent_checkpoint_id"] = cid2
            saver.put(cfg3, cp3, {"step": 3, "node": "crash_step"}, {})  # type: ignore[typeddict-unknown-key]
            saver.put_writes(
                cfg3,
                [("__error__", "Failure at step 3")],
                task_id="task-err",
            )

            with RelateForensics(saver) as forensics:
                report = forensics.trace_root_cause(thread_id)

            assert report.thread_id == thread_id
            assert report.error_checkpoint_id == cid3
            assert report.root_tool_call_id is None
            assert report.root_tool_name is None
