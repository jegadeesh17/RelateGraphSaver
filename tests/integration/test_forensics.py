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
            saver.put_writes(config, [("__error__", "Chain aborted")], task_id="task-long")

            started = time.perf_counter()
            report = saver.trace_root_cause(thread_id)
            elapsed = time.perf_counter() - started

        assert report.error_checkpoint_id == config["configurable"]["checkpoint_id"]
        assert report.root_tool_call_id == "call-long-chain-1"
        assert report.root_tool_name == "slow_api"
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
        # AC-164: distinct separate string elements in origin_nodes, not single joined string
        assert summary[0].origin_nodes == ("fetcher", "parser")
        # trace_argument_origin(...).node remains a joined string (unchanged public behavior)
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
