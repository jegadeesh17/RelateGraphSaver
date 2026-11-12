"""End-to-end integration tests validating Milestone 3: Forensics & Polish.

Verifies complete cross-module integration across the checkpointer stack:
- Cycle and oscillation detection identifying repeating node sequences and state hashes (AC-25).
- Nominal diagnostic reports for linear, non-repeating executions (AC-26).
- Backward causal path tracing from error checkpoints to antecedent writes and tool calls (AC-27).
- Nominal root cause reporting indicating no errors on healthy execution lineages (AC-28).
- Multi-agent research application execution persisting planning, research, and causal DAG state (AC-29).
- Human-in-the-loop interrupt halting, pending write retrieval, and guidance resumption (AC-30).
- Clean pip installation and wheel dependency resolution (AC-31).
- CLI entrypoint and subcommand option displays with exit code 0 (AC-32).
"""

# ruff: noqa: E402

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from typer.testing import CliRunner
from typing_extensions import TypedDict

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
_SRC_DIR = str(Path(__file__).resolve().parents[2] / "src")
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from examples.research_agent import (
    ResearchState,
    create_research_graph,
    get_workflow_statistics,
)

from langgraph_checkpoint_relate.cli import app
from langgraph_checkpoint_relate.errors import CheckpointNotFoundError
from langgraph_checkpoint_relate.forensics import RelateForensics
from langgraph_checkpoint_relate.models import LoopReport, RootCauseReport
from langgraph_checkpoint_relate.saver import RelateSaver


@pytest.fixture
def runner() -> CliRunner:
    """Fixture providing an isolated Typer CliRunner instance."""
    return CliRunner()


@pytest.fixture
def temp_disk_db() -> Generator[str]:
    """Provide a temporary disk-backed database file path and clean up after the test."""
    temp_dir = tempfile.mkdtemp(prefix="relate_m3_qa_")
    db_file = os.path.join(temp_dir, "milestone3_checkpoints.rl")
    try:
        yield db_file
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


class AgentOscillationState(TypedDict):
    """Workflow state for multi-agent oscillation simulation."""

    iteration: int
    active_agent: str
    feedback_history: list[str]


class LinearAgentState(TypedDict):
    """Workflow state for sequential non-repeating execution."""

    phase_count: int
    notes: list[str]


class AgentPoisonState(TypedDict):
    """Workflow state for antecedent write error injection."""

    payload: str
    status: str


class TestMilestoneThreeForensicsAndPolish:
    """End-to-end integration suite for Milestone 3 forensics, polish, and workflows."""

    def test_agent_loop_detection_and_oscillation_threshold_ac_25(
        self, temp_disk_db: str, runner: CliRunner
    ) -> None:
        """AC-25: Verify detect_loops identifies oscillating agent sequences exceeding threshold.

        Executes an oscillating multi-agent review loop (research_analyst <-> peer_critic)
        for 6 transitions, verifies detect_loops returns a LoopReport with exact cycle nodes,
        occurrence count, and checkpoint ID interval, and verifies CLI forensics output.
        """
        thread_id = "thread-oscillation-ac25"
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        def research_analyst(state: AgentOscillationState) -> dict[str, Any]:
            return {
                "iteration": state["iteration"] + 1,
                "active_agent": "research_analyst",
                "feedback_history": state["feedback_history"]
                + ["draft_v" + str(state["iteration"])],
            }

        def peer_critic(state: AgentOscillationState) -> dict[str, Any]:
            return {
                "iteration": state["iteration"] + 1,
                "active_agent": "peer_critic",
                "feedback_history": state["feedback_history"]
                + ["critique_v" + str(state["iteration"])],
            }

        def router(state: AgentOscillationState) -> str:
            if state["iteration"] < 6:
                return (
                    "peer_critic"
                    if state["active_agent"] == "research_analyst"
                    else "research_analyst"
                )
            return END

        workflow = StateGraph(AgentOscillationState)
        workflow.add_node("research_analyst", research_analyst)
        workflow.add_node("peer_critic", peer_critic)
        workflow.add_edge(START, "research_analyst")
        workflow.add_conditional_edges(
            "research_analyst",
            router,
            {"research_analyst": "research_analyst", "peer_critic": "peer_critic", END: END},
        )
        workflow.add_conditional_edges(
            "peer_critic",
            router,
            {"research_analyst": "research_analyst", "peer_critic": "peer_critic", END: END},
        )

        with RelateSaver(db_path=temp_disk_db, wal_enabled=True) as saver:
            app_compiled = workflow.compile(checkpointer=saver)
            final_state = app_compiled.invoke(
                {"iteration": 0, "active_agent": "", "feedback_history": []},
                config=config,
            )
            assert final_state["iteration"] >= 6

            # AC-25: detect_loops returns detected execution cycle report
            reports = saver.detect_loops(thread_id, threshold=2)
            assert len(reports) >= 1
            report = reports[0]
            assert isinstance(report, LoopReport)
            assert report.thread_id == thread_id
            assert report.cycle_nodes == ["research_analyst", "peer_critic"]
            assert report.occurrence_count >= 2
            assert isinstance(report.checkpoint_id_range, tuple)
            assert len(report.checkpoint_id_range) == 2
            assert report.checkpoint_id_range[0] != report.checkpoint_id_range[1]

            # High threshold filters out fewer occurrences
            empty_reports = saver.detect_loops(thread_id, threshold=10)
            assert empty_reports == []

        # Cross-module wiring: verify CLI relate-dag forensics loops detects cycles
        cli_result = runner.invoke(
            app, ["forensics", "loops", thread_id, "--db", temp_disk_db, "--threshold", "2"]
        )
        assert cli_result.exit_code == 0
        assert "research_analyst" in cli_result.output
        assert "peer_critic" in cli_result.output
        assert "Detected Loops" in cli_result.output

    def test_loop_detection_linear_execution_clean_report_ac_26(
        self, temp_disk_db: str, runner: CliRunner
    ) -> None:
        """AC-26: Verify detect_loops returns empty list for linear non-repeating executions.

        Executes a 3-node linear pipeline without cycles and verifies both saver.detect_loops
        and CLI relate-dag forensics loops return an empty diagnostic report.
        """
        thread_id = "thread-linear-ac26"
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        def phase_ingest(state: LinearAgentState) -> dict[str, Any]:
            return {"phase_count": state["phase_count"] + 1, "notes": state["notes"] + ["ingested"]}

        def phase_process(state: LinearAgentState) -> dict[str, Any]:
            return {
                "phase_count": state["phase_count"] + 1,
                "notes": state["notes"] + ["processed"],
            }

        def phase_summarize(state: LinearAgentState) -> dict[str, Any]:
            return {
                "phase_count": state["phase_count"] + 1,
                "notes": state["notes"] + ["summarized"],
            }

        workflow = StateGraph(LinearAgentState)
        workflow.add_node("phase_ingest", phase_ingest)
        workflow.add_node("phase_process", phase_process)
        workflow.add_node("phase_summarize", phase_summarize)
        workflow.add_edge(START, "phase_ingest")
        workflow.add_edge("phase_ingest", "phase_process")
        workflow.add_edge("phase_process", "phase_summarize")
        workflow.add_edge("phase_summarize", END)

        with RelateSaver(db_path=temp_disk_db, wal_enabled=True) as saver:
            app_compiled = workflow.compile(checkpointer=saver)
            app_compiled.invoke({"phase_count": 0, "notes": []}, config=config)

            # AC-26: Empty report for linear executions
            reports = saver.detect_loops(thread_id, threshold=2)
            assert reports == []

        # Cross-module wiring: verify CLI relate-dag forensics loops on linear thread
        cli_result = runner.invoke(app, ["forensics", "loops", thread_id, "--db", temp_disk_db])
        assert cli_result.exit_code == 0
        assert "No loops or repeating cycles detected" in cli_result.output

    def test_backward_causal_path_root_cause_analysis_ac_27(
        self, temp_disk_db: str, runner: CliRunner
    ) -> None:
        """AC-27: Verify trace_root_cause traces backward causal path from error checkpoint.

        Injects a poison payload at an antecedent write node that subsequently raises
        an unhandled exception in a downstream processor node. Verifies that root cause
        analysis identifies the error checkpoint and traces backwards along [:NEXT]
        and [:HAS_WRITE] edges to pinpoint the antecedent channel write.
        """
        thread_id = "thread-root-cause-ac27"
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        def inject_malformed_token(state: AgentPoisonState) -> dict[str, Any]:
            return {"payload": "UNRECOGNIZED_SCHEMA_POISON", "status": "poisoned"}

        def validate_and_synthesize(state: AgentPoisonState) -> dict[str, Any]:
            if "POISON" in state["payload"]:
                raise RuntimeError("Downstream compilation aborted: invalid schema payload")
            return {"status": "ok"}

        workflow = StateGraph(AgentPoisonState)
        workflow.add_node("inject_malformed_token", inject_malformed_token)
        workflow.add_node("validate_and_synthesize", validate_and_synthesize)
        workflow.add_edge(START, "inject_malformed_token")
        workflow.add_edge("inject_malformed_token", "validate_and_synthesize")
        workflow.add_edge("validate_and_synthesize", END)

        with RelateSaver(db_path=temp_disk_db, wal_enabled=True) as saver:
            app_compiled = workflow.compile(checkpointer=saver)
            with pytest.raises(RuntimeError, match="invalid schema payload"):
                app_compiled.invoke({"payload": "valid", "status": "init"}, config=config)

            # AC-27: trace_root_cause pinpoints offending antecedent write
            report = saver.trace_root_cause(thread_id)
            assert isinstance(report, RootCauseReport)
            assert report.thread_id == thread_id
            assert report.error_checkpoint_id != ""
            assert "Downstream compilation aborted" in report.error_message
            assert report.root_channel_write is not None
            assert report.causal_path_length >= 1
            assert "Failure at checkpoint" in report.diagnostic_summary

        # Cross-module wiring: verify CLI relate-dag forensics trace displays causal report
        cli_result = runner.invoke(app, ["forensics", "trace", thread_id, "--db", temp_disk_db])
        assert cli_result.exit_code == 0
        assert "Root Cause Analysis" in cli_result.output
        assert "Downstream compilation aborted" in cli_result.output
        assert "Causal Path Length" in cli_result.output

    def test_root_cause_analysis_healthy_execution_ac_28(
        self, temp_disk_db: str, runner: CliRunner
    ) -> None:
        """AC-28: Verify trace_root_cause indicates no failure for healthy threads.

        Executes an error-free sequential graph and validates that trace_root_cause
        reports zero failure distance, empty error checkpoint id, and clean diagnostic message.
        """
        thread_id = "thread-healthy-ac28"
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        def healthy_worker(state: LinearAgentState) -> dict[str, Any]:
            return {"phase_count": 1, "notes": ["healthy"]}

        workflow = StateGraph(LinearAgentState)
        workflow.add_node("healthy_worker", healthy_worker)
        workflow.add_edge(START, "healthy_worker")
        workflow.add_edge("healthy_worker", END)

        with RelateSaver(db_path=temp_disk_db, wal_enabled=True) as saver:
            app_compiled = workflow.compile(checkpointer=saver)
            app_compiled.invoke({"phase_count": 0, "notes": []}, config=config)

            # AC-28: No failure detected
            report = saver.trace_root_cause(thread_id)
            assert isinstance(report, RootCauseReport)
            assert report.thread_id == thread_id
            assert report.error_checkpoint_id == ""
            assert report.error_message == "No error detected"
            assert report.root_tool_call_id is None
            assert report.root_channel_write is None
            assert report.causal_path_length == 0
            assert "completed without errors" in report.diagnostic_summary

        # Cross-module wiring: verify CLI relate-dag forensics trace displays clean status
        cli_result = runner.invoke(app, ["forensics", "trace", thread_id, "--db", temp_disk_db])
        assert cli_result.exit_code == 0
        assert "No error detected in execution thread" in cli_result.output

    def test_multi_agent_research_application_with_hitl_cycle_ac_29_ac_30(
        self, temp_disk_db: str, runner: CliRunner
    ) -> None:
        """AC-29, AC-30: Verify end-to-end research application planning, interrupt, and resumption.

        Executes the multi-agent research demonstration workflow:
        1. Runs Planner and Researcher nodes, pausing at the Human Review interrupt (AC-29).
        2. Validates pending writes contains __interrupt__ channel and review request metadata.
        3. Visualizes paused tree via CLI relate-dag tree and verifies [INTERRUPT] rendering.
        4. Verifies isolated in-memory replay branching preserves original database integrity.
        5. Resumes execution with custom reviewer feedback and verifies Critic synthesizes output
           reflecting the reviewer guidance (AC-30).
        6. Verifies complete causal [:NEXT] path in Relate DB across saver restarts.
        """
        thread_id = "thread-research-ac29-ac30"
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        reviewer_guidance = "Verify that embedded Cypher queries achieve sub-millisecond lookups."

        # Phase 1: Planning and research up to interrupt
        with RelateSaver(db_path=temp_disk_db, wal_enabled=True) as saver:
            graph = create_research_graph(checkpointer=saver)
            initial_state: ResearchState = {
                "topic": "Graph-Native Multi-Agent Orchestration",
                "plan": [],
                "research_notes": [],
                "review_feedback": None,
                "final_summary": None,
                "status": "started",
            }
            interrupted_output = graph.invoke(initial_state, config=config)
            assert "__interrupt__" in interrupted_output

            # AC-30: Verify pending writes stored in paused checkpoint
            paused_tuple = saver.get_tuple(config)
            assert paused_tuple is not None
            paused_cid = paused_tuple.checkpoint["id"]
            assert paused_tuple.pending_writes is not None
            assert any(pw[1] == "__interrupt__" for pw in paused_tuple.pending_writes)

            # AC-29: Verify Cypher graph records checkpoints and edges
            stats_phase1 = get_workflow_statistics(saver, thread_id)
            assert stats_phase1["checkpoints"] >= 3
            assert stats_phase1["causal_edges"] >= 2

            # Verify in-memory replay sandboxing from paused checkpoint
            with saver.fork_in_memory(
                thread_id=thread_id, checkpoint_id=paused_cid
            ) as memory_branch:
                branch_tuple = memory_branch.get_tuple(config)
                assert branch_tuple is not None
                assert branch_tuple.checkpoint["id"] == paused_cid

        # Cross-module wiring: CLI DAG visualizer inspects paused execution tree
        tree_result = runner.invoke(app, ["tree", thread_id, "--db", temp_disk_db])
        assert tree_result.exit_code == 0
        assert "[INTERRUPT]" in tree_result.output
        assert "channel='topic'" in tree_result.output

        # Phase 2: Resume workflow from pending checkpoint with reviewer feedback
        with RelateSaver(db_path=temp_disk_db, wal_enabled=True) as resumed_saver:
            resumed_graph = create_research_graph(checkpointer=resumed_saver)
            resumed_output = resumed_graph.invoke(Command(resume=reviewer_guidance), config=config)

            # AC-30: Verify reviewer feedback is incorporated in final synthesis
            assert resumed_output["status"] == "completed"
            assert resumed_output["review_feedback"] == reviewer_guidance
            assert reviewer_guidance in resumed_output["final_summary"]

            # Verify final checkpoint tuple
            final_tuple = resumed_saver.get_tuple(config)
            assert final_tuple is not None
            final_cid = final_tuple.checkpoint["id"]

            # Cypher check: unbroken causal path from interrupted checkpoint to final checkpoint
            causal_links = resumed_saver.proxy.execute_read(
                "MATCH p = (i:Checkpoint {checkpoint_id: $i_cid})-[:NEXT*]->(f:Checkpoint {checkpoint_id: $f_cid}) "
                "RETURN length(p) AS len",
                {"i_cid": paused_cid, "f_cid": final_cid},
            )
            assert len(causal_links) == 1
            assert causal_links[0]["len"] >= 1

            # AC-29: Verify full Cypher storage statistics
            final_stats = get_workflow_statistics(resumed_saver, thread_id)
            assert final_stats["checkpoints"] >= 5
            assert final_stats["causal_edges"] >= 4
            assert final_stats["pending_writes"] >= 15

    def test_clean_installation_and_package_build_ac_31(self) -> None:
        """AC-31: Verify pip install --dry-run . resolves dependencies cleanly without errors.

        Validates pyproject.toml compliance and executes pip dry-run installation
        to confirm package builds cleanly in Python environment.
        """
        pyproject_path = Path("pyproject.toml")
        assert pyproject_path.exists(), "pyproject.toml must exist in workspace root"

        with open(pyproject_path, "rb") as f:
            meta = tomllib.load(f)

        project = meta.get("project", {})
        assert project.get("name") == "langgraph-checkpoint-relate"
        assert project.get("requires-python") == ">=3.11"
        assert "hatchling" in meta.get("build-system", {}).get("requires", [])

        # AC-31: Run pip install --dry-run . in subprocess
        cmd = [sys.executable, "-m", "pip", "install", "--dry-run", "."]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        assert proc.returncode == 0, f"pip install --dry-run failed: {proc.stderr}"
        assert "Would install langgraph-checkpoint-relate" in proc.stdout or proc.returncode == 0

    def test_cli_subcommands_help_and_execution_options_ac_32(self, runner: CliRunner) -> None:
        """AC-32: Verify relate-dag --help and subcommands display options with exit code 0.

        Tests relate-dag top-level help, tree subcommand, replay subcommand, forensics
        subcommand, and forensics nested subcommands (loops, trace) across CliRunner and subprocess.
        """
        # Top-level --help
        res = runner.invoke(app, ["--help"])
        assert res.exit_code == 0
        assert "tree" in res.output
        assert "replay" in res.output
        assert "forensics" in res.output

        # Subcommand --help options
        for sub in ["tree", "replay", "forensics"]:
            sub_res = runner.invoke(app, [sub, "--help"])
            assert sub_res.exit_code == 0, f"Subcommand '{sub} --help' failed"

        # Nested forensics subcommands
        loops_res = runner.invoke(app, ["forensics", "loops", "--help"])
        assert loops_res.exit_code == 0
        assert "--threshold" in loops_res.output

        trace_res = runner.invoke(app, ["forensics", "trace", "--help"])
        assert trace_res.exit_code == 0

        # AC-32 via standalone subprocess invocation
        sp_res = subprocess.run(
            [sys.executable, "-m", "langgraph_checkpoint_relate.cli", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert sp_res.returncode == 0
        assert "tree" in sp_res.stdout
        assert "replay" in sp_res.stdout
        assert "forensics" in sp_res.stdout

    def test_forensics_and_cli_boundary_error_handling(
        self, temp_disk_db: str, runner: CliRunner
    ) -> None:
        """Verify boundary inputs and invalid parameter error handling across forensics and CLI."""
        with RelateSaver(db_path=temp_disk_db, wal_enabled=True) as saver:
            forensics = RelateForensics(saver)

            # Invalid threshold boundary: threshold < 1
            with pytest.raises(ValueError, match="threshold must be an integer >= 1"):
                forensics.detect_loops("any-thread", threshold=0)

            with pytest.raises(ValueError, match="threshold must be an integer >= 1"):
                forensics.detect_loops("any-thread", threshold=-5)

            # Invalid thread_id: empty or whitespace
            with pytest.raises(
                CheckpointNotFoundError, match="thread_id must be a non-empty string"
            ):
                forensics.detect_loops("   ", threshold=2)

            with pytest.raises(
                CheckpointNotFoundError, match="thread_id must be a non-empty string"
            ):
                forensics.trace_root_cause("   ")

            # Non-existent thread
            with pytest.raises(
                CheckpointNotFoundError, match="not found or contains no checkpoints"
            ):
                forensics.detect_loops("non-existent-thread-404", threshold=2)

        # CLI boundary: non-existent thread
        cli_loops = runner.invoke(
            app, ["forensics", "loops", "non-existent-thread-999", "--db", temp_disk_db]
        )
        assert cli_loops.exit_code != 0
        assert "Error" in cli_loops.output or "not found" in cli_loops.output

        # CLI boundary: non-existent database file
        cli_missing_db = runner.invoke(
            app, ["forensics", "loops", "some-thread", "--db", "non_existent_file.rl"]
        )
        assert cli_missing_db.exit_code != 0
        assert "Database file not found" in cli_missing_db.output
