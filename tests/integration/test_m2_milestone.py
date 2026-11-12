"""End-to-end integration tests validating Milestone 2: Core Flows and Time-Travel Debugger.

Verifies end-to-end workflows across the checkpointer stack:
- Pending channel writes and human-in-the-loop interrupt persistence (AC-14, AC-15, AC-16).
- Hierarchical namespaced subgraphs and namespace-filtered queries (AC-17, AC-18).
- Isolated in-memory sandboxed time-travel replay and disk immutability (AC-19, AC-20).
- CheckpointNotFoundError diagnostics with candidate checkpoint listings (AC-21).
- Unicode execution tree visualization with status annotations (AC-22, AC-23).
- CLI error handling and non-existent thread boundary behavior (AC-24).
"""

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
    Checkpoint,
    empty_checkpoint,
)
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from typer.testing import CliRunner

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
_SRC_DIR = str(Path(__file__).resolve().parents[2] / "src")
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from langgraph_checkpoint_relate.cli import app
from langgraph_checkpoint_relate.errors import CheckpointNotFoundError
from langgraph_checkpoint_relate.saver import RelateSaver
from langgraph_checkpoint_relate.visualizer import DAGVisualizer


def _compute_sha256(path: str) -> str:
    """Compute the SHA-256 hash of a file on disk."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def _build_checkpoint(
    checkpoint_id: str,
    channel_values: dict[str, Any] | None = None,
    parent_checkpoint_id: str | None = None,
) -> Checkpoint:
    """Create a structured Checkpoint instance for testing."""
    base: dict[str, Any] = dict(empty_checkpoint())
    base["id"] = checkpoint_id
    if parent_checkpoint_id is not None:
        base["parent_checkpoint_id"] = parent_checkpoint_id
    if channel_values:
        base["channel_values"] = channel_values
        base["channel_versions"] = dict.fromkeys(channel_values, "1.0")
    return cast(Checkpoint, base)


@pytest.fixture
def temp_disk_db() -> Generator[str]:
    """Provide a temporary disk-backed database file path and clean up after the test."""
    temp_dir = tempfile.mkdtemp(prefix="relate_m2_qa_")
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


@pytest.fixture
def cli_runner() -> CliRunner:
    """Provide an isolated Typer CliRunner instance."""
    return CliRunner()


class ApprovalWorkflowState(TypedDict):
    """Execution state schema for human-in-the-loop workflows."""

    query: str
    approval: str
    result: str


class SubgraphWorkflowState(TypedDict):
    """Execution state schema for nested subgraph workflows."""

    input_text: str
    processed_text: str
    validation_status: str


class TestMilestone2Integration:
    """Milestone 2 integration test suite covering AC-14 through AC-24."""

    def test_human_in_the_loop_interrupt_and_resume_flow_ac_14_ac_15_ac_16(
        self, temp_disk_db: str
    ) -> None:
        """AC-14, AC-15, AC-16: End-to-end human-in-the-loop execution on disk.

        - AC-14: Pending writes stored via put_writes connected via :HAS_WRITE.
        - AC-15: get_tuple reconstructs pending writes matching stored pairs.
        - AC-16: Resumed checkpoint links interrupted checkpoint as parent via :NEXT.
        """

        def ask_human_step(state: ApprovalWorkflowState) -> dict[str, Any]:
            decision = interrupt("User authorization required to proceed.")
            return {"approval": decision}

        def process_step(state: ApprovalWorkflowState) -> dict[str, Any]:
            return {"result": f"executed:{state['query']}:{state['approval']}"}

        builder = StateGraph(ApprovalWorkflowState)
        builder.add_node("ask_human", ask_human_step)
        builder.add_node("process", process_step)
        builder.add_edge(START, "ask_human")
        builder.add_edge("ask_human", "process")
        builder.add_edge("process", END)

        thread_id = "thread-m2-hitl-001"
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        # Stage 1: Run workflow until interrupt on disk-backed saver
        with RelateSaver(db_path=temp_disk_db, wal_enabled=True) as saver:
            graph = builder.compile(checkpointer=saver)

            initial_result = graph.invoke(
                {"query": "transfer_100", "approval": "", "result": ""},
                config,
            )
            assert "__interrupt__" in initial_result

            # AC-15: get_tuple() reconstructs pending writes list matching stored pairs
            interrupted_tuple = saver.get_tuple(config)
            assert interrupted_tuple is not None
            assert interrupted_tuple.pending_writes is not None
            assert len(interrupted_tuple.pending_writes) >= 1
            pending_channels = [pw[1] for pw in interrupted_tuple.pending_writes]
            assert "__interrupt__" in pending_channels

            # AC-14: verify :HAS_WRITE edge in Relate database
            interrupted_cid = interrupted_tuple.checkpoint["id"]
            write_edges = saver.proxy.execute_read(
                "MATCH (c:Checkpoint {checkpoint_id: $cid})-[:HAS_WRITE]->(w:Write) "
                "WHERE w.channel = '__interrupt__' "
                "RETURN w.idx AS idx, w.task_path AS task_path",
                {"cid": interrupted_cid},
            )
            assert len(write_edges) == 1
            assert write_edges[0]["idx"] == -3

            # Stage 2: Resume workflow from interrupt
            resumed_result = graph.invoke(Command(resume="user_approved_token"), config)
            assert resumed_result["approval"] == "user_approved_token"
            assert resumed_result["result"] == "executed:transfer_100:user_approved_token"

            # AC-16: Resumed checkpoint links interrupted checkpoint as parent
            resumed_tuple = saver.get_tuple(config)
            assert resumed_tuple is not None
            resumed_cid = resumed_tuple.checkpoint["id"]

            direct_next = saver.proxy.execute_read(
                "MATCH (p:Checkpoint {checkpoint_id: $p_cid})-[:NEXT]->(c:Checkpoint) "
                "RETURN c.checkpoint_id AS child_id, c.parent_checkpoint_id AS parent_id",
                {"p_cid": interrupted_cid},
            )
            assert len(direct_next) == 1
            assert direct_next[0]["parent_id"] == interrupted_cid

            causal_path = saver.proxy.execute_read(
                "MATCH path = (p:Checkpoint {checkpoint_id: $p_cid})-[:NEXT*]->(c:Checkpoint {checkpoint_id: $c_cid}) "
                "RETURN length(path) AS path_len",
                {"p_cid": interrupted_cid, "c_cid": resumed_cid},
            )
            assert len(causal_path) == 1
            assert causal_path[0]["path_len"] >= 1

        # Stage 3: Durability restart check from disk
        with RelateSaver(db_path=temp_disk_db, wal_enabled=True) as reopened_saver:
            latest = reopened_saver.get_tuple(config)
            assert latest is not None
            assert latest.checkpoint["id"] == resumed_cid
            assert latest.checkpoint["channel_values"]["result"] == (
                "executed:transfer_100:user_approved_token"
            )

    def test_hierarchical_subgraphs_and_namespace_filtering_ac_17_ac_18(
        self, temp_disk_db: str
    ) -> None:
        """AC-17, AC-18: Checkpoints with varying checkpoint_ns and namespace filtering.

        - AC-17: Hierarchical subgraphs with varying checkpoint_ns persist attributes intact.
        - AC-18: list() filtering by checkpoint_ns yields only matching checkpoints.
        """
        thread_id = "thread-m2-namespaces-002"
        root_ns = ""
        subgraph_worker_ns = "worker_subgraph"
        subgraph_validator_ns = "validator_subgraph"

        with RelateSaver(db_path=temp_disk_db, wal_enabled=True) as saver:
            # 1. Root pipeline checkpoint
            cfg_root: RunnableConfig = {
                "configurable": {"thread_id": thread_id, "checkpoint_ns": root_ns}
            }
            cp_root = _build_checkpoint("cp-root-01", {"input_text": "data_payload"})
            saver.put(cfg_root, cp_root, {"step": 0, "node": "root_dispatcher"}, {})

            # 2. Worker subgraph checkpoints (AC-17)
            cfg_worker_1: RunnableConfig = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": subgraph_worker_ns,
                    "checkpoint_id": "cp-root-01",
                }
            }
            cp_worker_1 = _build_checkpoint(
                "cp-worker-01", {"processed_text": "data_payload_v1"}, "cp-root-01"
            )
            saver.put(
                cfg_worker_1,
                cp_worker_1,
                {
                    "step": 1,
                    "node": "worker_node",
                    "parents": {root_ns: "cp-root-01", subgraph_worker_ns: ""},
                },
                {},
            )

            cfg_worker_2: RunnableConfig = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": subgraph_worker_ns,
                    "checkpoint_id": "cp-worker-01",
                }
            }
            cp_worker_2 = _build_checkpoint(
                "cp-worker-02", {"processed_text": "data_payload_v2"}, "cp-worker-01"
            )
            saver.put(
                cfg_worker_2,
                cp_worker_2,
                {
                    "step": 2,
                    "node": "worker_node",
                    "parents": {subgraph_worker_ns: "cp-worker-01"},
                },
                {},
            )

            # 3. Validator subgraph checkpoint (AC-17)
            cfg_validator: RunnableConfig = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": subgraph_validator_ns,
                    "checkpoint_id": "cp-worker-02",
                }
            }
            cp_val = _build_checkpoint(
                "cp-validator-01", {"validation_status": "PASSED"}, "cp-worker-02"
            )
            saver.put(
                cfg_validator,
                cp_val,
                {
                    "step": 3,
                    "node": "validator_node",
                    "parents": {subgraph_validator_ns: "", subgraph_worker_ns: "cp-worker-02"},
                },
                {},
            )

            # AC-17: Verify namespace attribute stored correctly in Relate graph
            db_namespaces = saver.proxy.execute_read(
                "MATCH (c:Checkpoint {thread_id: $tid}) "
                "RETURN c.checkpoint_id AS cid, c.checkpoint_ns AS ns "
                "ORDER BY c.checkpoint_id ASC",
                {"tid": thread_id},
            )
            ns_dict = {row["cid"]: row["ns"] for row in db_namespaces}
            assert ns_dict["cp-root-01"] == root_ns
            assert ns_dict["cp-worker-01"] == subgraph_worker_ns
            assert ns_dict["cp-worker-02"] == subgraph_worker_ns
            assert ns_dict["cp-validator-01"] == subgraph_validator_ns

            # AC-18: list() filtering by checkpoint_ns
            worker_list = list(
                saver.list(
                    {"configurable": {"thread_id": thread_id}},
                    filter={"checkpoint_ns": subgraph_worker_ns},
                )
            )
            assert len(worker_list) == 2
            assert all(
                t.config["configurable"]["checkpoint_ns"] == subgraph_worker_ns for t in worker_list
            )
            assert [t.checkpoint["id"] for t in worker_list] == ["cp-worker-02", "cp-worker-01"]

            validator_list = list(
                saver.list(
                    {"configurable": {"thread_id": thread_id}},
                    filter={"checkpoint_ns": subgraph_validator_ns},
                )
            )
            assert len(validator_list) == 1
            assert validator_list[0].checkpoint["id"] == "cp-validator-01"

            root_list = list(
                saver.list(
                    {"configurable": {"thread_id": thread_id}},
                    filter={"checkpoint_ns": root_ns},
                )
            )
            assert len(root_list) == 1
            assert root_list[0].checkpoint["id"] == "cp-root-01"

            # Invalid input boundary check: querying a non-existent namespace returns empty list
            ghost_list = list(
                saver.list(
                    {"configurable": {"thread_id": thread_id}},
                    filter={"checkpoint_ns": "ghost_namespace"},
                )
            )
            assert ghost_list == []

    def test_time_travel_sandboxed_replay_and_disk_isolation_ac_19_ac_20(
        self, temp_disk_db: str
    ) -> None:
        """AC-19, AC-20: fork_in_memory isolated sandbox and disk bitwise immutability.

        - AC-19: fork_in_memory() clones subgraph up to target checkpoint into :memory: saver.
        - AC-20: In-memory sandbox writes leave .rl disk file bitwise unmodified.
        """
        thread_id = "thread-m2-timetravel-003"
        checkpoint_ids = ["cp-base-00", "cp-base-01", "cp-base-02"]

        # 1. Establish durable lineage on disk
        with RelateSaver(db_path=temp_disk_db, wal_enabled=True) as disk_saver:
            cfg: RunnableConfig = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}

            cp_0 = _build_checkpoint(checkpoint_ids[0], {"state": "init"})
            disk_saver.put(cfg, cp_0, {"step": 0, "node": "start_node"}, {})

            cp_1 = _build_checkpoint(checkpoint_ids[1], {"state": "step_1"}, checkpoint_ids[0])
            disk_saver.put(
                {"configurable": {"thread_id": thread_id, "checkpoint_id": checkpoint_ids[0]}},
                cp_1,
                {"step": 1, "node": "worker_node"},
                {},
            )

            cp_2 = _build_checkpoint(checkpoint_ids[2], {"state": "step_2"}, checkpoint_ids[1])
            disk_saver.put(
                {"configurable": {"thread_id": thread_id, "checkpoint_id": checkpoint_ids[1]}},
                cp_2,
                {"step": 2, "node": "end_node"},
                {},
            )

            # AC-19: Fork into isolated in-memory saver at cp-base-01 (excluding cp-base-02)
            sandbox_saver = disk_saver.fork_in_memory(
                thread_id=thread_id,
                checkpoint_id="cp-base-01",
            )

        # Baseline hash and file stats of the closed disk database
        baseline_hash = _compute_sha256(temp_disk_db)
        baseline_size = os.path.getsize(temp_disk_db)

        try:
            # AC-19: Verify ancestor graph is present in sandbox
            sb_0 = sandbox_saver.get_tuple(
                {"configurable": {"thread_id": thread_id, "checkpoint_id": "cp-base-00"}}
            )
            sb_1 = sandbox_saver.get_tuple(
                {"configurable": {"thread_id": thread_id, "checkpoint_id": "cp-base-01"}}
            )
            sb_2 = sandbox_saver.get_tuple(
                {"configurable": {"thread_id": thread_id, "checkpoint_id": "cp-base-02"}}
            )

            assert sb_0 is not None
            assert sb_1 is not None
            # cp-base-02 was created after target cp-base-01 and should not be in the historical clone
            assert sb_2 is None

            # AC-20: Mutate sandbox heavily with speculative steps
            for i in range(3, 8):
                parent_id = f"cp-speculative-{i - 1}" if i > 3 else "cp-base-01"
                spec_cp = _build_checkpoint(
                    f"cp-speculative-{i}", {"state": f"speculative_val_{i}"}, parent_id
                )
                sandbox_saver.put(
                    {"configurable": {"thread_id": thread_id, "checkpoint_id": parent_id}},
                    spec_cp,
                    {"step": i, "node": f"speculative_node_{i}"},
                    {},
                )

            # Confirm sandbox now holds the speculative branch
            sb_latest = sandbox_saver.get_tuple({"configurable": {"thread_id": thread_id}})
            assert sb_latest is not None
            assert sb_latest.checkpoint["id"] == "cp-speculative-7"

            # AC-20: Ensure durable disk file is 100% bitwise identical
            post_hash = _compute_sha256(temp_disk_db)
            post_size = os.path.getsize(temp_disk_db)
            assert post_hash == baseline_hash
            assert post_size == baseline_size

        finally:
            sandbox_saver.close()

        # Reopen disk file to verify zero leakage of speculative checkpoints
        with RelateSaver(db_path=temp_disk_db, wal_enabled=True) as reopened:
            disk_latest = reopened.get_tuple({"configurable": {"thread_id": thread_id}})
            assert disk_latest is not None
            assert disk_latest.checkpoint["id"] == "cp-base-02"
            leaked_tuple = reopened.get_tuple(
                {"configurable": {"thread_id": thread_id, "checkpoint_id": "cp-speculative-7"}}
            )
            assert leaked_tuple is None

    def test_checkpoint_not_found_with_candidate_list_ac_21(
        self, temp_disk_db: str, cli_runner: CliRunner
    ) -> None:
        """AC-21: Non-existent checkpoint raises CheckpointNotFoundError listing candidates."""
        thread_id = "thread-m2-candidates-004"
        checkpoint_ids = ["cp-cand-01", "cp-cand-02", "cp-cand-03"]

        with RelateSaver(db_path=temp_disk_db, wal_enabled=True) as saver:
            cfg: RunnableConfig = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
            parent: str | None = None
            for idx, cid in enumerate(checkpoint_ids):
                cp = _build_checkpoint(cid, {"v": idx}, parent)
                saver.put(cfg, cp, {"step": idx, "node": f"step_{idx}"}, {})
                parent = cid

            # Invoking fork_in_memory with a non-existent checkpoint must raise CheckpointNotFoundError
            with pytest.raises(CheckpointNotFoundError) as exc_info:
                saver.fork_in_memory(
                    thread_id=thread_id,
                    checkpoint_id="cp-non-existent-999",
                )

            error_msg = str(exc_info.value)
            # AC-21: Verify candidate list contains available checkpoints
            assert "cp-non-existent-999" in error_msg
            assert "Available checkpoints:" in error_msg
            for cid in checkpoint_ids:
                assert cid in error_msg

        # Also test via CLI replay subcommand (AC-21 boundary behavior)
        cli_result = cli_runner.invoke(
            app, ["replay", thread_id, "cp-non-existent-999", "--db", temp_disk_db]
        )
        assert cli_result.exit_code == 1
        assert "cp-non-existent-999" in cli_result.output
        for cid in checkpoint_ids:
            assert cid in cli_result.output

    def test_visualizer_and_cli_tree_ac_22_ac_23_ac_24(
        self, temp_disk_db: str, cli_runner: CliRunner
    ) -> None:
        """AC-22, AC-23, AC-24: Terminal visualization, status annotations, and missing thread handling.

        - AC-22: relate-dag tree renders structured Unicode DAG detailing step sequences and node names.
        - AC-23: Branches and pending writes annotated with status tags ([BRANCH], [INTERRUPT], [PENDING]).
        - AC-24: Non-existent thread exits with code 1.
        """
        thread_id = "thread-m2-visualizer-005"

        with RelateSaver(db_path=temp_disk_db, wal_enabled=True) as saver:
            # Step 1: Root
            cfg_1: RunnableConfig = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
            cp_1 = _build_checkpoint("cp-vis-01", {"data": "start"})
            saver.put(cfg_1, cp_1, {"step": 1, "node": "router_agent"}, {})

            # Step 2: Main path with interrupt write (AC-23: [INTERRUPT])
            cfg_2: RunnableConfig = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_id": "cp-vis-01",
                    "checkpoint_ns": "",
                }
            }
            cp_2 = _build_checkpoint("cp-vis-02", {"data": "paused"}, "cp-vis-01")
            saver.put(cfg_2, cp_2, {"step": 2, "node": "approval_gate"}, {})
            saver.put_writes(
                {"configurable": {"thread_id": thread_id, "checkpoint_id": "cp-vis-02"}},
                [("__interrupt__", {"reason": "Waiting on manager"})],
                task_id="task-vis-hitl",
            )

            # Step 3: Branch from Step 1 with a regular pending channel write (AC-23: [BRANCH], [PENDING])
            cfg_branch: RunnableConfig = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_id": "cp-vis-01",
                    "checkpoint_ns": "",
                }
            }
            cp_branch = _build_checkpoint("cp-vis-br01", {"data": "alt_path"}, "cp-vis-01")
            saver.put(cfg_branch, cp_branch, {"step": 2, "node": "fallback_worker"}, {})
            saver.put_writes(
                {"configurable": {"thread_id": thread_id, "checkpoint_id": "cp-vis-br01"}},
                [("messages", {"content": "pending fallback response"})],
                task_id="task-branch-pending",
            )

            # Verify DAGVisualizer Python API output (AC-22, AC-23)
            vis = DAGVisualizer(saver)
            vis_output = vis.render_to_string(thread_id)

            assert thread_id in vis_output
            assert "router_agent" in vis_output
            assert "approval_gate" in vis_output
            assert "fallback_worker" in vis_output
            assert "cp-vis-01" in vis_output
            assert "cp-vis-02" in vis_output
            assert "cp-vis-br01" in vis_output
            # AC-23 status tags
            assert "[INTERRUPT]" in vis_output
            assert "[BRANCH]" in vis_output or "Branch" in vis_output
            assert "[PENDING]" in vis_output

        # AC-22: CLI relate-dag tree invocation against the real disk file
        cli_result = cli_runner.invoke(app, ["tree", thread_id, "--db", temp_disk_db])
        assert cli_result.exit_code == 0
        cli_out = cli_result.output
        assert thread_id in cli_out
        assert "router_agent" in cli_out
        assert "approval_gate" in cli_out
        assert "fallback_worker" in cli_out
        assert "[INTERRUPT]" in cli_out

        # AC-24: Non-existent thread in relate-dag tree exits with code 1
        missing_result = cli_runner.invoke(
            app, ["tree", "non-existent-thread-id", "--db", temp_disk_db]
        )
        assert missing_result.exit_code == 1
        assert "Error:" in missing_result.output
        assert "non-existent-thread-id" in missing_result.output
        assert "not found" in missing_result.output.lower()

    def test_complete_cross_journey_wiring(self, temp_disk_db: str, cli_runner: CliRunner) -> None:
        """Full end-to-end integration journey combining all Milestone 2 capabilities.

        Drives:
        1. LangGraph StateGraph pipeline running to a human-in-the-loop interrupt.
        2. CLI tree visualization inspecting the interrupted state and pending write tags.
        3. TimeTravelEngine forking an in-memory sandbox at the interrupt.
        4. Speculative execution in the sandbox leaving disk bitwise identical.
        5. Resuming original workflow on disk with user input, connecting causal parent edges.
        6. CLI tree verification of the resumed, completed graph.
        """

        class PipelineState(TypedDict):
            action: str
            status: str
            output: str

        def step_plan(state: PipelineState) -> dict[str, Any]:
            return {"status": "planned", "output": f"plan_for_{state['action']}"}

        def step_review(state: PipelineState) -> dict[str, Any]:
            decision = interrupt(f"Approval for plan: {state['output']}")
            return {"status": "reviewed", "output": f"reviewed:{decision}"}

        def step_finalize(state: PipelineState) -> dict[str, Any]:
            return {"status": "completed", "output": f"final:{state['output']}"}

        workflow = StateGraph(PipelineState)
        workflow.add_node("plan", step_plan)
        workflow.add_node("review", step_review)
        workflow.add_node("finalize", step_finalize)
        workflow.add_edge(START, "plan")
        workflow.add_edge("plan", "review")
        workflow.add_edge("review", "finalize")
        workflow.add_edge("finalize", END)

        thread_id = "thread-m2-journey-full"
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        # Step 1: Run on disk until interrupt
        with RelateSaver(db_path=temp_disk_db, wal_enabled=True) as disk_saver:
            app_graph = workflow.compile(checkpointer=disk_saver)
            int_res = app_graph.invoke(
                {"action": "deploy_service", "status": "", "output": ""},
                config,
            )
            assert "__interrupt__" in int_res

            int_tuple = disk_saver.get_tuple(config)
            assert int_tuple is not None
            int_cid = int_tuple.checkpoint["id"]

            # Fork in-memory sandbox from the interrupt while disk_saver is open
            sandbox = disk_saver.fork_in_memory(
                thread_id=thread_id,
                checkpoint_id=int_cid,
            )

        # Step 2: CLI tree visualization displays interrupt (disk saver closed)
        tree_int = cli_runner.invoke(app, ["tree", thread_id, "--db", temp_disk_db])
        assert tree_int.exit_code == 0
        assert "[INTERRUPT]" in tree_int.output
        assert "review" in tree_int.output

        # Record disk file baseline hash right before sandbox mutations
        disk_hash_before_sandbox = _compute_sha256(temp_disk_db)

        try:
            # Step 3: Run speculative replay in the isolated sandbox
            speculative_cp = _build_checkpoint(
                "cp-sandbox-hypothetical",
                {"action": "deploy_service", "status": "rejected", "output": "sandbox_rejection"},
                int_cid,
            )
            sandbox.put(
                {"configurable": {"thread_id": thread_id, "checkpoint_id": int_cid}},
                speculative_cp,
                {"step": 99, "node": "sandbox_review"},
                {},
            )
            sandbox_tuple = sandbox.get_tuple({"configurable": {"thread_id": thread_id}})
            assert sandbox_tuple is not None
            assert sandbox_tuple.checkpoint["id"] == "cp-sandbox-hypothetical"

            # Disk file remains completely unchanged
            assert _compute_sha256(temp_disk_db) == disk_hash_before_sandbox
        finally:
            sandbox.close()

        # Step 4: Resume original workflow on disk with actual human input
        with RelateSaver(db_path=temp_disk_db, wal_enabled=True) as disk_saver:
            app_graph = workflow.compile(checkpointer=disk_saver)
            final_res = app_graph.invoke(Command(resume="production_approved"), config)
            assert final_res["status"] == "completed"
            assert final_res["output"] == "final:reviewed:production_approved"

            final_tuple = disk_saver.get_tuple(config)
            assert final_tuple is not None
            final_cid = final_tuple.checkpoint["id"]

            # Verify causal edge from interrupt to resumed path
            path = disk_saver.proxy.execute_read(
                "MATCH p = (i:Checkpoint {checkpoint_id: $i_cid})-[:NEXT*]->(f:Checkpoint {checkpoint_id: $f_cid}) "
                "RETURN length(p) AS len",
                {"i_cid": int_cid, "f_cid": final_cid},
            )
            assert len(path) == 1
            assert path[0]["len"] >= 1

        # Step 5: CLI tree inspection reflects full completed journey (disk saver closed)
        final_tree = cli_runner.invoke(app, ["tree", thread_id, "--db", temp_disk_db])
        assert final_tree.exit_code == 0
        assert "plan" in final_tree.output
        assert "review" in final_tree.output
        assert "finalize" in final_tree.output
