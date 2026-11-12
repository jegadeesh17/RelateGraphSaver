"""Integration tests for multi-agent research workflow and human-in-the-loop validation.

Verifies end-to-end execution of the multi-agent research graph (Planner, Researcher,
Human Review interrupt, and Critic) backed by RelateSaver graph persistence:
- AC-29: Multi-agent execution saves checkpoints, channel states, and causal :NEXT edges.
- AC-30: Human-in-the-loop interrupt halting, pending write retrieval, and resumption
  reflecting reviewer guidance in the final synthesized output.
- Durability across process and checkpointer restarts on disk.
- Cypher graph statistics and ASCII DAG tree visualization fidelity.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Generator

import pytest
from examples.research_agent import (
    DEFAULT_FEEDBACK,
    DEFAULT_TOPIC,
    ResearchState,
    create_research_graph,
    get_workflow_statistics,
    run_research_workflow,
)
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from langgraph_checkpoint_relate.saver import RelateSaver
from langgraph_checkpoint_relate.visualizer import DAGVisualizer


@pytest.fixture
def temp_disk_db() -> Generator[str]:
    """Provide a temporary disk-backed database file path and clean up after the test."""
    temp_dir = tempfile.mkdtemp(prefix="relate_workflow_qa_")
    db_file = os.path.join(temp_dir, "test_research.rl")
    try:
        yield db_file
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


class TestLangGraphResearchWorkflow:
    """End-to-end integration tests for multi-agent research workflows with RelateSaver."""

    def test_multi_agent_planning_and_research_execution_ac_29(self) -> None:
        """AC-29: Verify multi-agent planning and research steps persist to RelateSaver.

        Verifies that running the initial phase executes Planner and Researcher nodes,
        recording discrete checkpoints and causal :NEXT relationships in the Relate DB.
        """
        thread_id = "test-research-lineage-001"
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        custom_topic = "Quantum Computing in Graph Neural Networks"

        with RelateSaver(":memory:") as saver:
            graph = create_research_graph(checkpointer=saver)

            initial_state: ResearchState = {
                "topic": custom_topic,
                "plan": [],
                "research_notes": [],
                "review_feedback": None,
                "final_summary": None,
                "status": "started",
            }

            # Phase 1: Executes through Planner and Researcher, halting at Human Review interrupt
            result = graph.invoke(initial_state, config=config)
            assert "__interrupt__" in result

            # Verify paused checkpoint state
            tuple_paused = saver.get_tuple(config)
            assert tuple_paused is not None
            assert tuple_paused.checkpoint is not None

            channel_vals = tuple_paused.checkpoint.get("channel_values", {})
            assert channel_vals.get("topic") == custom_topic
            assert len(channel_vals.get("plan", [])) >= 3
            assert any("Literature review" in item for item in channel_vals["plan"])
            assert len(channel_vals.get("research_notes", [])) >= 3
            assert channel_vals.get("status") == "researched"

            # AC-29: Verify checkpoints and causal :NEXT relationships stored in Relate DB
            checkpoints = saver.proxy.execute_read(
                "MATCH (c:Checkpoint {thread_id: $tid}) RETURN c.checkpoint_id AS cid, c.step AS step ORDER BY c.step",
                {"tid": thread_id},
            )
            # Expect initial input step (-1), planner step (0), researcher step (1), and interrupt step (2)
            assert len(checkpoints) >= 3

            # Verify causal NEXT edge connections between consecutive steps
            causal_edges = saver.proxy.execute_read(
                "MATCH (p:Checkpoint {thread_id: $tid})-[:NEXT]->(c:Checkpoint {thread_id: $tid}) "
                "RETURN p.checkpoint_id AS parent, c.checkpoint_id AS child, p.step AS p_step, c.step AS c_step "
                "ORDER BY p.step",
                {"tid": thread_id},
            )
            assert len(causal_edges) >= 2
            for edge in causal_edges:
                assert edge["c_step"] > edge["p_step"]

    def test_human_in_the_loop_interrupt_and_resumption_ac_30(self) -> None:
        """AC-30: Verify interrupt halting, pending writes, and resumption with reviewer guidance.

        Halts at the human review prompt, verifies that get_tuple() reconstructs pending writes,
        supplies explicit reviewer feedback, and verifies execution completes with feedback
        incorporated into the final synthesized research summary.
        """
        thread_id = "test-hitl-resume-002"
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        reviewer_guidance = (
            "Prioritize transactional ACID benchmarking against SQLite and PostgreSQL."
        )

        with RelateSaver(":memory:") as saver:
            graph = create_research_graph(checkpointer=saver)

            initial_state: ResearchState = {
                "topic": DEFAULT_TOPIC,
                "plan": [],
                "research_notes": [],
                "review_feedback": None,
                "final_summary": None,
                "status": "started",
            }

            # Phase 1: Run until human review interrupt
            paused_output = graph.invoke(initial_state, config=config)
            assert "__interrupt__" in paused_output

            # AC-30: Verify get_tuple() contains pending writes for the interrupt
            paused_tuple = saver.get_tuple(config)
            assert paused_tuple is not None
            assert paused_tuple.pending_writes is not None
            assert len(paused_tuple.pending_writes) >= 1

            interrupted_channels = [pw[1] for pw in paused_tuple.pending_writes]
            assert "__interrupt__" in interrupted_channels

            # Verify database edge :HAS_WRITE points to Write node with __interrupt__ channel
            interrupted_cid = paused_tuple.checkpoint["id"]
            db_writes = saver.proxy.execute_read(
                "MATCH (c:Checkpoint {checkpoint_id: $cid})-[:HAS_WRITE]->(w:Write) "
                "WHERE w.channel = '__interrupt__' RETURN w.channel AS ch",
                {"cid": interrupted_cid},
            )
            assert len(db_writes) == 1
            assert db_writes[0]["ch"] == "__interrupt__"

            # Phase 2: Resume execution from pending checkpoint with reviewer guidance
            final_output = graph.invoke(Command(resume=reviewer_guidance), config=config)

            # AC-30: Verify the graph finishes with reviewer feedback reflected in output
            assert final_output.get("status") == "completed"
            assert final_output.get("review_feedback") == reviewer_guidance

            final_summary = final_output.get("final_summary", "")
            assert reviewer_guidance in final_summary
            assert "=== Synthesized Research Report:" in final_summary
            assert "Key Research Discoveries:" in final_summary

            # Verify causal NEXT edge links interrupted checkpoint to resumed checkpoint
            resumed_tuple = saver.get_tuple(config)
            assert resumed_tuple is not None
            resumed_cid = resumed_tuple.checkpoint["id"]

            causal_path = saver.proxy.execute_read(
                "MATCH p = (i:Checkpoint {checkpoint_id: $i_cid})-[:NEXT*]->(r:Checkpoint {checkpoint_id: $r_cid}) "
                "RETURN length(p) AS path_len",
                {"i_cid": interrupted_cid, "r_cid": resumed_cid},
            )
            assert len(causal_path) == 1
            assert causal_path[0]["path_len"] >= 1

    def test_durability_across_saver_restarts_ac_29_ac_30(self, temp_disk_db: str) -> None:
        """AC-29, AC-30: Verify durability across checkpointer restarts on disk.

        Runs up to human review interrupt on a disk-backed database file, closes the saver,
        re-opens a new saver instance, resumes the workflow, and verifies that the complete
        execution lineage is preserved across restarts.
        """
        thread_id = "test-durability-restart-003"
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        guidance = "Emphasize zero-cleanup in-memory time travel."

        # Session 1: Run up to interrupt on disk
        with RelateSaver(db_path=temp_disk_db, wal_enabled=True) as saver1:
            graph1 = create_research_graph(checkpointer=saver1)
            init_state: ResearchState = {
                "topic": "Distributed Graph Persistence",
                "plan": [],
                "research_notes": [],
                "review_feedback": None,
                "final_summary": None,
                "status": "started",
            }
            res1 = graph1.invoke(init_state, config=config)
            assert "__interrupt__" in res1

            tuple1 = saver1.get_tuple(config)
            assert tuple1 is not None
            cid1 = tuple1.checkpoint["id"]

        # Session 2: Fresh saver instance on the same database file
        with RelateSaver(db_path=temp_disk_db, wal_enabled=True) as saver2:
            tuple2 = saver2.get_tuple(config)
            assert tuple2 is not None
            assert tuple2.checkpoint["id"] == cid1
            assert tuple2.pending_writes is not None
            assert any(pw[1] == "__interrupt__" for pw in tuple2.pending_writes)

            # Resume workflow on reopened database
            graph2 = create_research_graph(checkpointer=saver2)
            final_res = graph2.invoke(Command(resume=guidance), config=config)
            assert final_res.get("status") == "completed"
            assert guidance in final_res.get("final_summary", "")

            tuple_after_resume = saver2.get_tuple(config)
            assert tuple_after_resume is not None
            cid2 = tuple_after_resume.checkpoint["id"]

        # Session 3: Verify complete persistent DAG lineage in a third fresh saver instance
        with RelateSaver(db_path=temp_disk_db, wal_enabled=True) as saver3:
            tuple3 = saver3.get_tuple(config)
            assert tuple3 is not None
            assert tuple3.checkpoint["id"] == cid2
            vals = tuple3.checkpoint.get("channel_values", {})
            assert vals.get("status") == "completed"
            assert vals.get("review_feedback") == guidance

            # Cypher check: Full unbroken causal chain from root checkpoint to final checkpoint
            root_records = saver3.proxy.execute_read(
                "MATCH (t:Thread {thread_id: $tid})-[:HAS_CHECKPOINT]->(c:Checkpoint) "
                "WHERE NOT ()-[:NEXT]->(c) RETURN c.checkpoint_id AS root_cid",
                {"tid": thread_id},
            )
            assert len(root_records) == 1
            root_cid = root_records[0]["root_cid"]

            chain = saver3.proxy.execute_read(
                "MATCH path = (root:Checkpoint {checkpoint_id: $root_cid})-[:NEXT*]->(leaf:Checkpoint {checkpoint_id: $leaf_cid}) "
                "RETURN length(path) AS chain_len",
                {"root_cid": root_cid, "leaf_cid": cid2},
            )
            assert len(chain) == 1
            assert chain[0]["chain_len"] >= 3

    def test_dag_visualization_and_graph_statistics(self) -> None:
        """Verify DAGVisualizer rendering and Cypher statistics on the research workflow."""
        thread_id = "test-vis-stats-004"
        with RelateSaver(":memory:") as saver:
            final_state = run_research_workflow(
                topic="Graph Lineage Diagnostics",
                thread_id=thread_id,
                auto_approve=True,
                saver=saver,
                print_output=False,
            )
            assert final_state["status"] == "completed"

            # Test DAGVisualizer render_to_string
            visualizer = DAGVisualizer(saver)
            tree_output = visualizer.render_to_string(thread_id)
            assert thread_id in tree_output
            assert "[INTERRUPT]" in tree_output
            assert "[OK]" in tree_output
            assert "channel='topic'" in tree_output
            assert "channel='review_feedback'" in tree_output

            # Test Cypher statistics helper
            stats = get_workflow_statistics(saver, thread_id)
            assert stats["checkpoints"] >= 5
            assert stats["causal_edges"] >= 4
            assert stats["pending_writes"] >= 15

    def test_cli_execution_in_memory_ac_29_ac_30(self) -> None:
        """AC-29, AC-30: Verify standalone CLI execution with --in-memory flag."""
        custom_topic = "Autonomous Multi-Agent Memory Architectures"
        cmd = [
            sys.executable,
            "examples/research_agent.py",
            "--in-memory",
            "--topic",
            custom_topic,
            "--auto-approve",
        ]
        env = dict(os.environ)
        env["PYTHONPATH"] = "src" + os.pathsep + env.get("PYTHONPATH", "")

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        assert proc.returncode == 0, f"CLI exited with error: {proc.stderr}"
        stdout = proc.stdout

        assert "RelateGraphSaver Multi-Agent Research Application Demo" in stdout
        assert custom_topic in stdout
        assert "[Phase 1 Complete] Paused at human_review interrupt" in stdout
        assert "[Phase 3 Complete] Critic synthesis finished." in stdout
        assert "=== Synthesized Research Report:" in stdout
        assert "Execution DAG Tree Topology" in stdout
        assert "[INTERRUPT]" in stdout
        assert "[OK]" in stdout
        assert "Relate Graph Storage Statistics" in stdout
        assert "Checkpoints Persisted:" in stdout

    def test_cli_execution_disk_database_ac_29_ac_30(self, temp_disk_db: str) -> None:
        """AC-29, AC-30: Verify standalone CLI execution with explicit --db and --feedback."""
        custom_guidance = "Verify that WAL write locks serialize multi-threaded graph updates."
        cmd = [
            sys.executable,
            "examples/research_agent.py",
            "--db",
            temp_disk_db,
            "--feedback",
            custom_guidance,
            "--auto-approve",
        ]
        env = dict(os.environ)
        env["PYTHONPATH"] = "src" + os.pathsep + env.get("PYTHONPATH", "")

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        assert proc.returncode == 0, f"CLI exited with error: {proc.stderr}"
        assert os.path.exists(temp_disk_db)
        assert os.path.getsize(temp_disk_db) > 0

        # Inspect persisted database to confirm reviewer feedback is reflected
        with RelateSaver(db_path=temp_disk_db, wal_enabled=True) as saver:
            threads = saver.proxy.execute_read("MATCH (t:Thread) RETURN t.thread_id AS tid")
            assert len(threads) >= 1
            active_tid = threads[0]["tid"]

            latest = saver.get_tuple({"configurable": {"thread_id": active_tid}})
            assert latest is not None
            vals = latest.checkpoint.get("channel_values", {})
            assert vals.get("review_feedback") == custom_guidance
            assert custom_guidance in vals.get("final_summary", "")

    def test_invalid_thread_id_boundary(self) -> None:
        """Verify boundary input validation: empty or whitespace thread_id is rejected."""
        with RelateSaver(":memory:") as saver:
            graph = create_research_graph(checkpointer=saver)
            invalid_config: RunnableConfig = {"configurable": {"thread_id": "   "}}

            initial_state: ResearchState = {
                "topic": DEFAULT_TOPIC,
                "plan": [],
                "research_notes": [],
                "review_feedback": None,
                "final_summary": None,
                "status": "started",
            }

            with pytest.raises(ValueError, match="thread_id must be a non-empty string"):
                graph.invoke(initial_state, config=invalid_config)

    def test_empty_topic_fallback(self) -> None:
        """Verify empty topic input gracefully falls back to default topic."""
        with RelateSaver(":memory:") as saver:
            final_state = run_research_workflow(
                topic="   ",
                auto_approve=True,
                saver=saver,
                print_output=False,
            )
            assert final_state["status"] == "completed"
            assert final_state["topic"] == DEFAULT_TOPIC
            assert DEFAULT_FEEDBACK in final_state["final_summary"]
