"""Multi-agent research demonstration application with graph-native persistence.

Demonstrates an end-to-end multi-agent research workflow orchestrated by LangGraph
and backed by RelateSaver. Incorporates a Human-in-the-Loop review interrupt where
reviewer feedback is supplied in console and execution resumes from the pending
checkpoint with full causal DAG lineage tracking.

Pipeline Topology:
    START -> planner -> researcher -> human_review (interrupt) -> critic -> END
"""

from __future__ import annotations

import argparse
import logging
import sys
import uuid
from collections.abc import Sequence
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

from langgraph_checkpoint_relate.saver import RelateSaver
from langgraph_checkpoint_relate.visualizer import DAGVisualizer

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "checkpoints.rl"
DEFAULT_TOPIC = "Graph Database Checkpointing in Multi-Agent AI Systems"
DEFAULT_FEEDBACK = (
    "LGTM! Proceed to synthesis. Emphasize ACID Write-Ahead Logging and Cypher graph traversals."
)


class ResearchState(TypedDict):
    """Typed execution state for the multi-agent research workflow."""

    topic: str
    plan: list[str]
    research_notes: list[str]
    review_feedback: str | None
    final_summary: str | None
    status: str


def planner(state: ResearchState) -> dict[str, Any]:
    """Formulate research objectives and an investigation outline based on the topic."""
    topic = (state.get("topic") or "").strip() or DEFAULT_TOPIC
    plan = [
        f"1. Literature review on {topic} state management and persistence.",
        "2. Architecture benchmark: directed property graph vs flat key-value blobs.",
        "3. Verification of ACID Write-Ahead Logging under concurrent actor writes.",
        "4. Time-travel replay and causal DAG branch forensics evaluation.",
    ]
    return {
        "topic": topic,
        "plan": plan,
        "status": "planned",
    }


def researcher(state: ResearchState) -> dict[str, Any]:
    """Produce structured research draft, technical findings, and data points."""
    topic = (state.get("topic") or "").strip() or DEFAULT_TOPIC
    research_notes = [
        f"Primary finding for '{topic}': Property graphs model state lineages as first-class entities.",
        "Relate stores checkpoints in one local file and needs no separate database server.",
        "[:NEXT] relationships record which checkpoint followed which.",
        "Cypher queries over that history find loops and the step a failure came from.",
    ]
    return {
        "research_notes": research_notes,
        "status": "researched",
    }


def human_review(state: ResearchState) -> dict[str, Any]:
    """Human-in-the-Loop review interrupt node.

    Halts workflow execution and yields a pending write checkpoint containing the review prompt.
    When resumed with a Command(resume=...), the returned value provides reviewer feedback.
    """
    review_request = {
        "prompt": "Please review the research plan and preliminary findings before synthesis.",
        "topic": state.get("topic", DEFAULT_TOPIC),
        "plan": state.get("plan", []),
        "research_notes": state.get("research_notes", []),
    }
    reviewer_feedback = interrupt(review_request)
    normalized_feedback = (
        str(reviewer_feedback).strip()
        if reviewer_feedback is not None and str(reviewer_feedback).strip()
        else DEFAULT_FEEDBACK
    )
    return {
        "review_feedback": normalized_feedback,
        "status": "reviewed",
    }


def critic(state: ResearchState) -> dict[str, Any]:
    """Evaluate reviewer feedback and synthesize the finalized research summary."""
    topic = (state.get("topic") or "").strip() or DEFAULT_TOPIC
    plan = state.get("plan", [])
    notes = state.get("research_notes", [])
    feedback = state.get("review_feedback") or DEFAULT_FEEDBACK

    report_lines = [
        f"=== Synthesized Research Report: {topic} ===",
        "Strategic Objectives:",
        *[f"  - {item}" for item in plan],
        "Key Research Discoveries:",
        *[f"  * {note}" for note in notes],
        "Human Reviewer Evaluation & Guidance:",
        f'  "{feedback}"',
        "Status: Synthesized and verified across causal graph lineage.",
    ]
    return {
        "final_summary": "\n".join(report_lines),
        "status": "completed",
    }


def create_research_graph(checkpointer: RelateSaver | None = None) -> Any:
    """Build and compile the multi-agent research workflow graph.

    Topology:
        START -> planner -> researcher -> human_review -> critic -> END

    Args:
        checkpointer: RelateSaver instance for graph-native state checkpointing.

    Returns:
        Compiled LangGraph executable graph.
    """
    builder = StateGraph(ResearchState)
    builder.add_node("planner", planner)
    builder.add_node("researcher", researcher)
    builder.add_node("human_review", human_review)
    builder.add_node("critic", critic)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "researcher")
    builder.add_edge("researcher", "human_review")
    builder.add_edge("human_review", "critic")
    builder.add_edge("critic", END)

    return builder.compile(checkpointer=checkpointer)


def get_workflow_statistics(saver: RelateSaver, thread_id: str) -> dict[str, int]:
    """Retrieve Cypher execution statistics for the research thread.

    Args:
        saver: Active RelateSaver instance.
        thread_id: Target workflow execution thread identifier.

    Returns:
        Dictionary mapping metric names to integer counts.
    """
    checkpoints = saver.proxy.execute_read(
        "MATCH (c:Checkpoint {thread_id: $tid}) RETURN count(c) AS cnt",
        {"tid": thread_id},
    )
    checkpoint_count = int(checkpoints[0]["cnt"]) if checkpoints else 0

    next_edges = saver.proxy.execute_read(
        "MATCH (:Checkpoint {thread_id: $tid})-[r:NEXT]->(:Checkpoint) RETURN count(r) AS cnt",
        {"tid": thread_id},
    )
    next_count = int(next_edges[0]["cnt"]) if next_edges else 0

    writes = saver.proxy.execute_read(
        "MATCH (:Checkpoint {thread_id: $tid})-[:HAS_WRITE]->(w:Write) RETURN count(w) AS cnt",
        {"tid": thread_id},
    )
    write_count = int(writes[0]["cnt"]) if writes else 0

    return {
        "checkpoints": checkpoint_count,
        "causal_edges": next_count,
        "pending_writes": write_count,
    }


def run_research_workflow(
    topic: str = DEFAULT_TOPIC,
    *,
    db_path: str = DEFAULT_DB_PATH,
    thread_id: str | None = None,
    feedback: str | None = None,
    auto_approve: bool = False,
    saver: RelateSaver | None = None,
    print_output: bool = True,
) -> dict[str, Any]:
    """Execute the multi-agent research workflow end-to-end with HITL validation.

    Args:
        topic: Research topic to plan, research, and review.
        db_path: Database path for Relate checkpoint storage (ignored if saver provided).
        thread_id: Optional custom thread identifier (auto-generated if None).
        feedback: Explicit reviewer feedback to provide upon resumption.
        auto_approve: If True, bypass interactive input and use default guidance.
        saver: Optional pre-configured RelateSaver instance.
        print_output: Whether to print progress, report, and visualizer tree to stdout.

    Returns:
        Final state dictionary containing plan, research_notes, feedback, and summary.
    """
    active_thread_id = (
        thread_id.strip() if thread_id and thread_id.strip() else f"research-{uuid.uuid4().hex[:8]}"
    )
    run_config: RunnableConfig = {"configurable": {"thread_id": active_thread_id}}
    normalized_topic = topic.strip() if topic and topic.strip() else DEFAULT_TOPIC

    initial_state: ResearchState = {
        "topic": normalized_topic,
        "plan": [],
        "research_notes": [],
        "review_feedback": None,
        "final_summary": None,
        "status": "started",
    }

    def _execute(active_saver: RelateSaver) -> dict[str, Any]:
        graph = create_research_graph(checkpointer=active_saver)

        if print_output:
            print("=" * 70)
            print(" RelateGraphSaver Multi-Agent Research Application Demo")
            print("=" * 70)
            print(f"[*] Thread ID:     {active_thread_id}")
            print(f"[*] Database:      {active_saver.db_path}")
            print(f"[*] Topic:         {normalized_topic}")
            print("\n[Phase 1] Executing Planner and Researcher agents...")

        # Phase 1: Run graph up to Human Review interrupt
        _ = graph.invoke(initial_state, config=run_config)

        if print_output:
            tuple_paused = active_saver.get_tuple(run_config)
            cid = tuple_paused.checkpoint["id"][:8] if tuple_paused else "unknown"
            print(f"[Phase 1 Complete] Paused at human_review interrupt (Checkpoint: {cid})")
            if tuple_paused and tuple_paused.checkpoint.get("channel_values"):
                vals = tuple_paused.checkpoint["channel_values"]
                print("\n  Objectives Formulated:")
                for step in vals.get("plan", []):
                    print(f"    {step}")
                print("\n  Preliminary Findings:")
                for note in vals.get("research_notes", []):
                    print(f"    {note}")

        # Human Review Guidance Selection
        chosen_feedback: str
        if feedback is not None and feedback.strip():
            chosen_feedback = feedback.strip()
        elif auto_approve or not sys.stdin.isatty():
            chosen_feedback = DEFAULT_FEEDBACK
            if print_output:
                print(f'\n[Phase 2] Automated human review guidance:\n  "{chosen_feedback}"')
        else:
            print("\n" + "-" * 70)
            print("[Phase 2] Human-in-the-Loop Review Required")
            print(f"Topic: {normalized_topic}")
            prompt_str = f"Enter review feedback [press Enter for '{DEFAULT_FEEDBACK}']: "
            user_input = input(prompt_str).strip()
            chosen_feedback = user_input if user_input else DEFAULT_FEEDBACK

        if print_output:
            print("\n[Phase 3] Resuming workflow with human review feedback...")

        # Phase 2: Resume workflow from pending interrupt
        final_state: dict[str, Any] = graph.invoke(
            Command(resume=chosen_feedback),
            config=run_config,
        )

        if print_output:
            print("\n[Phase 3 Complete] Critic synthesis finished.")
            print("\n" + str(final_state.get("final_summary", "")))

            # Render terminal DAG execution tree
            print("\n" + "=" * 70)
            print(" Execution DAG Tree Topology (via Relate DAGVisualizer)")
            print("=" * 70)
            visualizer = DAGVisualizer(active_saver)
            visualizer.print_tree(active_thread_id)

            # Summarize Cypher graph statistics
            stats = get_workflow_statistics(active_saver, active_thread_id)
            print("\n" + "=" * 70)
            print(" Relate Graph Storage Statistics")
            print("=" * 70)
            print(f"  Checkpoints Persisted:       {stats['checkpoints']}")
            print(f"  Causal [:NEXT] Relationships: {stats['causal_edges']}")
            print(f"  Pending Channel Writes:      {stats['pending_writes']}")
            print("=" * 70)

        return final_state

    if saver is not None:
        return _execute(saver)

    with RelateSaver(db_path=db_path) as newly_opened_saver:
        return _execute(newly_opened_saver)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for the research agent demonstration."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Research Application with RelateGraphSaver Persistence",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default=DEFAULT_TOPIC,
        help="Research topic to plan, investigate, and review",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=DEFAULT_DB_PATH,
        help=f"Path to Relate database file (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--in-memory",
        action="store_true",
        help="Run execution in an isolated in-memory database (:memory:)",
    )
    parser.add_argument(
        "--thread-id",
        type=str,
        default=None,
        help="Custom thread identifier (default: auto-generated UUID)",
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Automatically provide default human reviewer feedback without prompting",
    )
    parser.add_argument(
        "--feedback",
        type=str,
        default=None,
        help="Explicit feedback to supply when resuming from human review interrupt",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for the research demonstration application."""
    args = parse_args(argv)
    target_db = ":memory:" if args.in_memory else args.db

    try:
        run_research_workflow(
            topic=args.topic,
            db_path=target_db,
            thread_id=args.thread_id,
            feedback=args.feedback,
            auto_approve=args.auto_approve,
            print_output=True,
        )
        return 0
    except Exception as exc:
        print(f"\n[ERROR] Workflow failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
