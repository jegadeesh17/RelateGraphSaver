"""Key-free tool-calling demonstration with graph-native forensics.

A deterministic agent (no LLM, no API key) asks a ``ToolNode`` to run one tool. By default
the tool fails on purpose; ``ToolNode(handle_tool_errors=True)`` turns the exception into a
``ToolMessage`` with ``status="error"`` so the run completes normally. RelateSaver records the
tool call and its outcome as a ``:ToolCall`` node, and ``trace_root_cause()`` then names the
failing tool from data the run itself produced (AC-45, ADR-0005).

Topology:
    START -> agent -> tools -> agent -> END
"""

from __future__ import annotations

import argparse
import sys
import uuid
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from langgraph_checkpoint_relate.models import RootCauseReport
from langgraph_checkpoint_relate.saver import RelateSaver

DEFAULT_DB_PATH = "tool_agent.rl"
TOOL_NAME = "fetch_source"
OK_URL = "https://example.org/report"
UNREACHABLE_URL = "https://unreachable.invalid/report"


@tool
def fetch_source(url: str) -> str:
    """Fetch a document from a URL (simulated, no network access).

    Raises RuntimeError for the unreachable demo host so the tool fails on purpose.
    """
    if url == UNREACHABLE_URL:
        raise RuntimeError(f"upstream unreachable: could not fetch {url}")
    return f"Contents of {url}: Relate stores checkpoints as a causal graph."


def agent(state: MessagesState) -> dict[str, list[AnyMessage]]:
    """Deterministic stand-in for a model: request one tool call, then answer."""
    messages = state["messages"]
    if any(isinstance(message, ToolMessage) for message in messages):
        last_tool_message = next(m for m in reversed(messages) if isinstance(m, ToolMessage))
        if last_tool_message.status == "error":
            answer = f"The source could not be fetched: {last_tool_message.content}"
        else:
            answer = f"Source fetched: {last_tool_message.content}"
        return {"messages": [AIMessage(content=answer)]}

    url = next((str(m.content) for m in reversed(messages) if isinstance(m, HumanMessage)), OK_URL)
    call = {"name": TOOL_NAME, "args": {"url": url}, "id": f"call_{uuid.uuid4().hex}"}
    return {"messages": [AIMessage(content="", tool_calls=[call])]}


def route_after_agent(state: MessagesState) -> str:
    """Send the run to the tools node while the last AI message requests tool calls."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return END


def create_tool_agent_graph(
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Build and compile the tool-calling graph.

    Topology:
        START -> agent -> tools (when the last AI message has tool calls, else END)
        tools -> agent

    Args:
        checkpointer: Checkpoint saver, typically a RelateSaver.

    Returns:
        Compiled LangGraph executable graph.
    """
    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode([fetch_source], handle_tool_errors=True))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route_after_agent, ["tools", END])
    builder.add_edge("tools", "agent")
    return builder.compile(checkpointer=checkpointer)


def run_tool_agent(saver: RelateSaver, thread_id: str, *, fail: bool = True) -> dict[str, Any]:
    """Run the demo through ``graph.invoke`` and return the final state.

    Args:
        saver: Active RelateSaver instance the run is persisted with.
        thread_id: Thread identifier for the run.
        fail: When True (default) the tool raises and the ToolMessage carries an error.

    Returns:
        Final graph state containing the message history.
    """
    if not thread_id.strip():
        raise ValueError("thread_id must be a non-empty string")
    graph = create_tool_agent_graph(checkpointer=saver)
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    url = UNREACHABLE_URL if fail else OK_URL
    final_state: dict[str, Any] = graph.invoke(
        {"messages": [HumanMessage(content=url)]},
        config=config,
    )
    return final_state


def format_report(report: RootCauseReport) -> str:
    """Render a root-cause report as plain text lines."""
    return "\n".join(
        [
            f"Error Checkpoint ID: {report.error_checkpoint_id}",
            f"Error Message:       {report.error_message}",
            f"Root Tool Name:      {report.root_tool_name or 'None'}",
            f"Root Tool Call ID:   {report.root_tool_call_id or 'None'}",
            f"Diagnostic Summary:  {report.diagnostic_summary}",
        ]
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for the tool-calling demonstration."""
    parser = argparse.ArgumentParser(
        description="Key-free tool-calling demo with RelateGraphSaver root-cause forensics",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=DEFAULT_DB_PATH,
        help=f"Path to Relate database file (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--thread-id",
        type=str,
        default=None,
        help="Custom thread identifier (default: auto-generated)",
    )
    parser.add_argument(
        "--no-fail",
        action="store_true",
        help="Let the tool succeed instead of failing on purpose",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: run the demo, then print the root-cause analysis."""
    args = parse_args(argv)
    thread_id = args.thread_id.strip() if args.thread_id and args.thread_id.strip() else None
    thread_id = thread_id or f"tool-agent-{uuid.uuid4().hex[:8]}"

    try:
        with RelateSaver(db_path=args.db) as saver:
            print(f"[*] Thread ID: {thread_id}")
            print(f"[*] Database:  {args.db}")
            state = run_tool_agent(saver, thread_id, fail=not args.no_fail)
            print(f"[*] Final answer: {state['messages'][-1].content}")
            print("\nRoot-cause analysis:")
            print(format_report(saver.trace_root_cause(thread_id)))
        print(f"\nInspect later with: relate-dag forensics trace-error {thread_id} --db {args.db}")
        return 0
    except Exception as exc:
        print(f"\n[ERROR] Demo failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
