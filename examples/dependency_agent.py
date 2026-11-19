"""Key-free dependency audit that calls the real PyPI JSON API.

A deterministic agent (no LLM, no API key) audits the runtime dependencies of this project.
The planner lists the names from ``pyproject.toml`` and naively adds the lower-cased project
name, ``relategraphsaver``, which does not exist on PyPI. The lookup for that name fails with
an HTTP 404, ``ToolNode(handle_tool_errors=True)`` turns the failure into a ``ToolMessage``
with ``status="error"``, and the run still reaches the report. RelateSaver stores the failed
``lookup_pypi_package`` call, and ``trace_root_cause()`` follows its argument back to the
planner (AC-127..AC-132, ADR-0019).

Topology:
    START -> intake -> planner -> agent -> tools -> agent -> report -> END

The module never reads an environment variable or a key (AC-132).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from collections.abc import Callable, Sequence
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from langgraph_checkpoint_relate import __version__
from langgraph_checkpoint_relate.models import RootCauseReport
from langgraph_checkpoint_relate.saver import RelateSaver

DEFAULT_DB_PATH = "dependency_agent.rl"
TOOL_NAME = "lookup_pypi_package"
REQUEST_TEXT = "Audit the dependencies of RelateGraphSaver"
PYPI_URL_TEMPLATE = "https://pypi.org/pypi/{name}/json"
REQUEST_TIMEOUT_SECONDS = 10

# Runtime dependency names from pyproject.toml [project].dependencies.
RUNTIME_DEPENDENCIES = [
    "langgraph-checkpoint",
    "langgraph",
    "relate-db",
    "typer",
    "click",
    "langchain-core",
    "rich",
]

FetchJson = Callable[[str], dict[str, str]]


class DependencyState(MessagesState):
    """Graph state: the chat history plus the audit inputs and output."""

    project: str
    packages: list[str]
    report: str


def fetch_pypi_json(name: str) -> dict[str, str]:
    """Fetch a package's metadata from the PyPI JSON API.

    Sends one GET request to ``https://pypi.org/pypi/<name>/json`` with a 10 second timeout
    and a generic User-Agent that carries no personal data (AC-130).

    Args:
        name: Package name on PyPI.

    Returns:
        Only the ``name``, ``version`` and ``summary`` of the package.

    Raises:
        ValueError: If ``name`` is empty.
        LookupError: If PyPI answers HTTP 404 (the package does not exist).
        urllib.error.URLError: For any other network failure, unchanged.
    """
    if not name or not name.strip():
        raise ValueError("name must be a non-empty string")
    request = Request(
        PYPI_URL_TEMPLATE.format(name=quote(name.strip(), safe="")),
        headers={"User-Agent": f"RelateGraphSaver-demo/{__version__}"},
    )
    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            payload = json.load(response)
    except HTTPError as exc:
        if exc.code == 404:
            raise LookupError(f"PyPI returned HTTP 404: package '{name}' does not exist") from exc
        raise
    info = payload["info"]
    return {
        "name": str(info["name"]),
        "version": str(info["version"]),
        "summary": str(info.get("summary") or ""),
    }


def intake(state: DependencyState) -> dict[str, str]:
    """Read the project name from the audit request."""
    request = next(
        (str(m.content) for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        "",
    )
    match = re.search(r"dependencies of\s+(\S+)", request)
    if match is None:
        raise ValueError(f"could not find a project name in the request: {request!r}")
    return {"project": match.group(1)}


def planner(state: DependencyState) -> dict[str, list[str]]:
    """List the runtime dependencies and naively add the lower-cased project name."""
    return {"packages": [*RUNTIME_DEPENDENCIES, state["project"].lower()]}


def agent(state: DependencyState) -> dict[str, list[AnyMessage]]:
    """Deterministic stand-in for a model: request one lookup per package, then answer."""
    messages = state["messages"]
    if isinstance(messages[-1], ToolMessage):
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        failed = sum(1 for m in tool_messages if m.status == "error")
        answer = (
            f"Looked up {len(tool_messages)} packages: "
            f"{len(tool_messages) - failed} found, {failed} failed."
        )
        return {"messages": [AIMessage(content=answer)]}

    calls = [
        {"name": TOOL_NAME, "args": {"name": package}, "id": f"call_{uuid.uuid4().hex}"}
        for package in state["packages"]
    ]
    return {"messages": [AIMessage(content="", tool_calls=calls)]}


def route_after_agent(state: DependencyState) -> str:
    """Send the run to the tools node while the last AI message requests tool calls."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "report"


def _first_line(content: object) -> str:
    return str(content).strip().splitlines()[0] if str(content).strip() else ""


def report(state: DependencyState) -> dict[str, str]:
    """Summarize the versions found and the failed lookups."""
    messages = state["messages"]
    requested = {
        call["id"]: str(call["args"].get("name", "?"))
        for m in messages
        if isinstance(m, AIMessage)
        for call in m.tool_calls
    }
    found: list[str] = []
    failed: list[str] = []
    for m in messages:
        if not isinstance(m, ToolMessage):
            continue
        package = requested.get(m.tool_call_id, m.name or "?")
        if m.status == "error":
            failed.append(f"  {package}: {_first_line(m.content)}")
            continue
        try:
            info = json.loads(str(m.content))
            found.append(f"  {info['name']} {info['version']} - {info['summary']}")
        except (ValueError, KeyError, TypeError):
            found.append(f"  {package}: {_first_line(m.content)}")

    lines = [f"Dependency audit for {state['project']}", f"Found ({len(found)}):", *found]
    lines += [f"Failed lookups ({len(failed)}):", *failed]
    return {"report": "\n".join(lines)}


def _make_lookup_tool(fetch_json: FetchJson) -> BaseTool:
    """Create the lookup tool so that it calls the injected fetcher."""

    @tool
    def lookup_pypi_package(name: str) -> dict[str, str]:
        """Look up a package on PyPI and return its name, version and summary."""
        return fetch_json(name)

    return lookup_pypi_package


def create_dependency_graph(
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    *,
    fetch_json: FetchJson = fetch_pypi_json,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Build and compile the dependency audit graph.

    Topology:
        START -> intake -> planner -> agent -> tools -> agent -> report -> END

    Args:
        checkpointer: Checkpoint saver, typically a RelateSaver.
        fetch_json: Function the lookup tool calls for each package name; tests inject a fake.

    Returns:
        Compiled LangGraph executable graph.
    """
    builder = StateGraph(DependencyState)
    builder.add_node("intake", intake)
    builder.add_node("planner", planner)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode([_make_lookup_tool(fetch_json)], handle_tool_errors=True))
    builder.add_node("report", report)
    builder.add_edge(START, "intake")
    builder.add_edge("intake", "planner")
    builder.add_edge("planner", "agent")
    builder.add_conditional_edges("agent", route_after_agent, ["tools", "report"])
    builder.add_edge("tools", "agent")
    builder.add_edge("report", END)
    return builder.compile(checkpointer=checkpointer)


def run_dependency_agent(
    saver: RelateSaver,
    thread_id: str,
    *,
    fetch_json: FetchJson = fetch_pypi_json,
) -> dict[str, Any]:
    """Run the audit through ``graph.invoke`` and return the final state.

    Args:
        saver: Active RelateSaver instance the run is persisted with.
        thread_id: Thread identifier for the run.
        fetch_json: Function the lookup tool calls for each package name.

    Returns:
        Final graph state, including the ``report`` text.

    Raises:
        ValueError: If ``thread_id`` is empty.
    """
    if not thread_id.strip():
        raise ValueError("thread_id must be a non-empty string")
    graph = create_dependency_graph(checkpointer=saver, fetch_json=fetch_json)
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    final_state: dict[str, Any] = graph.invoke(
        {"messages": [HumanMessage(content=REQUEST_TEXT)]},
        config=config,
    )
    return final_state


def format_report(report: RootCauseReport) -> str:
    """Render a root-cause report as plain text lines."""
    lines = [
        f"Error Checkpoint ID: {report.error_checkpoint_id}",
        f"Error Message:       {report.error_message}",
        f"Root Tool Name:      {report.root_tool_name or 'None'}",
        f"Root Tool Call ID:   {report.root_tool_call_id or 'None'}",
    ]
    if report.origin_node is not None:
        lines.append(
            f"Argument Origin:     '{report.origin_argument}' from node "
            f"'{report.origin_node}' ({report.origin_hops} hops)"
        )
    lines.append(f"Diagnostic Summary:  {report.diagnostic_summary}")
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for the dependency audit demonstration."""
    parser = argparse.ArgumentParser(
        description="Key-free dependency audit against the real PyPI API, with root-cause forensics",
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
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: run the audit, then print the root-cause analysis."""
    args = parse_args(argv)
    thread_id = args.thread_id.strip() if args.thread_id and args.thread_id.strip() else None
    thread_id = thread_id or f"dependency-agent-{uuid.uuid4().hex[:8]}"

    try:
        with RelateSaver(db_path=args.db) as saver:
            print(f"[*] Thread ID: {thread_id}")
            print(f"[*] Database:  {args.db}")
            state = run_dependency_agent(saver, thread_id)
            print(f"\n{state['report']}")
            print("\nRoot-cause analysis:")
            print(format_report(saver.trace_root_cause(thread_id)))
        print(f"\nInspect later with: relate-dag forensics trace-error {thread_id} --db {args.db}")
        return 0
    except Exception as exc:
        print(f"\n[ERROR] Demo failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
