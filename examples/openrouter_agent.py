"""Optional recorded run of a real LLM agent through OpenRouter (AC-139..AC-141).

A model chosen by the caller is bound to the ``lookup_pypi_package`` tool from
``examples/dependency_agent.py`` and runs once, with every checkpoint and tool call stored by
RelateSaver. The run is meant to be recorded into its own store, ``examples/data/demo-llm.rl``,
separate from the key-free ``demo.rl`` that ``build_demo_store.py`` writes.

The API key comes only from the ``OPENROUTER_API_KEY`` environment variable of the running
process. This module never reads a ``.env`` file. With the key unset or empty it prints an
``Error:`` line to stderr and exits 1 before importing ``langchain_openai`` (the ``llm`` extra) and
before any network access (AC-140).

Usage:
    python examples/openrouter_agent.py --model <openrouter-model-id> [--db PATH] [--thread-id ID]
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

# Allow ``python examples/openrouter_agent.py`` to import the sibling example modules.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from examples.dependency_agent import (  # noqa: E402
    RUNTIME_DEPENDENCIES,
    _make_lookup_tool,
    fetch_pypi_json,
)
from langchain_core.messages import (  # noqa: E402
    AnyMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.runnables import RunnableConfig  # noqa: E402
from langgraph.checkpoint.base import BaseCheckpointSaver  # noqa: E402
from langgraph.graph import START, MessagesState, StateGraph  # noqa: E402
from langgraph.graph.state import CompiledStateGraph  # noqa: E402
from langgraph.prebuilt import ToolNode, tools_condition  # noqa: E402
from pydantic import SecretStr  # noqa: E402

from langgraph_checkpoint_relate.saver import RelateSaver  # noqa: E402

API_KEY_ENV = "OPENROUTER_API_KEY"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_DB_PATH = Path(__file__).resolve().parent / "data" / "demo-llm.rl"
DEFAULT_THREAD_ID = "demo-llm"
# Upper bound on graph steps, so a model that keeps calling tools cannot run up the bill.
RECURSION_LIMIT = 12
SYSTEM_PROMPT = (
    "You audit Python package dependencies. Use the lookup_pypi_package tool once for every "
    "package you are asked about. When all lookups are back, reply with a short summary of the "
    "versions found and the packages that could not be found."
)


class ToolBindableModel(Protocol):
    """The part of a chat model this agent needs: tool binding and invocation."""

    def bind_tools(self, tools: Sequence[Any]) -> Any:
        """Return a runnable that may answer with tool calls."""


def build_request() -> str:
    """Return the audit request: the runtime dependencies plus the project's own name."""
    packages = [*RUNTIME_DEPENDENCIES, "relategraphsaver"]
    return "Audit these packages on PyPI: " + ", ".join(packages)


def create_llm_graph(
    model: ToolBindableModel,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Build and compile ``START -> agent <-> tools`` around a tool-calling chat model.

    Args:
        model: Chat model with ``bind_tools``, for example ``ChatOpenAI`` pointed at OpenRouter.
        checkpointer: Checkpoint saver, typically a RelateSaver.

    Returns:
        Compiled LangGraph executable graph.
    """
    lookup_tool = _make_lookup_tool(fetch_pypi_json)
    model_with_tools = model.bind_tools([lookup_tool])

    def agent(state: MessagesState) -> dict[str, list[BaseMessage]]:
        reply = model_with_tools.invoke(state["messages"])
        return {"messages": [reply]}

    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode([lookup_tool], handle_tool_errors=True))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition, ["tools", "__end__"])
    builder.add_edge("tools", "agent")
    return builder.compile(checkpointer=checkpointer)


def run_llm_agent(
    model: ToolBindableModel,
    saver: RelateSaver,
    thread_id: str,
) -> dict[str, Any]:
    """Run the audit once through ``graph.invoke`` and return the final state.

    Raises:
        ValueError: If ``thread_id`` is empty.
    """
    if not thread_id.strip():
        raise ValueError("thread_id must be a non-empty string")
    graph = create_llm_graph(model, checkpointer=saver)
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": RECURSION_LIMIT,
    }
    messages: list[AnyMessage] = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=build_request()),
    ]
    final_state: dict[str, Any] = graph.invoke({"messages": messages}, config=config)
    return final_state


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for the recorded LLM run."""
    parser = argparse.ArgumentParser(
        description="Record one OpenRouter LLM run, with tool calls, into a Relate store",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="OpenRouter model id, for example vendor/model-name",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=str(DEFAULT_DB_PATH),
        help=f"Path to Relate database file (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--thread-id",
        type=str,
        default=DEFAULT_THREAD_ID,
        help=f"Thread identifier (default: {DEFAULT_THREAD_ID})",
    )
    return parser.parse_args(argv)


def _fail(message: str) -> int:
    print(f"Error: {message}", file=sys.stderr)
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: record one run and print the answer."""
    args = parse_args(argv)
    api_key = os.environ.get(API_KEY_ENV, "").strip()
    if not api_key:
        return _fail(
            f"{API_KEY_ENV} is not set. Export it in your shell before running this script; "
            "the script never reads a .env file."
        )
    model_id = args.model.strip()
    if not model_id:
        return _fail("--model must be a non-empty OpenRouter model id")
    thread_id = args.thread_id.strip()
    if not thread_id:
        return _fail("--thread-id must be a non-empty string")

    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        return _fail('langchain-openai is not installed. Install the extra: pip install ".[llm]"')

    model = ChatOpenAI(base_url=OPENROUTER_BASE_URL, api_key=SecretStr(api_key), model=model_id)

    try:
        with RelateSaver(db_path=args.db) as saver:
            print(f"[*] Model:     {model_id}")
            print(f"[*] Thread ID: {thread_id}")
            print(f"[*] Database:  {args.db}")
            state = run_llm_agent(model, saver, thread_id)
            print(f"\n{state['messages'][-1].content}")
        print(f"\nInspect later with: relate-dag tree {thread_id} --db {args.db}")
        return 0
    except Exception as exc:
        return _fail(f"recorded run failed: {exc}")


if __name__ == "__main__":
    sys.exit(main())
