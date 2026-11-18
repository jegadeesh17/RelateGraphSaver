"""Integration test proving the fork, edit and replay mechanism behind the replay engine.

A failing tool-agent run is stored in a file, forked into memory, its first human message is
replaced, and the sandbox is resumed with ``invoke(None, ...)``. The source file must stay
byte-identical (AC-103) and the replay must reach a working tool call (AC-99, AC-100, AC-149).
"""

# ruff: noqa: E402

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

from examples.tool_agent import OK_URL, TOOL_NAME, create_tool_agent_graph, run_tool_agent
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from langgraph_checkpoint_relate.saver import RelateSaver
from langgraph_checkpoint_relate.time_travel import TimeTravelEngine

THREAD_ID = "replay-spike-thread"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_update_state_in_fork_replaces_human_message_and_reaches_ok_tool(tmp_path: Path) -> None:
    """Editing the first human message in a sandbox fork replays to an ok tool call.

    The earliest checkpoint ('input', step -1) has no state values yet: the first human message
    is only a pending write there. The first checkpoint that holds it is step 0, so that is the
    fork point for editing it with ``as_node="__start__"``.
    """
    db_path = tmp_path / "source.rl"

    writer = RelateSaver(str(db_path))
    try:
        run_tool_agent(writer, THREAD_ID, fail=True)
    finally:
        writer.close()
    digest_before = _sha256(db_path)

    source = RelateSaver(str(db_path), read_only=True)
    try:
        list_config: RunnableConfig = {
            "configurable": {"thread_id": THREAD_ID, "checkpoint_ns": ""}
        }
        by_step = {
            int(c.metadata["step"]): str(c.config["configurable"]["checkpoint_id"])
            for c in source.list(list_config)
            if c.metadata is not None and "step" in c.metadata
        }
        assert {-1, 0} <= set(by_step)
        fork_checkpoint_id = by_step[0]

        sandbox = TimeTravelEngine.fork_in_memory(source, THREAD_ID, fork_checkpoint_id)
        try:
            graph = create_tool_agent_graph(checkpointer=sandbox)
            cfg: RunnableConfig = {
                "configurable": {
                    "thread_id": THREAD_ID,
                    "checkpoint_ns": "",
                    "checkpoint_id": fork_checkpoint_id,
                }
            }
            messages = graph.get_state(cfg).values.get("messages", [])
            assert messages, "the fork checkpoint must already hold the first human message"
            edited = messages[0].model_copy(update={"content": OK_URL})
            new_cfg = graph.update_state(cfg, {"messages": [edited]}, as_node="__start__")
            final = graph.invoke(
                None,
                {
                    "configurable": {
                        "thread_id": THREAD_ID,
                        "checkpoint_id": new_cfg["configurable"]["checkpoint_id"],
                    }
                },
            )

            final_messages = final["messages"]
            assert str(final_messages[-1].content).startswith("Source fetched")
            human = [m for m in final_messages if isinstance(m, HumanMessage)]
            assert len(human) == 1
            assert human[0].content == OK_URL
            tool_messages = [m for m in final_messages if isinstance(m, ToolMessage)]
            assert len(tool_messages) == 1
            assert tool_messages[0].status != "error"

            rows: list[dict[str, Any]] = sandbox.proxy.execute_read(
                "MATCH (tc:ToolCall {tool_name: $tool_name}) "
                "RETURN tc.result_blob AS result_blob, tc.error AS error",
                {"tool_name": TOOL_NAME},
            )
            # The fork also carries the recorded call, still without an outcome because its
            # result was not in the fork checkpoint's channel values; the replay adds its own.
            resolved = [row for row in rows if row["result_blob"] is not None]
            assert len(resolved) == 1
            assert not resolved[0]["error"]
            assert not any(row["error"] for row in rows)
        finally:
            sandbox.close()
    finally:
        source.close()

    assert _sha256(db_path) == digest_before
