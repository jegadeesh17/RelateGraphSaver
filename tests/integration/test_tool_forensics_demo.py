"""End-to-end test of root-cause forensics on a real tool-calling graph run.

The key-free demo agent runs through ``graph.invoke`` with RelateSaver, and nothing is
inserted into the database by any other means (AC-45, ADR-0005).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from examples.tool_agent import TOOL_NAME, run_tool_agent
from langchain_core.messages import AIMessage, ToolMessage
from typer.testing import CliRunner

from langgraph_checkpoint_relate.cli import app
from langgraph_checkpoint_relate.saver import RelateSaver


def _tool_call_id(state: dict[str, object]) -> str:
    """Return the id of the single tool call requested in the run's AI messages."""
    messages = state["messages"]
    assert isinstance(messages, list)
    ids = [call["id"] for m in messages if isinstance(m, AIMessage) for call in m.tool_calls]
    assert len(ids) == 1
    assert isinstance(ids[0], str)
    return ids[0]


def test_ac45_trace_root_cause_names_failing_tool_from_real_run() -> None:
    """AC-45: a failing tool in a real graph.invoke run is named by trace_root_cause()."""
    with RelateSaver(":memory:") as saver:
        state = run_tool_agent(saver, "tool-demo-fail", fail=True)

        tool_messages = [m for m in state["messages"] if isinstance(m, ToolMessage)]
        assert len(tool_messages) == 1
        assert tool_messages[0].status == "error"
        call_id = _tool_call_id(state)
        assert call_id.startswith("call_")

        report = saver.trace_root_cause("tool-demo-fail")

        assert report.root_tool_name == TOOL_NAME == "fetch_source"
        assert report.root_tool_call_id == call_id
        assert "RuntimeError" in report.error_message
        assert f"Failing tool: '{TOOL_NAME}'" in report.diagnostic_summary


def test_ac45_successful_tool_call_records_result_and_no_error() -> None:
    """AC-45: with fail=False the ToolCall node holds a result and a null error."""
    with RelateSaver(":memory:") as saver:
        state = run_tool_agent(saver, "tool-demo-ok", fail=False)
        call_id = _tool_call_id(state)

        rows = saver.proxy.execute_read(
            "MATCH (c:Checkpoint {thread_id: $tid})-[:INVOKED]->(tc:ToolCall) "
            "RETURN tc.call_id AS call_id, tc.tool_name AS tool_name, "
            "tc.result_blob AS result_blob, tc.error AS error",
            {"tid": "tool-demo-ok"},
        )

        assert len(rows) == 1
        assert rows[0]["call_id"] == call_id
        assert rows[0]["tool_name"] == TOOL_NAME
        assert rows[0]["result_blob"]
        assert rows[0]["error"] is None


def test_ac45_empty_thread_id_is_rejected() -> None:
    """AC-45: invalid input (blank thread id) fails fast instead of running the graph."""
    with RelateSaver(":memory:") as saver, pytest.raises(ValueError, match="thread_id"):
        run_tool_agent(saver, "   ")


def test_ac45_cli_trace_error_prints_failing_tool_and_call_id(tmp_path: Path) -> None:
    """AC-45: relate-dag forensics trace-error on the run's file database names the tool."""
    db_file = str(tmp_path / "tool_demo.rl")
    thread_id = "tool-demo-cli"
    with RelateSaver(db_path=db_file) as saver:
        state = run_tool_agent(saver, thread_id, fail=True)
        call_id = _tool_call_id(state)

    result = CliRunner().invoke(app, ["forensics", "trace-error", thread_id, "--db", db_file])

    assert result.exit_code == 0, result.output
    assert "Root Tool Name:" in result.output
    assert TOOL_NAME in result.output
    assert call_id in result.output
