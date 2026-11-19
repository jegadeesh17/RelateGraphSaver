"""Tests for the optional recorded OpenRouter run (AC-140).

No test reads ``OPENROUTER_API_KEY`` or reaches the network: the script is started with the key
removed from the environment, and the in-process run uses a fake chat model.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from examples import openrouter_agent
from examples.dependency_agent import TOOL_NAME
from examples.openrouter_agent import API_KEY_ENV, create_llm_graph, run_llm_agent
from langchain_core.messages import AIMessage, BaseMessage
from tests.fakes.fake_pypi import FakePyPI

from langgraph_checkpoint_relate.saver import RelateSaver

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "examples" / "openrouter_agent.py"

# Runs the script with every route to the network, and the langchain_openai import, replaced
# by a hard exit. Exit code 99 and a FORBIDDEN line on stderr mean the script tried one.
_GUARD = textwrap.dedent(
    """
    import importlib.abc, os, runpy, socket, sys

    def forbidden(*args, **kwargs):
        sys.stderr.write("FORBIDDEN: network or langchain_openai access\\n")
        sys.stderr.flush()
        os._exit(99)

    class BlockLlmExtra(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name.split(".")[0] == "langchain_openai":
                forbidden()
            return None

    sys.meta_path.insert(0, BlockLlmExtra())
    socket.socket.connect = forbidden
    socket.getaddrinfo = forbidden
    socket.create_connection = forbidden
    sys.argv = [sys.argv[1], *sys.argv[2:]]
    runpy.run_path(sys.argv[0], run_name="__main__")
    """
)


def _env_without_key(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != API_KEY_ENV}
    env.update(extra)
    return env


def _run_guarded(env: dict[str, str], cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", _GUARD, str(SCRIPT), *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_ac140_unset_key_exits_1_with_error_line_and_no_network(tmp_path: Path) -> None:
    """AC-140: with the key unset the script exits 1 with an ``Error:`` line and no network."""
    db = tmp_path / "llm.rl"
    result = _run_guarded(_env_without_key(), tmp_path, "--model", "vendor/model", "--db", str(db))

    assert result.returncode == 1, result.stderr
    assert "FORBIDDEN" not in result.stderr
    assert any(line.startswith("Error:") for line in result.stderr.splitlines()), result.stderr
    assert API_KEY_ENV in result.stderr
    assert not db.exists(), "no store may be created before the key check passes"


def test_ac140_empty_key_exits_1_with_error_line_and_no_network(tmp_path: Path) -> None:
    """AC-140: an empty or blank key is treated like an unset one."""
    for blank in ("", "   "):
        result = _run_guarded(
            _env_without_key(**{API_KEY_ENV: blank}), tmp_path, "--model", "vendor/model"
        )

        assert result.returncode == 1, result.stderr
        assert "FORBIDDEN" not in result.stderr
        assert any(line.startswith("Error:") for line in result.stderr.splitlines())


def test_ac140_model_is_required(tmp_path: Path) -> None:
    """AC-140: the script has no default model; omitting --model is a usage error."""
    result = _run_guarded(_env_without_key(), tmp_path)

    assert result.returncode == 2
    assert "--model" in result.stderr


def test_ac140_env_example_lists_the_key_name_empty() -> None:
    """AC-140: the example environment file lists OPENROUTER_API_KEY= with no value."""
    lines = (REPO_ROOT / ".env.example").read_text(encoding="utf-8").splitlines()

    assert f"{API_KEY_ENV}=" in lines


def test_ac140_script_loads_no_dotenv_file() -> None:
    """AC-140: the script loads no .env file; the key comes from the process environment."""
    source = SCRIPT.read_text(encoding="utf-8")

    assert "dotenv" not in source


class _ScriptedModel:
    """Chat model stand-in: requests one lookup per package, then answers."""

    def __init__(self) -> None:
        self.calls = 0

    def bind_tools(self, tools: Sequence[Any]) -> _ScriptedModel:
        assert [t.name for t in tools] == [TOOL_NAME]
        return self

    def invoke(self, messages: Sequence[BaseMessage]) -> AIMessage:
        self.calls += 1
        if self.calls == 1:
            calls = [
                {"name": TOOL_NAME, "args": {"name": name}, "id": f"call_{i}"}
                for i, name in enumerate(["typer", "relategraphsaver"])
            ]
            return AIMessage(content="", tool_calls=calls)
        return AIMessage(content="typer found, relategraphsaver missing")


def test_in_process_run_records_the_tool_calls_with_a_fake_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The graph stores the model's tool calls through RelateSaver and ends on its answer."""
    fake = FakePyPI()
    monkeypatch.setattr(openrouter_agent, "fetch_pypi_json", fake)
    db = tmp_path / "llm.rl"

    with RelateSaver(db_path=str(db)) as saver:
        state = run_llm_agent(_ScriptedModel(), saver, "llm-thread")
        rows = saver.proxy.execute_read(
            "MATCH (c:Checkpoint)-[:INVOKED]->(tc:ToolCall) "
            "RETURN tc.tool_name AS name, tc.error AS error",
            {},
        )

    assert state["messages"][-1].content == "typer found, relategraphsaver missing"
    assert sorted((str(r["name"]), r["error"] is not None) for r in rows) == [
        (TOOL_NAME, False),
        (TOOL_NAME, True),
    ]
    assert fake.calls == ["typer", "relategraphsaver"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["llm.rl"]


def test_run_rejects_empty_thread_id(tmp_path: Path) -> None:
    """An empty thread id is refused before any run starts."""
    with RelateSaver(db_path=str(tmp_path / "llm.rl")) as saver:
        with pytest.raises(ValueError, match="thread_id"):
            run_llm_agent(_ScriptedModel(), saver, "  ")


def test_graph_compiles_without_a_checkpointer() -> None:
    """The graph can be built without a saver, for inspection."""
    assert create_llm_graph(_ScriptedModel()) is not None
