"""Tests for the optional recorded OpenRouter run (AC-140).

No test reads ``OPENROUTER_API_KEY`` or reaches the network: the script is started with the key
removed from the environment, and the in-process run uses a fake chat model.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import textwrap
import types
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from examples import openrouter_agent
from examples.build_demo_store import build_demo_store
from examples.dependency_agent import TOOL_NAME
from examples.openrouter_agent import (
    API_KEY_ENV,
    OPENROUTER_BASE_URL,
    create_llm_graph,
    main,
    run_llm_agent,
)
from langchain_core.messages import AIMessage, BaseMessage
from tests.fakes.fake_pypi import FakePyPI
from typer.testing import CliRunner

from langgraph_checkpoint_relate.cli import app
from langgraph_checkpoint_relate.saver import RelateSaver

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "examples" / "openrouter_agent.py"
COMMITTED_LLM_STORE = REPO_ROOT / "examples" / "data" / "demo-llm.rl"
LLM_THREAD = "demo-llm"

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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _invoke(*args: str) -> tuple[int, str]:
    result = CliRunner().invoke(app, list(args))
    return result.exit_code, result.output


@pytest.fixture
def llm_store_copy(tmp_path: Path) -> Path:
    """A copy of the committed recorded run; skipped when the optional store is not committed."""
    if not COMMITTED_LLM_STORE.is_file():
        pytest.skip("examples/data/demo-llm.rl is not committed (optional recorded run, AC-141)")
    store = tmp_path / "demo-llm.rl"
    shutil.copyfile(COMMITTED_LLM_STORE, store)
    return store


def test_ac141_committed_llm_store_holds_one_recorded_run() -> None:
    """AC-141: the recorded store holds only the demo-llm thread, eight lookups, one 404 failure."""
    if not COMMITTED_LLM_STORE.is_file():
        pytest.skip("examples/data/demo-llm.rl is not committed (optional recorded run, AC-141)")
    with RelateSaver(str(COMMITTED_LLM_STORE), read_only=True) as saver:
        threads = saver.proxy.execute_read(
            "MATCH (c:Checkpoint) RETURN DISTINCT c.thread_id AS tid", {}
        )
        rows = saver.proxy.execute_read(
            "MATCH (c:Checkpoint)-[:INVOKED]->(tc:ToolCall) "
            "RETURN tc.tool_name AS name, tc.args_blob AS args, tc.error AS error",
            {},
        )
        calls = [
            (str(r["name"]), str(saver.serde.loads(bytes(r["args"]))["name"]), r["error"])
            for r in rows
        ]

    assert {str(t["tid"]) for t in threads} == {LLM_THREAD}
    assert len(calls) == 8
    assert {name for name, _, _ in calls} == {TOOL_NAME}
    failures = [(package, error) for _, package, error in calls if error is not None]
    assert [package for package, _ in failures] == ["relategraphsaver"]
    assert "404" in failures[0][1]


def test_ac141_committed_llm_store_is_read_only_inspectable(llm_store_copy: Path) -> None:
    """AC-141: the read-only commands answer on the recorded store and leave its bytes alone."""
    store = llm_store_copy
    before = _sha256(store)
    commands = [
        ["tree", LLM_THREAD],
        ["forensics", "failures"],
        ["forensics", "loops", LLM_THREAD],
        ["forensics", "trace-error", LLM_THREAD],
    ]
    outputs: dict[str, str] = {}
    for command in commands:
        code, output = _invoke(*command, "--db", str(store))
        assert code == 0, (command, output)
        assert _sha256(store) == before, command
        assert [p.name for p in store.parent.iterdir()] == ["demo-llm.rl"], command
        outputs[" ".join(command)] = output

    assert "[OK]" in outputs[f"tree {LLM_THREAD}"]
    assert "lookup_pypi_package" in outputs["forensics failures"]
    flat = " ".join(outputs[f"forensics trace-error {LLM_THREAD}"].replace("|", " ").split())
    assert "Root Tool Name: lookup_pypi_package" in flat
    assert "'relategraphsaver' does not exist" in flat
    assert "404" in flat


def test_ac141_unknown_thread_on_llm_store_exits_1(llm_store_copy: Path) -> None:
    """AC-141 (invalid input): asking for a thread the recorded store does not hold exits 1."""
    code, output = _invoke("tree", "no-such-thread", "--db", str(llm_store_copy))
    assert code == 1
    assert "not found" in output
    assert _sha256(llm_store_copy) == _sha256(COMMITTED_LLM_STORE)


def test_llm_run_leaves_a_sibling_demo_store_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC-141: running the recorded-run path beside demo.rl leaves demo.rl byte-identical."""
    data = tmp_path / "data"
    data.mkdir()
    demo = data / "demo.rl"
    build_demo_store(demo, fetch_json=FakePyPI())
    before = _sha256(demo)
    monkeypatch.setattr(openrouter_agent, "fetch_pypi_json", FakePyPI())

    with RelateSaver(db_path=str(data / "demo-llm.rl")) as saver:
        run_llm_agent(_ScriptedModel(), saver, LLM_THREAD)

    assert _sha256(demo) == before
    assert sorted(p.name for p in data.iterdir()) == ["demo-llm.rl", "demo.rl"]


def test_main_builds_the_chat_model_from_the_key_and_writes_a_cli_readable_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Wiring: main() passes the key and OpenRouter URL to ChatOpenAI, and the store it writes
    is read back by the CLI. The chat model is a stand-in; no request leaves the process.
    """
    placeholder = "placeholder-for-test-only"
    monkeypatch.setenv(API_KEY_ENV, placeholder)
    captured: dict[str, Any] = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        def bind_tools(self, tools: Sequence[Any]) -> _ScriptedModel:
            return _ScriptedModel().bind_tools(tools)

    fake_module: Any = types.ModuleType("langchain_openai")
    fake_module.ChatOpenAI = FakeChatOpenAI
    monkeypatch.setitem(sys.modules, "langchain_openai", fake_module)
    monkeypatch.setattr(openrouter_agent, "fetch_pypi_json", FakePyPI())
    db = tmp_path / "llm.rl"

    assert main(["--model", "vendor/model", "--db", str(db), "--thread-id", "llm-cli"]) == 0

    assert captured["base_url"] == OPENROUTER_BASE_URL
    assert captured["model"] == "vendor/model"
    assert captured["api_key"].get_secret_value() == placeholder
    out = capsys.readouterr().out
    assert f"relate-dag tree llm-cli --db {db}" in out
    assert [p.name for p in tmp_path.iterdir()] == ["llm.rl"]

    code, output = _invoke("forensics", "failures", "--db", str(db))
    assert code == 0, output
    assert "lookup_pypi_package" in output
    code, output = _invoke("tree", "llm-cli", "--db", str(db))
    assert code == 0, output
    assert "[OK]" in output


@pytest.fixture
def recorded_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A store in tmp_path that already holds a finished run on ``LLM_THREAD``."""
    monkeypatch.setattr(openrouter_agent, "fetch_pypi_json", FakePyPI())
    store = tmp_path / "recorded.rl"
    with RelateSaver(db_path=str(store)) as saver:
        run_llm_agent(_ScriptedModel(), saver, LLM_THREAD)
    return store


def test_ac141_existing_thread_is_refused_before_model_import_or_network(
    recorded_store: Path,
) -> None:
    """AC-141: a thread that already holds a run is refused with an ``Error:`` line.

    The langchain_openai import and every network route are blocked, so a refusal that came
    after them would exit 99. The store file must be byte-identical afterwards.
    """
    before = _sha256(recorded_store)
    env = _env_without_key(**{API_KEY_ENV: "placeholder-for-test-only"})

    result = _run_guarded(
        env,
        recorded_store.parent,
        "--model",
        "vendor/model",
        "--db",
        str(recorded_store),
        "--thread-id",
        LLM_THREAD,
    )

    assert result.returncode == 1, result.stderr
    assert "FORBIDDEN" not in result.stderr
    error_lines = [line for line in result.stderr.splitlines() if line.startswith("Error:")]
    assert len(error_lines) == 1, result.stderr
    assert "already holds a recorded run" in error_lines[0]
    assert "--thread-id" in error_lines[0]
    assert "--db" in error_lines[0]
    assert _sha256(recorded_store) == before
    assert [p.name for p in recorded_store.parent.iterdir()] == ["recorded.rl"]


def test_ac141_fresh_thread_on_a_recorded_store_is_not_refused(
    recorded_store: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """AC-141: a new thread id on the same store passes the guard and records its own run."""
    monkeypatch.setenv(API_KEY_ENV, "placeholder-for-test-only")

    class FakeChatOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def bind_tools(self, tools: Sequence[Any]) -> _ScriptedModel:
            return _ScriptedModel().bind_tools(tools)

    fake_module: Any = types.ModuleType("langchain_openai")
    fake_module.ChatOpenAI = FakeChatOpenAI
    monkeypatch.setitem(sys.modules, "langchain_openai", fake_module)
    argv = ["--model", "vendor/model", "--db", str(recorded_store), "--thread-id", "second-run"]

    assert main(argv) == 0

    assert "Error:" not in capsys.readouterr().err
    with RelateSaver(db_path=str(recorded_store), read_only=True) as saver:
        rows = saver.proxy.execute_read(
            "MATCH (c:Checkpoint) RETURN DISTINCT c.thread_id AS tid", {}
        )
    assert {str(r["tid"]) for r in rows} == {LLM_THREAD, "second-run"}
