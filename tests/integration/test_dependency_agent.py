"""Tests for the key-free dependency agent and its PyPI fetcher (AC-127..AC-132)."""

from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
import tomllib
import urllib.error
from pathlib import Path
from typing import Any

import pytest
from examples import dependency_agent
from examples.dependency_agent import (
    TOOL_NAME,
    create_dependency_graph,
    fetch_pypi_json,
    main,
    run_dependency_agent,
)
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from tests.fakes.fake_pypi import FakePyPI, UnreachablePyPI

from langgraph_checkpoint_relate import __version__
from langgraph_checkpoint_relate.saver import RelateSaver

REPO_ROOT = Path(__file__).resolve().parents[2]
THREAD_ID = "dependency-agent-test"
THIS_FILE = "tests/integration/test_dependency_agent.py"


class _FakeResponse(io.BytesIO):
    """Minimal ``urlopen`` response: a context manager that can be read as JSON."""


def _http_error(url: str, code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(url, code, "msg", None, None)  # type: ignore[arg-type]


def _node_order(saver: RelateSaver, fake: FakePyPI, thread_id: str) -> list[str]:
    """Return the nodes the stored run executed, oldest first, from each snapshot's ``next``."""
    graph = create_dependency_graph(saver, fetch_json=fake)
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    history = list(graph.get_state_history(config))
    return [node for snapshot in reversed(history) for node in snapshot.next if node != "__start__"]


def test_ac127_stored_run_visits_nodes_in_order_and_plans_relategraphsaver() -> None:
    """AC-127: the stored history shows the node order and the planner's package list."""
    fake = FakePyPI()
    with RelateSaver(":memory:") as saver:
        state = run_dependency_agent(saver, THREAD_ID, fetch_json=fake)
        order = _node_order(saver, fake, THREAD_ID)

    assert order == ["intake", "planner", "agent", "tools", "agent", "report"]
    assert state["project"] == "RelateGraphSaver"
    assert "relategraphsaver" in state["packages"]
    assert state["packages"][:-1] == dependency_agent.RUNTIME_DEPENDENCIES
    assert sorted(fake.calls) == sorted(state["packages"])


def test_ac127_tool_calls_use_the_lookup_tool_with_call_ids() -> None:
    """AC-127: one lookup_pypi_package call per package, with call_<hex> ids."""
    with RelateSaver(":memory:") as saver:
        state = run_dependency_agent(saver, THREAD_ID, fetch_json=FakePyPI())

    calls = [c for m in state["messages"] if isinstance(m, AIMessage) for c in m.tool_calls]
    assert len(calls) == len(state["packages"])
    assert {c["name"] for c in calls} == {TOOL_NAME} == {"lookup_pypi_package"}
    assert all(str(c["id"]).startswith("call_") for c in calls)


def test_ac128_failed_lookup_is_stored_and_reported(tmp_path: Path) -> None:
    """AC-128: the 404 lookup is stored with an error and the run still reaches report."""
    db_file = str(tmp_path / "dependency.rl")
    with RelateSaver(db_path=db_file) as saver:
        state = run_dependency_agent(saver, THREAD_ID, fetch_json=FakePyPI())
        rows = saver.proxy.execute_read(
            "MATCH (c:Checkpoint {thread_id: $tid})-[:INVOKED]->(tc:ToolCall) "
            "RETURN tc.tool_name AS tool_name, tc.error AS error",
            {"tid": THREAD_ID},
        )

    failures = [r for r in rows if r["error"]]
    assert len(rows) == len(state["packages"])
    assert len(failures) == 1
    assert failures[0]["tool_name"] == TOOL_NAME
    assert "404" in failures[0]["error"]
    tool_errors = [
        m for m in state["messages"] if isinstance(m, ToolMessage) and m.status == "error"
    ]
    assert len(tool_errors) == 1
    assert "Failed lookups (1)" in state["report"]
    assert "relategraphsaver" in state["report"]
    assert "404" in state["report"]
    assert "langgraph-checkpoint 1.2.3" in state["report"]


def test_ac129_root_cause_points_at_the_planner() -> None:
    """AC-129: trace_root_cause names the tool and the planner as the argument's origin."""
    with RelateSaver(":memory:") as saver:
        run_dependency_agent(saver, THREAD_ID, fetch_json=FakePyPI())
        report = saver.trace_root_cause(THREAD_ID)

    assert report.root_tool_name == "lookup_pypi_package"
    assert report.origin_node == "planner"
    assert report.origin_hops is not None and report.origin_hops >= 2


def test_ac127_blank_thread_id_is_rejected() -> None:
    """AC-127: invalid input (blank thread id) fails before the graph runs."""
    fake = FakePyPI()
    with RelateSaver(":memory:") as saver, pytest.raises(ValueError, match="thread_id"):
        run_dependency_agent(saver, "  ", fetch_json=fake)
    assert fake.calls == []


def test_ac128_unreachable_network_becomes_tool_errors_for_every_package() -> None:
    """AC-128: network errors from the fetcher are recorded as tool errors, not raised."""
    with RelateSaver(":memory:") as saver:
        state = run_dependency_agent(saver, THREAD_ID, fetch_json=UnreachablePyPI())

    errors = [m for m in state["messages"] if isinstance(m, ToolMessage) and m.status == "error"]
    assert len(errors) == len(state["packages"])
    assert "Found (0)" in state["report"]


def test_ac130_fetch_requests_json_url_with_timeout_and_generic_user_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC-130: URL, 10 s timeout, generic User-Agent, and only name/version/summary returned."""
    seen: dict[str, Any] = {}
    body = {
        "info": {"name": "relate-db", "version": "1.0.2", "summary": "A graph db", "author": "x"},
        "releases": {"1.0.2": []},
    }

    def fake_urlopen(request: Any, timeout: float | None = None) -> _FakeResponse:
        seen["url"] = request.full_url
        seen["timeout"] = timeout
        seen["user_agent"] = request.get_header("User-agent")
        return _FakeResponse(json.dumps(body).encode())

    monkeypatch.setattr(dependency_agent, "urlopen", fake_urlopen)

    result = fetch_pypi_json("relate-db")

    assert seen["url"] == "https://pypi.org/pypi/relate-db/json"
    assert seen["timeout"] == 10
    assert seen["user_agent"] == f"RelateGraphSaver-demo/{__version__}"
    assert result == {"name": "relate-db", "version": "1.0.2", "summary": "A graph db"}


def test_ac130_http_404_raises_lookup_error_and_other_errors_propagate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC-130: HTTP 404 becomes LookupError('...404...'); other failures are not rewritten."""

    def raising(code: int) -> Any:
        def fake_urlopen(request: Any, timeout: float | None = None) -> Any:
            raise _http_error(request.full_url, code)

        return fake_urlopen

    monkeypatch.setattr(dependency_agent, "urlopen", raising(404))
    with pytest.raises(LookupError, match="404"):
        fetch_pypi_json("relategraphsaver")

    monkeypatch.setattr(dependency_agent, "urlopen", raising(500))
    with pytest.raises(urllib.error.HTTPError):
        fetch_pypi_json("relate-db")


def test_ac130_blank_package_name_is_rejected() -> None:
    """AC-130: an empty package name fails before any request is made."""
    with pytest.raises(ValueError, match="name"):
        fetch_pypi_json("   ")


def test_ac131_pyproject_registers_the_network_marker() -> None:
    """AC-131: the network marker is registered next to slow, so no unknown-marker warning."""
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    options = config["tool"]["pytest"]["ini_options"]

    assert any(m.startswith("network:") for m in options["markers"])
    assert any(m.startswith("slow:") for m in options["markers"])
    assert "-m" not in options["addopts"]


@pytest.mark.slow
def test_ac131_default_and_fast_runs_deselect_the_live_test() -> None:
    """AC-131: collection under the default and fast commands leaves the network test out."""
    unknown_marker = ["-W", "error::pytest.PytestUnknownMarkWarning"]
    for extra in ([], ["-m", "not slow"]):
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q", "-o", "addopts="]
            + [THIS_FILE, *unknown_marker, *extra],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "test_ac131_live_pypi" not in result.stdout
        assert "test_ac130_blank_package_name" in result.stdout

    selected = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-o", "addopts=", "-m", "network"]
        + [THIS_FILE],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert selected.returncode == 0, selected.stdout + selected.stderr
    assert "test_ac131_live_pypi" in selected.stdout
    assert "test_ac130_blank_package_name" not in selected.stdout


@pytest.mark.network
def test_ac131_live_pypi_returns_404_for_the_missing_name_and_data_for_relate_db() -> None:
    """AC-131: pypi.org answers 404 for relategraphsaver and a real result for relate-db."""
    with pytest.raises(LookupError, match="404"):
        fetch_pypi_json("relategraphsaver")

    result = fetch_pypi_json("relate-db")
    assert set(result) == {"name", "version", "summary"}
    assert result["name"] == "relate-db"
    assert result["version"]


def test_ac132_agent_runs_without_api_keys_and_reads_no_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC-132: with every *_API_KEY variable removed the agent completes; source reads no env."""
    for name in [n for n in os.environ if n.endswith("_API_KEY")]:
        monkeypatch.delenv(name)

    with RelateSaver(":memory:") as saver:
        state = run_dependency_agent(saver, THREAD_ID, fetch_json=FakePyPI())
    assert "Dependency audit for RelateGraphSaver" in state["report"]

    source = Path(dependency_agent.__file__).read_text(encoding="utf-8")
    assert "os.environ" not in source
    assert "getenv" not in source
    assert "import os" not in source


def test_ac127_main_prints_thread_id_and_root_cause_lines(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """AC-127: main() runs through the real fetcher (urlopen stubbed) and prints root cause."""

    def fake_urlopen(request: Any, timeout: float | None = None) -> _FakeResponse:
        name = request.full_url.split("/")[-2]
        if name == "relategraphsaver":
            raise _http_error(request.full_url, 404)
        info = {"name": name, "version": "9.9.9", "summary": f"summary of {name}"}
        return _FakeResponse(json.dumps({"info": info}).encode())

    monkeypatch.setattr(dependency_agent, "urlopen", fake_urlopen)
    db_file = str(tmp_path / "main.rl")

    code = main(["--db", db_file, "--thread-id", "main-thread"])

    out = capsys.readouterr().out
    assert code == 0
    assert "Thread ID: main-thread" in out
    assert "Failed lookups (1)" in out
    assert "Root Tool Name:      lookup_pypi_package" in out
    assert "from node 'planner'" in out
    assert Path(db_file).exists()


def test_ac127_planner_package_names_match_pyproject_runtime_dependencies() -> None:
    """AC-127: the planner's package list is the runtime dependencies declared in pyproject.toml."""
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    declared = []
    for spec in config["project"]["dependencies"]:
        match = re.match(r"[A-Za-z0-9._-]+", spec)
        assert match is not None, spec
        declared.append(match.group(0).lower().replace("_", "-"))

    assert list(dependency_agent.RUNTIME_DEPENDENCIES) == declared
