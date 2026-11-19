"""Tests for the committed demo store and the script that regenerates it.

Covers AC-69..AC-75 and AC-133..AC-138. ``examples/build_demo_store.py`` needs network access to
pypi.org for its ``demo-real-api`` thread, so every build in the fast suite injects ``FakePyPI``
and stays offline; only the test marked ``network`` runs the script against the real API.
Comparisons use checkpoint counts, tool names and error/no-error outcomes, never result text,
which changes with the live package index.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
from pathlib import Path
from typing import Any

import pytest
from examples import dependency_agent
from examples.build_demo_store import (
    DEMO_THREADS,
    MISSING_PACKAGE,
    REAL_API_THREAD,
    RESEARCH_THREAD,
    TOOL_FAIL_THREAD,
    TOOL_OK_THREAD,
    PyPIUnreachableError,
    build_demo_store,
    main,
)
from examples.tool_agent import run_tool_agent
from tests.fakes.fake_pypi import FakePyPI, UnreachablePyPI
from typer.testing import CliRunner

from langgraph_checkpoint_relate.cli import app
from langgraph_checkpoint_relate.saver import RelateSaver

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMITTED_STORE = REPO_ROOT / "examples" / "data" / "demo.rl"
CHECKPOINT_ID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _listing(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir())


def _checkpoint_counts(db_path: Path) -> dict[str, int]:
    """Checkpoint count per thread id in a store, opened read-only."""
    with RelateSaver(str(db_path), read_only=True) as saver:
        rows = saver.proxy.execute_read(
            "MATCH (c:Checkpoint) RETURN c.thread_id AS tid, count(c) AS cnt", {}
        )
    return {str(r["tid"]): int(r["cnt"]) for r in rows}


def _tool_calls(db_path: Path) -> list[tuple[str, str, str, bool]]:
    """(thread id, tool name, call id, failed) for every :ToolCall in a store."""
    with RelateSaver(str(db_path), read_only=True) as saver:
        rows = saver.proxy.execute_read(
            "MATCH (c:Checkpoint)-[:INVOKED]->(tc:ToolCall) "
            "RETURN c.thread_id AS tid, tc.tool_name AS name, tc.call_id AS call_id, "
            "tc.error AS error",
            {},
        )
    return sorted(
        (str(r["tid"]), str(r["name"]), str(r["call_id"]), r["error"] is not None) for r in rows
    )


@pytest.fixture(scope="module")
def built_store(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A freshly built demo store in an otherwise empty directory."""
    out = tmp_path_factory.mktemp("demo_store") / "demo.rl"
    build_demo_store(out, fetch_json=FakePyPI())
    return out


def _invoke(*args: str) -> tuple[int, str]:
    result = CliRunner().invoke(app, list(args))
    return result.exit_code, result.output


def test_ac69_committed_store_is_not_ignored_by_git() -> None:
    """AC-69: examples/data/demo.rl is not gitignored, while other .rl files stay ignored."""
    kept = subprocess.run(
        ["git", "check-ignore", "-q", "examples/data/demo.rl"], cwd=REPO_ROOT, check=False
    )
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", "examples/data/other.rl"], cwd=REPO_ROOT, check=False
    )
    assert kept.returncode == 1, "examples/data/demo.rl must not be ignored"
    assert ignored.returncode == 0, "other .rl files must stay ignored"


def test_ac69_committed_store_is_a_single_file_without_sidecars() -> None:
    """AC-69, AC-141: examples/data holds demo.rl and at most the optional demo-llm.rl.

    Passes with or without demo-llm.rl; any sidecar or other file beside them fails.
    """
    assert COMMITTED_STORE.is_file()
    listing = _listing(COMMITTED_STORE.parent)
    assert "demo.rl" in listing
    assert set(listing) <= {"demo.rl", "demo-llm.rl"}


def test_ac141_demo_llm_store_is_not_ignored_by_git() -> None:
    """AC-141: demo-llm.rl has its own gitignore negation while other .rl files stay ignored."""
    kept = subprocess.run(
        ["git", "check-ignore", "-q", "examples/data/demo-llm.rl"], cwd=REPO_ROOT, check=False
    )
    assert kept.returncode == 1, "examples/data/demo-llm.rl must not be ignored"


def test_ac141_builder_never_mentions_the_llm_store() -> None:
    """AC-141: build_demo_store.py neither reads nor writes demo-llm.rl."""
    source = (REPO_ROOT / "examples" / "build_demo_store.py").read_text(encoding="utf-8")
    assert "demo-llm" not in source


def test_ac73_script_builds_closed_single_file_store_with_four_threads(built_store: Path) -> None:
    """AC-73: the script writes a closed single-file store holding exactly the four threads."""
    assert built_store.is_file()
    assert _listing(built_store.parent) == ["demo.rl"]
    assert set(_checkpoint_counts(built_store)) == set(DEMO_THREADS)
    assert DEMO_THREADS == (TOOL_FAIL_THREAD, TOOL_OK_THREAD, RESEARCH_THREAD, REAL_API_THREAD)


@pytest.mark.network
def test_ac134_script_runs_as_a_program_against_pypi_without_api_keys(tmp_path: Path) -> None:
    """AC-134: ``python examples/build_demo_store.py --out`` exits 0 with no LLM keys set.

    Calls the real https://pypi.org, so it only runs with ``-m network``.
    """
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.endswith("_API_KEY") and not k.endswith("_API_TOKEN")
    }
    out = tmp_path / "cli" / "demo.rl"
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "examples" / "build_demo_store.py"), "--out", str(out)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert _listing(out.parent) == ["demo.rl"]
    assert set(_checkpoint_counts(out)) == set(DEMO_THREADS)


def test_ac134_main_builds_the_store_in_process_with_a_fetcher_patched_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """AC-134: ``main`` resolves the fetcher when called, and needs no API key to build."""
    monkeypatch.setattr("examples.dependency_agent.fetch_pypi_json", FakePyPI())
    for key in [k for k in os.environ if k.endswith(("_API_KEY", "_API_TOKEN"))]:
        monkeypatch.delenv(key)
    out = tmp_path / "cli" / "demo.rl"
    assert main(["--out", str(out)]) == 0
    assert _listing(out.parent) == ["demo.rl"]
    assert set(_checkpoint_counts(out)) == set(DEMO_THREADS)
    assert REAL_API_THREAD in capsys.readouterr().out


def test_ac70_trace_error_names_failing_tool_and_call_id(built_store: Path) -> None:
    """AC-70: trace-error on demo-tool-fail exits 0 and names fetch_source and its call id."""
    code, output = _invoke("forensics", "trace-error", TOOL_FAIL_THREAD, "--db", str(built_store))
    flat = " ".join(output.split())
    assert code == 0, output
    assert "fetch_source" in flat
    assert re.search(r"call_[0-9a-f]{32}", flat), flat


def test_ac71_trace_error_reports_no_failure_for_ok_thread(built_store: Path) -> None:
    """AC-71: trace-error on demo-tool-ok exits 0 and reports no tool failure."""
    code, output = _invoke("forensics", "trace-error", TOOL_OK_THREAD, "--db", str(built_store))
    assert code == 0, output
    assert "fetch_source" not in output.split("Root Tool Name:")[-1].split("\n")[0]


def test_ac72_tree_shows_checkpoints_for_research_thread(built_store: Path) -> None:
    """AC-72: tree demo-research exits 0 and shows at least one checkpoint."""
    code, output = _invoke("tree", RESEARCH_THREAD, "--db", str(built_store))
    assert code == 0, output
    assert CHECKPOINT_ID.search(output.replace("\n", "").replace(" ", ""))


def test_ac74_ac137_two_builds_hold_the_same_threads_and_tool_outcomes(tmp_path: Path) -> None:
    """AC-74/AC-137: rebuilding keeps thread ids, checkpoint counts, tool names and outcomes."""
    first = tmp_path / "a" / "demo.rl"
    second = tmp_path / "b" / "demo.rl"
    build_demo_store(first, fetch_json=FakePyPI())
    build_demo_store(second, fetch_json=FakePyPI())
    counts = _checkpoint_counts(first)
    assert counts == _checkpoint_counts(second)
    assert set(counts) == set(DEMO_THREADS)

    calls_one, calls_two = _tool_calls(first), _tool_calls(second)
    # Call ids are random, so they decide the row order; compare the outcomes in a stable order.
    outcomes_one = sorted((t, n, failed) for t, n, _, failed in calls_one)
    outcomes_two = sorted((t, n, failed) for t, n, _, failed in calls_two)
    assert outcomes_one == outcomes_two
    assert (TOOL_FAIL_THREAD, "fetch_source", True) in outcomes_one
    assert (TOOL_OK_THREAD, "fetch_source", False) in outcomes_one
    real_api = [(n, failed) for t, n, failed in outcomes_one if t == REAL_API_THREAD]
    assert {n for n, _ in real_api} == {"lookup_pypi_package"}
    assert sorted(failed for _, failed in real_api) == [False] * (len(real_api) - 1) + [True]
    # Call ids are random per build, but every call has one.
    assert all(call_id for _, _, call_id, _ in calls_one + calls_two)


def test_ac73_overwrite_replaces_existing_store_and_leaves_no_sidecars(tmp_path: Path) -> None:
    """AC-73: an existing file is replaced cleanly and nothing is left beside it."""
    out = tmp_path / "demo.rl"
    out.write_bytes(b"not a store")
    build_demo_store(out, fetch_json=FakePyPI())
    assert _listing(tmp_path) == ["demo.rl"]
    assert set(_checkpoint_counts(out)) == set(DEMO_THREADS)
    build_demo_store(out, fetch_json=FakePyPI())
    assert _listing(tmp_path) == ["demo.rl"]


def test_ac73_failed_build_keeps_existing_store_and_cleans_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC-73: if a build fails the previous file survives and no staging file remains."""
    out = tmp_path / "demo.rl"
    build_demo_store(out, fetch_json=FakePyPI())
    before = _sha256(out)

    def boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("scripted failure")

    monkeypatch.setattr("examples.build_demo_store.run_research_workflow", boom)
    with pytest.raises(RuntimeError, match="scripted failure"):
        build_demo_store(out, fetch_json=FakePyPI())
    assert _listing(tmp_path) == ["demo.rl"]
    assert _sha256(out) == before


def test_build_without_network_leaves_store_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """AC-136: with pypi.org unreachable the build exits 1, names it, and changes nothing."""
    out = tmp_path / "demo.rl"
    build_demo_store(out, fetch_json=FakePyPI())
    before = _sha256(out)
    unreachable = UnreachablePyPI()

    with pytest.raises(PyPIUnreachableError, match="unreachable"):
        build_demo_store(out, fetch_json=unreachable)
    assert unreachable.calls, "the real-API thread must have tried the lookups"
    assert _sha256(out) == before
    assert _listing(tmp_path) == ["demo.rl"]

    # Through main() the same outage is one stderr line and exit code 1.
    capsys.readouterr()
    monkeypatch.setattr("examples.dependency_agent.fetch_pypi_json", unreachable)
    assert main(["--out", str(out)]) == 1
    err = capsys.readouterr().err
    assert err.startswith("Error: could not reach pypi.org:")
    assert "unreachable" in err
    assert len(err.strip().splitlines()) == 1
    assert _sha256(out) == before
    assert _listing(tmp_path) == ["demo.rl"]


def test_build_with_no_existing_store_and_no_network_creates_nothing(tmp_path: Path) -> None:
    """AC-136: a failed first build leaves neither a store nor a staging file."""
    with pytest.raises(PyPIUnreachableError):
        build_demo_store(tmp_path / "demo.rl", fetch_json=UnreachablePyPI())
    assert _listing(tmp_path) == []


def test_ac135_a_lookup_that_does_not_return_404_is_rejected(tmp_path: Path) -> None:
    """AC-135: a fetcher that answers for relategraphsaver (no 404) does not pass the check."""

    def answers_everything(name: str) -> dict[str, str]:
        return {"name": name, "version": "1", "summary": "x"}

    with pytest.raises(PyPIUnreachableError, match="404"):
        build_demo_store(tmp_path / "demo.rl", fetch_json=answers_everything)
    assert _listing(tmp_path) == []


def test_ac73_main_rejects_empty_output_path(capsys: pytest.CaptureFixture[str]) -> None:
    """AC-73 (invalid input): an empty --out is reported and exits non-zero."""
    assert main(["--out", "  "]) == 1
    assert "out_path" in capsys.readouterr().err


def test_ac73_non_rl_output_is_rejected_and_nothing_is_created(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """AC-73 (invalid input): a --out without the .rl suffix exits 1 and creates nothing."""
    target_dir = tmp_path / "new_parent"
    for name in ("demo", "demo.db", "demo.rl.bak"):
        out = target_dir / name
        assert main(["--out", str(out)]) == 1
        err = capsys.readouterr().err
        assert str(out) in err and ".rl" in err
        assert len(err.strip().splitlines()) == 1
    assert not target_dir.exists()
    assert _listing(tmp_path) == []
    with pytest.raises(ValueError, match=r'must end in "\.rl"'):
        build_demo_store(tmp_path / "demo.db")
    assert _listing(tmp_path) == []


def test_ac73_failed_build_cleanup_never_masks_the_original_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """AC-73: a cleanup that fails is reported, and the build error still propagates."""
    out = tmp_path / "we[ird].rl"
    staging = tmp_path / "we[ird].building.rl"
    attempted: list[str] = []

    def failing_build(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("scripted failure")

    def failing_remove(path: Path) -> None:
        # Only the cleanup of the staging file the build created fails; removing a stale one
        # before the build is a no-op.
        if path == staging and path.exists():
            attempted.append(path.name)
            raise PermissionError("scripted cleanup failure")

    monkeypatch.setattr("examples.build_demo_store.run_research_workflow", failing_build)
    monkeypatch.setattr("examples.build_demo_store._remove_if_exists", failing_remove)
    with pytest.raises(RuntimeError, match="scripted failure"):
        build_demo_store(out, fetch_json=FakePyPI())
    assert attempted == [staging.name]
    err = capsys.readouterr().err
    assert "[WARN] Could not remove leftover" in err
    assert str(staging) in err


def test_ac73_failed_build_removes_only_its_own_staging_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC-73: files and folders that merely share the staging name as a prefix survive."""
    out = tmp_path / "demo.rl"
    staging_name = "demo.building.rl"
    unrelated_file = tmp_path / f"{staging_name}-mine"
    unrelated_dir = tmp_path / f"{staging_name}.keep"
    unrelated_wal_named = tmp_path / f"{staging_name}-wal"
    unrelated_file.write_bytes(b"mine")
    unrelated_wal_named.write_bytes(b"also mine")
    unrelated_dir.mkdir()
    (unrelated_dir / "notes.txt").write_bytes(b"keep me")
    before = _listing(tmp_path)

    def boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("scripted failure")

    monkeypatch.setattr("examples.build_demo_store.run_research_workflow", boom)
    with pytest.raises(RuntimeError, match="scripted failure"):
        build_demo_store(out, fetch_json=FakePyPI())

    assert _listing(tmp_path) == before
    assert not (tmp_path / staging_name).exists()
    assert unrelated_file.read_bytes() == b"mine"
    assert unrelated_wal_named.read_bytes() == b"also mine"
    assert (unrelated_dir / "notes.txt").read_bytes() == b"keep me"


def test_ac73_successful_build_leaves_unrelated_neighbours_alone(tmp_path: Path) -> None:
    """AC-73: a build that succeeds adds only demo.rl and leaves its neighbours as they were."""
    neighbour_dir = tmp_path / "demo.building.rl.keep"
    neighbour_dir.mkdir()
    (neighbour_dir / "notes.txt").write_bytes(b"keep me")
    neighbour_file = tmp_path / "demo.building.rl-mine"
    neighbour_file.write_bytes(b"mine")

    build_demo_store(tmp_path / "demo.rl", fetch_json=FakePyPI())

    assert _listing(tmp_path) == ["demo.building.rl-mine", "demo.building.rl.keep", "demo.rl"]
    assert neighbour_file.read_bytes() == b"mine"
    assert (neighbour_dir / "notes.txt").read_bytes() == b"keep me"


def test_ac73_single_file_store_creates_no_files_beside_itself(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC-73: while the saver is open and after it closes, only the one store file exists."""
    seen: list[list[str]] = []
    real_run_tool_agent = run_tool_agent

    def spying_run_tool_agent(saver: RelateSaver, thread_id: str, *, fail: bool) -> dict[str, Any]:
        result = real_run_tool_agent(saver, thread_id, fail=fail)
        seen.append(_listing(tmp_path))
        return result

    monkeypatch.setattr("examples.build_demo_store.run_tool_agent", spying_run_tool_agent)
    build_demo_store(tmp_path / "demo.rl", fetch_json=FakePyPI())
    assert seen == [["demo.building.rl"], ["demo.building.rl"]]
    assert _listing(tmp_path) == ["demo.rl"]


def test_ac133_committed_store_holds_the_four_threads() -> None:
    """AC-69/AC-72/AC-133: the committed store contains exactly the four demo threads."""
    counts = _checkpoint_counts(COMMITTED_STORE)
    assert set(counts) == set(DEMO_THREADS)
    assert all(n >= 1 for n in counts.values())


def test_ac135_committed_real_api_thread_holds_a_404_lookup_and_a_result() -> None:
    """AC-135: relategraphsaver failed with a 404; at least one other lookup has a result."""
    with RelateSaver(str(COMMITTED_STORE), read_only=True) as saver:
        rows = saver.proxy.execute_read(
            "MATCH (c:Checkpoint {thread_id: $tid})-[:INVOKED]->(tc:ToolCall) "
            "RETURN tc.tool_name AS name, tc.args_blob AS args, tc.error AS error, "
            "tc.result_blob AS result",
            {"tid": REAL_API_THREAD},
        )
        lookups = [(saver.serde.loads(bytes(r["args"]))["name"], r) for r in rows]
    assert {r["name"] for _, r in lookups} == {"lookup_pypi_package"}
    missing = [r for name, r in lookups if name == "relategraphsaver"]
    assert len(missing) == 1
    assert missing[0]["error"] and "404" in missing[0]["error"]
    others = [r for name, r in lookups if name != "relategraphsaver"]
    assert others and all(r["error"] is None and r["result"] is not None for r in others)


def test_ac124_ac138_committed_store_failures_table_lists_both_tools() -> None:
    """AC-124: forensics failures shows fetch_source and lookup_pypi_package rows."""
    code, output = _invoke("forensics", "failures", "--db", str(COMMITTED_STORE))
    assert code == 0, output
    assert "fetch_source" in output
    assert "lookup_pypi_package" in output


def test_ac138_committed_real_api_thread_traces_the_bad_argument_to_the_planner() -> None:
    """AC-138: tree marks completed steps [OK]; trace-error shows the argument origin hops."""
    path = str(COMMITTED_STORE)
    code, tree_output = _invoke("tree", REAL_API_THREAD, "--db", path)
    assert code == 0, tree_output
    assert "[OK]" in tree_output

    code, output = _invoke("forensics", "trace-error", REAL_API_THREAD, "--db", path)
    assert code == 0, output
    flat = " ".join(output.replace("|", " ").split())
    match = re.search(r"Bad argument origin:.*?(\d+) hops before", flat)
    assert match, flat
    assert int(match.group(1)) >= 2
    assert "lookup_pypi_package" in flat


def test_ac70_ac71_ac72_committed_store_answers_the_readme_commands() -> None:
    """AC-70..AC-72: the committed store works with the documented commands."""
    path = str(COMMITTED_STORE)
    code, output = _invoke("forensics", "trace-error", TOOL_FAIL_THREAD, "--db", path)
    assert code == 0 and "fetch_source" in " ".join(output.split())
    code, _ = _invoke("forensics", "trace-error", TOOL_OK_THREAD, "--db", path)
    assert code == 0
    code, output = _invoke("tree", RESEARCH_THREAD, "--db", path)
    assert code == 0 and CHECKPOINT_ID.search(output.replace("\n", "").replace(" ", ""))


def test_ac75_cli_leaves_committed_store_bytes_and_directory_unchanged(tmp_path: Path) -> None:
    """AC-75: tree, replay and forensics leave the store byte-identical and add no sidecars."""
    work = tmp_path / "copy"
    work.mkdir()
    store = work / "demo.rl"
    shutil.copyfile(COMMITTED_STORE, store)
    before = _sha256(store)

    _, tree_output = _invoke("tree", TOOL_FAIL_THREAD, "--db", str(store))
    flat = tree_output.replace("\n", "").replace(" ", "")
    checkpoint_id = CHECKPOINT_ID.findall(flat)[0]

    commands = [
        ["tree", RESEARCH_THREAD],
        ["tree", TOOL_FAIL_THREAD],
        ["replay", TOOL_FAIL_THREAD, checkpoint_id],
        ["forensics", TOOL_FAIL_THREAD],
        ["forensics", "loops", TOOL_FAIL_THREAD],
        ["forensics", "trace-error", TOOL_FAIL_THREAD],
        ["forensics", "trace-error", TOOL_OK_THREAD],
        ["tree", REAL_API_THREAD],
        ["forensics", "trace-error", REAL_API_THREAD],
        ["forensics", "failures"],
    ]
    for command in commands:
        code, output = _invoke(*command, "--db", str(store))
        assert code == 0, (command, output)
        assert _sha256(store) == before, command
        assert _listing(work) == ["demo.rl"], command


LATENCY_PHRASES = (
    "sub-millisecond",
    "instantaneous",
    "eliminate branch-tracking ambiguities",
    "millisecond",
    "latency",
    "instant",
    "fast",
    "faster",
    "real-time",
    "zero-latency",
)


def _real_api_lookups(db_path: Path) -> list[tuple[str, str | None, bool]]:
    """(package, error, has a result) for each demo-real-api lookup, read from a store."""
    with RelateSaver(str(db_path), read_only=True) as saver:
        rows = saver.proxy.execute_read(
            "MATCH (c:Checkpoint {thread_id: $tid})-[:INVOKED]->(tc:ToolCall) "
            "RETURN tc.args_blob AS args, tc.error AS error, tc.result_blob AS result",
            {"tid": REAL_API_THREAD},
        )
        return [
            (
                str(saver.serde.loads(bytes(r["args"]))["name"]),
                r["error"],
                r["result"] is not None,
            )
            for r in rows
        ]


class _PyPIJsonResponse(io.BytesIO):
    """A ``urlopen`` response body that ``json.load`` can read."""


def _stub_pypi_urlopen(monkeypatch: pytest.MonkeyPatch, requested: list[str]) -> None:
    """Stub only the network boundary: pypi.org answers 404 for the missing name, else JSON."""

    def fake_urlopen(request: Any, timeout: float | None = None) -> _PyPIJsonResponse:
        url = str(request.full_url)
        requested.append(url)
        name = url.split("/")[-2]
        if name == MISSING_PACKAGE:
            raise urllib.error.HTTPError(url, 404, "Not Found", None, None)  # type: ignore[arg-type]
        info = {"name": name, "version": "1.2.3", "summary": f"summary of {name}"}
        return _PyPIJsonResponse(json.dumps({"info": info}).encode())

    monkeypatch.setattr("examples.dependency_agent.urlopen", fake_urlopen)


def test_ac126_committed_research_thread_carries_no_latency_claims() -> None:
    """AC-126: the research thread in the committed store holds the revised notes only."""
    with RelateSaver(str(COMMITTED_STORE), read_only=True) as saver:
        checkpoint = saver.get_tuple({"configurable": {"thread_id": RESEARCH_THREAD}})
    assert checkpoint is not None
    notes = checkpoint.checkpoint["channel_values"]["research_notes"]
    assert notes
    text = "\n".join(notes).lower()
    for phrase in LATENCY_PHRASES:
        assert phrase not in text, phrase


def test_ac134_ac135_main_builds_through_the_real_fetcher_and_urlopen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC-134/AC-135: with only urlopen stubbed, main() runs the real fetcher end to end."""
    requested: list[str] = []
    _stub_pypi_urlopen(monkeypatch, requested)
    out = tmp_path / "demo.rl"

    assert main(["--out", str(out)]) == 0

    packages = [*dependency_agent.RUNTIME_DEPENDENCIES, MISSING_PACKAGE]
    assert sorted(requested) == sorted(f"https://pypi.org/pypi/{p}/json" for p in packages)
    lookups = _real_api_lookups(out)
    assert sorted(name for name, _, _ in lookups) == sorted(packages)
    failed = [(name, error) for name, error, _ in lookups if error is not None]
    assert [name for name, _ in failed] == [MISSING_PACKAGE]
    assert failed[0][1] is not None and "404" in failed[0][1]
    assert all(has_result for name, error, has_result in lookups if error is None)
    assert _listing(tmp_path) == ["demo.rl"]


def test_ac136_main_with_urlopen_failing_exits_1_and_keeps_the_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """AC-136: a real URLError from urlopen names the network failure and changes nothing."""
    out = tmp_path / "demo.rl"
    build_demo_store(out, fetch_json=FakePyPI())
    before = _sha256(out)

    def offline(request: Any, timeout: float | None = None) -> None:
        raise urllib.error.URLError("[Errno 11001] getaddrinfo failed")

    monkeypatch.setattr("examples.dependency_agent.urlopen", offline)
    capsys.readouterr()
    assert main(["--out", str(out)]) == 1
    err = capsys.readouterr().err
    assert err.startswith("Error: could not reach pypi.org:")
    assert "getaddrinfo failed" in err
    assert _sha256(out) == before
    assert _listing(tmp_path) == ["demo.rl"]


def test_ac141_builder_leaves_a_sibling_llm_store_untouched(tmp_path: Path) -> None:
    """AC-141: building demo.rl beside demo-llm.rl neither reads nor changes the LLM store."""
    data = tmp_path / "data"
    data.mkdir()
    llm_store = data / "demo-llm.rl"
    with RelateSaver(str(llm_store)) as saver:
        run_tool_agent(saver, "demo-llm", fail=False)
    before = _sha256(llm_store)

    build_demo_store(data / "demo.rl", fetch_json=FakePyPI())

    assert _sha256(llm_store) == before
    assert _listing(data) == ["demo-llm.rl", "demo.rl"]
    assert set(_checkpoint_counts(llm_store)) == {"demo-llm"}
