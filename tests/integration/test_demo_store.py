"""Tests for the committed demo store and the script that regenerates it (AC-69..AC-75).

The store is built offline by ``examples/build_demo_store.py``; nothing here needs an API key
or network access.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from examples.build_demo_store import (
    DEMO_THREADS,
    RESEARCH_THREAD,
    TOOL_FAIL_THREAD,
    TOOL_OK_THREAD,
    build_demo_store,
    main,
)
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
    build_demo_store(out)
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
    """AC-69: the working tree holds demo.rl as a file, with no sidecar beside it."""
    assert COMMITTED_STORE.is_file()
    assert _listing(COMMITTED_STORE.parent) == ["demo.rl"]


def test_ac73_script_builds_closed_single_file_store_with_three_threads(built_store: Path) -> None:
    """AC-73: the script writes a closed single-file store holding exactly the three threads."""
    assert built_store.is_file()
    assert _listing(built_store.parent) == ["demo.rl"]
    assert set(_checkpoint_counts(built_store)) == set(DEMO_THREADS)
    assert DEMO_THREADS == (TOOL_FAIL_THREAD, TOOL_OK_THREAD, RESEARCH_THREAD)


def test_ac73_script_runs_as_a_program_without_api_keys(tmp_path: Path) -> None:
    """AC-73: ``python examples/build_demo_store.py --out`` exits 0 with no LLM keys set."""
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


def test_ac74_two_builds_hold_the_same_threads_and_tool_outcomes(tmp_path: Path) -> None:
    """AC-74: rebuilding keeps thread ids, checkpoint counts and tool-call outcomes."""
    first = tmp_path / "a" / "demo.rl"
    second = tmp_path / "b" / "demo.rl"
    build_demo_store(first)
    build_demo_store(second)
    counts = _checkpoint_counts(first)
    assert counts == _checkpoint_counts(second)
    assert set(counts) == set(DEMO_THREADS)

    calls_one, calls_two = _tool_calls(first), _tool_calls(second)
    expected = [(TOOL_FAIL_THREAD, "fetch_source", True), (TOOL_OK_THREAD, "fetch_source", False)]
    assert [(t, n, failed) for t, n, _, failed in calls_one] == expected
    assert [(t, n, failed) for t, n, _, failed in calls_two] == expected
    # Call ids are random per build, but every call has one.
    assert all(call_id for _, _, call_id, _ in calls_one + calls_two)


def test_ac73_overwrite_replaces_existing_store_and_leaves_no_sidecars(tmp_path: Path) -> None:
    """AC-73: an existing file is replaced cleanly and nothing is left beside it."""
    out = tmp_path / "demo.rl"
    out.write_bytes(b"not a store")
    build_demo_store(out)
    assert _listing(tmp_path) == ["demo.rl"]
    assert set(_checkpoint_counts(out)) == set(DEMO_THREADS)
    build_demo_store(out)
    assert _listing(tmp_path) == ["demo.rl"]


def test_ac73_failed_build_keeps_existing_store_and_cleans_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC-73: if a build fails the previous file survives and no staging file remains."""
    out = tmp_path / "demo.rl"
    build_demo_store(out)
    before = _sha256(out)

    def boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("scripted failure")

    monkeypatch.setattr("examples.build_demo_store.run_research_workflow", boom)
    with pytest.raises(RuntimeError, match="scripted failure"):
        build_demo_store(out)
    assert _listing(tmp_path) == ["demo.rl"]
    assert _sha256(out) == before


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
    leftover = tmp_path / "we[ird].building.rl-extra"
    attempted: list[str] = []

    def failing_build(*_args: object, **_kwargs: object) -> None:
        # Guarantee a leftover (on a name containing a glob character) for the cleanup to find.
        leftover.write_bytes(b"x")
        raise RuntimeError("scripted failure")

    def failing_remove(path: Path) -> None:
        # Only the cleanup of the leftover fails; the pre-build removal of a stale file is a no-op.
        if path == leftover:
            attempted.append(path.name)
            raise PermissionError("scripted cleanup failure")

    monkeypatch.setattr("examples.build_demo_store.run_research_workflow", failing_build)
    monkeypatch.setattr("examples.build_demo_store._remove_if_exists", failing_remove)
    with pytest.raises(RuntimeError, match="scripted failure"):
        build_demo_store(out)
    assert leftover.name in attempted
    assert "Could not remove leftover" in capsys.readouterr().err


def test_ac69_committed_store_holds_the_three_threads() -> None:
    """AC-69/AC-72: the committed store contains exactly the three demo threads."""
    counts = _checkpoint_counts(COMMITTED_STORE)
    assert set(counts) == set(DEMO_THREADS)
    assert all(n >= 1 for n in counts.values())


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
    ]
    for command in commands:
        code, output = _invoke(*command, "--db", str(store))
        assert code == 0, (command, output)
        assert _sha256(store) == before, command
        assert _listing(work) == ["demo.rl"], command
