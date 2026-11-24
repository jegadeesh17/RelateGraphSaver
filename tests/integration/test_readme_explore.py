"""The README explore commands, run through the installed relate-dag console script.

tests/unit/test_readme.py runs the same commands in-process with the Typer runner. This file runs
the exact command lines copied from README.md in a subprocess from the repository root, the way a
reader would, so it also covers the console-script entry point and the working directory. Covers
AC-147 and AC-148, plus invalid input to the same commands.
"""

import hashlib
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
README = REPO_ROOT / "README.md"
DATA_DIR = REPO_ROOT / "examples" / "data"
DEMO_STORE = "examples/data/demo.rl"
_COMMAND_TIMEOUT_SECONDS = 120


def _console_script() -> str:
    script = shutil.which("relate-dag", path=str(Path(sys.executable).parent))
    assert script is not None, 'relate-dag is not installed; run pip install -e ".[dev]"'
    return script


def _readme_explore_commands() -> list[list[str]]:
    text = README.read_text(encoding="utf-8")
    heading = "### Explore the demo store"
    block = text[text.index(heading) + len(heading) :]
    end = re.search(r"^#{2,3} ", block, flags=re.MULTILINE)
    block = block[: end.start()] if end else block
    commands = [line.split()[1:] for line in block.splitlines() if line.startswith("relate-dag ")]
    assert len(commands) == 4, f"expected four explore commands, found {commands}"
    return commands


def _store_digests() -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(DATA_DIR.iterdir())
        if path.is_file()
    }


def _run_relate_dag(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [_console_script(), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=_COMMAND_TIMEOUT_SECONDS,
        check=False,
    )


@pytest.mark.slow
def test_readme_explore_commands_exit_zero_through_console_script_ac_147_ac_148() -> None:
    """AC-147, AC-148: the four README commands exit 0, replay shows the fix, the store is unchanged."""
    before = _store_digests()
    replay_output = ""
    for args in _readme_explore_commands():
        result = _run_relate_dag(args)
        output = result.stdout + result.stderr
        command = " ".join(["relate-dag", *args])
        assert result.returncode == 0, f"`{command}` exited {result.returncode}:\n{output}"
        if args[0] == "replay":
            replay_output = output
    assert "fetch_source #1" in replay_output
    assert "error -> ok" in replay_output
    assert _store_digests() == before, "an explore command changed examples/data"


@pytest.mark.slow
@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["tree", "no-such-thread", "--db", DEMO_STORE], "no-such-thread"),
        (
            [
                "replay",
                "demo-tool-fail",
                "00000000-0000-0000-0000-000000000000",
                "--graph",
                "examples/tool_agent.py:create_tool_agent_graph",
                "--db",
                DEMO_STORE,
            ],
            "00000000-0000-0000-0000-000000000000",
        ),
    ],
)
def test_explore_commands_reject_unknown_ids_and_leave_store_unchanged_ac_148(
    args: list[str], expected: str
) -> None:
    """AC-148: an unknown thread or checkpoint id exits 1 with an Error line and changes nothing."""
    before = _store_digests()
    result = _run_relate_dag(args)
    output = result.stdout + result.stderr
    assert result.returncode == 1, f"exited {result.returncode}:\n{output}"
    assert "Error:" in output
    assert expected in output
    assert _store_digests() == before
