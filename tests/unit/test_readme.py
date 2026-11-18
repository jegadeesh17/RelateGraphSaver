"""README content checks for the use case, Relate rationale, data location, limits and benchmark.

Covers AC-39 (how and why Relate is used), AC-40 (single-process limit), AC-58 (published
benchmark results) and AC-76 to AC-80 (use case, mode, data location and the explore commands).
"""

import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from langgraph_checkpoint_relate.cli import app

REPO_ROOT = Path(__file__).resolve().parents[2]
README_PATH = REPO_ROOT / "README.md"
DEMO_STORE = REPO_ROOT / "examples" / "data" / "demo.rl"
DEMO_THREADS = ("demo-tool-fail", "demo-tool-ok", "demo-research")
REPLAY_COMMAND = re.compile(
    r"^relate-dag replay (?P<thread>\S+) (?P<checkpoint>[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12})"
    r" --db examples/data/demo\.rl$",
    flags=re.MULTILINE,
)

RELATE_HEADING = "How we use Relate, and why"
BENCHMARK_HEADING = "Benchmark"
METRIC_ROWS = (
    "Put throughput",
    "Get p50",
    "Get p95",
    "On-disk size",
    "Ancestry query",
)


def _section(heading: str) -> str:
    """Return the body of the level-2 README section with the given title."""
    text = README_PATH.read_text(encoding="utf-8")
    match = re.search(rf"^## {re.escape(heading)}\s*$", text, flags=re.MULTILINE)
    assert match is not None, f"README.md has no '## {heading}' section"
    rest = text[match.end() :]
    next_heading = re.search(r"^## ", rest, flags=re.MULTILINE)
    return rest[: next_heading.start()] if next_heading else rest


def test_relate_section_names_each_capability_ac_39() -> None:
    """AC-39: the section names every Relate capability the package relies on."""
    section = _section(RELATE_HEADING)
    for term in ("single-file", "WAL", "Cypher", ":memory:", "constraints", "indexes"):
        assert term in section, f"'{term}' missing from '{RELATE_HEADING}'"
    for relationship in ("NEXT", "FORKED_TO", "HAS_WRITE", "INVOKED"):
        assert relationship in section, f"relationship {relationship} missing"


def test_relate_section_states_single_process_limit_ac_40() -> None:
    """AC-40: the section states the single-process limit and recommends PostgresSaver."""
    section = _section(RELATE_HEADING)
    assert "single-process" in section
    assert "PostgresSaver" in section
    assert "multi-worker" in section and "multi-pod" in section


def test_benchmark_section_has_metric_rows_for_both_savers_ac_58() -> None:
    """AC-58: the benchmark section has all metric rows, both savers, N and environment."""
    section = _section(BENCHMARK_HEADING)
    header = next(line for line in section.splitlines() if line.startswith("| Metric"))
    assert "RelateSaver" in header and "SqliteSaver" in header
    for metric in METRIC_ROWS:
        rows = [line for line in section.splitlines() if line.startswith(f"| {metric}")]
        assert len(rows) == 1, f"expected exactly one '{metric}' row"
        cells = [cell.strip() for cell in rows[0].strip("|").split("|")]
        assert len(cells) == len(header.strip("|").split("|"))
        assert all(re.fullmatch(r"\d+(\.\d+)?", cell) for cell in cells[1:]), rows[0]


def test_benchmark_section_states_n_environment_and_command_ac_58() -> None:
    """AC-58: N, the environment and the reproduction command are stated."""
    section = _section(BENCHMARK_HEADING)
    assert re.search(r"^- Checkpoints \(N\): \d+$", section, flags=re.MULTILINE)
    for key in ("OS", "CPU", "Python", "relate-db", "SQLite"):
        assert re.search(rf"^- {re.escape(key)}.*: ", section, flags=re.MULTILINE), key
    assert "python benchmarks/bench_checkpointers.py --checkpoints 1000" in section


def test_missing_section_is_reported_ac_58() -> None:
    """Invalid input: asking for a heading that does not exist fails with a clear message."""
    try:
        _section("No Such Section")
    except AssertionError as exc:
        assert "No Such Section" in str(exc)
    else:
        raise AssertionError("expected an AssertionError for a missing section")


def test_use_case_section_names_problem_and_relationships_ac_76() -> None:
    """AC-76: a Use case section states the problem and the relationships that matter."""
    section = _section("Use case")
    assert "fail" in section
    for relationship in ("NEXT", "HAS_WRITE", "INVOKED", "FORKED_TO"):
        assert relationship in section, f"relationship {relationship} missing from Use case"


def test_relate_section_states_mode_reason_and_model_ac_77() -> None:
    """AC-77: the Relate section has `Mode: single file`, a reason and what the graph models."""
    section = _section(RELATE_HEADING)
    assert re.search(r"^Mode: single file$", section, flags=re.MULTILINE)
    assert "portable" in section and "no server" in section
    assert "What the graph models" in section
    for node in (":Checkpoint", ":Write", ":ToolCall"):
        assert node in section, f"node label {node} missing"
    assert "read-only" in section and "ADR-0015" in section


def test_where_the_data_lives_names_store_and_threads_ac_78() -> None:
    """AC-78: the data section names the committed store and its three thread ids."""
    section = _section("Where the data lives")
    assert "examples/data/demo.rl" in section
    for thread_id in DEMO_THREADS:
        assert thread_id in section, f"thread {thread_id} missing"
    assert "python examples/build_demo_store.py" in section
    assert DEMO_STORE.is_file(), "the committed demo store is missing"


def _readme_demo_commands() -> list[list[str]]:
    """Return the `relate-dag` commands the README runs against the committed store."""
    text = README_PATH.read_text(encoding="utf-8")
    block = text[text.index("### Explore the demo store") :]
    block = block[: block.index("\n## ")]
    commands = [
        line.split()[1:]
        for line in block.splitlines()
        if line.startswith("relate-dag ") and "examples/data/demo.rl" in line
    ]
    assert len(commands) == 4, f"expected 4 explore commands, found {commands}"
    return commands


def test_explore_commands_exit_zero_on_committed_store_ac_79() -> None:
    """AC-79: each documented tree, trace-error and replay command exits 0, no agent run first."""
    runner = CliRunner()
    verbs = set()
    for args in _readme_demo_commands():
        args = [*args[:-1], str(DEMO_STORE)]
        result = runner.invoke(app, args)
        assert result.exit_code == 0, f"{args} exited {result.exit_code}: {result.output}"
        verbs.add(args[0] if args[0] != "forensics" else f"forensics {args[1]}")
    assert verbs == {"tree", "forensics trace-error", "replay"}


def test_readme_replay_id_appears_in_tree_output_ac_80() -> None:
    """AC-80: the replay checkpoint id in the README is one printed by tree for that thread."""
    text = README_PATH.read_text(encoding="utf-8")
    match = REPLAY_COMMAND.search(text)
    assert match is not None, "README has no replay command with a checkpoint id"
    result = CliRunner().invoke(app, ["tree", match["thread"], "--db", str(DEMO_STORE)])
    assert result.exit_code == 0, result.output
    # The tree wraps long lines; compare with whitespace and tree glyphs removed.
    flat = re.sub(r"[\s|+`-]", "", result.output)
    assert re.sub(r"-", "", match["checkpoint"]) in flat


def test_where_the_data_lives_rejects_missing_section() -> None:
    """A heading that is not in the README fails loudly instead of returning an empty body."""
    with pytest.raises(AssertionError, match="no '## Nonexistent heading' section"):
        _section("Nonexistent heading")
