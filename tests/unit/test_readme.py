"""README content checks for the Relate rationale, limits and benchmark sections.

Covers AC-39 (how and why Relate is used), AC-40 (single-process limit) and
AC-58 (published benchmark results).
"""

import re
from pathlib import Path

README_PATH = Path(__file__).resolve().parents[2] / "README.md"

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
