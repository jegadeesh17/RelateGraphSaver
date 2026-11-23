"""README content checks for the use case, Relate rationale, data location, limits and benchmark.

Covers AC-39 (how and why Relate is used), AC-40 (single-process limit), AC-58 (published
benchmark results), AC-76 to AC-80 (use case, mode, data location and the explore commands) and
AC-144 to AC-150 (source install, plain tagline, "Why not X?", the four explore commands and the
four demo threads).
"""

import hashlib
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from langgraph_checkpoint_relate.cli import app

REPO_ROOT = Path(__file__).resolve().parents[2]
README_PATH = REPO_ROOT / "README.md"
DEMO_STORE = REPO_ROOT / "examples" / "data" / "demo.rl"
DEMO_THREADS = ("demo-tool-fail", "demo-tool-ok", "demo-research", "demo-real-api")
REPLAY_COMMAND = re.compile(
    r"^relate-dag replay (?P<thread>\S+) (?P<checkpoint>[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12})"
    r"(?P<flags>(?: \S+)*?) --db examples/data/demo\.rl$",
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
    """AC-78: the data section names the committed store and its thread ids."""
    section = _section("Where the data lives")
    assert "examples/data/demo.rl" in section
    for thread_id in DEMO_THREADS:
        assert thread_id in section, f"thread {thread_id} missing"
    assert "python examples/build_demo_store.py" in section
    assert DEMO_STORE.is_file(), "the committed demo store is missing"


def test_rebuild_warning_follows_the_rebuild_command_ac_78() -> None:
    """AC-78: beside the rebuild command, the README says a rebuild changes ids and sample output."""
    section = _section("Where the data lives")
    command_at = section.index("python examples/build_demo_store.py")
    after = " ".join(section[command_at:].split())
    assert "new checkpoint ids" in after
    assert "replay checkpoint id" in after and "sample output" in after
    assert "output of the `tree` command" in after
    # The warning sits before the next paragraph about the other demo agents' files.
    assert after.index("new checkpoint ids") < after.index("checkpoints.rl")


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


def _data_state() -> tuple[list[str], dict[str, str]]:
    """Return the sorted file names in examples/data and the sha256 of each file."""
    data_dir = DEMO_STORE.parent
    names = sorted(path.name for path in data_dir.iterdir())
    digests = {
        name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
        for name in names
        if (data_dir / name).is_file()
    }
    return names, digests


def test_explore_commands_exit_zero_on_committed_store_ac_79(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC-79, AC-148: the four documented commands exit 0 and leave examples/data untouched."""
    # The replay command names its graph by a path relative to the repository root.
    monkeypatch.chdir(REPO_ROOT)
    before = _data_state()
    runner = CliRunner()
    verbs = set()
    outputs = {}
    for args in _readme_demo_commands():
        args = [*args[:-1], str(DEMO_STORE)]
        result = runner.invoke(app, args)
        assert result.exit_code == 0, f"{args} exited {result.exit_code}: {result.output}"
        verb = args[0] if args[0] != "forensics" else f"forensics {args[1]}"
        verbs.add(verb)
        outputs[verb] = result.output
    assert verbs == {"tree", "forensics trace-error", "forensics failures", "replay"}
    assert "fetch_source #1" in outputs["replay"]
    assert "error -> ok" in outputs["replay"]
    assert _data_state() == before, "an explore command changed examples/data"


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


def _readme_text() -> str:
    return README_PATH.read_text(encoding="utf-8")


def test_readme_has_source_install_and_no_pypi_install_ac_144() -> None:
    """AC-144: the README has a source install line and no PyPI install line."""
    text = _readme_text()
    assert "pip install langgraph-checkpoint-relate" not in text
    assert re.search(r'^pip install -e "\.\[dev\]"$', text, flags=re.MULTILINE)


def test_tagline_is_plain_language_ac_145() -> None:
    """AC-145: the tagline names a failing step, finding its cause from one file, offline."""
    lines = _readme_text().splitlines()
    title_at = next(i for i, line in enumerate(lines) if line.startswith("# "))
    tagline = next(
        line
        for line in lines[title_at + 1 :]
        if line.strip() and not line.lstrip().startswith(("[![", "!["))
    )
    lowered = tagline.lower()
    for banned in ("wal", "dag", "graph-native", "relate-dag"):
        assert banned not in lowered, f"tagline contains '{banned}': {tagline}"
    assert "fail" in lowered and "step" in lowered
    assert "one file" in lowered and "offline" in lowered


def test_why_not_x_table_has_four_rows_ac_146() -> None:
    """AC-146: the comparison table covers the four alternatives and says Neo4jSaver needs a server."""
    section = _section("Why not X?")
    lines = section.splitlines()
    for alternative in (
        "Neo4jSaver",
        "SqliteSaver",
        "LangGraph built-in time travel",
        "Tracing tools",
    ):
        rows = [line for line in lines if line.startswith(f"| {alternative}")]
        assert len(rows) == 1, f"expected exactly one '{alternative}' row, found {rows}"
    neo4j_row = next(line for line in lines if line.startswith("| Neo4jSaver"))
    assert "Needs a Neo4j server" in neo4j_row


def test_why_not_x_rows_state_graph_labels_and_trace_gap_ac_146() -> None:
    """AC-146: PREVIOUS is described as a relationship and the tracing row names the manual link."""
    lines = _section("Why not X?").splitlines()
    neo4j_row = next(line for line in lines if line.startswith("| Neo4jSaver"))
    assert "`Thread`, `Checkpoint`, `Branch` and `PendingWrite` nodes" in neo4j_row
    assert "checkpoints linked by `PREVIOUS`" in neo4j_row
    assert "`PREVIOUS`, `Branch` nodes" not in neo4j_row
    tracing_row = next(line for line in lines if line.startswith("| Tracing tools"))
    assert "Traces are stored separately from the checkpoints" in tracing_row
    assert "linking a failed trace back to the saved state is manual" in tracing_row


def test_explore_block_holds_exactly_the_four_commands_ac_147() -> None:
    """AC-147: the explore block has the four documented commands, every flag before --db."""
    commands = [" ".join(args) for args in _readme_demo_commands()]
    match = REPLAY_COMMAND.search(_readme_text())
    assert match is not None, "README has no replay command with a checkpoint id"
    replay_id = match["checkpoint"]
    assert commands == [
        "tree demo-real-api --db examples/data/demo.rl",
        "forensics trace-error demo-real-api --db examples/data/demo.rl",
        "forensics failures --db examples/data/demo.rl",
        f"replay demo-tool-fail {replay_id} --graph examples/tool_agent.py:create_tool_agent_graph"
        " --set-message 0=https://example.org/report --db examples/data/demo.rl",
    ]
    for args in _readme_demo_commands():
        assert args[-2] == "--db", f"--db must be the last flag: {args}"


def test_readme_replay_id_is_step_zero_of_demo_tool_fail_ac_149() -> None:
    """AC-149: the README replay id is demo-tool-fail's step 0 checkpoint, not the input step."""
    match = REPLAY_COMMAND.search(_readme_text())
    assert match is not None, "README has no replay command with a checkpoint id"
    assert match["thread"] == "demo-tool-fail"
    result = CliRunner().invoke(app, ["tree", "demo-tool-fail", "--db", str(DEMO_STORE)])
    assert result.exit_code == 0, result.output
    # The tree wraps long lines; remove all whitespace before matching.
    flat = re.sub(r"\s+", "", result.output)
    step_ids = dict(re.findall(r"Step(-?\d+):source:\w+\(id:([0-9a-f-]{36})\)", flat))
    assert "0" in step_ids and "-1" in step_ids, step_ids
    assert match["checkpoint"] == step_ids["0"]
    assert match["checkpoint"] != step_ids["-1"]


def test_where_the_data_lives_names_all_four_threads_ac_150() -> None:
    """AC-150: the data section names all four thread ids of the committed store."""
    section = _section("Where the data lives")
    for thread_id in ("demo-tool-fail", "demo-tool-ok", "demo-research", "demo-real-api"):
        assert thread_id in section, f"thread {thread_id} missing"


def test_readme_tree_sample_uses_flat_layout_ac_185() -> None:
    """AC-185: the README tree excerpt matches the real flat output, steps at one depth."""
    text = _readme_text()
    block = text[text.index("Thread: demo-real-api") :]
    excerpt = block[: block.index("```")].splitlines()
    # The excerpt ends with an ellipsis line; everything before it is real output.
    assert excerpt[-1] == "..."
    sample = [line.rstrip() for line in excerpt[:-1]]

    result = CliRunner().invoke(app, ["tree", "demo-real-api", "--db", str(DEMO_STORE)])
    assert result.exit_code == 0, result.output
    # The README shows the ASCII glyphs; the runner may print the Unicode ones.
    ascii_glyphs = str.maketrans({"├": "+", "└": "`", "─": "-", "│": "|"})
    actual = [line.translate(ascii_glyphs).rstrip() for line in result.output.splitlines()]
    assert actual[: len(sample)] == sample

    step_lines = [line for line in sample if "] Step " in line]
    assert len(step_lines) >= 3
    assert {line.index("[") for line in step_lines} == {4}
    assert not any(line.startswith("    ") for line in sample)


LLM_STORE = REPO_ROOT / "examples" / "data" / "demo-llm.rl"


def _llm_section() -> str:
    """Return the body of the README's recorded LLM run subsection."""
    text = _readme_text()
    heading = "### Explore the recorded LLM run"
    rest = text[text.index(heading) + len(heading) :]
    end = re.search(r"^#{2,3} ", rest, flags=re.MULTILINE)
    return rest[: end.start()] if end else rest


def test_llm_run_section_commands_exit_zero_and_leave_store_unchanged_ac_141() -> None:
    """AC-141: the recorded LLM run commands read demo-llm.rl and never change it."""
    section = _llm_section()
    commands = [
        line.split()[1:]
        for line in section.splitlines()
        if line.startswith("relate-dag ") and "examples/data/demo-llm.rl" in line
    ]
    assert [args[:3] for args in commands] == [
        ["tree", "demo-llm", "--db"],
        ["forensics", "trace-error", "demo-llm"],
    ]
    before = _data_state()
    assert before[0] == ["demo-llm.rl", "demo.rl"], before[0]
    for args in commands:
        assert args[-2] == "--db"
        args = [*args[:-1], str(REPO_ROOT / args[-1])]
        result = CliRunner().invoke(app, args)
        assert result.exit_code == 0, result.output
        assert "demo-llm" in result.output
    assert _data_state() == before


def test_llm_run_section_names_only_existing_thread_ac_141() -> None:
    """AC-141: the thread id the README names for demo-llm.rl is the one stored in it."""
    result = CliRunner().invoke(app, ["forensics", "failures", "--db", str(LLM_STORE)])
    assert result.exit_code == 0, result.output
    assert "demo-llm" in result.output
    assert "demo-llm" in _llm_section()


def test_agent_instructions_cover_key_handling_ac_140() -> None:
    """AC-140: the README shows the llm extra install, the env var and the missing-key exit."""
    text = _readme_text()
    section = text[text.index("### Running the LLM agent") :]
    section = section[: section.index("\n## ")]
    assert 'pip install -e ".[dev,llm]"' in section
    assert "OPENROUTER_API_KEY" in section
    assert "python examples/openrouter_agent.py --model" in section
    assert "`Error:` line" in section and "exit" in section
    assert ".env.example" in section
    # The key is only ever a placeholder: no assignment of a real-looking value.
    assert not re.search(r"OPENROUTER_API_KEY\s*=\s*(?!<)[^\s<]", section.replace('"', ""))
    assert "OPENROUTER_API_KEY=" in (REPO_ROOT / ".env.example").read_text(encoding="utf-8")


def test_dependency_agent_instructions_name_the_command_ac_132() -> None:
    """AC-132: the README documents running the key-free dependency audit."""
    text = _readme_text()
    section = text[text.index("### Running the dependency audit") :]
    section = section[: section.index("\n### ")]
    assert "python examples/dependency_agent.py" in section
    assert "needs no API key" in section
    assert "relate-dag forensics trace-error" in section


def _cli_section() -> str:
    """Return the README's "Command-line interface" subsection."""
    text = _readme_text()
    section = text[text.index("### Command-line interface") :]
    return section[: section.index("\n### Running the multi-agent demo")]


def _help(*args: str) -> str:
    """Return `relate-dag <args> --help` output."""
    result = CliRunner().invoke(app, [*args, "--help"])
    assert result.exit_code == 0, result.output
    return result.output


def test_cli_reference_documents_forensics_failures_rc_10() -> None:
    """RC-10: the CLI reference lists `forensics failures`, which takes only --db."""
    section = _cli_section()
    assert "relate-dag forensics failures --db checkpoints.rl" in section
    assert "failures" in _help("forensics")
    options = _help("forensics", "failures")
    assert "--db" in options and "--threshold" not in options


def test_cli_reference_replay_flags_match_help_rc_10() -> None:
    """RC-10: every replay flag the README names exists in `replay --help`, and vice versa."""
    section = _cli_section()
    replay_help = _help("replay")
    documented = {"--graph", "--set-message", "--update", "--as-node", "--checkpoint", "--branch"}
    for flag in documented:
        assert f"`{flag}" in section, f"README CLI reference does not document {flag}"
        assert flag in replay_help, f"{flag} is documented but missing from replay --help"
    help_flags = set(re.findall(r"--[a-z][a-z-]*", replay_help)) - {"--help", "--file", "--db"}
    assert help_flags == documented


def test_cli_reference_defaults_match_help_rc_10() -> None:
    """RC-10: the documented --db and --threshold defaults are the ones --help prints."""
    section = _cli_section()
    assert "defaults to `checkpoints.rl`" in section
    assert "[default: checkpoints.rl]" in _help("forensics", "failures")
    assert "`--threshold` (short `-t`, default `2`)" in section
    assert "[default: 2]" in _help("forensics", "loops")


def test_cli_reference_replay_examples_use_documented_flags_rc_10() -> None:
    """RC-10: the replay usage examples use --graph, --set-message, --update and --as-node."""
    section = _cli_section()
    examples = [line for line in section.splitlines() if line.startswith("relate-dag replay ")]
    joined = "\n".join(examples)
    for flag in ("--graph", "--set-message", "--update", "--as-node"):
        assert flag in joined, f"no replay example uses {flag}"
    for line in examples:
        if any(flag in line for flag in ("--set-message", "--update", "--as-node")):
            assert "--graph" in line, f"edit flags need --graph: {line}"


def test_cli_reference_edit_flags_without_graph_are_rejected_rc_10() -> None:
    """RC-10: the README says the edit flags need --graph; the CLI rejects them without it."""
    assert "Needs `--graph`" in _cli_section()
    result = CliRunner().invoke(
        app, ["replay", "t", "c", "--set-message", "0=x", "--db", str(DEMO_STORE)]
    )
    assert result.exit_code != 0
    assert "--graph" in result.output


def test_features_list_names_the_invoked_relationship_rc_10() -> None:
    """RC-10: the feature list names `:INVOKED` beside the other graph relationships."""
    text = _readme_text()
    features = text[text.index("## Features") : text.index("## Why not X?")]
    for relationship in (":NEXT", ":FORKED_TO", ":HAS_WRITE", ":INVOKED"):
        assert f"`{relationship}`" in features, relationship


def test_research_demo_documents_auto_approve_flag_rc_10() -> None:
    """RC-10: the multi-agent demo section names `--auto-approve`, which the script accepts."""
    text = _readme_text()
    start = text.index("### Running the multi-agent demo")
    section = text[start : text.index("### Running the tool-calling demo")]
    assert "--auto-approve" in section
    script = (README_PATH.parent / "examples" / "research_agent.py").read_text(encoding="utf-8")
    assert '"--auto-approve"' in script


def test_readme_environment_variables_section_rejects_missing_heading_rc_10() -> None:
    """RC-10: a heading that is not in the README is reported, not silently empty."""
    with pytest.raises(ValueError):
        _readme_text().index("### No such section")
