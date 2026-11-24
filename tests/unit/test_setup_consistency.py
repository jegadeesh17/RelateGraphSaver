"""Consistency checks for the documented fresh setup, the relate-db requirement and README claims.

Reads the README, docs/ARCHITECTURE.md, docs/DECISIONS.md, pyproject.toml, the example
script's install hint, the installed package metadata and the installed SQLite checkpointer.
Nothing here touches the network. Covers AC-31 (metadata), AC-37 (mypy gate configuration),
AC-139 (optional llm extra), AC-142 to AC-144 (URLs, relate-db requirement and source install)
and the README comparison claims in Journey 24.
"""

import re
import sqlite3
import tomllib
from importlib import metadata
from pathlib import Path
from typing import Any

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import Version

from langgraph_checkpoint_relate.saver import RelateSaver

REPO_ROOT = Path(__file__).resolve().parents[2]
README = REPO_ROOT / "README.md"
ARCHITECTURE = REPO_ROOT / "docs" / "ARCHITECTURE.md"
DECISIONS = REPO_ROOT / "docs" / "DECISIONS.md"
PYPROJECT = REPO_ROOT / "pyproject.toml"
OPENROUTER_EXAMPLE = REPO_ROOT / "examples" / "openrouter_agent.py"
ENV_EXAMPLE = REPO_ROOT / ("." + "env.example")
CHANGELOG = REPO_ROOT / "CHANGELOG.md"

_SPECIFIER_OPERATOR = r"(?:>=|<=|==|~=|!=|<|>)"
_SPECIFIER_TEXT = rf"{_SPECIFIER_OPERATOR}[^\s`,;)]+(?:\s*,\s*{_SPECIFIER_OPERATOR}[^\s`,;)]+)*"
_RELATE_DB_SPECIFIER = re.compile(rf"relate-db\s*({_SPECIFIER_TEXT})")


def _pyproject() -> dict[str, Any]:
    with PYPROJECT.open("rb") as handle:
        return tomllib.load(handle)


def _names(requirements: list[Requirement]) -> set[str]:
    return {canonicalize_name(requirement.name) for requirement in requirements}


def _extra(data: dict[str, Any], extra: str) -> list[Requirement]:
    entries = data["project"].get("optional-dependencies", {}).get(extra, [])
    return [Requirement(entry) for entry in entries]


def _relate_db_specifier(data: dict[str, Any]) -> SpecifierSet:
    matches = [
        Requirement(entry)
        for entry in data["project"]["dependencies"]
        if canonicalize_name(Requirement(entry).name) == "relate-db"
    ]
    assert len(matches) == 1, f"expected exactly one relate-db requirement, found {matches}"
    return matches[0].specifier


def _readme_section(heading: str) -> str:
    text = README.read_text(encoding="utf-8")
    match = re.search(rf"^## {re.escape(heading)}\s*$", text, flags=re.MULTILINE)
    assert match is not None, f"README.md has no '## {heading}' section"
    rest = text[match.end() :]
    following = re.search(r"^## ", rest, flags=re.MULTILINE)
    return rest[: following.start()] if following else rest


def _adr_section(number: str) -> str:
    text = DECISIONS.read_text(encoding="utf-8")
    match = re.search(rf"^## ADR-{number}:.*$", text, flags=re.MULTILINE)
    assert match is not None, f"DECISIONS.md has no ADR-{number} heading"
    rest = text[match.end() :]
    following = re.search(r"^## ", rest, flags=re.MULTILINE)
    return rest[: following.start()] if following else rest


def test_readme_source_install_names_only_declared_extras_ac_144() -> None:
    """AC-144: the source install command names declared extras, including dev, and not llm."""
    readme = README.read_text(encoding="utf-8")
    match = re.search(r'^pip install -e "\.\[([^\]]+)\]"$', readme, flags=re.MULTILINE)
    assert match is not None, 'README has no source install line pip install -e ".[...]"'
    extras = set(match.group(1).split(","))
    declared = set(_pyproject()["project"].get("optional-dependencies", {}))
    assert extras <= declared, f"README installs undeclared extras {extras - declared}"
    assert "dev" in extras, "the README install must include the dev extra for the test suite"
    assert "llm" not in extras, (
        "the llm extra is optional and must not be part of the default install"
    )


def test_architecture_setup_names_the_same_install_command_ac_144() -> None:
    """AC-144: the ARCHITECTURE setup section gives the same editable dev install as the README."""
    architecture = ARCHITECTURE.read_text(encoding="utf-8")
    assert 'pip install -e ".[dev]"' in architecture


def test_dev_extra_holds_the_tools_the_setup_section_names_ac_37() -> None:
    """AC-37: the dev extra brings the lint, type and test tools the setup section says it brings."""
    architecture = ARCHITECTURE.read_text(encoding="utf-8")
    assert "The `dev` extra brings pytest, ruff, mypy and `langgraph-checkpoint-sqlite`" in (
        architecture
    )
    dev = _names(_extra(_pyproject(), "dev"))
    assert {"pytest", "ruff", "mypy", "langgraph-checkpoint-sqlite"} <= dev


def test_llm_extra_stays_optional_and_its_hint_names_it_ac_139() -> None:
    """AC-139: langchain-openai is only in the llm extra, and the example's install hint names it."""
    data = _pyproject()
    assert "langchain-openai" not in _names(_extra(data, "dev"))
    assert "langchain-openai" in _names(_extra(data, "llm"))
    hint = re.search(
        r'pip install "\.\[([^\]]+)\]"', OPENROUTER_EXAMPLE.read_text(encoding="utf-8")
    )
    assert hint is not None, "examples/openrouter_agent.py has no install hint for its extra"
    assert hint.group(1) in data["project"]["optional-dependencies"]
    assert "langchain-openai" in _names(_extra(data, hint.group(1)))


def test_mypy_override_ignores_the_optional_llm_import_ac_37() -> None:
    """AC-37: the mypy override that lets a dev-only install pass covers langchain_openai."""
    overrides = _pyproject()["tool"]["mypy"]["overrides"]
    covering = [o for o in overrides if "langchain_openai" in o.get("module", [])]
    assert len(covering) == 1, f"expected one mypy override for langchain_openai, found {covering}"
    assert covering[0].get("ignore_missing_imports") is True


def test_pyproject_urls_match_the_clone_command_in_readme_ac_142() -> None:
    """AC-142: the README clone command points at the repository named in the project URLs."""
    readme = README.read_text(encoding="utf-8")
    match = re.search(r"^git clone (\S+)$", readme, flags=re.MULTILINE)
    assert match is not None, "README has no git clone line"
    repository = _pyproject()["project"]["urls"]["Repository"]
    assert match.group(1) == f"{repository}.git"


def test_installed_metadata_matches_pyproject_ac_142_ac_143() -> None:
    """AC-142, AC-143: the installed distribution carries the declared version, URLs and requirement.

    A stale editable install would show the old URLs or floor here, so this also catches an
    environment that was not reinstalled after pyproject.toml changed.
    """
    data = _pyproject()
    installed = metadata.metadata("langgraph-checkpoint-relate")
    assert installed["Version"] == data["project"]["version"]

    installed_urls = set(installed.get_all("Project-URL") or [])
    declared_urls = {f"{key}, {value}" for key, value in data["project"]["urls"].items()}
    assert installed_urls == declared_urls

    runtime = [
        Requirement(entry)
        for entry in installed.get_all("Requires-Dist") or []
        if Requirement(entry).marker is None
    ]
    installed_relate_db = [r for r in runtime if canonicalize_name(r.name) == "relate-db"]
    assert len(installed_relate_db) == 1
    assert installed_relate_db[0].specifier == _relate_db_specifier(data)


def test_installed_relate_db_satisfies_the_declared_requirement_ac_143() -> None:
    """AC-143: the relate-db version the suite runs against is inside the declared requirement."""
    installed = Version(metadata.version("relate-db"))
    specifier = _relate_db_specifier(_pyproject())
    assert installed in specifier, f"relate-db {installed} is outside the requirement {specifier}"


def test_adr_0020_is_accepted_and_names_the_declared_requirement_ac_143() -> None:
    """AC-143: ADR-0020 is Accepted and its outcome names the requirement in pyproject.toml."""
    section = _adr_section("0020")
    assert re.search(r"^Status: Accepted\s*$", section, flags=re.MULTILINE) is not None
    outcome = re.search(r"requirement is now `(relate-db[^`]+)`", section)
    assert outcome is not None, "ADR-0020 does not state the requirement it set"
    adr_requirement = Requirement(outcome.group(1))
    assert canonicalize_name(adr_requirement.name) == "relate-db"
    assert adr_requirement.specifier == _relate_db_specifier(_pyproject())


def test_readme_and_architecture_state_the_declared_relate_db_requirement_ac_143() -> None:
    """AC-143: every relate-db specifier written in README or ARCHITECTURE is the declared one.

    ARCHITECTURE's "Expensive to change" list still gave the 1.0.2 floor from ADR-0015 when
    this check was written, so a stale floor in a living document fails here.
    """
    declared = _relate_db_specifier(_pyproject())
    mismatches = [
        f"{path.name}: relate-db{match.group(1)}"
        for path in (README, ARCHITECTURE)
        for match in _RELATE_DB_SPECIFIER.finditer(path.read_text(encoding="utf-8"))
        if SpecifierSet(match.group(1)) != declared
    ]
    assert mismatches == [], f"pyproject declares relate-db{declared}; stated instead: {mismatches}"


def test_architecture_relate_db_stack_line_is_settled_ac_143() -> None:
    """AC-143: the ARCHITECTURE stack line states one requirement and does not say it is pending."""
    architecture = ARCHITECTURE.read_text(encoding="utf-8")
    lines = [
        line
        for line in architecture.splitlines()
        if "relate-db" in line and "requirement is" in line
    ]
    assert len(lines) == 1, f"expected one relate-db requirement line, found {len(lines)}"
    assert "pending" not in lines[0], "ARCHITECTURE says the relate-db decision is pending"
    stated = re.findall(r"`(>=[^`]+)`", lines[0])
    assert len(stated) == 1, f"ARCHITECTURE states more than one relate-db requirement: {stated}"
    assert SpecifierSet(stated[0]) == _relate_db_specifier(_pyproject())


def test_python_versions_agree_across_readme_architecture_and_pyproject() -> None:
    """AC-31: the README prerequisite list, the classifiers and requires-python name the same versions."""
    project = _pyproject()["project"]
    spec = SpecifierSet(project["requires-python"])
    readme = README.read_text(encoding="utf-8")
    prerequisite = re.search(r"^- Python (.+)$", readme, flags=re.MULTILINE)
    assert prerequisite is not None, "README has no 'Python' prerequisite line"
    listed = set(re.findall(r"3\.\d+", prerequisite.group(1)))
    classified = {
        classifier.removeprefix("Programming Language :: Python :: ")
        for classifier in project["classifiers"]
        if classifier.startswith("Programming Language :: Python :: 3.")
    }
    assert listed == classified == {"3.11", "3.12", "3.13"}
    assert all(Version(v) in spec for v in listed)
    assert Version("3.10") not in spec

    architecture = ARCHITECTURE.read_text(encoding="utf-8")
    stated = re.search(r'requires-python = "([^"]+)"', architecture)
    assert stated is not None
    assert SpecifierSet(stated.group(1)) == spec


def test_readme_benchmark_supports_the_why_not_x_sqlite_claim_ac_146() -> None:
    """AC-146: the SqliteSaver row says it is faster and the benchmark table shows it on every speed metric.

    Table columns are: RelateSaver (default), RelateSaver (wal_enabled=False), SqliteSaver.
    """
    rows: dict[str, list[float]] = {}
    for line in _readme_section("Benchmark").splitlines():
        if not line.startswith("| ") or line.startswith(("| Metric", "| :")):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        rows[cells[0]] = [float(cell) for cell in cells[1:]]

    def row(prefix: str) -> list[float]:
        found = [values for label, values in rows.items() if label.startswith(prefix)]
        assert len(found) == 1, f"expected one benchmark row starting '{prefix}', found {found}"
        return found[0]

    throughput = row("Put throughput")
    assert throughput[2] > max(throughput[0], throughput[1])
    for prefix in ("Get p50", "Get p95", "Ancestry query"):
        latency = row(prefix)
        assert latency[2] < min(latency[0], latency[1]), prefix
    size = row("On-disk size")
    assert size[2] < min(size[0], size[1])

    sqlite_row = next(
        line
        for line in _readme_section("Why not X?").splitlines()
        if line.startswith("| SqliteSaver")
    )
    assert "faster than RelateSaver" in sqlite_row


def test_sqlite_row_tables_match_the_installed_checkpointer_ac_146() -> None:
    """AC-146: the SqliteSaver row names the two tables the installed langgraph-checkpoint-sqlite creates."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    connection = sqlite3.connect(":memory:")
    try:
        SqliteSaver(connection).setup()
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
            if not row[0].startswith("sqlite_")
        }
    finally:
        connection.close()

    assert tables == {"checkpoints", "writes"}
    sqlite_row = next(
        line
        for line in _readme_section("Why not X?").splitlines()
        if line.startswith("| SqliteSaver")
    )
    assert "`checkpoints` and `writes`" in sqlite_row


RELATE_ENV_VARS = (
    "RELATE_DB_PATH",
    "RELATE_WAL_ENABLED",
    "RELATE_PROXY_TIMEOUT_SEC",
    "RELATE_LOG_LEVEL",
)


def _readme_env_table() -> dict[str, str]:
    """Return the README environment-variable table as {variable: default cell}."""
    text = README.read_text(encoding="utf-8")
    start = text.index("### Environment variables")
    section = text[start : text.index("\n## ", start)]
    rows: dict[str, str] = {}
    for line in section.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if line.startswith("| `RELATE_") and len(cells) >= 2:
            rows[cells[0].strip("`")] = cells[1].strip("`")
    return rows


def test_env_example_lists_each_variable_with_a_comment_and_no_value_rc_10() -> None:
    """RC-10: each RELATE_* variable is a commented `# NAME=` line, preceded by a comment.

    Copying the file to `.env` must not set an empty RELATE_* value, which RelateSaver rejects.
    """
    lines = ENV_EXAMPLE.read_text(encoding="utf-8").splitlines()
    for name in RELATE_ENV_VARS:
        assert f"# {name}=" in lines, name
        assert lines[lines.index(f"# {name}=") - 1].startswith("#"), name
    assert "OPENROUTER_API_KEY=" in lines
    assert lines[lines.index("OPENROUTER_API_KEY=") - 1].startswith("#")
    assignments = [line for line in lines if line and not line.startswith("#")]
    assert all(line.endswith("=") and line.count("=") == 1 for line in assignments)


def test_env_example_has_no_active_relate_assignment_rc_10(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RC-10: no RELATE_* line is active, so a copied `.env` leaves the saver defaults intact."""
    lines = ENV_EXAMPLE.read_text(encoding="utf-8").splitlines()
    assert [line for line in lines if line.startswith("RELATE_")] == []
    # An active empty value is what the file used to ship; the saver rejects it.
    monkeypatch.setenv("RELATE_WAL_ENABLED", "")
    with pytest.raises(ValueError, match="RELATE_WAL_ENABLED"):
        RelateSaver()


def test_env_example_drops_unused_provider_keys_rc_10() -> None:
    """RC-10: OPENAI_API_KEY and ANTHROPIC_API_KEY are gone, in comments as well."""
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    assert "OPENAI_API_KEY" not in text
    assert "ANTHROPIC_API_KEY" not in text


def test_readme_env_table_covers_every_relate_variable_rc_10() -> None:
    """RC-10: the README table names exactly the four RELATE_* variables the saver reads."""
    assert set(_readme_env_table()) == set(RELATE_ENV_VARS)


def test_readme_env_defaults_match_the_saver_rc_10(monkeypatch: pytest.MonkeyPatch) -> None:
    """RC-10: each default in the README table is the value an unset variable resolves to."""
    for name in RELATE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    table = _readme_env_table()
    with RelateSaver() as saver:
        assert table["RELATE_DB_PATH"].startswith(str(saver.db_path))
        assert table["RELATE_WAL_ENABLED"] == str(saver.wal_enabled).lower()
        assert float(table["RELATE_PROXY_TIMEOUT_SEC"]) == saver.default_timeout
        assert table["RELATE_LOG_LEVEL"] == saver.log_level


@pytest.mark.parametrize(
    ("name", "bad"),
    [
        ("RELATE_WAL_ENABLED", "maybe"),
        ("RELATE_PROXY_TIMEOUT_SEC", "0"),
        ("RELATE_LOG_LEVEL", "LOUD"),
    ],
)
def test_readme_states_invalid_env_values_raise_rc_10(
    monkeypatch: pytest.MonkeyPatch, name: str, bad: str
) -> None:
    """RC-10: the README says an invalid value raises ValueError; the saver does."""
    assert "`ValueError`" in README.read_text(encoding="utf-8")
    monkeypatch.setenv(name, bad)
    with pytest.raises(ValueError, match=name):
        RelateSaver()


def test_changelog_semver_link_points_to_spec_2_0_0_rc_10() -> None:
    """RC-10: the changelog links Semantic Versioning 2.0.0."""
    text = CHANGELOG.read_text(encoding="utf-8")
    assert "[Semantic Versioning](https://semver.org/spec/v2.0.0.html)" in text
    assert "spec/v2.0.4" not in text
