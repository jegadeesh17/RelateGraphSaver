"""Packaging standards and PyPI distribution verification tests.

Verifies pyproject.toml compliance, package metadata completeness,
and documentation requirements under AC-31 and AC-32.
"""

import ast
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest


def test_pyproject_metadata_ac_31() -> None:
    """AC-31: Verify pyproject.toml has all required PyPI packaging metadata."""
    pyproject_path = Path("pyproject.toml")
    assert pyproject_path.exists(), "pyproject.toml must exist in repo root"

    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    # Build system
    assert "build-system" in data
    assert data["build-system"]["requires"] == ["hatchling"]
    assert data["build-system"]["build-backend"] == "hatchling.build"

    # Project core metadata
    project = data.get("project", {})
    assert project.get("name") == "langgraph-checkpoint-relate"
    assert project.get("requires-python") == ">=3.11"
    assert project.get("license") == {"text": "MIT"}
    assert project.get("authors") is not None
    assert len(project["authors"]) > 0

    # Classifiers
    classifiers = project.get("classifiers", [])
    expected_classifiers = [
        "Framework :: LangGraph",
        "Topic :: Database",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ]
    for ec in expected_classifiers:
        assert ec in classifiers, f"Missing required classifier: {ec}"

    # Keywords
    keywords = project.get("keywords", [])
    expected_keywords = [
        "langgraph",
        "relate",
        "neo4j",
        "checkpointer",
        "persistence",
        "agent",
        "dag",
        "cypher",
    ]
    for ek in expected_keywords:
        assert ek in keywords, f"Missing required keyword: {ek}"

    # Project URLs
    urls = project.get("urls", {})
    assert "Homepage" in urls
    assert "Repository" in urls
    assert "Issues" in urls

    # CLI Entrypoints
    scripts = project.get("scripts", {})
    assert "relate-dag" in scripts
    assert scripts["relate-dag"] == "langgraph_checkpoint_relate.cli:app"

    # Wheel target packaging
    hatch_packages = (
        data.get("tool", {})
        .get("hatch", {})
        .get("build", {})
        .get("targets", {})
        .get("wheel", {})
        .get("packages", [])
    )
    assert hatch_packages == ["src/langgraph_checkpoint_relate"]


def test_readme_documentation_completeness_ac_31() -> None:
    """AC-31: Verify root README.md exists and contains required sections."""
    readme_path = Path("README.md")
    assert readme_path.exists(), "Root README.md must exist for PyPI packaging"

    content = readme_path.read_text(encoding="utf-8")
    assert len(content.strip()) > 500, "README.md should have comprehensive documentation"

    # Required sections & keywords
    assert "RelateGraphSaver" in content or "langgraph-checkpoint-relate" in content
    assert "Relate by Neo4j" in content or "relate-db" in content
    assert "pip install" in content
    assert "RelateSaver" in content
    assert "relate-dag tree" in content
    assert "relate-dag replay" in content
    assert "relate-dag forensics" in content
    assert "fork_in_memory" in content


def test_packaging_metadata_missing_fields_validation() -> None:
    """Verify validation boundary: detects incomplete or malformed pyproject tables."""
    incomplete_config: dict[str, object] = {
        "project": {
            "name": "langgraph-checkpoint-relate",
            # missing requires-python, license, authors, etc.
        }
    }

    project = incomplete_config.get("project", {})
    assert isinstance(project, dict)
    with pytest.raises(AssertionError):
        assert project.get("requires-python") == ">=3.11"
    with pytest.raises(AssertionError):
        assert project.get("license") == {"text": "MIT"}


def test_cli_entrypoint_help_ac_32() -> None:
    """AC-32: Verify relate-dag --help executes cleanly and displays commands."""
    res = subprocess.run(
        [sys.executable, "-m", "langgraph_checkpoint_relate.cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "tree" in res.stdout
    assert "replay" in res.stdout
    assert "forensics" in res.stdout


_REPO_ROOT = Path(__file__).resolve().parents[2]

# Import name to distribution name for every third-party module the package may import.
_MODULE_TO_DISTRIBUTION: dict[str, str] = {
    "relate": "relate-db",
    "langchain_core": "langchain-core",
    "langgraph": "langgraph",
    "typer": "typer",
    "click": "click",
    "rich": "rich",
}


def _normalize_name(name: str) -> str:
    """Normalize a distribution name per PEP 503."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirement_name(requirement: str) -> str:
    """Return the normalized distribution name of a PEP 508 requirement string."""
    match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement)
    assert match is not None, f"Unparseable requirement: {requirement!r}"
    return _normalize_name(match.group(1))


def _runtime_dependency_names(data: dict[str, Any]) -> set[str]:
    dependencies = data.get("project", {}).get("dependencies", [])
    return {_requirement_name(requirement) for requirement in dependencies}


def _imported_top_level_modules(source_root: Path) -> dict[str, Path]:
    """Map each absolute top-level import under source_root to a file that imports it."""
    modules: dict[str, Path] = {}
    for path in sorted(source_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                modules.setdefault(name.split(".")[0], path)
    return modules


def test_click_is_runtime_dependency_ac_34() -> None:
    """AC-34: click is declared in [project].dependencies, not only in extras."""
    with open(_REPO_ROOT / "pyproject.toml", "rb") as f:
        data = tomllib.load(f)

    assert "click" in _runtime_dependency_names(data)


def test_click_check_rejects_extras_only_declaration_ac_34() -> None:
    """AC-34: a click requirement that lives only in an extra is not a runtime dependency."""
    config: dict[str, Any] = {
        "project": {
            "dependencies": ["typer>=0.12.0"],
            "optional-dependencies": {"dev": ["click>=8.1.0"]},
        }
    }

    assert "click" not in _runtime_dependency_names(config)


def test_requirement_name_normalization_ac_34() -> None:
    """AC-34: requirement names are normalized so Click and click[extra]>=8 both match."""
    assert _requirement_name("Click>=8.1.0") == "click"
    assert _requirement_name("langchain_core >= 0.2.0") == "langchain-core"
    assert _requirement_name("click[extra]>=8.1; python_version>='3.11'") == "click"


def test_src_imports_are_declared_dependencies_ac_34() -> None:
    """AC-34: every third-party module imported under src/ maps to a declared dependency."""
    with open(_REPO_ROOT / "pyproject.toml", "rb") as f:
        declared = _runtime_dependency_names(tomllib.load(f))

    own_name = "langgraph_checkpoint_relate"
    imported = _imported_top_level_modules(_REPO_ROOT / "src")
    assert imported, "No imports found under src/; the scan is not looking at the package"

    for module, path in sorted(imported.items()):
        if module in sys.stdlib_module_names or module == own_name or module == "__future__":
            continue
        assert module in _MODULE_TO_DISTRIBUTION, (
            f"Module {module!r} imported in {path} has no entry in the module-to-distribution table"
        )
        distribution = _MODULE_TO_DISTRIBUTION[module]
        assert _normalize_name(distribution) in declared, (
            f"Module {module!r} imported in {path} needs {distribution!r} in [project].dependencies"
        )


def _dev_extra_names(data: dict[str, Any]) -> set[str]:
    dev = data.get("project", {}).get("optional-dependencies", {}).get("dev", [])
    return {_requirement_name(requirement) for requirement in dev}


def test_sqlite_checkpointer_is_dev_only_ac_57() -> None:
    """AC-57: langgraph-checkpoint-sqlite is in the dev extra and not a runtime dependency."""
    with open(_REPO_ROOT / "pyproject.toml", "rb") as f:
        data = tomllib.load(f)

    assert "langgraph-checkpoint-sqlite" in _dev_extra_names(data)
    assert "langgraph-checkpoint-sqlite" not in _runtime_dependency_names(data)


def test_sqlite_dev_only_check_rejects_runtime_declaration_ac_57() -> None:
    """AC-57: the check tells a runtime declaration apart from a dev-extra one."""
    config: dict[str, Any] = {
        "project": {
            "dependencies": ["langgraph-checkpoint-sqlite>=2.0.0"],
            "optional-dependencies": {"dev": ["pytest>=8.3.0"]},
        }
    }

    assert "langgraph-checkpoint-sqlite" in _runtime_dependency_names(config)
    assert "langgraph-checkpoint-sqlite" not in _dev_extra_names(config)
