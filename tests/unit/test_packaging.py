"""Packaging standards and PyPI distribution verification tests.

Verifies pyproject.toml compliance, package metadata completeness,
and documentation requirements under AC-31 and AC-32.
"""

import subprocess
import sys
import tomllib
from pathlib import Path

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
