"""Clean-install verification for the relate-dag command (AC-33).

Installs the package into a fresh virtual environment with no extras and checks that
the console script starts without missing-module errors.
"""

import subprocess
import sys
import venv
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_INSTALL_TIMEOUT_SECONDS = 600
_HELP_TIMEOUT_SECONDS = 120


@pytest.mark.slow
def test_clean_install_relate_dag_help_ac_33(tmp_path: Path) -> None:
    """AC-33: pip install in a fresh venv, then relate-dag --help exits 0 with subcommands."""
    env_dir = tmp_path / "clean-venv"
    venv.create(env_dir, with_pip=True)
    bin_dir = env_dir / ("Scripts" if sys.platform == "win32" else "bin")
    suffix = ".exe" if sys.platform == "win32" else ""
    python = bin_dir / f"python{suffix}"
    relate_dag = bin_dir / f"relate-dag{suffix}"

    install = subprocess.run(
        [str(python), "-m", "pip", "install", str(_REPO_ROOT)],
        capture_output=True,
        text=True,
        timeout=_INSTALL_TIMEOUT_SECONDS,
    )
    assert install.returncode == 0, f"pip install failed:\n{install.stdout}\n{install.stderr}"
    assert relate_dag.exists(), f"relate-dag executable not found at {relate_dag}"

    result = subprocess.run(
        [str(relate_dag), "--help"],
        capture_output=True,
        text=True,
        timeout=_HELP_TIMEOUT_SECONDS,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, f"relate-dag --help failed:\n{output}"
    for subcommand in ("tree", "replay", "forensics"):
        assert subcommand in result.stdout, f"Subcommand {subcommand!r} missing from help output"
    assert "ImportError" not in output
    assert "ModuleNotFoundError" not in output
