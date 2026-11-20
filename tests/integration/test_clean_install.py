"""Clean-install verification for the relate-dag command (AC-33).

Installs the package into a fresh virtual environment with no extras and checks that
the console script starts without missing-module errors. The install step is skipped,
not failed, only when pip's output shows the package index is unreachable (ADR-0013).
"""

import subprocess
import sys
import venv
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_INSTALL_TIMEOUT_SECONDS = 600
_HELP_TIMEOUT_SECONDS = 120

# Case-sensitive substrings that pip prints when it cannot reach the package index.
# Resolver messages such as "No matching distribution found" are deliberately absent:
# on their own they point to a real dependency problem (ADR-0013).
_INDEX_UNREACHABLE_MARKERS = (
    "Failed to establish a new connection",
    "NewConnectionError",
    "ConnectTimeoutError",
    "Max retries exceeded with url",
    "Temporary failure in name resolution",
    "getaddrinfo failed",
    "Name or service not known",
    "nodename nor servname provided",
    "Network is unreachable",
    "No route to host",
    "ProxyError",
)


def _index_unreachable(output: str) -> str | None:
    """Return the first output line naming a connection failure, or None."""
    for line in output.splitlines():
        if any(marker in line for marker in _INDEX_UNREACHABLE_MARKERS):
            return line.strip()
    return None


def test_index_unreachable_returns_matching_line_ac_33() -> None:
    """AC-33: offline pip output yields the first line naming the connection failure."""
    output = (
        "Collecting relate-db>=1.0\n"
        "  WARNING: Retrying (Retry(total=0)) after connection broken by "
        "'NewConnectionError(<HTTPSConnection object>: Failed to establish a new "
        "connection: [Errno -3] Temporary failure in name resolution)': /simple/relate-db/\n"
        "ERROR: Could not find a version that satisfies the requirement relate-db>=1.0\n"
        "ERROR: No matching distribution found for relate-db>=1.0\n"
    )
    line = _index_unreachable(output)
    assert line is not None
    assert line.startswith("WARNING: Retrying")
    assert "Failed to establish a new connection" in line
    assert "\n" not in line
    assert _index_unreachable("ProxyError('Cannot connect to proxy.')") is not None


@pytest.mark.parametrize(
    "output",
    [
        "ERROR: No matching distribution found for relate-db>=99\n",
        "ERROR: Could not find a version that satisfies the requirement relate-db>=99\n"
        "ERROR: No matching distribution found for relate-db>=99\n",
        "SSLError(SSLCertVerificationError: certificate verify failed)\n",
        "subprocess.TimeoutExpired: Command 'pip install' timed out after 600 seconds\n",
        "error: subprocess-exited-with-error\n  Failed building wheel for something\n",
        "relate-dag --help failed:\nModuleNotFoundError: No module named 'rich'\n",
        "",
    ],
)
def test_index_unreachable_ignores_other_failures_ac_33(output: str) -> None:
    """AC-33: resolver, SSL, build, timeout and --help failures are never treated as offline."""
    assert _index_unreachable(output) is None


def _fresh_venv(env_dir: Path) -> tuple[Path, Path]:
    """Create a virtual environment with pip; return its python executable and script directory."""
    venv.create(env_dir, with_pip=True)
    bin_dir = env_dir / ("Scripts" if sys.platform == "win32" else "bin")
    suffix = ".exe" if sys.platform == "win32" else ""
    return bin_dir / f"python{suffix}", bin_dir


def _pip_install_or_skip(python: Path, pip_args: list[str], cwd: Path | None = None) -> None:
    """Run pip install in the environment; skip only when the package index is unreachable."""
    install = subprocess.run(
        [str(python), "-m", "pip", "install", *pip_args],
        capture_output=True,
        text=True,
        cwd=cwd,
        timeout=_INSTALL_TIMEOUT_SECONDS,
    )
    install_output = f"{install.stdout}\n{install.stderr}"
    if install.returncode != 0:
        unreachable_line = _index_unreachable(install_output)
        if unreachable_line is not None:
            pytest.skip(
                f"package index unreachable, clean install not verified: {unreachable_line}"
            )
    assert install.returncode == 0, f"pip install failed:\n{install_output}"


@pytest.mark.slow
def test_clean_install_relate_dag_help_ac_33(tmp_path: Path) -> None:
    """AC-33: pip install in a fresh venv, then relate-dag --help exits 0 with subcommands."""
    python, bin_dir = _fresh_venv(tmp_path / "clean-venv")
    suffix = ".exe" if sys.platform == "win32" else ""
    relate_dag = bin_dir / f"relate-dag{suffix}"

    _pip_install_or_skip(python, [str(_REPO_ROOT)])
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


_MYPY_TIMEOUT_SECONDS = 900


@pytest.mark.slow
def test_dev_extra_install_passes_mypy_gate_without_llm_extra_ac_37(tmp_path: Path) -> None:
    """AC-37: in a venv with only the dev extra (no langchain-openai), `mypy src tests` exits 0.

    The check first proves the venv really lacks langchain_openai, so a pass cannot come from
    an extra that happens to be installed.
    """
    python, _ = _fresh_venv(tmp_path / "dev-venv")
    _pip_install_or_skip(python, ["-e", ".[dev]"], cwd=_REPO_ROOT)

    probe = subprocess.run(
        [
            str(python),
            "-c",
            "import importlib.util, sys; "
            "sys.exit(importlib.util.find_spec('langchain_openai') is not None)",
        ],
        capture_output=True,
        text=True,
        timeout=_HELP_TIMEOUT_SECONDS,
    )
    assert probe.returncode == 0, "langchain_openai is installed; this venv is not dev-only"

    mypy = subprocess.run(
        [str(python), "-m", "mypy", "src", "tests"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=_MYPY_TIMEOUT_SECONDS,
    )
    output = f"{mypy.stdout}{mypy.stderr}"
    assert mypy.returncode == 0, f"mypy failed in a dev-only install:\n{output}"
    assert "Success:" in output
