"""Integration tests for environment parsing in the CLI and timed-out saver writes.

Verifies AC-194 and AC-195 through the entry points a user runs:
- relate-dag reads RELATE_LOG_LEVEL, RELATE_WAL_ENABLED and RELATE_PROXY_TIMEOUT_SEC the way
  the library does, and ignores RELATE_DB_PATH because it takes the store from --db (ADR-0033).
- A RelateSaver.put whose proxy write is still running when the caller's timeout expires raises
  RelateConcurrencyError saying the write may have committed, and the write does commit
  (ADR-0033).
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest
import relate
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import Checkpoint, empty_checkpoint
from typer.testing import CliRunner

from langgraph_checkpoint_relate.cli import app
from langgraph_checkpoint_relate.errors import RelateConcurrencyError
from langgraph_checkpoint_relate.saver import RelateSaver

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_STORE = REPO_ROOT / "examples" / "data" / "demo.rl"
DEMO_THREAD = "demo-tool-fail"
PACKAGE_LOGGER = "langgraph_checkpoint_relate"
RELATE_ENV_NAMES = (
    "RELATE_DB_PATH",
    "RELATE_WAL_ENABLED",
    "RELATE_PROXY_TIMEOUT_SEC",
    "RELATE_LOG_LEVEL",
)


@pytest.fixture
def clean_relate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every RELATE_* variable so each test sets only the ones it needs."""
    for name in RELATE_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def restore_package_logger() -> Iterator[None]:
    """Restore the package logger level, because a valid RELATE_LOG_LEVEL changes it."""
    package_logger = logging.getLogger(PACKAGE_LOGGER)
    saved_level = package_logger.level
    yield
    package_logger.setLevel(saved_level)


@pytest.fixture
def demo_copy(tmp_path: Path) -> Path:
    """Copy the committed demo store so the CLI never touches the file under examples/."""
    copy_path = tmp_path / "demo_copy.rl"
    shutil.copy(DEMO_STORE, copy_path)
    return copy_path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestCliReadsEnvironmentLikeLibrary:
    """AC-194: relate-dag tree parses only the RELATE_* variables it uses."""

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("RELATE_LOG_LEVEL", "LOUD"),
            ("RELATE_WAL_ENABLED", "maybe"),
            ("RELATE_PROXY_TIMEOUT_SEC", "0"),
        ],
    )
    def test_invalid_env_value_fails_tree_with_named_variable_ac_194(
        self,
        clean_relate_env: None,
        monkeypatch: pytest.MonkeyPatch,
        demo_copy: Path,
        name: str,
        value: str,
    ) -> None:
        """AC-194: an invalid value for a variable the CLI reads exits 1 and names it."""
        monkeypatch.setenv(name, value)
        before = _sha256(demo_copy)

        result = CliRunner().invoke(app, ["tree", DEMO_THREAD, "--db", str(demo_copy)])

        assert result.exit_code == 1
        assert f"Invalid value for {name}" in result.output
        assert DEMO_THREAD not in result.output
        assert _sha256(demo_copy) == before

    def test_invalid_db_path_env_is_ignored_by_cli_ac_194(
        self, clean_relate_env: None, monkeypatch: pytest.MonkeyPatch, demo_copy: Path
    ) -> None:
        """AC-194: RELATE_DB_PATH is not read by relate-dag, so an unusable path does not fail."""
        monkeypatch.setenv("RELATE_DB_PATH", str(demo_copy.parent / "no_such_dir" / "x.rl"))

        result = CliRunner().invoke(app, ["tree", DEMO_THREAD, "--db", str(demo_copy)])

        assert result.exit_code == 0, result.output
        assert DEMO_THREAD in result.output

    def test_lowercase_valid_log_level_is_read_and_applied_ac_194(
        self,
        clean_relate_env: None,
        monkeypatch: pytest.MonkeyPatch,
        demo_copy: Path,
        restore_package_logger: None,
    ) -> None:
        """AC-194: a valid RELATE_LOG_LEVEL is accepted case-insensitively and sets the logger."""
        monkeypatch.setenv("RELATE_LOG_LEVEL", "debug")

        result = CliRunner().invoke(app, ["tree", DEMO_THREAD, "--db", str(demo_copy)])

        assert result.exit_code == 0, result.output
        assert DEMO_THREAD in result.output
        assert logging.getLogger(PACKAGE_LOGGER).level == logging.DEBUG


class _HoldingConnection:
    """Wraps a real connection; once armed, the first statement blocks until released."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.armed = False
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()

    @property
    def autocommit(self) -> bool:
        return bool(self._inner.autocommit)

    @autocommit.setter
    def autocommit(self, value: bool) -> None:
        self._inner.autocommit = value

    def execute(self, cypher: str, *args: Any) -> Any:
        if self.armed and not self.started.is_set():
            self.started.set()
            assert self.release.wait(timeout=10.0), "test never released the held statement"
            try:
                return self._inner.execute(cypher, *args)
            finally:
                self.finished.set()
        return self._inner.execute(cypher, *args)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _checkpoint(checkpoint_id: str) -> Checkpoint:
    base: dict[str, Any] = dict(empty_checkpoint())
    base["id"] = checkpoint_id
    base["channel_values"] = {"data": "held"}
    base["channel_versions"] = {"data": "1.0"}
    return cast(Checkpoint, base)


class TestTimedOutSaverWriteAc195:
    """AC-195: a put held past the caller's timeout says it may have committed."""

    def test_put_held_past_timeout_reports_may_have_committed_ac_195(self) -> None:
        """AC-195: the put raises a timeout that says the write may have committed, and it did."""
        connections: list[_HoldingConnection] = []

        def factory(db_path: str) -> _HoldingConnection:
            conn = _HoldingConnection(relate.connect(database=db_path))
            connections.append(conn)
            return conn

        saver = RelateSaver(
            db_path=":memory:",
            wal_enabled=False,
            connection_factory=factory,
            default_timeout=0.5,
        )
        gate = connections[0]
        config: RunnableConfig = {"configurable": {"thread_id": "ac195-held", "checkpoint_ns": ""}}
        try:
            gate.armed = True
            with pytest.raises(RelateConcurrencyError) as exc_info:
                saver.put(config, _checkpoint("cp-ac195-held"), {"step": 1}, {})

            assert gate.started.is_set(), "the write never started, so the test proves nothing"
            message = str(exc_info.value)
            assert "may have committed" in message
            assert "not executed" not in message

            gate.release.set()
            assert gate.finished.wait(timeout=10.0)
            saver.proxy.flush(timeout=10.0)
            restored = saver.get_tuple(config)
            assert restored is not None
            assert restored.checkpoint["id"] == "cp-ac195-held"
        finally:
            gate.release.set()
            saver.close()
