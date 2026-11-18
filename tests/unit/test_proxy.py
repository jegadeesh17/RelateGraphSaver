"""Unit tests for RelateThreadProxy actor concurrency engine."""

# ruff: noqa: E402

from __future__ import annotations

import concurrent.futures
import logging
import sys
import threading
from pathlib import Path
from typing import Any

import pytest
import relate

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from tests.fakes.fake_relate_driver import InMemoryRelateDriver

from langgraph_checkpoint_relate.errors import (
    RelateCheckpointError,
    RelateConcurrencyError,
    RelateConnectionError,
)
from langgraph_checkpoint_relate.proxy import RelateThreadProxy


class TestProxyLifecycle:
    """Test suite for RelateThreadProxy actor lifecycle, ping, and shutdown semantics."""

    def test_proxy_lifecycle_ac_08(self) -> None:
        """AC-08: Verify actor initializes worker, opens DB, initializes schema, and shuts down."""
        proxy = RelateThreadProxy(db_path=":memory:")
        assert proxy.is_alive()
        assert proxy.ping()

        res = proxy.execute_read("RETURN 1 AS active")
        assert res == [{"active": 1}]

        proxy.close()
        assert not proxy.is_alive()
        assert not proxy.ping()

    def test_proxy_context_manager_ac_08(self) -> None:
        """AC-08: Verify RelateThreadProxy works cleanly as a context manager."""
        with RelateThreadProxy(db_path=":memory:") as proxy:
            assert proxy.is_alive()
            res = proxy.execute("RETURN 'ctx' AS label")
            assert res == [{"label": "ctx"}]

        assert not proxy.is_alive()

    def test_proxy_close_idempotency_ac_08(self) -> None:
        """AC-08: Verify multiple close calls are safe and idempotent."""
        proxy = RelateThreadProxy(db_path=":memory:")
        proxy.close()
        proxy.close()
        proxy.stop()
        assert not proxy.is_alive()

    def test_proxy_startup_failure_raises_connection_error(self) -> None:
        """Verify startup exception in connection factory raises RelateConnectionError."""

        def broken_factory(path: str) -> Any:
            raise RuntimeError("Database driver failed to initialize")

        with pytest.raises(RelateConnectionError) as exc_info:
            RelateThreadProxy(db_path=":memory:", connection_factory=broken_factory)
        assert "Database driver failed to initialize" in str(exc_info.value)


class TestProxyExecution:
    """Test suite for Cypher query and batch transaction execution."""

    def test_execute_read_and_write_ac_08(self) -> None:
        """AC-08: Verify execute_write commits nodes and execute_read retrieves them."""
        with RelateThreadProxy(db_path=":memory:") as proxy:
            write_res = proxy.execute_write(
                "CREATE (t:Thread {thread_id: $tid}) RETURN t.thread_id AS id",
                {"tid": "thread-101"},
            )
            assert write_res == [{"id": "thread-101"}]

            read_res = proxy.execute_read(
                "MATCH (t:Thread {thread_id: $tid}) RETURN t.thread_id AS id",
                {"tid": "thread-101"},
            )
            assert read_res == [{"id": "thread-101"}]

    def test_execute_general_method_ac_08(self) -> None:
        """AC-08: Verify general execute() method works for reads and writes."""
        with RelateThreadProxy(db_path=":memory:") as proxy:
            res1 = proxy.execute("CREATE (t:Thread {thread_id: $tid})", {"tid": "thread-102"})
            assert isinstance(res1, list)

            res2 = proxy.execute(
                "MATCH (t:Thread {thread_id: $tid}) RETURN t.thread_id AS id", {"tid": "thread-102"}
            )
            assert res2 == [{"id": "thread-102"}]

    def test_execute_batch_atomicity_ac_08(self) -> None:
        """AC-08: Verify execute_batch executes multiple statements in a single transaction."""
        with RelateThreadProxy(db_path=":memory:") as proxy:
            statements: list[tuple[str, dict[str, Any]]] = [
                ("CREATE (t:Thread {thread_id: 'batch-1'})", {}),
                ("CREATE (t:Thread {thread_id: 'batch-2'})", {}),
                ("CREATE (t:Thread {thread_id: 'batch-3'})", {}),
            ]
            batch_res = proxy.execute_batch(statements)
            assert len(batch_res) == 3

            count_res = proxy.execute_read("MATCH (t:Thread) RETURN count(t) AS total")
            assert count_res[0]["total"] == 3

    def test_transaction_alias_ac_08(self) -> None:
        """AC-08: Verify transaction() alias behaves identically to execute_batch()."""
        with RelateThreadProxy(db_path=":memory:") as proxy:
            statements: list[tuple[str, dict[str, Any]]] = [
                ("CREATE (t:Thread {thread_id: 'tx-1'})", {}),
                ("CREATE (t:Thread {thread_id: 'tx-2'})", {}),
            ]
            tx_res = proxy.transaction(statements)
            assert len(tx_res) == 2

            count_res = proxy.execute_read("MATCH (t:Thread) RETURN count(t) AS total")
            assert count_res[0]["total"] == 2

    def test_batch_rollback_on_failure_ac_08(self) -> None:
        """AC-08: Verify failed statement in transaction rolls back all statements."""
        with RelateThreadProxy(db_path=":memory:") as proxy:
            # Statement 1 succeeds, Statement 2 has syntax error
            statements: list[tuple[str, dict[str, Any]]] = [
                ("CREATE (t:Thread {thread_id: 'rollback-item'})", {}),
                ("INVALID CYPHER SYNTAX", {}),
            ]
            with pytest.raises(RelateCheckpointError):
                proxy.transaction(statements)

            count_res = proxy.execute_read(
                "MATCH (t:Thread {thread_id: 'rollback-item'}) RETURN count(t) AS cnt"
            )
            assert count_res[0]["cnt"] == 0

    def test_empty_batch_returns_empty_list_ac_08(self) -> None:
        """AC-08: Verify empty batch returns an empty list immediately."""
        with RelateThreadProxy(db_path=":memory:") as proxy:
            res = proxy.execute_batch([])
            assert res == []

    def test_tuple_protocol_execution_ac_08(self) -> None:
        """AC-08: Verify legacy (query, params, event, result_holder) actor protocol."""
        with RelateThreadProxy(db_path=":memory:") as proxy:
            event = threading.Event()
            holder: dict[str, Any] = {}
            proxy._queue.put(("RETURN 42 AS answer", {}, event, holder))
            assert event.wait(timeout=5.0)
            assert holder.get("error") is None
            assert holder.get("result") == [{"answer": 42}]

    def test_dict_protocol_execution_ac_08(self) -> None:
        """AC-08: Verify command dictionary message protocol (execute, transaction, is_alive)."""
        with RelateThreadProxy(db_path=":memory:") as proxy:
            # Test execute dict
            exec_fut: concurrent.futures.Future[Any] = concurrent.futures.Future()
            proxy._queue.put(
                {
                    "type": "execute",
                    "cypher": "CREATE (t:Thread {thread_id: 'dict-1'}) RETURN t.thread_id AS id",
                    "params": {},
                    "future": exec_fut,
                }
            )
            assert exec_fut.result(timeout=5.0) == [{"id": "dict-1"}]

            # Test transaction dict
            tx_fut: concurrent.futures.Future[Any] = concurrent.futures.Future()
            proxy._queue.put(
                {
                    "type": "transaction",
                    "statements": [("CREATE (t:Thread {thread_id: 'dict-2'})", {})],
                    "future": tx_fut,
                }
            )
            assert len(tx_fut.result(timeout=5.0)) == 1

            # Test is_alive dict
            alive_fut: concurrent.futures.Future[Any] = concurrent.futures.Future()
            proxy._queue.put({"type": "is_alive", "future": alive_fut})
            assert alive_fut.result(timeout=5.0) is True


class TestProxyValidationAndErrors:
    """Test suite for boundary input validation and exception wrapping."""

    def test_invalid_cypher_input_raises(self) -> None:
        """Verify invalid or empty Cypher inputs raise RelateCheckpointError."""
        with RelateThreadProxy(db_path=":memory:") as proxy:
            with pytest.raises(RelateCheckpointError, match="non-empty string"):
                proxy.execute("")

            with pytest.raises(RelateCheckpointError, match="non-empty string"):
                proxy.execute("   \n  ")

            with pytest.raises(RelateCheckpointError, match="non-empty string"):
                proxy.execute(None)  # type: ignore[arg-type]

    def test_invalid_parameters_input_raises(self) -> None:
        """Verify malformed parameters raise RelateCheckpointError."""
        with RelateThreadProxy(db_path=":memory:") as proxy:
            with pytest.raises(RelateCheckpointError, match="must be a dictionary"):
                proxy.execute("RETURN 1", params=123)  # type: ignore[arg-type]

    def test_invalid_batch_input_raises(self) -> None:
        """Verify malformed batch inputs raise RelateCheckpointError."""
        with RelateThreadProxy(db_path=":memory:") as proxy:
            with pytest.raises(RelateCheckpointError, match="must be a sequence"):
                proxy.execute_batch("not a list")  # type: ignore[arg-type]

            with pytest.raises(RelateCheckpointError, match="2-tuple"):
                proxy.execute_batch(["invalid item"])  # type: ignore[list-item]

            with pytest.raises(RelateCheckpointError, match="invalid or empty query"):
                proxy.execute_batch([("", {})])

            with pytest.raises(RelateCheckpointError, match="parameters must be a dictionary"):
                proxy.execute_batch([("RETURN 1", "not a dict")])  # type: ignore[list-item]

    def test_cypher_syntax_error_wrapped(self) -> None:
        """Verify database syntax errors are wrapped into RelateCheckpointError."""
        with RelateThreadProxy(db_path=":memory:") as proxy:
            with pytest.raises(RelateCheckpointError) as exc_info:
                proxy.execute("MALFORMED QUERY STATEMENT")
            assert "Relate database error" in str(exc_info.value)

    def test_schema_constraint_violation_wrapped(self) -> None:
        """Verify schema uniqueness violations are wrapped into RelateCheckpointError."""
        with RelateThreadProxy(db_path=":memory:") as proxy:
            proxy.execute_write("CREATE (t:Thread {thread_id: 'dup-id'})")
            with pytest.raises(RelateCheckpointError) as exc_info:
                proxy.execute_write("CREATE (t:Thread {thread_id: 'dup-id'})")
            assert (
                "ConstraintValidationFailed" in str(exc_info.value)
                or "unique" in str(exc_info.value).lower()
            )

    def test_closed_proxy_operations_raise_relate_connection_error(self) -> None:
        """Verify calling methods on a closed proxy raises RelateConnectionError."""
        proxy = RelateThreadProxy(db_path=":memory:")
        proxy.close()

        with pytest.raises(RelateConnectionError, match="RelateThreadProxy is closed"):
            proxy.execute("RETURN 1")

        with pytest.raises(RelateConnectionError, match="RelateThreadProxy is closed"):
            proxy.execute_read("RETURN 1")

        with pytest.raises(RelateConnectionError, match="RelateThreadProxy is closed"):
            proxy.execute_write("CREATE (t:Thread {thread_id: 'x'})")

        with pytest.raises(RelateConnectionError, match="RelateThreadProxy is closed"):
            proxy.execute_batch([("RETURN 1", {})])

    def test_execution_timeout_raises_relate_concurrency_error(self) -> None:
        """Verify operations exceeding timeout raise RelateConcurrencyError."""
        with RelateThreadProxy(db_path=":memory:") as proxy:
            block_event = threading.Event()

            # Enqueue a blocking callable onto the actor queue
            proxy._queue.put(lambda: block_event.wait(timeout=2.0))

            # Execute with an immediate timeout while worker is blocked
            with pytest.raises(RelateConcurrencyError, match="timed out"):
                proxy.execute("RETURN 1", timeout=0.01)

            block_event.set()


class TestProxyFakeDriverCompatibility:
    """Test suite verifying RelateThreadProxy compatibility with InMemoryRelateDriver."""

    def test_proxy_with_in_memory_fake_driver_ac_08(self) -> None:
        """AC-08: Verify RelateThreadProxy works with InMemoryRelateDriver."""
        fake_driver = InMemoryRelateDriver()
        with RelateThreadProxy(
            db_path=":memory:",
            connection_factory=lambda _: fake_driver.connect(),
            auto_init_schema=True,
        ) as proxy:
            proxy.execute_write("CREATE (t:Thread {thread_id: 'fake-thread'})")
            res = proxy.execute_read(
                "MATCH (t:Thread {thread_id: $tid}) RETURN t.thread_id AS id",
                {"tid": "fake-thread"},
            )
            assert len(res) == 1
            assert res[0]["id"] == "fake-thread"


class _SpyConnection:
    """Wraps a real connection and records commit, flush and close calls in order."""

    def __init__(self, inner: Any, events: list[str], flush_error: Exception | None = None) -> None:
        self._inner = inner
        self._events = events
        self._flush_error = flush_error

    @property
    def autocommit(self) -> bool:
        return bool(self._inner.autocommit)

    @autocommit.setter
    def autocommit(self, value: bool) -> None:
        self._inner.autocommit = value

    def commit(self) -> None:
        self._events.append("commit")
        self._inner.commit()

    def flush(self) -> None:
        self._events.append("flush")
        if self._flush_error is not None:
            raise self._flush_error
        self._inner.flush()

    def close(self) -> None:
        self._events.append("close")
        self._inner.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _spy_proxy(
    events: list[str], *, wal_enabled: bool = True, flush_error: Exception | None = None
) -> RelateThreadProxy:
    def factory(db_path: str) -> _SpyConnection:
        return _SpyConnection(relate.connect(database=db_path), events, flush_error)

    return RelateThreadProxy(
        db_path=":memory:", wal_enabled=wal_enabled, connection_factory=factory
    )


_CREATE: list[tuple[str, dict[str, Any]]] = [("CREATE (:Thread {thread_id: $t})", {"t": "spy"})]
_COUNT = "MATCH (t:Thread {thread_id: 'spy'}) RETURN count(t) AS n"


class TestProxyFlushControl:
    """execute_batch(flush=False) commits without a flush; close() flushes, abort() does not."""

    def test_batch_flushes_by_default_adr_0014(self) -> None:
        """ADR-0014: the default batch commits and then flushes."""
        events: list[str] = []
        proxy = _spy_proxy(events)
        try:
            events.clear()
            proxy.execute_batch(_CREATE)
            assert events == ["commit", "flush"]
        finally:
            proxy.abort()

    def test_batch_without_flush_still_commits_adr_0014(self) -> None:
        """ADR-0014: flush=False commits before returning and skips the flush."""
        events: list[str] = []
        proxy = _spy_proxy(events)
        try:
            events.clear()
            proxy.execute_batch(_CREATE, flush=False)
            assert events == ["commit"]
            assert proxy.execute_read(_COUNT)[0]["n"] == 1
        finally:
            proxy.abort()

    def test_failed_batch_without_flush_rolls_back_adr_0014(self) -> None:
        """ADR-0014: a failing flush=False batch rolls back as one transaction."""
        events: list[str] = []
        proxy = _spy_proxy(events)
        try:
            events.clear()
            with pytest.raises(Exception):  # noqa: B017
                proxy.execute_batch([*_CREATE, ("THIS IS NOT CYPHER", {})], flush=False)
            assert "flush" not in events
            assert proxy.execute_read(_COUNT)[0]["n"] == 0
        finally:
            proxy.abort()

    def test_close_flushes_before_closing_adr_0014(self) -> None:
        """ADR-0014: close() flushes explicitly, then closes the connection."""
        events: list[str] = []
        proxy = _spy_proxy(events)
        proxy.execute_batch(_CREATE, flush=False)
        events.clear()
        proxy.close()
        assert events == ["flush", "close"]

    def test_close_does_not_flush_without_wal_adr_0014(self) -> None:
        """ADR-0014: with wal_enabled=False nothing flushes, including close()."""
        events: list[str] = []
        proxy = _spy_proxy(events, wal_enabled=False)
        proxy.execute_batch(_CREATE)
        events.clear()
        proxy.close()
        assert events == ["close"]

    def test_abort_does_not_flush_adr_0014(self) -> None:
        """ADR-0014: abort() closes the connection without flushing."""
        events: list[str] = []
        proxy = _spy_proxy(events)
        proxy.execute_batch(_CREATE, flush=False)
        events.clear()
        proxy.abort()
        assert events == ["close"]

    def test_close_flush_failure_is_logged_and_connection_closed_adr_0014(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """ADR-0014: a failing flush on close logs a WARNING and the connection still closes."""
        events: list[str] = []
        proxy = _spy_proxy(events, flush_error=RuntimeError("disk full"))
        events.clear()
        with caplog.at_level(logging.WARNING):
            proxy.close()
        assert events == ["flush", "close"]
        assert any(
            r.levelno == logging.WARNING and "disk full" in r.getMessage() for r in caplog.records
        )
