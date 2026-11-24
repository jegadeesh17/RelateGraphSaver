"""Integration tests verifying RelateThreadProxy worker cancellation and timeout behavior (AC-179)."""

# ruff: noqa: E402

from __future__ import annotations

import concurrent.futures
import os
import sys
import tempfile
import threading
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from langgraph_checkpoint_relate.errors import (
    RelateCheckpointError,
    RelateConcurrencyError,
    RelateConnectionError,
)
from langgraph_checkpoint_relate.proxy import BatchCommand, DbCommand, RelateThreadProxy


@pytest.fixture
def temp_db_path() -> Generator[str, None, None]:
    """Provide a unique temporary disk-backed .rl database file path."""
    temp_dir = tempfile.mkdtemp(prefix="relate_proxy_test_")
    db_file = os.path.join(temp_dir, "proxy_test.rl")
    yield db_file

    for root, _, files in os.walk(temp_dir, topdown=False):
        for f in files:
            try:
                os.remove(os.path.join(root, f))
            except OSError:
                pass
    try:
        os.rmdir(temp_dir)
    except OSError:
        pass


class TestProxyCancellationOnTimeout:
    """AC-179: On caller timeout, cancel future and skip execution in worker thread."""

    def test_write_timeout_cancels_future_and_skips_execution_ac_179(
        self, temp_db_path: str
    ) -> None:
        """AC-179: When write times out, future is cancelled and worker skips execution."""
        with RelateThreadProxy(db_path=temp_db_path) as proxy:
            block_event = threading.Event()

            # Enqueue a blocking callable to temporarily stall the worker thread
            proxy._queue.put(lambda: block_event.wait(timeout=5.0))

            # Trigger a mutating write command with a short caller timeout
            with pytest.raises(RelateConcurrencyError, match="Query timed out"):
                proxy.execute_write(
                    "CREATE (n:CancelledNode {node_id: 'cancel-1'})",
                    timeout=0.05,
                )

            # Inspect the queued command while worker is still blocked: future must be cancelled
            queued_items = list(proxy._queue.queue)
            db_commands = [item for item in queued_items if isinstance(item, DbCommand)]
            assert len(db_commands) == 1, "Expected queued DbCommand awaiting worker"
            assert db_commands[0].future.cancelled() is True, (
                "Future must be marked cancelled after caller timeout"
            )

            # Unblock the worker so it dequeues the cancelled command
            block_event.set()

            # Issue a subsequent read query (deterministic synchronization via FIFO actor queue)
            res = proxy.execute_read(
                "MATCH (n:CancelledNode {node_id: 'cancel-1'}) RETURN count(n) AS cnt"
            )
            assert res == [{"cnt": 0}], (
                "Mutation must NOT have executed or persisted to the database"
            )

    def test_batch_write_timeout_cancels_future_and_skips_execution_ac_179(
        self, temp_db_path: str
    ) -> None:
        """AC-179: When batch write times out, future is cancelled and worker skips execution."""
        with RelateThreadProxy(db_path=temp_db_path) as proxy:
            block_event = threading.Event()

            # Enqueue a blocking callable to temporarily stall the worker thread
            proxy._queue.put(lambda: block_event.wait(timeout=5.0))

            # Trigger a batch write transaction with a short caller timeout
            statements: list[tuple[str, dict[str, Any]]] = [
                ("CREATE (n:BatchCancelled {id: 'b-1'})", {}),
                ("CREATE (n:BatchCancelled {id: 'b-2'})", {}),
            ]
            with pytest.raises(RelateConcurrencyError, match="Batch transaction timed out"):
                proxy.execute_batch(statements, timeout=0.05)

            # Inspect queued batch command while worker is still blocked
            queued_items = list(proxy._queue.queue)
            batch_commands = [item for item in queued_items if isinstance(item, BatchCommand)]
            assert len(batch_commands) == 1, "Expected queued BatchCommand awaiting worker"
            assert batch_commands[0].future.cancelled() is True, (
                "Batch future must be marked cancelled after caller timeout"
            )

            # Unblock worker to dequeue and skip the cancelled batch command
            block_event.set()

            # Verify no nodes from the cancelled transaction were created or committed
            res = proxy.execute_read("MATCH (n:BatchCancelled) RETURN count(n) AS cnt")
            assert res == [{"cnt": 0}], (
                "Batch mutation must NOT have executed or persisted to the database"
            )

    def test_read_timeout_cancels_future_ac_179(self, temp_db_path: str) -> None:
        """AC-179: When read times out, future is marked cancelled."""
        with RelateThreadProxy(db_path=temp_db_path) as proxy:
            block_event = threading.Event()

            proxy._queue.put(lambda: block_event.wait(timeout=5.0))

            with pytest.raises(RelateConcurrencyError, match="Query timed out"):
                proxy.execute_read("RETURN 1 AS num", timeout=0.05)

            queued_items = list(proxy._queue.queue)
            db_commands = [item for item in queued_items if isinstance(item, DbCommand)]
            assert len(db_commands) == 1
            assert db_commands[0].future.cancelled() is True

            block_event.set()

            # Confirm proxy resumes normal query execution cleanly
            res = proxy.execute_read("RETURN 42 AS answer")
            assert res == [{"answer": 42}]

    def test_pre_cancelled_db_command_skipped_by_worker_ac_179(self, temp_db_path: str) -> None:
        """AC-179: A DbCommand cancelled prior to processing is skipped without execution."""
        with RelateThreadProxy(db_path=temp_db_path) as proxy:
            fut: concurrent.futures.Future[list[dict[str, Any]]] = concurrent.futures.Future()
            fut.cancel()

            cmd = DbCommand(
                cypher="CREATE (n:PreCancelled {id: 'pre-1'})",
                params={},
                is_write=True,
                future=fut,
            )
            proxy._queue.put(cmd)

            # Sync by running another read query
            res = proxy.execute_read("MATCH (n:PreCancelled {id: 'pre-1'}) RETURN count(n) AS cnt")
            assert res == [{"cnt": 0}]

    def test_pre_cancelled_batch_command_skipped_by_worker_ac_179(self, temp_db_path: str) -> None:
        """AC-179: A BatchCommand cancelled prior to processing is skipped without execution."""
        with RelateThreadProxy(db_path=temp_db_path) as proxy:
            fut: concurrent.futures.Future[list[list[dict[str, Any]]]] = concurrent.futures.Future()
            fut.cancel()

            cmd = BatchCommand(
                statements=[("CREATE (n:PreCancelledBatch {id: 'pre-b-1'})", {})],
                future=fut,
                flush=True,
            )
            proxy._queue.put(cmd)

            # Sync by running another read query
            res = proxy.execute_read(
                "MATCH (n:PreCancelledBatch {id: 'pre-b-1'}) RETURN count(n) AS cnt"
            )
            assert res == [{"cnt": 0}]


class TestProxyLifecycleAndWorkerTimeout:
    """Verifies proxy worker initialization timeouts and atomic lifecycle coordination."""

    def test_startup_timeout_signals_worker_exit_and_closes_connection(self) -> None:
        """On worker startup timeout, proxy signals worker thread to stop and closes connection handle."""

        class MockRelateConn:
            def __init__(self) -> None:
                self.closed = False

            def close(self) -> None:
                self.closed = True

        mock_conn = MockRelateConn()
        worker_threads: list[threading.Thread] = []

        def slow_factory(_path: str) -> Any:
            worker_threads.append(threading.current_thread())
            time.sleep(0.1)
            return mock_conn

        with pytest.raises(RelateConnectionError, match="failed to initialize within"):
            RelateThreadProxy(
                db_path=":memory:",
                connection_factory=slow_factory,
                default_timeout=0.03,
                auto_init_schema=False,
            )

        assert len(worker_threads) == 1
        worker_thread = worker_threads[0]
        # Worker thread must have joined and exited cleanly
        assert not worker_thread.is_alive()
        # Connection handle must have been closed
        assert mock_conn.closed is True

    def test_lifecycle_lock_prevents_race_conditions_with_close(self, temp_db_path: str) -> None:
        """Unified lifecycle lock coordinates active check, command enqueue, and close atomically."""
        proxy = RelateThreadProxy(db_path=temp_db_path)
        assert proxy.is_alive() is True

        # Pre-seed a test record
        proxy.execute_write("CREATE (n:RaceTest {id: 'init'})")

        # Hold lifecycle lock from an external thread, simulating an active command enqueue
        lock_held = threading.Event()
        release_lock = threading.Event()

        def hold_lock() -> None:
            with proxy.lifecycle_lock:
                lock_held.set()
                release_lock.wait(timeout=2.0)

        t = threading.Thread(target=hold_lock)
        t.start()
        assert lock_held.wait(timeout=1.0) is True

        # While lifecycle lock is held, proxy.close() must block
        close_finished = threading.Event()

        def do_close() -> None:
            proxy.close()
            close_finished.set()

        close_thread = threading.Thread(target=do_close)
        close_thread.start()

        # Verify close has not completed while lock is held
        assert close_finished.wait(timeout=0.05) is False

        # Release lock; close() should proceed cleanly
        release_lock.set()
        t.join(timeout=1.0)
        close_thread.join(timeout=1.0)
        assert close_finished.is_set() is True

        # Post-close, active checks immediately reject all operations atomically
        with pytest.raises(RelateConnectionError, match="is closed"):
            proxy.execute_read("RETURN 1")

        with pytest.raises(RelateConnectionError, match="is closed"):
            proxy.execute_write("CREATE (n:RaceTest {id: 'post'})")

        with pytest.raises(RelateConnectionError, match="is closed"):
            proxy.execute_batch([("CREATE (n:RaceTest {id: 'batch'})", {})])

        with pytest.raises(RelateConnectionError, match="is closed"):
            proxy.execute_batch([])

        with pytest.raises(RelateConnectionError, match="is closed"):
            proxy.flush()

        assert proxy.ping() is False
        assert proxy.is_alive() is False
        assert proxy._queue.empty() is True

    def test_concurrent_execution_and_close_no_races_or_deadlocks(self, temp_db_path: str) -> None:
        """High-concurrency query execution racing with close() completes cleanly without hangs."""
        proxy = RelateThreadProxy(db_path=temp_db_path)
        proxy.execute_write("CREATE (n:ConcurrentItem {count: 0})")

        stop_signal = threading.Event()
        errors: list[Exception] = []
        operations_count = 0
        ops_lock = threading.Lock()

        def worker(worker_id: int) -> None:
            nonlocal operations_count
            while not stop_signal.is_set():
                try:
                    op_type = worker_id % 4
                    if op_type == 0:
                        proxy.execute_read("MATCH (n:ConcurrentItem) RETURN count(n) AS cnt")
                    elif op_type == 1:
                        proxy.execute_write(
                            "CREATE (n:ConcurrentItem {id: $id})",
                            {"id": f"item-{worker_id}-{time.perf_counter()}"},
                        )
                    elif op_type == 2:
                        proxy.execute_batch(
                            [
                                ("CREATE (n:ConcurrentBatch {worker: $w})", {"w": worker_id}),
                            ]
                        )
                    else:
                        proxy.flush()
                    with ops_lock:
                        operations_count += 1
                except RelateConnectionError as conn_err:
                    # Expected once proxy is closed
                    assert "is closed" in str(conn_err) or "not running" in str(conn_err)
                    break
                except Exception as exc:
                    errors.append(exc)
                    break

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for th in threads:
            th.start()

        # Let workers run for a brief moment, then close the proxy concurrently
        time.sleep(0.05)
        proxy.close(timeout=3.0)
        stop_signal.set()

        for th in threads:
            th.join(timeout=3.0)
            assert not th.is_alive(), "Worker thread hung during concurrent execution and close"

        assert len(errors) == 0, f"Unexpected exceptions during concurrent close: {errors}"
        assert proxy.is_alive() is False
        assert proxy._is_closed is True


class TestProxyExceptionClassification:
    """Verifies proxy exception mapping classifies concurrency and connection errors accurately."""

    def test_proxy_avoids_false_positive_concurrency_on_substrings(self) -> None:
        """Verify queries failing with 'unlocked', 'clock', 'block', or 'timeout_ms' do not raise RelateConcurrencyError."""

        class MockFailingConn:
            def __init__(self) -> None:
                self.error: Exception | None = None

            def execute(self, _cypher: str, _params: Any = None) -> Any:
                if self.error:
                    raise self.error
                return []

            def close(self) -> None:
                pass

        mock_conn = MockFailingConn()
        with RelateThreadProxy(
            db_path=":memory:",
            connection_factory=lambda _p: mock_conn,
            auto_init_schema=False,
        ) as proxy:
            for false_match in [
                "node state is unlocked",
                "system clock drift detected",
                "syntax error in code block",
                "invalid configuration parameter timeout_ms",
            ]:
                mock_conn.error = RuntimeError(false_match)
                with pytest.raises(RelateCheckpointError) as exc_info:
                    proxy.execute_read("RETURN 1")
                assert not isinstance(exc_info.value, RelateConcurrencyError)
                assert false_match in str(exc_info.value)

    def test_proxy_classifies_concurrency_and_connection_errors(self) -> None:
        """Verify queries failing with lock, timeout, or closed connection are classified accurately."""

        class MockFailingConn:
            def __init__(self) -> None:
                self.error: Exception | None = None

            def execute(self, _cypher: str, _params: Any = None) -> Any:
                if self.error:
                    raise self.error
                return []

            def close(self) -> None:
                pass

        mock_conn = MockFailingConn()
        with RelateThreadProxy(
            db_path=":memory:",
            connection_factory=lambda _p: mock_conn,
            auto_init_schema=False,
        ) as proxy:
            # Concurrency errors
            for conc_msg in [
                "database is locked",
                "write lock contention",
                "query timed out",
            ]:
                mock_conn.error = RuntimeError(conc_msg)
                with pytest.raises(RelateConcurrencyError) as exc_info:
                    proxy.execute_write("CREATE (n:Test)")
                assert conc_msg in str(exc_info.value)

            # Connection errors
            mock_conn.error = RuntimeError("connection closed")
            with pytest.raises(RelateConnectionError) as conn_info:
                proxy.execute_read("RETURN 1")
            assert "connection closed" in str(conn_info.value)
