"""Integration tests verifying RelateThreadProxy worker cancellation and timeout behavior (AC-179)."""

# ruff: noqa: E402

from __future__ import annotations

import concurrent.futures
import os
import sys
import tempfile
import threading
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from langgraph_checkpoint_relate.errors import RelateConcurrencyError
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
            res = proxy.execute_read(
                "MATCH (n:BatchCancelled) RETURN count(n) AS cnt"
            )
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
            res = proxy.execute_read(
                "MATCH (n:PreCancelled {id: 'pre-1'}) RETURN count(n) AS cnt"
            )
            assert res == [{"cnt": 0}]

    def test_pre_cancelled_batch_command_skipped_by_worker_ac_179(
        self, temp_db_path: str
    ) -> None:
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
