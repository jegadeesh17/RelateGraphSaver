"""Dedicated actor proxy for single-writer thread-affinity and concurrency serialization."""

from __future__ import annotations

import concurrent.futures
import logging
import os
import queue
import re
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import relate

from langgraph_checkpoint_relate.errors import (
    RelateCheckpointError,
    RelateConcurrencyError,
    RelateConnectionError,
    RelateReadOnlyError,
)
from langgraph_checkpoint_relate.schema import initialize_schema

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DbCommand:
    """Cypher execution command message for actor queue."""

    cypher: str
    params: dict[str, Any]
    is_write: bool
    future: concurrent.futures.Future[Any]


@dataclass(frozen=True)
class BatchCommand:
    """Batch transaction execution command message for actor queue."""

    statements: list[tuple[str, dict[str, Any]]]
    future: concurrent.futures.Future[Any]
    flush: bool = True


class _ShutdownSentinel:
    """Sentinel indicator instructing the background worker thread to terminate."""


def _format_record(record: Any) -> dict[str, Any]:
    """Convert a database record or mapping to a standard dictionary."""
    if isinstance(record, dict):
        if hasattr(record, "data") and callable(record.data):
            return dict(record.data())
        return dict(record)
    if hasattr(record, "items"):
        return dict(record.items())
    if hasattr(record, "__dict__"):
        return dict(record.__dict__)
    return {"value": record}


def _format_records(records: Any) -> list[dict[str, Any]]:
    """Format raw fetchall results into a list of dictionaries."""
    if not records:
        return []
    return [_format_record(r) for r in records]


_CONCURRENCY_RE = re.compile(
    r"\b(lock(ed)?|busy|contention|deadlock|collision|timeout|timed out)\b",
    re.IGNORECASE,
)
_CONNECTION_RE = re.compile(r"\b(closed|connection)\b", re.IGNORECASE)


def _wrap_exception(exc: Exception) -> RelateCheckpointError:
    """Map database, driver, or runtime exceptions to appropriate RelateCheckpointError."""
    if isinstance(exc, RelateCheckpointError):
        return exc

    msg = str(exc)

    err: RelateCheckpointError
    if isinstance(exc, (TimeoutError, concurrent.futures.TimeoutError)) or _CONCURRENCY_RE.search(
        msg
    ):
        err = RelateConcurrencyError(f"Concurrency or lock failure: {msg}")
        err.__cause__ = exc
        return err

    if _CONNECTION_RE.search(msg) or exc.__class__.__name__ in (
        "InterfaceError",
        "RelateConnectionError",
    ):
        err = RelateConnectionError(f"Database connection error: {msg}")
        err.__cause__ = exc
        return err

    err = RelateCheckpointError(f"Relate database error: {msg}")
    err.__cause__ = exc
    return err


class RelateThreadProxy:
    """Actor proxy maintaining a dedicated worker thread for thread-affine Relate operations."""

    def __init__(
        self,
        db_path: str = ":memory:",
        wal_enabled: bool = True,
        connection_factory: Callable[..., Any] | None = None,
        auto_init_schema: bool = True,
        default_timeout: float | None = 30.0,
        *,
        read_only: bool = False,
    ) -> None:
        if read_only and db_path == ":memory:":
            raise ValueError(
                "read_only=True requires a file path; ':memory:' has nothing to inspect"
            )
        self.db_path = db_path
        self.read_only = read_only
        self.wal_enabled = wal_enabled
        self.auto_init_schema = auto_init_schema
        self.default_timeout = default_timeout
        self._connection_factory = connection_factory

        self._queue: queue.Queue[Any] = queue.Queue()
        self._writer_lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._is_closed = False
        self._conn: Any = None
        self._worker_thread: threading.Thread | None = None
        self._ready_event = threading.Event()
        self._startup_error: Exception | None = None

        self._start_worker()

    @property
    def writer_lock(self) -> threading.Lock:
        """Exclusive writer lock protecting mutating database operations."""
        return self._writer_lock

    @property
    def lifecycle_lock(self) -> threading.Lock:
        """Lifecycle lock protecting state transitions, active checks, and enqueuing."""
        return self._lifecycle_lock

    def _start_worker(self) -> None:
        """Spawn the dedicated background worker thread and wait for initialization."""
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name=f"RelateThreadProxy-{id(self)}",
            daemon=True,
        )
        self._worker_thread.start()

        ready = self._ready_event.wait(timeout=self.default_timeout)
        if not ready:
            self._is_closed = True
            self._queue.put(_ShutdownSentinel())
            if self._worker_thread is not None and self._worker_thread.is_alive():
                self._worker_thread.join(timeout=2.0)
            raise RelateConnectionError(
                f"RelateThreadProxy worker failed to initialize within {self.default_timeout} seconds"
            )
        if self._startup_error is not None:
            self._is_closed = True
            wrapped = _wrap_exception(self._startup_error)
            if isinstance(wrapped, (RelateConcurrencyError, RelateConnectionError)):
                raise wrapped from self._startup_error
            raise RelateConnectionError(
                f"Failed to initialize Relate database connection: {self._startup_error}"
            ) from self._startup_error

    def _open_connection(self) -> Any:
        """Open database connection on the worker thread."""
        if self.db_path != ":memory:" and not self.read_only:
            parent_dir = os.path.dirname(os.path.abspath(self.db_path))
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)

        if self._connection_factory is not None:
            return self._connection_factory(self.db_path)
        if self.read_only:
            return relate.connect(database=self.db_path, read_only=True)
        return relate.connect(database=self.db_path)

    def _worker_loop(self) -> None:
        """Worker thread loop executing commands strictly on this thread."""
        try:
            self._conn = self._open_connection()
            if self._is_closed:
                self._cleanup_connection()
                return
            if self.auto_init_schema and not self.read_only:
                initialize_schema(self._conn)
            self._ready_event.set()
        except Exception as exc:
            self._startup_error = exc
            self._ready_event.set()
            self._cleanup_connection()
            return

        while True:
            try:
                item = self._queue.get()
                if isinstance(item, _ShutdownSentinel) or item is None:
                    self._queue.task_done()
                    break

                self._process_item(item)
                self._queue.task_done()
            except Exception as loop_exc:
                logger.exception("Unexpected error in RelateThreadProxy worker loop: %s", loop_exc)

        self._cleanup_connection()

    def _process_item(self, item: Any) -> None:
        """Dispatch a queued work item."""
        if callable(item):
            try:
                item()
            except Exception as call_exc:
                logger.debug("Callable work item raised exception: %s", call_exc)
            return
        if isinstance(item, DbCommand):
            if item.future.cancelled():
                return
            if item.is_write and self.read_only:
                item.future.set_exception(self._read_only_error())
                return
            self._handle_db_command(item)
        elif isinstance(item, BatchCommand):
            if item.future.cancelled():
                return
            if self.read_only:
                item.future.set_exception(self._read_only_error())
                return
            self._handle_batch_command(item)
        elif isinstance(item, tuple) and len(item) == 4:
            if self.read_only:
                self._fail_tuple_command(item, self._read_only_error())
                return
            self._handle_tuple_command(item)
        elif isinstance(item, dict):
            self._handle_dict_command(item)
        else:
            logger.warning("Unrecognized message type submitted to actor queue: %r", type(item))

    @staticmethod
    def _read_only_error() -> RelateReadOnlyError:
        return RelateReadOnlyError("RelateThreadProxy is read-only; writes are not allowed")

    @staticmethod
    def _fail_tuple_command(item: tuple[Any, Any, Any, Any], err: RelateCheckpointError) -> None:
        """Fail a (query, params, event, result_holder) item without touching the driver."""
        _, _, event, holder = item
        if isinstance(holder, dict):
            holder["result"] = None
            holder["error"] = err
        elif isinstance(holder, list):
            holder.append(err)
        if hasattr(event, "set"):
            event.set()

    def _handle_db_command(self, cmd: DbCommand) -> None:
        """Execute a single Cypher command."""
        if cmd.future.cancelled():
            return
        try:
            if cmd.is_write:
                with self._writer_lock:
                    cur = (
                        self._conn.execute(cmd.cypher, cmd.params)
                        if cmd.params
                        else self._conn.execute(cmd.cypher)
                    )
                    records = cur.fetchall() if hasattr(cur, "fetchall") else []
                    if hasattr(self._conn, "commit") and not getattr(
                        self._conn, "autocommit", True
                    ):
                        self._conn.commit()
                    if self.wal_enabled and hasattr(self._conn, "flush"):
                        self._conn.flush()
            else:
                cur = (
                    self._conn.execute(cmd.cypher, cmd.params)
                    if cmd.params
                    else self._conn.execute(cmd.cypher)
                )
                records = cur.fetchall() if hasattr(cur, "fetchall") else []

            if not cmd.future.cancelled():
                formatted = _format_records(records)
                cmd.future.set_result(formatted)
        except Exception as exc:
            if hasattr(self._conn, "rollback"):
                try:
                    self._conn.rollback()
                except Exception:
                    pass
            if not cmd.future.cancelled():
                cmd.future.set_exception(_wrap_exception(exc))

    def _handle_batch_command(self, cmd: BatchCommand) -> None:
        """Execute a batch of Cypher commands atomically."""
        if cmd.future.cancelled():
            return
        try:
            with self._writer_lock:
                prev_autocommit = getattr(self._conn, "autocommit", True)
                if hasattr(self._conn, "autocommit"):
                    self._conn.autocommit = False

                results: list[list[dict[str, Any]]] = []
                try:
                    for cypher, params in cmd.statements:
                        cur = (
                            self._conn.execute(cypher, params)
                            if params
                            else self._conn.execute(cypher)
                        )
                        raw = cur.fetchall() if hasattr(cur, "fetchall") else []
                        results.append(_format_records(raw))

                    if hasattr(self._conn, "commit"):
                        self._conn.commit()
                    if cmd.flush and self.wal_enabled and hasattr(self._conn, "flush"):
                        self._conn.flush()
                except Exception:
                    if hasattr(self._conn, "rollback"):
                        try:
                            self._conn.rollback()
                        except Exception:
                            pass
                    raise
                finally:
                    if hasattr(self._conn, "autocommit"):
                        self._conn.autocommit = prev_autocommit

            if not cmd.future.cancelled():
                cmd.future.set_result(results)
        except Exception as exc:
            if not cmd.future.cancelled():
                cmd.future.set_exception(_wrap_exception(exc))

    def _handle_tuple_command(self, item: tuple[Any, Any, Any, Any]) -> None:
        """Handle legacy/internal (query, params, event, result_holder) protocol."""
        query, params, event, result_holder = item
        try:
            with self._writer_lock:
                cur = self._conn.execute(query, params) if params else self._conn.execute(query)
                raw = cur.fetchall() if hasattr(cur, "fetchall") else []
                if hasattr(self._conn, "commit") and not getattr(self._conn, "autocommit", True):
                    self._conn.commit()
                if self.wal_enabled and hasattr(self._conn, "flush"):
                    self._conn.flush()

            formatted = _format_records(raw)
            if isinstance(result_holder, dict):
                result_holder["result"] = formatted
                result_holder["error"] = None
            elif isinstance(result_holder, list):
                result_holder.append(formatted)
        except Exception as exc:
            wrapped = _wrap_exception(exc)
            if isinstance(result_holder, dict):
                result_holder["result"] = None
                result_holder["error"] = wrapped
            elif isinstance(result_holder, list):
                result_holder.append(wrapped)
        finally:
            if hasattr(event, "set"):
                event.set()

    def _handle_dict_command(self, msg: dict[str, Any]) -> None:
        """Handle command dictionary messages."""
        msg_type = msg.get("type")
        future: concurrent.futures.Future[Any] | None = msg.get("future")

        if msg_type == "is_alive":
            if future is not None:
                future.set_result(True)
            return

        if msg_type == "execute":
            db_cmd = DbCommand(
                cypher=msg["cypher"],
                params=msg.get("params") or {},
                is_write=msg.get("is_write", True),
                future=future or concurrent.futures.Future(),
            )
            if db_cmd.is_write and self.read_only:
                db_cmd.future.set_exception(self._read_only_error())
                return
            self._handle_db_command(db_cmd)
            return

        if msg_type == "transaction":
            if self.read_only:
                if future is not None:
                    future.set_exception(self._read_only_error())
                return
            batch_cmd = BatchCommand(
                statements=msg["statements"],
                future=future or concurrent.futures.Future(),
            )
            self._handle_batch_command(batch_cmd)
            return

        if msg_type == "close":
            if future is not None:
                future.set_result(True)
            return

        if future is not None:
            future.set_exception(
                RelateCheckpointError(f"Unsupported command message type: {msg_type}")
            )

    def _cleanup_connection(self) -> None:
        """Close connection and fail any remaining pending commands."""
        if self._conn is not None:
            # Batches committed without their own flush become durable here.
            # A read-only connection never flushes.
            if self.wal_enabled and not self.read_only and hasattr(self._conn, "flush"):
                try:
                    self._conn.flush()
                except Exception as flush_exc:
                    logger.warning("Error flushing Relate connection before close: %s", flush_exc)
            try:
                if hasattr(self._conn, "close"):
                    self._conn.close()
            except Exception as close_exc:
                logger.warning("Error closing Relate connection: %s", close_exc)
            finally:
                self._conn = None

        # Drain any remaining commands in the queue
        while not self._queue.empty():
            try:
                pending = self._queue.get_nowait()
                err = RelateConnectionError(
                    "RelateThreadProxy closed before operation could execute"
                )
                if isinstance(pending, (DbCommand, BatchCommand)):
                    if not pending.future.done():
                        pending.future.set_exception(err)
                elif isinstance(pending, tuple) and len(pending) == 4:
                    _, _, event, holder = pending
                    if isinstance(holder, dict):
                        holder["error"] = err
                    if hasattr(event, "set"):
                        event.set()
                elif isinstance(pending, dict) and "future" in pending:
                    fut = pending["future"]
                    if isinstance(fut, concurrent.futures.Future) and not fut.done():
                        fut.set_exception(err)
                self._queue.task_done()
            except queue.Empty:
                break

    def is_alive(self) -> bool:
        """Return True if the background worker thread is running and proxy is active."""
        return (
            not self._is_closed
            and self._worker_thread is not None
            and self._worker_thread.is_alive()
        )

    def ping(self, timeout: float | None = 5.0) -> bool:
        """Ping the worker thread through the actor queue to verify responsiveness."""
        fut: concurrent.futures.Future[bool] = concurrent.futures.Future()
        with self._lifecycle_lock:
            if not self.is_alive():
                return False
            self._queue.put({"type": "is_alive", "future": fut})
        try:
            return fut.result(timeout=timeout or self.default_timeout)
        except Exception:
            return False

    def execute_read(
        self,
        cypher: str,
        params: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a read-only Cypher query."""
        return self._dispatch_execute(cypher, params, is_write=False, timeout=timeout)

    def execute_write(
        self,
        cypher: str,
        params: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a mutating Cypher query under the exclusive writer lock."""
        self._check_writable()
        return self._dispatch_execute(cypher, params, is_write=True, timeout=timeout)

    def execute(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a Cypher statement (read or write)."""
        self._check_writable()
        return self._dispatch_execute(query, params, is_write=True, timeout=timeout)

    def _dispatch_execute(
        self,
        cypher: str,
        params: dict[str, Any] | None,
        is_write: bool,
        timeout: float | None,
    ) -> list[dict[str, Any]]:
        """Validate input, build command, and await completion."""
        if not isinstance(cypher, str) or not cypher.strip():
            raise RelateCheckpointError("Cypher statement must be a non-empty string")

        if params is not None and not isinstance(params, (dict, Mapping)):
            raise RelateCheckpointError("Parameters must be a dictionary or None")

        bound_params = dict(params) if params is not None else {}
        fut: concurrent.futures.Future[list[dict[str, Any]]] = concurrent.futures.Future()
        cmd = DbCommand(cypher=cypher, params=bound_params, is_write=is_write, future=fut)

        with self._lifecycle_lock:
            self._check_active()
            self._queue.put(cmd)

        effective_timeout = timeout if timeout is not None else self.default_timeout
        try:
            return fut.result(timeout=effective_timeout)
        except (TimeoutError, concurrent.futures.TimeoutError) as to_exc:
            fut.cancel()
            raise RelateConcurrencyError(
                f"Query timed out after {effective_timeout}s: {cypher[:60]}"
            ) from to_exc

    _execute_internal = _dispatch_execute

    def execute_batch(
        self,
        statements: list[tuple[str, dict[str, Any]]],
        timeout: float | None = None,
        *,
        flush: bool = True,
    ) -> list[list[dict[str, Any]]]:
        """Execute a batch of statements atomically within a single transaction.

        With ``flush=False`` the batch still commits before this returns, but the
        per-write flush is skipped; the data is flushed by the next flushed write
        or by ``close()``.
        """
        self._check_writable()

        if isinstance(statements, (str, bytes)) or not isinstance(statements, Sequence):
            raise RelateCheckpointError("Statements must be a sequence of (query, params) tuples")

        validated_statements: list[tuple[str, dict[str, Any]]] = []
        for idx, item in enumerate(statements):
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise RelateCheckpointError(
                    f"Statement at index {idx} must be a (query, params) 2-tuple, got {type(item)}"
                )
            stmt_query, stmt_params = item
            if not isinstance(stmt_query, str) or not stmt_query.strip():
                raise RelateCheckpointError(f"Statement at index {idx} has invalid or empty query")
            if stmt_params is not None and not isinstance(stmt_params, (dict, Mapping)):
                raise RelateCheckpointError(
                    f"Statement at index {idx} parameters must be a dictionary or None"
                )
            validated_statements.append(
                (stmt_query, dict(stmt_params) if stmt_params is not None else {})
            )

        fut: concurrent.futures.Future[list[list[dict[str, Any]]]] = concurrent.futures.Future()
        cmd = BatchCommand(statements=validated_statements, future=fut, flush=flush)

        with self._lifecycle_lock:
            self._check_active()
            if not statements:
                return []
            self._queue.put(cmd)

        effective_timeout = timeout if timeout is not None else self.default_timeout
        try:
            return fut.result(timeout=effective_timeout)
        except (TimeoutError, concurrent.futures.TimeoutError) as to_exc:
            fut.cancel()
            raise RelateConcurrencyError(
                f"Batch transaction timed out after {effective_timeout}s ({len(statements)} statements)"
            ) from to_exc

    def transaction(
        self,
        queries_and_params: list[tuple[str, dict[str, Any]]],
        timeout: float | None = None,
    ) -> list[list[dict[str, Any]]]:
        """Alias for execute_batch providing transaction semantics."""
        return self.execute_batch(queries_and_params, timeout=timeout)

    def _check_writable(self) -> None:
        """Refuse writes on a read-only proxy before anything is queued."""
        if self.read_only:
            raise self._read_only_error()

    def _check_active(self) -> None:
        """Verify the actor proxy is active and worker thread is alive."""
        if self._is_closed:
            raise RelateConnectionError("RelateThreadProxy is closed")
        if self._worker_thread is None or not self._worker_thread.is_alive():
            raise RelateConnectionError("RelateThreadProxy background worker thread is not running")

    def flush(self, timeout: float | None = None) -> None:
        """Explicitly flush pending changes to disk WAL (a no-op when read-only)."""
        fut: concurrent.futures.Future[None] = concurrent.futures.Future()

        def do_flush() -> None:
            if fut.cancelled():
                return
            try:
                with self._writer_lock:
                    if self._conn is not None and hasattr(self._conn, "flush"):
                        self._conn.flush()
                if not fut.cancelled():
                    fut.set_result(None)
            except Exception as exc:
                if not fut.cancelled():
                    fut.set_exception(_wrap_exception(exc))

        with self._lifecycle_lock:
            self._check_active()
            if self.read_only:
                return
            self._queue.put(do_flush)

        effective_timeout = timeout if timeout is not None else self.default_timeout
        try:
            fut.result(timeout=effective_timeout)
        except (TimeoutError, concurrent.futures.TimeoutError) as to_exc:
            fut.cancel()
            raise RelateConcurrencyError(f"Flush timed out after {effective_timeout}s") from to_exc

    def abort(self, timeout: float | None = 2.0) -> None:
        """Abruptly terminate the proxy and connection without graceful flush."""
        with self._lifecycle_lock:
            if self._is_closed:
                return
            self._is_closed = True

        def do_abort() -> None:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None

        try:
            self._queue.put(do_abort)
        except Exception:
            pass

        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._queue.put(_ShutdownSentinel())
            effective_timeout = timeout if timeout is not None else 2.0
            self._worker_thread.join(timeout=effective_timeout)

    def stop(self, timeout: float | None = 5.0) -> None:
        """Alias for close."""
        self.close(timeout=timeout)

    def close(self, timeout: float | None = 5.0) -> None:
        """Cleanly shut down worker actor thread and close Relate connection."""
        with self._lifecycle_lock:
            if self._is_closed:
                return
            self._is_closed = True
            self._queue.put(_ShutdownSentinel())

        if self._worker_thread is not None and self._worker_thread.is_alive():
            effective_timeout = timeout if timeout is not None else 5.0
            self._worker_thread.join(timeout=effective_timeout)

    def __enter__(self) -> RelateThreadProxy:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


__all__ = [
    "BatchCommand",
    "DbCommand",
    "RelateThreadProxy",
]
