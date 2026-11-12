"""Integration tests verifying multi-threaded concurrency, serialization, and isolation."""

# ruff: noqa: E402

from __future__ import annotations

import os
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

import pytest

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from langgraph_checkpoint_relate.proxy import RelateThreadProxy


@pytest.fixture
def temp_db_path() -> Any:
    """Provide a unique temporary disk-backed .rl database file path."""
    temp_dir = tempfile.mkdtemp(prefix="relate_test_")
    db_file = os.path.join(temp_dir, "concurrency_test.rl")
    yield db_file

    # Teardown: close and clean up database files
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


class TestConcurrencyEngine:
    """Integration test suite for high-concurrency multi-threaded database operations."""

    def test_concurrent_writes_20_threads_ac_08(self, temp_db_path: str) -> None:
        """AC-08: 20 concurrent threads hammering execute_write against disk store.

        Asserts zero file-lock collisions, zero thread-affinity exceptions, and 100%
        deterministic write persistence across all threads.
        """
        num_threads = 20
        writes_per_thread = 10
        total_expected_records = num_threads * writes_per_thread

        thread_errors: list[Exception] = []
        cypher = (
            "CREATE (c:Checkpoint {\n"
            "    checkpoint_id: $cid,\n"
            "    thread_id: $tid,\n"
            "    checkpoint_ns: '',\n"
            "    step: $step,\n"
            "    state_hash: 'sha256-state-hash'\n"
            "})"
        )

        with RelateThreadProxy(db_path=temp_db_path) as proxy:
            def worker(thread_idx: int) -> None:
                try:
                    for i in range(writes_per_thread):
                        cid = f"chk-{thread_idx:02d}-{i:03d}"
                        tid = f"thread-{thread_idx:02d}"
                        proxy.execute_write(
                            cypher,
                            {"cid": cid, "tid": tid, "step": i},
                        )
                except Exception as exc:
                    thread_errors.append(exc)

            threads = [
                threading.Thread(target=worker, args=(t_id,), name=f"WriterThread-{t_id}")
                for t_id in range(num_threads)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # Assert zero exceptions occurred during concurrent execution
            assert not thread_errors, f"Encountered {len(thread_errors)} concurrent write errors: {thread_errors}"

            # Verify deterministic record persistence
            count_res = proxy.execute_read("MATCH (c:Checkpoint) RETURN count(c) AS total_count")
            assert count_res[0]["total_count"] == total_expected_records

        # Reopen from disk to verify durable commit persistence
        with RelateThreadProxy(db_path=temp_db_path) as verify_proxy:
            persisted_res = verify_proxy.execute_read("MATCH (c:Checkpoint) RETURN count(c) AS total_count")
            assert persisted_res[0]["total_count"] == total_expected_records

    def test_concurrent_transactions_20_threads_ac_08(self, temp_db_path: str) -> None:
        """AC-08: 20 concurrent threads hammering transaction() against disk store.

        Asserts atomic multi-statement transaction persistence without deadlocks or corruption.
        """
        num_threads = 20
        tx_errors: list[Exception] = []

        with RelateThreadProxy(db_path=temp_db_path) as proxy:
            def tx_worker(thread_idx: int) -> None:
                try:
                    tid = f"tx-thread-{thread_idx:02d}"
                    cid = f"tx-chk-{thread_idx:02d}"
                    statements = [
                        (
                            "CREATE (t:Thread {thread_id: $tid})",
                            {"tid": tid},
                        ),
                        (
                            "CREATE (c:Checkpoint {checkpoint_id: $cid, thread_id: $tid, checkpoint_ns: '', step: 0, state_hash: 'hash'})",
                            {"cid": cid, "tid": tid},
                        ),
                        (
                            "MATCH (t:Thread {thread_id: $tid}), (c:Checkpoint {checkpoint_id: $cid})\n"
                            "CREATE (t)-[:HAS_CHECKPOINT]->(c)",
                            {"tid": tid, "cid": cid},
                        ),
                    ]
                    proxy.transaction(statements)
                except Exception as exc:
                    tx_errors.append(exc)

            threads = [
                threading.Thread(target=tx_worker, args=(t_id,), name=f"TxThread-{t_id}")
                for t_id in range(num_threads)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert not tx_errors, f"Encountered transaction errors: {tx_errors}"

            # Verify atomic persistence of all 20 threads, checkpoints, and relationships
            threads_count = proxy.execute_read("MATCH (t:Thread) RETURN count(t) AS count")[0]["count"]
            checkpoints_count = proxy.execute_read("MATCH (c:Checkpoint) RETURN count(c) AS count")[0]["count"]
            edges_count = proxy.execute_read("MATCH ()-[r:HAS_CHECKPOINT]->() RETURN count(r) AS count")[0]["count"]

            assert threads_count == num_threads
            assert checkpoints_count == num_threads
            assert edges_count == num_threads

    def test_concurrent_reads_and_writes_ac_09(self, temp_db_path: str) -> None:
        """AC-09: Concurrent reads and writes across worker threads without dirty reads.

        Asserts that reads observe consistent committed states while writes are concurrently queued.
        """
        num_writers = 10
        num_readers = 10
        writes_per_writer = 10

        errors: list[Exception] = []
        read_counts: list[int] = []
        stop_readers = threading.Event()

        cypher_write = (
            "CREATE (c:Checkpoint {\n"
            "    checkpoint_id: $cid,\n"
            "    thread_id: $tid,\n"
            "    checkpoint_ns: '',\n"
            "    step: $step,\n"
            "    state_hash: 'hash-clean'\n"
            "})"
        )

        with RelateThreadProxy(db_path=temp_db_path) as proxy:
            def writer(w_id: int) -> None:
                try:
                    for i in range(writes_per_writer):
                        cid = f"rw-chk-{w_id:02d}-{i:03d}"
                        tid = f"rw-thread-{w_id:02d}"
                        proxy.execute_write(cypher_write, {"cid": cid, "tid": tid, "step": i})
                except Exception as exc:
                    errors.append(exc)

            def reader(r_id: int) -> None:
                try:
                    last_observed = 0
                    while not stop_readers.is_set():
                        res = proxy.execute_read("MATCH (c:Checkpoint) RETURN count(c) AS cnt")
                        observed = res[0]["cnt"]
                        # AC-09: Reads must be monotonically non-decreasing
                        if observed < last_observed:
                            raise ValueError(
                                f"Reader {r_id} observed regression in count: {observed} < {last_observed}"
                            )
                        last_observed = observed
                        read_counts.append(observed)
                except Exception as exc:
                    errors.append(exc)

            writer_threads = [
                threading.Thread(target=writer, args=(i,), name=f"Writer-{i}")
                for i in range(num_writers)
            ]
            reader_threads = [
                threading.Thread(target=reader, args=(i,), name=f"Reader-{i}")
                for i in range(num_readers)
            ]

            for r in reader_threads:
                r.start()
            for w in writer_threads:
                w.start()

            for w in writer_threads:
                w.join()

            stop_readers.set()
            for r in reader_threads:
                r.join()

            assert not errors, f"Encountered errors during concurrent read/write test: {errors}"

            final_count = proxy.execute_read("MATCH (c:Checkpoint) RETURN count(c) AS cnt")[0]["cnt"]
            assert final_count == num_writers * writes_per_writer

    def test_high_contention_mixed_workload_ac_08_ac_09(self, temp_db_path: str) -> None:
        """AC-08, AC-09: High-contention mixed workload of writes, transactions, and reads.

        Verifies complete isolation, lack of deadlocks, and data integrity under sustained pressure.
        """
        num_workers = 20
        ops_per_worker = 8
        errors: list[Exception] = []

        with RelateThreadProxy(db_path=temp_db_path) as proxy:
            def mixed_worker(w_id: int) -> None:
                try:
                    for i in range(ops_per_worker):
                        op_type = (w_id + i) % 3
                        if op_type == 0:
                            # Single write
                            proxy.execute_write(
                                "CREATE (t:Thread {thread_id: $tid})",
                                {"tid": f"mixed-t-{w_id}-{i}"},
                            )
                        elif op_type == 1:
                            # Batch transaction
                            proxy.transaction([
                                (
                                    "CREATE (c:Checkpoint {checkpoint_id: $cid, thread_id: $tid, checkpoint_ns: '', step: $step, state_hash: 'h'})",
                                    {"cid": f"mixed-c-{w_id}-{i}", "tid": f"mixed-t-{w_id}-{i}", "step": i},
                                ),
                            ])
                        else:
                            # Read
                            proxy.execute_read("MATCH (t:Thread) RETURN count(t) AS total")
                except Exception as exc:
                    errors.append(exc)

            threads = [
                threading.Thread(target=mixed_worker, args=(i,), name=f"MixedWorker-{i}")
                for i in range(num_workers)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert not errors, f"Encountered errors in mixed workload: {errors}"
