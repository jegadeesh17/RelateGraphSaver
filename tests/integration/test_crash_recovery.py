"""Integration tests verifying Write-Ahead Log durability and crash recovery."""

# ruff: noqa: E402

from __future__ import annotations

import multiprocessing
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

from langgraph_checkpoint_relate.errors import RelateCheckpointError
from langgraph_checkpoint_relate.proxy import RelateThreadProxy
from langgraph_checkpoint_relate.saver import RelateSaver


@pytest.fixture
def temp_disk_db() -> Generator[str]:
    """Provide a temporary disk-backed .rl database file path and clean up on teardown."""
    temp_dir = tempfile.mkdtemp(prefix="relate_crash_test_")
    db_file = os.path.join(temp_dir, "crash_durability.rl")
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


def _worker_multistep_workflow_and_crash(db_path: str, thread_id: str, exit_code: int = 42) -> None:
    """Subprocess worker that executes a multi-step LangGraph workflow and crashes abruptly."""
    saver = RelateSaver(db_path=db_path, wal_enabled=True)
    base_config: Any = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}

    # Step 0: Initial user input
    chk_0: Any = {
        "v": 1,
        "id": "chk-crash-000",
        "ts": "2026-11-12T10:00:00Z",
        "channel_values": {
            "messages": ["User query: Analyze portfolio"],
            "agent_state": "ready",
            "step_count": 0,
        },
        "channel_versions": {
            "messages": "00000001",
            "agent_state": "00000001",
            "step_count": "00000001",
        },
        "versions_seen": {},
        "pending_sends": [],
    }
    saver.put(base_config, chk_0, {"step": 0, "node": "input"}, {})  # type: ignore[typeddict-unknown-key]

    # Step 1: Agent planning with channel writes
    cfg_step1: Any = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": "",
            "checkpoint_id": "chk-crash-000",
        }
    }
    saver.put_writes(
        cfg_step1,
        [("plan_notes", "Risk profile is moderate. Querying market data.")],
        task_id="task-planner-1",
    )
    chk_1: Any = {
        "v": 1,
        "id": "chk-crash-001",
        "ts": "2026-11-12T10:00:05Z",
        "parent_checkpoint_id": "chk-crash-000",
        "channel_values": {
            "messages": ["User query: Analyze portfolio", "Plan: Search allocations"],
            "agent_state": "planning",
            "step_count": 1,
        },
        "channel_versions": {
            "messages": "00000002",
            "agent_state": "00000002",
            "step_count": "00000002",
        },
        "versions_seen": {},
        "pending_sends": [],
    }
    saver.put(cfg_step1, chk_1, {"step": 1, "node": "planner"}, {})  # type: ignore[typeddict-unknown-key]

    # Step 2: Tool execution with structured writes
    cfg_step2: Any = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": "",
            "checkpoint_id": "chk-crash-001",
        }
    }
    saver.put_writes(
        cfg_step2,
        [("tool_payload", {"allocations": [40, 40, 20], "variance": 0.05})],
        task_id="task-tool-2",
    )
    chk_2: Any = {
        "v": 1,
        "id": "chk-crash-002",
        "ts": "2026-11-12T10:00:10Z",
        "parent_checkpoint_id": "chk-crash-001",
        "channel_values": {
            "messages": [
                "User query: Analyze portfolio",
                "Plan: Search allocations",
                "Tool result: Allocations returned",
            ],
            "agent_state": "executing",
            "step_count": 2,
        },
        "channel_versions": {
            "messages": "00000003",
            "agent_state": "00000003",
            "step_count": "00000003",
        },
        "versions_seen": {},
        "pending_sends": [],
    }
    saver.put(cfg_step2, chk_2, {"step": 2, "node": "tools"}, {})  # type: ignore[typeddict-unknown-key]

    # Step 3: Synthesis / completion
    cfg_step3: Any = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": "",
            "checkpoint_id": "chk-crash-002",
        }
    }
    chk_3: Any = {
        "v": 1,
        "id": "chk-crash-003",
        "ts": "2026-11-12T10:00:15Z",
        "parent_checkpoint_id": "chk-crash-002",
        "channel_values": {
            "messages": [
                "User query: Analyze portfolio",
                "Plan: Search allocations",
                "Tool result: Allocations returned",
                "Final synthesis: Portfolio balanced at 40/40/20.",
            ],
            "agent_state": "completed",
            "step_count": 3,
        },
        "channel_versions": {
            "messages": "00000004",
            "agent_state": "00000004",
            "step_count": "00000004",
        },
        "versions_seen": {},
        "pending_sends": [],
    }
    saver.put(cfg_step3, chk_3, {"step": 3, "node": "synthesizer"}, {})  # type: ignore[typeddict-unknown-key]

    # Abrupt process termination: do not call close() or flush()
    os._exit(exit_code)


def _worker_committed_then_uncommitted_crash(
    db_path: str,
    thread_id: str,
    uncommitted_cid: str,
    exit_code: int = 99,
) -> None:
    """Subprocess worker that commits valid checkpoints then crashes with uncommitted write in flight."""
    saver = RelateSaver(db_path=db_path, wal_enabled=True)
    base_config: Any = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}

    # Commit step 0 cleanly
    chk_0: Any = {
        "v": 1,
        "id": "chk-valid-000",
        "ts": "2026-11-12T11:00:00Z",
        "channel_values": {"status": "valid_step_0"},
        "channel_versions": {"status": "00000001"},
        "versions_seen": {},
        "pending_sends": [],
    }
    saver.put(base_config, chk_0, {"step": 0}, {})

    # Commit step 1 cleanly
    cfg_step1: Any = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": "",
            "checkpoint_id": "chk-valid-000",
        }
    }
    chk_1: Any = {
        "v": 1,
        "id": "chk-valid-001",
        "ts": "2026-11-12T11:00:05Z",
        "parent_checkpoint_id": "chk-valid-000",
        "channel_values": {"status": "valid_step_1"},
        "channel_versions": {"status": "00000002"},
        "versions_seen": {},
        "pending_sends": [],
    }
    saver.put(cfg_step1, chk_1, {"step": 1}, {})

    # Queue an uncommitted mutation directly onto the connection without commit or flush
    def execute_uncommitted_mutation() -> None:
        conn = saver.proxy._conn
        conn.autocommit = False
        conn.execute(
            "CREATE (c:Checkpoint {checkpoint_id: $cid, thread_id: $tid, checkpoint_ns: '', step: 999})",
            {"cid": uncommitted_cid, "tid": thread_id},
        )
        # Intentionally no commit() and no flush()

    saver.proxy._queue.put(execute_uncommitted_mutation)
    time.sleep(0.3)

    # Abrupt exit mid-transaction
    os._exit(exit_code)


def _worker_concurrent_threads_and_crash(
    db_path: str,
    num_threads: int,
    checkpoints_per_thread: int,
    exit_code: int = 55,
) -> None:
    """Subprocess worker that executes concurrent checkpoint writes across threads and crashes."""
    saver = RelateSaver(db_path=db_path, wal_enabled=True)
    errors: list[Exception] = []

    def thread_task(t_idx: int) -> None:
        try:
            tid = f"mp-thread-{t_idx:02d}"
            last_cid: str | None = None
            for step in range(checkpoints_per_thread):
                cid = f"chk-{t_idx:02d}-{step:03d}"
                cfg: Any = {
                    "configurable": {
                        "thread_id": tid,
                        "checkpoint_ns": "",
                        "checkpoint_id": last_cid,
                    }
                }
                chk: Any = {
                    "v": 1,
                    "id": cid,
                    "ts": f"2026-11-12T12:00:{step:02d}Z",
                    "parent_checkpoint_id": last_cid,
                    "channel_values": {"thread": tid, "step": step},
                    "channel_versions": {"thread": f"{step:08d}"},
                    "versions_seen": {},
                    "pending_sends": [],
                }
                saver.put(cfg, chk, {"step": step}, {})
                last_cid = cid
        except Exception as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=thread_task, args=(i,), name=f"WorkerThread-{i}")
        for i in range(num_threads)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if errors:
        os._exit(1)

    os._exit(exit_code)


class TestCrashRecovery:
    """Integration test suite verifying Write-Ahead Log durability and crash recovery."""

    @pytest.mark.slow
    def test_multistep_workflow_crash_recovery_ac_10_ac_11(self, temp_disk_db: str) -> None:
        """AC-10, AC-11: Abrupt process termination mid-workflow recovers state and lineage.

        Verifies that after an ungraceful shutdown (SIGKILL / os._exit):
        1. A newly spawned RelateSaver instance pointing to the same .rl store recovers all committed checkpoints.
        2. get_tuple() retrieves the exact last committed state with byte-for-byte fidelity.
        3. Pending channel writes and metadata blobs remain intact.
        4. Causal [:NEXT] edges and parent_checkpoint_id lineage are completely preserved.
        """
        thread_id = "thread-crash-recovery-001"
        crash_code = 42

        # Spawn subprocess to execute multi-step workflow and terminate abruptly
        proc = multiprocessing.Process(
            target=_worker_multistep_workflow_and_crash,
            args=(temp_disk_db, thread_id, crash_code),
        )
        proc.start()
        proc.join(timeout=30.0)

        assert proc.exitcode == crash_code, (
            f"Expected crash exitcode {crash_code}, got {proc.exitcode}"
        )

        # Reopen brand-new checkpointer instance on the exact same disk file
        with RelateSaver(db_path=temp_disk_db, wal_enabled=True) as recovered_saver:
            config: Any = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}

            # Verify AC-11: get_tuple retrieves exact last committed checkpoint (step 3)
            latest_tuple = recovered_saver.get_tuple(config)
            assert latest_tuple is not None, (
                "Failed to retrieve latest checkpoint from recovered store"
            )

            assert latest_tuple.checkpoint["id"] == "chk-crash-003"
            assert latest_tuple.metadata["step"] == 3
            assert latest_tuple.metadata["node"] == "synthesizer"  # type: ignore[typeddict-item]

            expected_messages = [
                "User query: Analyze portfolio",
                "Plan: Search allocations",
                "Tool result: Allocations returned",
                "Final synthesis: Portfolio balanced at 40/40/20.",
            ]
            assert latest_tuple.checkpoint["channel_values"]["messages"] == expected_messages
            assert latest_tuple.checkpoint["channel_values"]["agent_state"] == "completed"
            assert latest_tuple.checkpoint["channel_values"]["step_count"] == 3

            # Verify pending writes for intermediate steps
            cfg_chk0: Any = {
                "configurable": {"thread_id": thread_id, "checkpoint_id": "chk-crash-000"}
            }
            chk0_tuple: Any = recovered_saver.get_tuple(cfg_chk0)
            assert chk0_tuple is not None
            assert len(chk0_tuple.pending_writes) == 1
            assert chk0_tuple.pending_writes[0][0] == "task-planner-1"
            assert chk0_tuple.pending_writes[0][1] == "plan_notes"
            assert (
                chk0_tuple.pending_writes[0][2] == "Risk profile is moderate. Querying market data."
            )

            cfg_chk1: Any = {
                "configurable": {"thread_id": thread_id, "checkpoint_id": "chk-crash-001"}
            }
            chk1_tuple: Any = recovered_saver.get_tuple(cfg_chk1)
            assert chk1_tuple is not None
            assert len(chk1_tuple.pending_writes) == 1
            assert chk1_tuple.pending_writes[0][0] == "task-tool-2"
            assert chk1_tuple.pending_writes[0][1] == "tool_payload"
            assert chk1_tuple.pending_writes[0][2] == {
                "allocations": [40, 40, 20],
                "variance": 0.05,
            }

            # Verify causal [:NEXT] lineage graph structure
            query_lineage = (
                "MATCH (c:Checkpoint {thread_id: $tid}) "
                "OPTIONAL MATCH (c)-[:NEXT]->(n:Checkpoint) "
                "RETURN c.checkpoint_id AS cid, c.step AS step, n.checkpoint_id AS next_cid "
                "ORDER BY c.step ASC"
            )
            lineage_rows = recovered_saver.proxy.execute_read(query_lineage, {"tid": thread_id})
            assert len(lineage_rows) == 4

            assert lineage_rows[0]["cid"] == "chk-crash-000"
            assert lineage_rows[0]["next_cid"] == "chk-crash-001"
            assert lineage_rows[1]["cid"] == "chk-crash-001"
            assert lineage_rows[1]["next_cid"] == "chk-crash-002"
            assert lineage_rows[2]["cid"] == "chk-crash-002"
            assert lineage_rows[2]["next_cid"] == "chk-crash-003"
            assert lineage_rows[3]["cid"] == "chk-crash-003"
            assert lineage_rows[3]["next_cid"] is None

            # Verify list() yields all 4 checkpoints in reverse-chronological order
            all_tuples = list(recovered_saver.list(config))
            assert len(all_tuples) == 4
            assert [t.metadata["step"] for t in all_tuples] == [3, 2, 1, 0]

    def test_seamless_continuation_after_crash_recovery_ac_11(self, temp_disk_db: str) -> None:
        """AC-11: Checkpointing resumes seamlessly on recovered thread without schema or constraint violations."""
        thread_id = "thread-continuation-002"
        crash_code = 42

        # 1. Execute initial workflow in crashed subprocess
        proc = multiprocessing.Process(
            target=_worker_multistep_workflow_and_crash,
            args=(temp_disk_db, thread_id, crash_code),
        )
        proc.start()
        proc.join(timeout=30.0)
        assert proc.exitcode == crash_code

        # 2. Reopen and append new steps to the recovered thread
        with RelateSaver(db_path=temp_disk_db, wal_enabled=True) as saver:
            config_latest: Any = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
            current_tuple = saver.get_tuple(config_latest)
            assert current_tuple is not None
            assert current_tuple.checkpoint["id"] == "chk-crash-003"

            # Append Step 4 (human feedback)
            cfg_step4: Any = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": "",
                    "checkpoint_id": "chk-crash-003",
                }
            }
            saver.put_writes(
                cfg_step4,
                [("feedback", "Looks great, please re-run next quarter.")],
                task_id="task-human-feedback-4",
            )
            chk_4: Any = {
                "v": 1,
                "id": "chk-resumed-004",
                "ts": "2026-11-12T10:01:00Z",
                "parent_checkpoint_id": "chk-crash-003",
                "channel_values": {
                    "messages": current_tuple.checkpoint["channel_values"]["messages"]
                    + ["User feedback: Looks great, please re-run next quarter."],
                    "agent_state": "feedback_received",
                    "step_count": 4,
                },
                "channel_versions": {
                    "messages": "00000005",
                    "agent_state": "00000005",
                    "step_count": "00000005",
                },
                "versions_seen": {},
                "pending_sends": [],
            }
            saver.put(cfg_step4, chk_4, {"step": 4, "node": "human_input"}, {})  # type: ignore[typeddict-unknown-key]

            # Append Step 5 (acknowledgement)
            cfg_step5: Any = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": "",
                    "checkpoint_id": "chk-resumed-004",
                }
            }
            chk_5: Any = {
                "v": 1,
                "id": "chk-resumed-005",
                "ts": "2026-11-12T10:01:05Z",
                "parent_checkpoint_id": "chk-resumed-004",
                "channel_values": {
                    "messages": chk_4["channel_values"]["messages"]
                    + ["Agent: Scheduled quarterly review."],
                    "agent_state": "finalized",
                    "step_count": 5,
                },
                "channel_versions": {
                    "messages": "00000006",
                    "agent_state": "00000006",
                    "step_count": "00000006",
                },
                "versions_seen": {},
                "pending_sends": [],
            }
            saver.put(cfg_step5, chk_5, {"step": 5, "node": "final_agent"}, {})  # type: ignore[typeddict-unknown-key]

            # 3. Verify step 5 is now the active head
            head_tuple = saver.get_tuple(config_latest)
            assert head_tuple is not None
            assert head_tuple.checkpoint["id"] == "chk-resumed-005"
            assert head_tuple.metadata["step"] == 5

            # 4. Verify continuous unbroken lineage from step 0 to step 5
            all_steps = list(saver.list(config_latest))
            assert len(all_steps) == 6
            assert [t.metadata["step"] for t in all_steps] == [5, 4, 3, 2, 1, 0]

    def test_uncommitted_transaction_rollback_on_crash_ac_10(self, temp_disk_db: str) -> None:
        """AC-10: WAL recovery rolls back uncommitted transaction interrupted by abrupt termination."""
        thread_id = "thread-uncommitted-003"
        uncommitted_cid = "chk-ghost-uncommitted-999"
        crash_code = 99

        proc = multiprocessing.Process(
            target=_worker_committed_then_uncommitted_crash,
            args=(temp_disk_db, thread_id, uncommitted_cid, crash_code),
        )
        proc.start()
        proc.join(timeout=30.0)
        assert proc.exitcode == crash_code

        # Reopen with RelateSaver to inspect WAL recovery result
        with RelateSaver(db_path=temp_disk_db, wal_enabled=True) as saver:
            config: Any = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}

            # Verify committed checkpoints are intact
            latest = saver.get_tuple(config)
            assert latest is not None
            assert latest.checkpoint["id"] == "chk-valid-001"
            assert latest.checkpoint["channel_values"]["status"] == "valid_step_1"

            # Verify uncommitted checkpoint was rolled back and does not exist
            uncommitted_check = saver.proxy.execute_read(
                "MATCH (c:Checkpoint {checkpoint_id: $cid}) RETURN count(c) AS cnt",
                {"cid": uncommitted_cid},
            )
            assert uncommitted_check[0]["cnt"] == 0, (
                "Uncommitted checkpoint leaked into persistent store"
            )

            # Verify total checkpoint count for this thread is exactly 2
            total_check = saver.proxy.execute_read(
                "MATCH (c:Checkpoint {thread_id: $tid}) RETURN count(c) AS cnt",
                {"tid": thread_id},
            )
            assert total_check[0]["cnt"] == 2

            # Verify store can continue appending without conflict
            cfg_step2: Any = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": "",
                    "checkpoint_id": "chk-valid-001",
                }
            }
            chk_2: Any = {
                "v": 1,
                "id": "chk-valid-002",
                "ts": "2026-11-12T11:00:10Z",
                "parent_checkpoint_id": "chk-valid-001",
                "channel_values": {"status": "valid_step_2_after_recovery"},
                "channel_versions": {"status": "00000003"},
                "versions_seen": {},
                "pending_sends": [],
            }
            saver.put(cfg_step2, chk_2, {"step": 2}, {})

            new_latest = saver.get_tuple(config)
            assert new_latest is not None
            assert new_latest.checkpoint["id"] == "chk-valid-002"

    def test_in_process_abrupt_proxy_close_recovery_ac_10_ac_11(self, temp_disk_db: str) -> None:
        """AC-10, AC-11: Abrupt in-process proxy shutdown without graceful close flushes WAL safely."""
        thread_id = "thread-in-process-004"

        saver1 = RelateSaver(db_path=temp_disk_db, wal_enabled=True)
        config: Any = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
        chk_0: Any = {
            "v": 1,
            "id": "chk-inproc-000",
            "ts": "2026-11-12T11:30:00Z",
            "channel_values": {"data": "step0"},
            "channel_versions": {"data": "00000001"},
            "versions_seen": {},
            "pending_sends": [],
        }
        saver1.put(config, chk_0, {"step": 0}, {})

        # Simulate sudden disconnect by aborting proxy without graceful flush
        saver1.abort()

        # Reopen with brand new saver instance
        with RelateSaver(db_path=temp_disk_db, wal_enabled=True) as saver2:
            tup = saver2.get_tuple(config)
            assert tup is not None
            assert tup.checkpoint["id"] == "chk-inproc-000"
            assert tup.checkpoint["channel_values"]["data"] == "step0"

    @pytest.mark.slow
    def test_concurrent_multithreaded_crash_recovery_ac_10_ac_11(self, temp_disk_db: str) -> None:
        """AC-10, AC-11: Multi-threaded concurrent writers recover all committed state after crash."""
        num_threads = 6
        checkpoints_per_thread = 5
        crash_code = 55

        proc = multiprocessing.Process(
            target=_worker_concurrent_threads_and_crash,
            args=(temp_disk_db, num_threads, checkpoints_per_thread, crash_code),
        )
        proc.start()
        proc.join(timeout=30.0)
        assert proc.exitcode == crash_code

        # Reopen and verify all threads recovered completely
        with RelateSaver(db_path=temp_disk_db, wal_enabled=True) as saver:
            for t_idx in range(num_threads):
                tid = f"mp-thread-{t_idx:02d}"
                cfg: Any = {"configurable": {"thread_id": tid, "checkpoint_ns": ""}}
                tup = saver.get_tuple(cfg)
                assert tup is not None, f"Failed to retrieve tuple for thread {tid}"
                assert tup.metadata["step"] == checkpoints_per_thread - 1
                assert tup.checkpoint["channel_values"]["thread"] == tid

                # Verify all steps present in reverse order
                steps = list(saver.list(cfg))
                assert len(steps) == checkpoints_per_thread
                assert [s.metadata["step"] for s in steps] == list(
                    reversed(range(checkpoints_per_thread))
                )

    def test_failed_batch_transaction_rolls_back_cleanly_ac_10(self, temp_disk_db: str) -> None:
        """AC-10: A failing transaction rolls back cleanly without leaving partial records."""
        with RelateThreadProxy(db_path=temp_disk_db, wal_enabled=True) as proxy:
            proxy.execute_write("CREATE (t:Thread {thread_id: 'clean-t-1'})")

            # Execute batch where the second statement has intentional Cypher syntax error
            failing_statements: list[tuple[str, dict[str, Any]]] = [
                ("CREATE (c:Checkpoint {checkpoint_id: 'ghost-c-1', thread_id: 'clean-t-1'})", {}),
                ("INVALID SYNTAX STATEMENT THAT FAILS PARSER", {}),
            ]
            with pytest.raises(RelateCheckpointError):
                proxy.transaction(failing_statements)

            # Assert first statement was rolled back
            res = proxy.execute_read(
                "MATCH (c:Checkpoint {checkpoint_id: 'ghost-c-1'}) RETURN count(c) AS cnt"
            )
            assert res[0]["cnt"] == 0, (
                "Partial batch write was committed despite subsequent statement failure"
            )

    def test_wal_flush_method_guarantees_durability(self, temp_disk_db: str) -> None:
        """Verify that explicit flush() synchronously commits and syncs WAL to disk."""
        with RelateSaver(db_path=temp_disk_db, wal_enabled=True) as saver:
            config: Any = {"configurable": {"thread_id": "flush-test-01", "checkpoint_ns": ""}}
            chk: Any = {
                "v": 1,
                "id": "chk-flush-01",
                "ts": "2026-11-12T12:00:00Z",
                "channel_values": {"status": "flushed"},
                "channel_versions": {"status": "00000001"},
                "versions_seen": {},
                "pending_sends": [],
            }
            saver.put(config, chk, {"step": 0}, {})
            # Explicitly call flush()
            saver.flush()

        # Reopen store and verify data was persisted
        with RelateSaver(db_path=temp_disk_db, wal_enabled=True) as reopen_saver:
            tup = reopen_saver.get_tuple({"configurable": {"thread_id": "flush-test-01"}})
            assert tup is not None
            assert tup.checkpoint["channel_values"]["status"] == "flushed"
